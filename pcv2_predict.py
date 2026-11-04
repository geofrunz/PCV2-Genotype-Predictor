#!/usr/bin/env python3
"""
PCV2 genotype prediction from FASTA/FASTQ files.

Typical use:
    pcv2_predict sample.fasta
    pcv2_predict sample.fastq
    pcv2_predict folder_with_sequences
    pcv2_predict input.fasta -o results.csv

The validated joblib model is automatically searched next to this script.
Missing Python packages are installed automatically only when necessary.
On Linux, the pcv2_predict launcher creates and uses an isolated virtual environment.
"""

from __future__ import annotations

# =============================================================================
# BOOTSTRAP: standard-library imports only
# =============================================================================

import argparse
import csv
import gzip
import importlib
import math
import os
import re
import subprocess
import sys
import warnings
from collections import Counter
from pathlib import Path

APP_NAME = "PCV2 genotype predictor"
EXPECTED_SKLEARN_VERSION = "1.9.0"

DEPENDENCIES = {
    "numpy": "numpy>=1.26",
    "pandas": "pandas>=2.0",
    "joblib": "joblib>=1.3",
    "sklearn": f"scikit-learn=={EXPECTED_SKLEARN_VERSION}",
    "imblearn": "imbalanced-learn>=0.14,<0.15",
}


def running_in_virtualenv() -> bool:
    return (
        getattr(sys, "base_prefix", sys.prefix) != sys.prefix
        or hasattr(sys, "real_prefix")
        or os.environ.get("CONDA_PREFIX") is not None
    )


def ensure_dependencies() -> None:
    """Install only missing imports, then restart the current command."""
    missing_packages: list[str] = []
    for import_name, pip_name in DEPENDENCIES.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing_packages.append(pip_name)

    if not missing_packages:
        return

    if "--no-install" in sys.argv:
        raise RuntimeError(
            "Pacchetti Python mancanti: " + ", ".join(missing_packages)
        )

    print("Pacchetti mancanti rilevati. Installazione automatica:")
    for package in missing_packages:
        print(f"  - {package}")

    command = [sys.executable, "-m", "pip", "install"]
    if not running_in_virtualenv():
        command.append("--user")
    command.extend(missing_packages)

    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Installazione automatica non riuscita. Comando eseguito:\n"
            + " ".join(command)
        ) from exc

    print("Installazione completata. Riavvio del comando...\n")
    os.execv(sys.executable, [sys.executable, *sys.argv])


ensure_dependencies()

# =============================================================================
# Third-party imports
# =============================================================================

import joblib
import numpy as np
import pandas as pd
import sklearn
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


# =============================================================================
# Defaults and supported formats
# =============================================================================

ALPHABET = "ACGT"
IUPAC_DNA = set("ACGTRYSWKMBDHVN")
GAP_CHARS = {"-", "."}

FASTA_EXTENSIONS = {".fa", ".fasta", ".fas", ".fna", ".ffn"}
FASTQ_EXTENSIONS = {".fq", ".fastq"}
TEXT_EXTENSIONS = {".txt"}
COMPRESSED_EXTENSION = ".gz"

EXPECTED_ORF2_MIN_LENGTH = 650
EXPECTED_ORF2_MAX_LENGTH = 750
MIN_LENGTH_FOR_RELIABLE_CALL = 600
MAX_AMBIGUOUS_FRACTION = 0.10
MIN_VALID_KMERS = 50
MIN_VALID_KMER_FRACTION = 0.70
MIN_MEAN_PHRED = 20.0

DEFAULT_ACCEPT_PROBABILITY = 0.80
DEFAULT_ACCEPT_MARGIN = 0.10
DEFAULT_STRONG_PROBABILITY = 0.90
DEFAULT_STRONG_MARGIN = 0.20
DEFAULT_BATCH_SIZE = 500

MODEL_FILENAMES = (
    "final_random_forest_full_dataset.joblib",
    "final_random_forest_full_dataset(1).joblib",
)


# =============================================================================
# Custom classes required by the serialized model
# =============================================================================

class KmerVectorizer(BaseEstimator, TransformerMixin):
    """Convert nucleotide sequences to binary k-mers or relative frequencies."""

    def __init__(
        self,
        k=6,
        alphabet="ACGT",
        skip_ambiguous=True,
        binary=True,
        use_tfidf=False,
        min_df=1,
    ):
        self.k = k
        self.alphabet = alphabet
        self.skip_ambiguous = skip_ambiguous
        self.binary = binary
        self.use_tfidf = use_tfidf
        self.min_df = min_df

    def _tokenize(self, sequence):
        sequence = str(sequence).upper()
        allowed = set(self.alphabet)
        tokens = []
        if len(sequence) < self.k:
            return tokens
        for i in range(len(sequence) - self.k + 1):
            kmer = sequence[i:i + self.k]
            if self.skip_ambiguous and any(base not in allowed for base in kmer):
                continue
            tokens.append(kmer)
        return tokens

    @staticmethod
    def _counts_to_relative_frequencies(X_counts):
        X_freq = X_counts.astype(float).copy()
        row_sums = np.asarray(X_freq.sum(axis=1)).ravel()
        nonzero = row_sums > 0
        if np.any(nonzero):
            inverse = np.zeros_like(row_sums, dtype=float)
            inverse[nonzero] = 1.0 / row_sums[nonzero]
            X_freq = X_freq.multiply(inverse[:, np.newaxis])
        return X_freq.tocsr()

    def _make_base_matrix(self, X_counts):
        if self.binary:
            return X_counts.astype(float)
        return self._counts_to_relative_frequencies(X_counts)

    def fit(self, X, y=None):
        self.vectorizer_ = CountVectorizer(
            analyzer=self._tokenize,
            lowercase=False,
            binary=self.binary,
            min_df=self.min_df,
        )
        X_counts = self.vectorizer_.fit_transform(X)
        X_base = self._make_base_matrix(X_counts)
        if self.use_tfidf:
            self.tfidf_ = TfidfTransformer()
            self.tfidf_.fit(X_base)
        else:
            self.tfidf_ = None
        return self

    def transform(self, X):
        X_counts = self.vectorizer_.transform(X)
        X_base = self._make_base_matrix(X_counts)
        if self.use_tfidf:
            return self.tfidf_.transform(X_base)
        return X_base

    def get_feature_names_out(self):
        return self.vectorizer_.get_feature_names_out()


class SafeSMOTE(BaseEstimator):
    """Compatibility class for the training pipeline; unused during prediction."""

    def __init__(self, coefficient=0.0, k_neighbors=5, random_state=123):
        self.coefficient = coefficient
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit_resample(self, X, y):
        coefficient = float(self.coefficient)
        if coefficient <= 0:
            return X, y
        counts = Counter(y)
        max_n = max(counts.values())
        strategy = {}
        for cls, n in counts.items():
            if n >= max_n or n <= self.k_neighbors:
                continue
            target_n = int(np.ceil(n + coefficient * (max_n - n)))
            if target_n > n:
                strategy[cls] = target_n
        if not strategy:
            return X, y
        effective_k = min(
            self.k_neighbors,
            min(counts[cls] - 1 for cls in strategy),
        )
        sampler = SMOTE(
            sampling_strategy=strategy,
            k_neighbors=effective_k,
            random_state=self.random_state,
        )
        return sampler.fit_resample(X, y)


# =============================================================================
# Input discovery and parsing
# =============================================================================


def application_directory() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def natural_key(text: str):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", text)]


def open_text(path: Path):
    if path.suffix.lower() == COMPRESSED_EXTENSION:
        return gzip.open(path, "rt", encoding="utf-8-sig", errors="replace")
    return path.open("r", encoding="utf-8-sig", errors="replace")


def logical_suffix(path: Path) -> str:
    if path.suffix.lower() == COMPRESSED_EXTENSION:
        return Path(path.stem).suffix.lower()
    return path.suffix.lower()


def detect_sequence_format(path: Path) -> str:
    suffix = logical_suffix(path)
    if suffix in FASTA_EXTENSIONS:
        return "FASTA"
    if suffix in FASTQ_EXTENSIONS:
        return "FASTQ"

    with open_text(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                return "FASTA"
            if line.startswith("@"):
                return "FASTQ"
            break
    raise ValueError(
        f"Formato non riconosciuto per {path}. Il file deve essere FASTA o FASTQ."
    )


def is_candidate_sequence_file(path: Path) -> bool:
    suffix = logical_suffix(path)
    return suffix in FASTA_EXTENSIONS | FASTQ_EXTENSIONS | TEXT_EXTENSIONS


def expand_input_argument(value: str) -> list[Path]:
    if any(char in value for char in "*?["):
        import glob
        matches = [Path(item) for item in glob.glob(value, recursive=True)]
        return matches
    return [Path(value).expanduser()]


def discover_input_files(input_values: list[str], recursive: bool) -> list[Path]:
    discovered: list[Path] = []
    for value in input_values:
        for input_path in expand_input_argument(value):
            if not input_path.exists():
                raise FileNotFoundError(f"Input non trovato: {input_path}")
            if input_path.is_file():
                discovered.append(input_path)
                continue
            iterator = input_path.rglob("*") if recursive else input_path.glob("*")
            discovered.extend(
                path for path in iterator
                if path.is_file() and is_candidate_sequence_file(path)
            )

    unique = {str(path.resolve()): path.resolve() for path in discovered}
    files = sorted(unique.values(), key=lambda item: natural_key(str(item)))
    if not files:
        raise FileNotFoundError("Nessun file FASTA/FASTQ trovato negli input indicati.")
    return files


def read_fasta(path: Path):
    header = None
    sequence_parts: list[str] = []
    record_number = 0
    with open_text(path) as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    record_number += 1
                    yield record_number, header, "".join(sequence_parts), None
                header = line[1:].strip()
                sequence_parts = []
            else:
                if header is None:
                    raise ValueError(
                        f"FASTA non valido, riga {line_number}: manca l'header '>'."
                    )
                sequence_parts.append(re.sub(r"\s+", "", line))
    if header is not None:
        record_number += 1
        yield record_number, header, "".join(sequence_parts), None
    if record_number == 0:
        raise ValueError("Nessuna sequenza FASTA trovata.")


def read_fastq(path: Path):
    """Read standard or wrapped FASTQ records."""
    record_number = 0
    with open_text(path) as handle:
        while True:
            header_line = handle.readline()
            while header_line and not header_line.strip():
                header_line = handle.readline()
            if not header_line:
                break
            if not header_line.startswith("@"):
                raise ValueError(
                    f"FASTQ non valido: atteso header '@', trovato {header_line[:30]!r}."
                )
            header = header_line[1:].strip()

            sequence_parts: list[str] = []
            while True:
                line = handle.readline()
                if not line:
                    raise ValueError("FASTQ troncato prima della riga '+'.")
                if line.startswith("+"):
                    break
                sequence_parts.append(re.sub(r"\s+", "", line))
            sequence = "".join(sequence_parts)
            if not sequence:
                raise ValueError("Record FASTQ con sequenza vuota.")

            quality_parts: list[str] = []
            quality_length = 0
            while quality_length < len(sequence):
                line = handle.readline()
                if not line:
                    raise ValueError("FASTQ troncato nella stringa di qualità.")
                quality = line.rstrip("\r\n")
                quality_parts.append(quality)
                quality_length += len(quality)
            quality_string = "".join(quality_parts)
            if len(quality_string) != len(sequence):
                raise ValueError(
                    "Lunghezza qualità FASTQ diversa dalla lunghezza della sequenza."
                )

            record_number += 1
            yield record_number, header, sequence, quality_string

    if record_number == 0:
        raise ValueError("Nessun record FASTQ trovato.")


def read_sequence_file(path: Path):
    sequence_format = detect_sequence_format(path)
    reader = read_fasta if sequence_format == "FASTA" else read_fastq
    for record_number, header, sequence, quality in reader(path):
        yield sequence_format, record_number, header, sequence, quality


# =============================================================================
# Sequence metrics
# =============================================================================


def clean_sequence(raw_sequence: str) -> str:
    sequence = re.sub(r"\s+", "", str(raw_sequence)).upper().replace("U", "T")
    return "".join(base for base in sequence if base not in GAP_CHARS)


def reverse_complement(sequence: str) -> str:
    complement = str.maketrans(
        "ACGTRYSWKMBDHVN",
        "TGCAYRSWMKVHDBN",
    )
    return sequence.translate(complement)[::-1]


def count_valid_kmers(sequence: str, k: int) -> int:
    if len(sequence) < k:
        return 0
    allowed = set(ALPHABET)
    return sum(
        all(base in allowed for base in sequence[index:index + k])
        for index in range(len(sequence) - k + 1)
    )


def phred_metrics(quality_string: str | None) -> tuple[float, int] | tuple[None, None]:
    if quality_string is None:
        return None, None
    scores = [ord(character) - 33 for character in quality_string]
    if not scores:
        return None, None
    return float(np.mean(scores)), int(min(scores))


def normalized_entropy(probabilities: np.ndarray) -> float:
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities[probabilities > 0]
    if probabilities.size <= 1:
        return 0.0
    entropy = -np.sum(probabilities * np.log(probabilities))
    return float(entropy / math.log(len(probabilities)))


def sequence_metadata(
    source_file: Path,
    input_format: str,
    record_number: int,
    header: str,
    raw_sequence: str,
    quality_string: str | None,
    kmer_size: int,
) -> dict:
    clean = clean_sequence(raw_sequence)
    raw_no_whitespace = re.sub(r"\s+", "", str(raw_sequence)).upper()
    sequence_id = header.split()[0] if header.strip() else f"record_{record_number}"
    acgt_count = sum(base in ALPHABET for base in clean)
    ambiguous_count = sum(base not in ALPHABET for base in clean)
    iupac_ambiguous_count = sum(
        base in IUPAC_DNA and base not in ALPHABET for base in clean
    )
    invalid_symbol_count = sum(base not in IUPAC_DNA for base in clean)
    gap_count = sum(base in GAP_CHARS for base in raw_no_whitespace)
    possible_kmers = max(0, len(clean) - kmer_size + 1)
    valid_kmers = count_valid_kmers(clean, kmer_size)
    mean_phred, min_phred = phred_metrics(quality_string)

    return {
        "source_file": source_file.name,
        "source_path": str(source_file.resolve()),
        "input_format": input_format,
        "record_number": record_number,
        "sequence_id": sequence_id,
        "full_header": header,
        "raw_length_including_gaps": len(raw_no_whitespace),
        "ungapped_length": len(clean),
        "gap_count": gap_count,
        "acgt_count": acgt_count,
        "ambiguous_count": ambiguous_count,
        "iupac_ambiguous_count": iupac_ambiguous_count,
        "invalid_symbol_count": invalid_symbol_count,
        "ambiguous_fraction": ambiguous_count / len(clean) if clean else np.nan,
        "gc_fraction_acgt_only": (
            (clean.count("G") + clean.count("C")) / acgt_count
            if acgt_count else np.nan
        ),
        "possible_kmers": possible_kmers,
        "valid_kmers": valid_kmers,
        "valid_kmer_fraction": valid_kmers / possible_kmers if possible_kmers else 0.0,
        "mean_phred": mean_phred,
        "minimum_phred": min_phred,
        "clean_sequence": clean,
    }


# =============================================================================
# Model loading and prediction
# =============================================================================


def find_model(explicit_model: str | None) -> Path:
    if explicit_model:
        model_path = Path(explicit_model).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Modello non trovato: {model_path}")
        return model_path

    search_directories = [application_directory(), Path.cwd()]
    for directory in search_directories:
        for filename in MODEL_FILENAMES:
            candidate = directory / filename
            if candidate.exists():
                return candidate.resolve()

    joblibs = []
    for directory in search_directories:
        joblibs.extend(directory.glob("*.joblib"))
    unique = sorted({path.resolve() for path in joblibs})
    if len(unique) == 1:
        return unique[0]

    raise FileNotFoundError(
        "Modello .joblib non trovato. Collocare "
        "final_random_forest_full_dataset.joblib nella stessa cartella del programma "
        "oppure usare --model PERCORSO."
    )


def load_and_validate_model(model_path: Path):
    if sklearn.__version__ != EXPECTED_SKLEARN_VERSION:
        warnings.warn(
            f"Il modello è stato salvato con scikit-learn {EXPECTED_SKLEARN_VERSION}, "
            f"ma l'ambiente usa {sklearn.__version__}. Il launcher Linux pcv2_predict "
            "crea automaticamente un ambiente isolato con la versione corretta."
        )

    model = joblib.load(model_path)
    if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
        raise TypeError("Il joblib non contiene il classificatore PCV2 atteso.")

    classes = np.asarray(model.classes_, dtype=str)
    if classes.size < 2:
        raise ValueError("Il modello contiene meno di due classi.")

    kmer_size = 6
    if hasattr(model, "named_steps") and "features" in model.named_steps:
        features = model.named_steps["features"]
        kmer_size = int(getattr(features, "k", 6))

    if hasattr(model, "named_steps") and "clf" in model.named_steps:
        classifier = model.named_steps["clf"]
        if hasattr(classifier, "n_jobs"):
            classifier.n_jobs = -1

    return model, classes, kmer_size


def predict_probabilities(model, sequences: list[str], batch_size: int) -> np.ndarray:
    chunks = []
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start:start + batch_size]
        chunks.append(np.asarray(model.predict_proba(batch), dtype=float))
    return np.vstack(chunks)


def choose_orientation(
    forward_probabilities: np.ndarray,
    reverse_probabilities: np.ndarray | None,
):
    forward_max = forward_probabilities.max(axis=1)
    if reverse_probabilities is None:
        return (
            forward_probabilities,
            ["forward"] * len(forward_probabilities),
            forward_max,
            np.full(len(forward_probabilities), np.nan),
        )

    reverse_max = reverse_probabilities.max(axis=1)
    use_reverse = reverse_max > forward_max
    selected = forward_probabilities.copy()
    selected[use_reverse] = reverse_probabilities[use_reverse]
    orientations = np.where(use_reverse, "reverse_complement", "forward").tolist()
    return selected, orientations, forward_max, reverse_max


def quality_flags(row: dict) -> list[str]:
    flags: list[str] = []
    length = int(row["ungapped_length"])
    if length == 0:
        flags.append("empty_sequence")
    if length < MIN_LENGTH_FOR_RELIABLE_CALL:
        flags.append("partial_or_short_sequence")
    if not (EXPECTED_ORF2_MIN_LENGTH <= length <= EXPECTED_ORF2_MAX_LENGTH):
        flags.append("outside_expected_ORF2_length")
    if row["ambiguous_fraction"] > MAX_AMBIGUOUS_FRACTION:
        flags.append("high_ambiguity")
    if row["valid_kmers"] < MIN_VALID_KMERS:
        flags.append("too_few_valid_kmers")
    if row["valid_kmer_fraction"] < MIN_VALID_KMER_FRACTION:
        flags.append("low_valid_kmer_fraction")
    if row["invalid_symbol_count"] > 0:
        flags.append("non_IUPAC_symbols")
    if row.get("mean_phred") is not None and not pd.isna(row.get("mean_phred")):
        if float(row["mean_phred"]) < MIN_MEAN_PHRED:
            flags.append("low_mean_FASTQ_quality")
    if row.get("orientation_used") == "reverse_complement":
        flags.append("reverse_complement_selected")
    return flags


def interpret_prediction(
    row: dict,
    flags: list[str],
    accept_probability: float,
    accept_margin: float,
    strong_probability: float,
    strong_margin: float,
) -> tuple[str, str]:
    fatal_flags = {"empty_sequence", "too_few_valid_kmers"}
    warning_flags = {
        "partial_or_short_sequence",
        "outside_expected_ORF2_length",
        "high_ambiguity",
        "low_valid_kmer_fraction",
        "non_IUPAC_symbols",
        "low_mean_FASTQ_quality",
    }

    if any(flag in fatal_flags for flag in flags):
        return "NOT_EVALUABLE", "not_evaluable_insufficient_sequence_information"

    top_probability = float(row["top_probability"])
    margin = float(row["probability_margin"])
    accepted = top_probability >= accept_probability and margin >= accept_margin
    strong = top_probability >= strong_probability and margin >= strong_margin
    has_warning = any(flag in warning_flags for flag in flags)

    if not accepted:
        return "AMBIGUOUS", "weak_or_ambiguous_model_support"

    predicted = str(row["predicted_genotype_raw"])
    if strong and not has_warning:
        return predicted, "strong_model_support"
    if strong and has_warning:
        return predicted, "strong_support_with_sequence_quality_warning"
    if has_warning:
        return predicted, "moderate_support_with_sequence_quality_warning"
    return predicted, "moderate_model_support"


def build_prediction_table(
    model,
    records: list[dict],
    classes: np.ndarray,
    auto_orient: bool,
    batch_size: int,
    accept_probability: float,
    accept_margin: float,
    strong_probability: float,
    strong_margin: float,
    save_all_probabilities: bool,
) -> pd.DataFrame:
    sequences = [record["clean_sequence"] for record in records]
    forward_probabilities = predict_probabilities(model, sequences, batch_size)

    reverse_probabilities = None
    if auto_orient:
        reverse_sequences = [reverse_complement(sequence) for sequence in sequences]
        reverse_probabilities = predict_probabilities(model, reverse_sequences, batch_size)

    selected, orientations, forward_max, reverse_max = choose_orientation(
        forward_probabilities,
        reverse_probabilities,
    )

    output_rows: list[dict] = []
    for index, metadata in enumerate(records):
        probabilities = selected[index]
        order = np.argsort(probabilities)[::-1]
        top_index = int(order[0])
        second_index = int(order[1])

        row = {key: value for key, value in metadata.items() if key != "clean_sequence"}
        row.update({
            "orientation_used": orientations[index],
            "forward_top_probability": float(forward_max[index]),
            "reverse_top_probability": (
                float(reverse_max[index]) if not np.isnan(reverse_max[index]) else np.nan
            ),
            "orientation_probability_gain": (
                float(reverse_max[index] - forward_max[index])
                if orientations[index] == "reverse_complement" else 0.0
            ),
            "predicted_genotype_raw": str(classes[top_index]),
            "top_probability": float(probabilities[top_index]),
            "second_genotype": str(classes[second_index]),
            "second_probability": float(probabilities[second_index]),
            "probability_margin": float(
                probabilities[top_index] - probabilities[second_index]
            ),
            "top2_probability_sum": float(
                probabilities[top_index] + probabilities[second_index]
            ),
            "normalized_probability_entropy": normalized_entropy(probabilities),
        })

        flags = quality_flags(row)
        reported, interpretation = interpret_prediction(
            row=row,
            flags=flags,
            accept_probability=accept_probability,
            accept_margin=accept_margin,
            strong_probability=strong_probability,
            strong_margin=strong_margin,
        )
        row["reported_genotype"] = reported
        row["interpretation"] = interpretation
        row["quality_flags"] = ";".join(flags) if flags else "none"

        if save_all_probabilities:
            for class_index, class_name in enumerate(classes):
                row[f"probability_{class_name}"] = float(probabilities[class_index])

        output_rows.append(row)

    result = pd.DataFrame(output_rows)
    preferred_columns = [
        "source_file", "input_format", "record_number", "sequence_id", "full_header",
        "raw_length_including_gaps", "ungapped_length", "gap_count", "acgt_count",
        "ambiguous_count", "ambiguous_fraction", "valid_kmers", "possible_kmers",
        "valid_kmer_fraction", "mean_phred", "minimum_phred", "orientation_used",
        "predicted_genotype_raw", "reported_genotype", "top_probability",
        "second_genotype", "second_probability", "probability_margin",
        "normalized_probability_entropy", "interpretation", "quality_flags",
        "gc_fraction_acgt_only", "iupac_ambiguous_count", "invalid_symbol_count",
        "forward_top_probability", "reverse_top_probability",
        "orientation_probability_gain", "top2_probability_sum", "source_path",
    ]
    probability_columns = [
        f"probability_{class_name}" for class_name in classes
        if f"probability_{class_name}" in result.columns
    ]
    remaining_columns = [
        column for column in result.columns
        if column not in preferred_columns and column not in probability_columns
    ]
    return result[preferred_columns + probability_columns + remaining_columns]


# =============================================================================
# CLI
# =============================================================================


def valid_probability(value: str) -> float:
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("Il valore deve essere compreso tra 0 e 1.")
    return number


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="pcv2_predict",
        description=(
            "Predice il genotipo PCV2 da FASTA/FASTQ usando il modello validato. "
            "Il modello viene cercato automaticamente accanto al programma."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Esempi:\n"
            "  pcv2_predict campione.fasta\n"
            "  pcv2_predict campione.fastq -o risultati.csv\n"
            "  pcv2_predict cartella_sequenze\n"
            "  pcv2_predict file1.fasta file2.fasta\n\n"
            "Nota: un FASTQ di reads grezze non è equivalente a una sequenza ORF2 "
            "consensus. Il modello è stato validato su sequenze ORF2."
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Uno o più file FASTA/FASTQ, cartelle o wildcard.",
    )
    parser.add_argument("-o", "--output", help="CSV di output.")
    parser.add_argument("-m", "--model", help="Percorso alternativo del modello joblib.")
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Non cercare nelle sottocartelle.",
    )
    parser.add_argument(
        "--no-auto-orient",
        action="store_true",
        help="Non confrontare il reverse complement.",
    )
    parser.add_argument(
        "--no-all-probabilities",
        action="store_true",
        help="Non salvare una colonna per ogni genotipo.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Sequenze per batch (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--accept-probability",
        type=valid_probability,
        default=DEFAULT_ACCEPT_PROBABILITY,
        help=f"Probabilità minima per accettare la chiamata (default: {DEFAULT_ACCEPT_PROBABILITY}).",
    )
    parser.add_argument(
        "--accept-margin",
        type=valid_probability,
        default=DEFAULT_ACCEPT_MARGIN,
        help=f"Margine minimo sulla seconda classe (default: {DEFAULT_ACCEPT_MARGIN}).",
    )
    parser.add_argument(
        "--strong-probability",
        type=valid_probability,
        default=DEFAULT_STRONG_PROBABILITY,
        help=f"Probabilità per supporto forte (default: {DEFAULT_STRONG_PROBABILITY}).",
    )
    parser.add_argument(
        "--strong-margin",
        type=valid_probability,
        default=DEFAULT_STRONG_MARGIN,
        help=f"Margine per supporto forte (default: {DEFAULT_STRONG_MARGIN}).",
    )
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="Non installare automaticamente eventuali librerie mancanti.",
    )
    return parser.parse_args()


def default_output_path(inputs: list[str]) -> Path:
    if len(inputs) == 1 and not any(character in inputs[0] for character in "*?["):
        input_path = Path(inputs[0]).expanduser()
        if input_path.exists() and input_path.is_file():
            stem = input_path.name
            if stem.lower().endswith(".gz"):
                stem = Path(stem).stem
            stem = Path(stem).stem
            return input_path.resolve().with_name(
                f"{stem}_PCV2_genotype_predictions.csv"
            )
        if input_path.exists() and input_path.is_dir():
            return input_path.resolve() / "PCV2_genotype_predictions.csv"
    return Path.cwd() / "PCV2_genotype_predictions.csv"


def main() -> int:
    args = parse_arguments()
    if args.batch_size < 1:
        raise ValueError("--batch-size deve essere almeno 1.")

    model_path = find_model(args.model)
    input_files = discover_input_files(args.inputs, recursive=not args.non_recursive)
    output_path = Path(args.output).expanduser().resolve() if args.output else default_output_path(args.inputs)

    print(f"{APP_NAME}")
    print(f"Modello: {model_path}")
    print(f"File di input: {len(input_files)}")

    model, classes, kmer_size = load_and_validate_model(model_path)
    print("Classi:", ", ".join(classes))
    print(f"k-mer: {kmer_size}")

    all_records: list[dict] = []
    file_errors: list[dict] = []
    format_counts = Counter()

    for input_file in input_files:
        file_record_count = 0
        try:
            for input_format, record_number, header, sequence, quality in read_sequence_file(input_file):
                all_records.append(
                    sequence_metadata(
                        source_file=input_file,
                        input_format=input_format,
                        record_number=record_number,
                        header=header,
                        raw_sequence=sequence,
                        quality_string=quality,
                        kmer_size=kmer_size,
                    )
                )
                file_record_count += 1
                format_counts[input_format] += 1
            print(f"  {input_file.name}: {file_record_count} sequenze")
        except Exception as exc:
            warnings.warn(f"File saltato: {input_file} ({exc})")
            file_errors.append({"source_path": str(input_file), "error": str(exc)})

    if not all_records:
        raise RuntimeError("Nessuna sequenza valida disponibile per la predizione.")

    if format_counts["FASTQ"] > 1000:
        warnings.warn(
            "Il FASTQ contiene oltre 1000 record. Se si tratta di reads grezze, "
            "il modello non deve essere usato direttamente: produrre prima una "
            "sequenza consensus ORF2."
        )

    print(f"Predizione di {len(all_records)} sequenze...")
    result = build_prediction_table(
        model=model,
        records=all_records,
        classes=classes,
        auto_orient=not args.no_auto_orient,
        batch_size=args.batch_size,
        accept_probability=args.accept_probability,
        accept_margin=args.accept_margin,
        strong_probability=args.strong_probability,
        strong_margin=args.strong_margin,
        save_all_probabilities=not args.no_all_probabilities,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
        quoting=csv.QUOTE_MINIMAL,
    )

    print("\nPredizione completata.")
    print(f"CSV: {output_path}")
    print("\nRiepilogo:")
    print(result["reported_genotype"].value_counts(dropna=False).to_string())

    if file_errors:
        error_path = output_path.with_name(f"{output_path.stem}_file_errors.csv")
        pd.DataFrame(file_errors).to_csv(error_path, index=False, encoding="utf-8-sig")
        print(f"File con errori: {error_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nOperazione interrotta dall'utente.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nERRORE: {exc}", file=sys.stderr)
        raise SystemExit(1)
