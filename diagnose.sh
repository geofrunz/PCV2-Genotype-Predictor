#!/usr/bin/env bash
set -u
APP_DIR="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
printf 'PCV2 predictor - diagnostica Linux\n'
printf 'Data: %s\n' "$(date -Is 2>/dev/null || date)"
printf 'Sistema: %s\n' "$(uname -a)"
printf 'Cartella programma: %s\n' "$APP_DIR"
printf 'PATH: %s\n' "$PATH"
printf '\nFile essenziali:\n'
for f in pcv2_predict pcv2_predict.py final_random_forest_full_dataset.joblib requirements.txt; do
    if [[ -e "$APP_DIR/$f" ]]; then
        printf '  OK      %s\n' "$f"
    else
        printf '  MANCANTE %s\n' "$f"
    fi
done
printf '\nPython disponibili:\n'
for p in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$p" >/dev/null 2>&1; then
        printf '  %s -> %s (%s)\n' "$p" "$(command -v "$p")" "$($p --version 2>&1)"
    fi
done
printf '\nAmbiente isolato:\n'
if [[ -x "$APP_DIR/.pcv2_env/bin/python" ]]; then
    "$APP_DIR/.pcv2_env/bin/python" - <<'PY'
import sys
print("  Python:", sys.version.replace("\n", " "))
for name in ("numpy", "pandas", "joblib", "sklearn", "imblearn"):
    try:
        module = __import__(name)
        print(f"  {name}: {getattr(module, '__version__', 'versione non disponibile')}")
    except Exception as exc:
        print(f"  {name}: ERRORE - {exc}")
PY
else
    printf '  Non ancora creato. Eseguire ./install.sh oppure ./pcv2_predict --setup-only\n'
fi
