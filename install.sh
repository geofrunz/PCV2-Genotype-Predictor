#!/usr/bin/env bash
# Install the pcv2_predict command for the current Linux user.
set -Eeuo pipefail

APP_DIR="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
LAUNCHER="$APP_DIR/pcv2_predict"
USER_BIN="$HOME/.local/bin"
COMMAND_LINK="$USER_BIN/pcv2_predict"
PROFILE_FILE="$HOME/.profile"
MARKER_BEGIN="# >>> PCV2 predictor PATH >>>"
MARKER_END="# <<< PCV2 predictor PATH <<<"

chmod +x "$LAUNCHER" "$APP_DIR/install.sh" "$APP_DIR/uninstall.sh" "$APP_DIR/diagnose.sh"
mkdir -p "$USER_BIN"
ln -sfn "$LAUNCHER" "$COMMAND_LINK"

case ":$PATH:" in
    *":$USER_BIN:"*) ;;
    *)
        if ! grep -Fq "$MARKER_BEGIN" "$PROFILE_FILE" 2>/dev/null; then
            {
                printf '\n%s\n' "$MARKER_BEGIN"
                printf 'export PATH="$HOME/.local/bin:$PATH"\n'
                printf '%s\n' "$MARKER_END"
            } >> "$PROFILE_FILE"
        fi
        ;;
esac

printf 'Configuro l’ambiente Python del programma...\n'
"$LAUNCHER" --setup-only

printf '\nInstallazione completata.\n'
printf 'Comando installato: %s\n' "$COMMAND_LINK"
printf '\nPer usarlo subito in questo terminale eseguire:\n'
printf '  export PATH="$HOME/.local/bin:$PATH"\n'
printf '\nPoi, ad esempio:\n'
printf '  pcv2_predict campione.fasta\n'
printf '  pcv2_predict campione.fastq -o risultati.csv\n'
