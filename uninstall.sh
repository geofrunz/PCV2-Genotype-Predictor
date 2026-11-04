#!/usr/bin/env bash
set -Eeuo pipefail
APP_DIR="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
LINK="$HOME/.local/bin/pcv2_predict"

if [[ -L "$LINK" ]]; then
    TARGET="$(readlink -f -- "$LINK" 2>/dev/null || true)"
    EXPECTED="$(readlink -f -- "$APP_DIR/pcv2_predict")"
    if [[ "$TARGET" == "$EXPECTED" ]]; then
        rm -f -- "$LINK"
        printf 'Comando rimosso: %s\n' "$LINK"
    else
        printf 'Il link %s punta a un altro programma e non è stato rimosso.\n' "$LINK"
    fi
fi

if [[ -d "$APP_DIR/.pcv2_env" ]]; then
    rm -rf -- "$APP_DIR/.pcv2_env"
    printf 'Ambiente Python isolato rimosso.\n'
fi

printf 'I file del programma restano in: %s\n' "$APP_DIR"
