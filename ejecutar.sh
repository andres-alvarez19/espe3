#!/usr/bin/env bash
set -euo pipefail

trap 'unset API_KEY' EXIT

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then PYTHON="$PROJECT_ROOT/.venv/bin/python"; else PYTHON="python3"; fi

case "${1:-}" in
  --audit|--new-run|--dry-run) exec "$PYTHON" "$PROJECT_ROOT/agente.py" "$1" ;;
  "") exec "$PYTHON" "$PROJECT_ROOT/agente.py" ;;
  *) echo "Uso: ./ejecutar.sh [--audit|--new-run|--dry-run]" >&2; exit 2 ;;
esac
