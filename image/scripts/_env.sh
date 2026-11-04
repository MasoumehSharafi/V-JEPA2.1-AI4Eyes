# Sourced by every script: cd to image/, load image/.env, pick the interpreter, define `step`.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [ -f .env ]; then set -a; . ./.env; set +a; fi
PY=${PY:-python}
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
step() { echo "=== $(date '+%F %H:%M:%S') $*"; }
