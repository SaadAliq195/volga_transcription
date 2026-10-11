#!/usr/bin/env bash
# Runs every verification offline (macOS / Linux).
#   bash scripts/run_checks.sh           # everything
#   bash scripts/run_checks.sh --quick   # skip the crash test
cd "$(dirname "$0")/.."
PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python
export HF_HUB_OFFLINE=1   # prove nothing needs the network

API_ARGS=()
[ "$1" = "--quick" ] && API_ARGS=(--skip-crash)

status=0
summary=""
run() {
  local name="$1"; shift
  echo; echo "==== $name ===="
  "$PY" "$@"
  local code=$?
  if [ $code -eq 2 ]; then echo; echo "Stopped: see the message above (model or test audio missing)."; exit 2; fi
  if [ $code -eq 0 ]; then summary+="$name: all passed"$'\n'; else summary+="$name: some checks failed"$'\n'; status=1; fi
}
run "Part 1, variant 1" tests/verify.py --variant 1
run "Part 1, variant 2" tests/verify.py --variant 2
run "Part 1, variant 3" tests/verify.py --variant 3
run "Part 2, HTTP API"  tests/verify_api.py "${API_ARGS[@]}"
echo; echo "==== Summary ===="; printf "%s" "$summary"
exit $status
