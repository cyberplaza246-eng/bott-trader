#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PY="$ROOT_DIR/.venv/bin/python"
LOG_FILE="$ROOT_DIR/logs/bot_supervisor.log"

mkdir -p "$ROOT_DIR/logs"

if [[ ! -x "$VENV_PY" ]]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Python venv not found at $VENV_PY" | tee -a "$LOG_FILE"
  exit 1
fi

cd "$ROOT_DIR" || exit 1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting bot supervisor loop" | tee -a "$LOG_FILE"

while true; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching bot..." | tee -a "$LOG_FILE"
  "$VENV_PY" -m scripts.run_bot >> "$LOG_FILE" 2>&1
  exit_code=$?
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Bot exited with code $exit_code. Restarting in 5s..." | tee -a "$LOG_FILE"
  sleep 5
done
