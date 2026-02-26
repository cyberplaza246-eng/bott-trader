#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_NAME="ai-bot"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}.service"
TEMPLATE_PATH="$ROOT_DIR/deploy/systemd/ai-bot.service.template"
ENV_FILE="${ROOT_DIR}/.env"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
RUN_USER="${SUDO_USER:-$(whoami)}"

if [[ ! -f "$TEMPLATE_PATH" ]]; then
  echo "Template not found: $TEMPLATE_PATH"
  exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing .env file at $ENV_FILE"
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing virtualenv Python at $PYTHON_BIN"
  exit 1
fi

TMP_SERVICE="$(mktemp)"
cp "$TEMPLATE_PATH" "$TMP_SERVICE"

sed -i "s|{{USER}}|${RUN_USER}|g" "$TMP_SERVICE"
sed -i "s|{{WORKDIR}}|${ROOT_DIR}|g" "$TMP_SERVICE"
sed -i "s|{{ENV_FILE}}|${ENV_FILE}|g" "$TMP_SERVICE"
sed -i "s|{{PYTHON_BIN}}|${PYTHON_BIN}|g" "$TMP_SERVICE"

echo "Installing systemd service to ${SERVICE_PATH}"
sudo mkdir -p "${ROOT_DIR}/logs"
sudo cp "$TMP_SERVICE" "$SERVICE_PATH"
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl restart "$SERVICE_NAME"

echo "Service installed and restarted."
echo "Check status: sudo systemctl status ${SERVICE_NAME}"
echo "Follow logs:   sudo journalctl -u ${SERVICE_NAME} -f"

rm -f "$TMP_SERVICE"
