#!/usr/bin/env bash
# Push code + config + models to the IoT server. Run from Git Bash on your PC:
#   ./deploy/deploy.sh user@iot-server
# Not copied on purpose: .env (create it on the server from .env.example), state/ (fresh alert
# state in prod), venv/, .git, data/ and notebooks (training only).
set -euo pipefail
HOST="${1:?usage: deploy.sh user@host}"
DEST=/opt/pdm-xgboost
cd "$(dirname "$0")/../.."          # -> pdm/

ssh "$HOST" "sudo mkdir -p $DEST && sudo chown \$USER $DEST"
tar czf - \
  --exclude='pdm-services/.env' --exclude='pdm-services/state' --exclude='pdm-services/venv' \
  --exclude='pdm-services/.git' --exclude='__pycache__' \
  pdm-services toe-lasting/config toe-lasting/models \
  | ssh "$HOST" "tar xzf - -C $DEST"
echo "Copied. Next on the server: sudo $DEST/pdm-services/deploy/install.sh"
