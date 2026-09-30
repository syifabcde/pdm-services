#!/usr/bin/env bash
# Run ON the server after deploy.sh:  sudo /opt/pdm-xgboost/pdm-services/deploy/install.sh
set -euo pipefail
BASE=/opt/pdm-xgboost
SVC=$BASE/pdm-services
PY="${PYTHON:-python3}"    # use the same Python minor version the models were trained with

id pdm &>/dev/null || useradd -r -s /usr/sbin/nologin pdm
[ -f "$SVC/.env" ] || { cp "$SVC/.env.example" "$SVC/.env"; echo ">> Fill in $SVC/.env (DB + Telegram), then re-run."; chown pdm: "$SVC/.env"; chmod 600 "$SVC/.env"; exit 1; }

[ -d "$SVC/venv" ] || "$PY" -m venv "$SVC/venv"
"$SVC/venv/bin/pip" install -q -r "$SVC/deploy/requirements-api.txt"
mkdir -p "$SVC/state"
chown -R pdm: "$BASE"; chmod 600 "$SVC/.env"

cp "$SVC"/deploy/*.service "$SVC"/deploy/*.timer /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now pdm-xgboost.service
sleep 3
curl -fsS http://127.0.0.1:8000/health && echo
systemctl enable --now pdm-predict-1.timer pdm-predict-2.timer pdm-drift@1.timer pdm-drift@2.timer
systemctl list-timers 'pdm-*'
