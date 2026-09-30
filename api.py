"""
FastAPI wrapper around the existing RUL scoring pipeline (machines/toe_lasting/rul_predict.py).

Calling GET /predict/{iddev} does exactly what opening that device's dashboard page already does:
fetch recent sensor rows, score every (sensor, subtype), log each triggered prediction (rul_store),
and send a debounced Telegram alert on a new trigger episode (rul_notify) -- see rul_predict.
score_and_notify's docstring, which is now the ONE place this logic lives (dashboard.py calls the
same function). This file only adds an HTTP entry point so another process (the Node.js /opt/api
service, once this runs on the IoT server) can call it instead of re-implementing any of this in JS
-- that caller becomes a thin bridge: relay this response to whatever needs it (dashboard, etc.),
nothing more. Telegram credentials (TELEGRAM_BOT_TOKEN/CHAT_ID in .env) decide where alerts land --
point them at the production bot/chat once this runs on the IoT server, not the dev/test one.

Run locally for testing:
    uvicorn api:app --reload --port 8000

Then in another terminal (or a browser):
    curl http://127.0.0.1:8000/predict/2

On the IoT server this same file runs under systemd, bound to 127.0.0.1 only --
see deploy notes (not written yet, comes after this is validated locally).
"""
import math

from fastapi import FastAPI, HTTPException

from core import db
from machines.toe_lasting import band_drift_watch, config as tl_config, db as tl_db
from machines.toe_lasting.rul_predict import load_models, score_and_notify

app = FastAPI(title="PDM XGBoost RUL service")
_engine = db.get_engine()

FETCH_LIMIT = 2000  # same window rul_predict's CLI test and the dashboard use


def _serialize(results):
    out = []
    for r in results:
        r = dict(r)
        for key in ("as_of", "eta", "eta_low", "eta_high"):
            if key in r:
                r[key] = r[key].isoformat()
        for key in ("current_value", "band_progress", "predicted_rul_minutes", "rul_smooth_minutes"):
            if key in r:
                r[key] = float(r[key])
        if "n_used" in r:
            r["n_used"] = int(r["n_used"])
        out.append(r)
    return out


@app.get("/")
def root():
    return {"status": "PDM XGBoost RUL service is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict/{iddev}")
def predict(iddev: int):
    try:
        bundles = load_models(iddev)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    raw = tl_db.fetch_recent(_engine, iddev, limit=FETCH_LIMIT)
    if raw.empty:
        return {"iddev": iddev, "results": []}

    building, cell = tl_db.fetch_device_location(_engine, iddev)
    results = score_and_notify(raw, iddev, bundles, building=building, cell=cell)
    return {"iddev": iddev, "results": _serialize(results)}


@app.get("/drift/{iddev}")
def drift(iddev: int, notify: bool = True):
    """Advisory band-drift check (band_drift_watch) for every sensor of one device -- what the
    dashboard's render_drift_banner does, but callable without anyone opening the dashboard.
    Sends at most ONE Telegram message per sensor per drift episode (see
    band_drift_watch.update_drift_state_and_notify), so it's safe to call as often as you like;
    hourly is plenty since the check looks at LOOKBACK_DAYS of history. notify=false is a dry
    run: returns the result without touching drift state or Telegram."""
    try:
        cadence_s = tl_config.device(iddev)["cadence_s"]
    except KeyError:
        raise HTTPException(status_code=404, detail=f"iddev{iddev} tidak ada di devices.toml")

    limit = math.ceil(band_drift_watch.LOOKBACK_DAYS * 24 * 60 / (cadence_s / 60))
    raw = tl_db.fetch_recent(_engine, iddev, limit=limit)
    if raw.empty:
        return {"iddev": iddev, "sensors": {}}

    building, cell = tl_db.fetch_device_location(_engine, iddev)
    sensors = {}
    for sensor in band_drift_watch.SENSORS:
        result = band_drift_watch.check_band_drift(raw, iddev, sensor)
        message = None
        if notify:
            message = band_drift_watch.update_drift_state_and_notify(
                iddev, result, sensor, building=building, cell=cell)
        sensors[sensor] = {**result, "notified": message is not None}
    return {"iddev": iddev, "sensors": sensors}
