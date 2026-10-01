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
from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from core import db
from machines.toe_lasting import band_drift_watch, config as tl_config, db as tl_db, rul_smooth, rul_store
from machines.toe_lasting.rul_features import SUBTYPES
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


ACTIVE_CADENCES = 2  # a logged prediction counts as "still triggered" while newer than this many cadences


def _clean(value):
    """JSON-safe scalar: Timestamp -> ISO string, NaN/NaT -> None, numpy numbers -> python."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value.item() if hasattr(value, "item") else value


def _device_or_404(iddev):
    try:
        return tl_config.device(iddev)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"iddev{iddev} tidak ada di devices.toml")


@app.get("/rul/{iddev}/latest")
def rul_latest(iddev: int):
    """Read-only: newest smoothed RUL per (sensor, subtype), straight from rul_predictions.sqlite.
    Unlike /predict this does NOT score, log or send Telegram -- the pdm-predict@ timers stay the only
    writer. rul_smooth recomputes the displayed RUL/ETA from the logged raw predictions, same as
    /predict returns. `active` = newest logged reading is within ACTIVE_CADENCES x the device's
    cadence; sensor/subtype pairs that were never triggered have no row and are simply absent
    (a 'normal' status leaves no trace in the table)."""
    dev = _device_or_404(iddev)
    now = pd.Timestamp.now()  # naive local, same clock as created_at (see rul_features.data_age_minutes)
    max_age_min = ACTIVE_CADENCES * dev["cadence_s"] / 60

    results = []
    for sensor, subtype in SUBTYPES:
        recent = rul_store.get_predictions(iddev, sensor=sensor, subtype=subtype, limit=rul_smooth.SMOOTH_READINGS)
        if recent.empty:
            continue
        s = rul_smooth.smooth_rul(recent, dev["session_break_min"]).iloc[-1]
        age_min = (now - s["created_at"]).total_seconds() / 60
        results.append({
            "sensor": sensor, "subtype": subtype,
            "as_of": _clean(s["created_at"]),
            "age_minutes": round(age_min, 1),
            "active": age_min <= max_age_min,
            "current_value": _clean(s["current_value"]),
            "band_progress": _clean(s["band_progress"]),
            "predicted_rul_minutes": _clean(s["predicted_rul_minutes"]),
            "rul_smooth_minutes": _clean(s["rul_smooth_min"]),
            "eta": _clean(s["eta_hat"]), "eta_low": _clean(s["eta_low"]), "eta_high": _clean(s["eta_high"]),
            "n_used": _clean(s["n_used"]),
            "model_version": _clean(s["model_version"]),
        })
    return {"iddev": iddev, "server_time": now.isoformat(), "results": results}


@app.get("/rul/{iddev}/history")
def rul_history(iddev: int, sensor: Optional[str] = None, subtype: Optional[str] = None,
                since: Optional[datetime] = None, limit: int = Query(500, ge=1, le=5000)):
    """Read-only: the last `limit` logged predictions (oldest first), each with its smoothed RUL/ETA,
    for the dashboard's trend chart. `since` is naive local time, e.g. 2026-10-01T00:00:00. The raw
    `features` JSON is left out on purpose (bulky; read the sqlite file when tracing a jump)."""
    dev = _device_or_404(iddev)
    df = rul_store.get_predictions(iddev, sensor=sensor, subtype=subtype, since=since, limit=limit)
    if df.empty:
        return {"iddev": iddev, "rows": []}
    df = rul_smooth.smooth_history(df, dev["session_break_min"]).sort_values("created_at")
    cols = ["created_at", "sensor", "subtype", "current_value", "band_progress", "predicted_rul_minutes",
            "rul_smooth_min", "eta_hat", "eta_low", "eta_high", "n_used", "model_version"]
    rows = [{c: _clean(v) for c, v in rec.items()} for rec in df[cols].to_dict("records")]
    return {"iddev": iddev, "rows": rows}


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
