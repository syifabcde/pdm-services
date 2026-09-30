"""
Live RUL scoring -- local test against the real Postgres pdm_tl_log table.

Mirrors predict.py's shape (load model bundle, fetch recent rows, compute
features, predict, print) but for the RUL model instead of the temperature
forecasting models.

UPDATED 2026-09-22: there is now ONE pooled model (xgb_pooled_v2.joblib,
sensor+subtype as categorical features) covering all 5 gradual subtypes in
the new strict-PDM-Category taxonomy, replacing the old 4 separate
per-subtype .joblib files. load_models() still returns a dict keyed by
(sensor, subtype) -- every key just maps to the SAME bundle now -- so
dashboard.py / rul_features.score_all_sessions() didn't need to change their
`bundles[key]` call pattern. The old 4 per-subtype .joblib files are left in
toe-lasting/models/rul/ (not deleted) but are no longer loaded by anything.

UPDATED 2026-09-24: one model per device, models/rul/iddevN/xgb_pooled_v2.joblib, chosen
by load_models(iddev). Each device's model was trained on that device's own bands
(config/bands.toml), so scoring a device with another device's model would give
confident-looking but wrong minutes -- a device with no model raises instead of
falling back (unlike rul_bands.py's band fallback, which is only a placeholder
threshold, not a prediction).

This is a read-only local test: it queries the real sensor log but does not
write anything back to the IoT database, and does not yet persist its own
predictions anywhere (predictions_store.py-style logging can be added once
this is validated against live data).

Usage (from pdm-services/):
    python -m machines.toe_lasting.rul_predict --iddev 2
"""
import argparse
import hashlib
import io
import os

import joblib

from . import config, db, rul_notify, rul_smooth, rul_store
from .rul_features import (
    ALL_SENSORS,
    ALL_SUBTYPES,
    SUBTYPES,
    feature_cols,
    latest_rul_inputs,
    session_break_minutes,
)


def model_path(iddev):
    return os.path.join(config.RUL_MODEL_DIR, f"iddev{iddev}", "xgb_pooled_v2.joblib")


def load_models(iddev):
    path = model_path(iddev)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No RUL model for iddev{iddev} (expected {path}). Train one with "
            f"toe-lasting/pipeline/save_final_rul_model.py --iddev {iddev}."
        )
    # read the bytes once so the version hash is of exactly the model that gets loaded
    with open(path, "rb") as f:
        data = f.read()
    bundle = joblib.load(io.BytesIO(data))
    # identity of this model file (name + content hash), stored with every logged RUL prediction so a
    # series can be split by model after a retrain; computed at load time, so it names the model that
    # is actually scoring even if the file is replaced while the dashboard keeps its cached copy
    bundle["model_version"] = f"{os.path.basename(path).removesuffix('.joblib')}@{hashlib.sha256(data).hexdigest()[:10]}"
    # bundles saved before 2026-09-24 (iddev2's) carry no "iddev" key
    if bundle.get("iddev", iddev) != iddev:
        raise ValueError(f"{path} was trained for iddev{bundle['iddev']}, not iddev{iddev}")
    # categorical encodings must line up with the live feature rows, or XGBoost silently mis-encodes
    if bundle["sensor_categories"] != ALL_SENSORS or bundle["subtype_categories"] != ALL_SUBTYPES:
        raise ValueError(f"{path}: sensor/subtype categories differ from rul_features' ({ALL_SENSORS}, {ALL_SUBTYPES})")
    # the model was trained on this exact feature set (window names included); a mismatch means the
    # device's roll_windows_min changed without retraining
    if bundle["feature_cols"] != feature_cols(iddev) + ["sensor", "subtype"]:
        raise ValueError(f"{path}: trained on {bundle['feature_cols']}, but iddev{iddev} now needs {feature_cols(iddev)} -- retrain")
    return {(sensor, subtype): bundle for sensor, subtype in SUBTYPES}


def predict_latest(engine, bundles, iddev, limit=2000):
    raw = db.fetch_recent(engine, iddev, limit=limit)
    if raw.empty:
        print(f"No rows found for iddev={iddev}")
        return {}

    print(f"Fetched {len(raw)} rows for iddev={iddev}, "
          f"range {raw['created_at'].min()} -> {raw['created_at'].max()}\n")

    inputs = latest_rul_inputs(raw, iddev)
    results = {}

    for sensor, subtype in SUBTYPES:
        key = (sensor, subtype)
        info = inputs[key]
        label = f"{sensor}/{subtype}"

        if not info["triggered"]:
            print(f"[{label}] not triggered -- {info['reason']}")
            continue

        if info["row"] is None:
            print(f"[{label}] TRIGGERED but no prediction yet -- {info['reason']}")
            continue

        bundle = bundles[key]
        row = info["row"]
        x = row[bundle["feature_cols"]]
        # RUL can't be negative -- XGBoost is a plain regressor with no such
        # constraint, and can overshoot slightly below 0 when band_progress
        # is already near 1 (close to/at failure).
        pred_seconds = max(0.0, float(bundle["model"].predict(x)[0]))

        results[key] = {
            "as_of": row["created_at"].iloc[0],
            "current_value": row["value"].iloc[0],
            "band_progress": row["band_progress"].iloc[0],
            "predicted_rul_minutes": pred_seconds / 60,
        }
        print(f"[{label}] TRIGGERED as_of={results[key]['as_of']} "
              f"current={results[key]['current_value']:.3f} "
              f"band_progress={results[key]['band_progress']:.2f} "
              f"-> predicted RUL = {results[key]['predicted_rul_minutes']:.1f} min")

    return results


def score_and_notify(raw, iddev, bundles, building=None, cell=None):
    """Score every (sensor, subtype) for iddev against `raw`, with the exact same log/smooth/notify
    side effects dashboard.py's render_status_cards did inline -- moved here so dashboard.py and any
    other caller (e.g. the HTTP API) share one implementation instead of two copies drifting apart.

    Returns one dict per (sensor, subtype). `status` is "normal", "triggered_no_data" (sustained
    out-of-band but not enough session history for a real RUL number yet), or "triggered".
    `building`/`cell` are passed straight through to rul_notify for the alert text (see its
    docstring -- both optional, no device->location lookup wired in yet).
    """
    inputs = latest_rul_inputs(raw, iddev)
    results = []

    for sensor, subtype in SUBTYPES:
        key = (sensor, subtype)
        info = inputs[key]

        if not info["triggered"]:
            rul_notify.reset_if_cleared(iddev, sensor, subtype)
            results.append({"sensor": sensor, "subtype": subtype, "status": "normal"})
            continue

        if info["row"] is None:
            results.append({
                "sensor": sensor, "subtype": subtype, "status": "triggered_no_data",
                "reason": info["reason"],
            })
            continue

        bundle = bundles[key]
        row = info["row"]
        x = row[bundle["feature_cols"]]
        pred_minutes = max(0.0, float(bundle["model"].predict(x)[0]) / 60)
        created_at = row["created_at"].iloc[0]
        current_value = row["value"].iloc[0]
        band_prog = row["band_progress"].iloc[0]

        rul_store.log_prediction(
            iddev, created_at, sensor, subtype, current_value, band_prog, pred_minutes,
            model_version=bundle["model_version"],
            features=x.drop(columns=bundle["cat_cols"]).iloc[0].to_dict(),
        )

        # what is reported is the smoothed RUL (see rul_smooth.py), pooled from the logged raw predictions
        recent = rul_store.get_predictions(iddev, sensor=sensor, subtype=subtype, limit=rul_smooth.SMOOTH_READINGS)
        s = rul_smooth.smooth_rul(recent, session_break_minutes(iddev)).iloc[-1]

        rul_notify.notify_if_new_trigger(iddev, sensor, subtype, created_at, current_value, band_prog, s,
                                          building=building, cell=cell)

        results.append({
            "sensor": sensor, "subtype": subtype, "status": "triggered",
            "as_of": created_at, "current_value": current_value, "band_progress": band_prog,
            "predicted_rul_minutes": pred_minutes, "rul_smooth_minutes": s["rul_smooth_min"],
            "eta": s["eta_hat"], "eta_low": s["eta_low"], "eta_high": s["eta_high"], "n_used": s["n_used"],
        })

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iddev", type=int, required=True, help="device to score")
    args = ap.parse_args()
    engine = db.get_engine()
    bundles = load_models(args.iddev)
    predict_latest(engine, bundles, args.iddev)
