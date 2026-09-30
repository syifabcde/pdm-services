import argparse

import joblib

from . import config, db, features

MODEL_FILES = {
    "h30": "temperature_rf_h30_retrain_20260810.joblib",
    "h60": "temperature_rf_h60_retrain_20260810.joblib",
}


def load_models():
    return {
        name: joblib.load(config.MODEL_DIR / filename)
        for name, filename in MODEL_FILES.items()
    }


def predict_latest(engine, models, iddev):
    raw = db.fetch_recent(engine, iddev)
    if raw.empty:
        print(f"No rows found for iddev={iddev}")
        return

    results = {}
    for name, bundle in models.items():
        row = features.latest_feature_row(raw, bundle["feature_cols"], bundle["session_break_min"],
                                           bundle["lags"], bundle["roll_window"])
        if row is None:
            print(f"[{name}] not enough history in current session yet "
                  f"(need >= {max(bundle['lags'])} on-samples since session start)")
            continue

        x = row[bundle["feature_cols"]]
        pred = bundle["model"].predict(x)[0]
        results[name] = {
            "as_of": row["created_at"].iloc[0],
            "current_temperature": row["temperature"].iloc[0],
            "predicted_temperature": pred,
            "horizon_minutes": bundle["horizon_minutes"],
        }

    for name, r in results.items():
        print(f"[{name}] as_of={r['as_of']} current={r['current_temperature']:.2f} "
              f"-> predicted in {r['horizon_minutes']}min = {r['predicted_temperature']:.2f}")

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iddev", type=int, required=True)
    args = ap.parse_args()
    engine = db.get_engine()
    models = load_models()
    predict_latest(engine, models, args.iddev)
