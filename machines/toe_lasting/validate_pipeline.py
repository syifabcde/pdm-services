"""One-off sanity check: run the h30/h60 models against *past* sessions in the
already-fetched window, where the real future temperature is already known,
to confirm the feature pipeline + model loading matches training -- without
waiting for the live session to accumulate enough history on its own.
"""
import argparse

import pandas as pd

from . import db, features
from .predict import load_models

TOLERANCE_MIN = 3


def attach_actual_future(feat, df_on, horizon_minutes):
    rows = []
    for sess_id, g in feat.groupby("session_id"):
        session_raw = (
            df_on[df_on["session_id"] == sess_id][["created_at", "temperature"]]
            .sort_values("created_at")
            .rename(columns={"created_at": "target_time_actual", "temperature": "actual_future_temp"})
        )
        g = g.sort_values("created_at").copy()
        g["target_time"] = g["created_at"] + pd.Timedelta(minutes=horizon_minutes)
        merged = pd.merge_asof(
            g[["created_at", "target_time"]],
            session_raw,
            left_on="target_time",
            right_on="target_time_actual",
            direction="nearest",
            tolerance=pd.Timedelta(minutes=TOLERANCE_MIN),
        )
        rows.append(g.merge(merged[["created_at", "actual_future_temp"]], on="created_at", how="left"))
    return pd.concat(rows, ignore_index=True)


def validate(raw, models):
    df_on = features.add_session_id(raw)
    feat = features.build_features(df_on)

    for name, bundle in models.items():
        valid = feat.dropna(subset=bundle["feature_cols"]).copy()
        if valid.empty:
            print(f"[{name}] no session in this window has enough history to build a feature row")
            continue

        valid = attach_actual_future(valid, df_on, bundle["horizon_minutes"]).dropna(subset=["actual_future_temp"])
        if valid.empty:
            print(f"[{name}] rows had features, but none had a real future reading within "
                  f"+/-{TOLERANCE_MIN}min of the {bundle['horizon_minutes']}min horizon to compare against")
            continue

        preds = bundle["model"].predict(valid[bundle["feature_cols"]])
        valid["predicted"] = preds
        valid["abs_error"] = (valid["predicted"] - valid["actual_future_temp"]).abs()

        print(f"[{name}] validated on {len(valid)} historical rows (horizon={bundle['horizon_minutes']}min) "
              f"| MAE={valid['abs_error'].mean():.2f}")
        print(valid[["created_at", "temperature", "predicted", "actual_future_temp", "abs_error"]]
              .tail(5).to_string(index=False))
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iddev", type=int, required=True)
    args = ap.parse_args()
    engine = db.get_engine()
    raw = db.fetch_recent(engine, args.iddev)
    models = load_models()
    validate(raw, models)
