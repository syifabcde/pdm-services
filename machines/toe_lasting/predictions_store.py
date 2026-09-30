"""Local log of predictions the dashboard has made, so the 'predicted' line
can keep its history across Streamlit reruns instead of only showing
whatever was computed in the current cycle. Separate SQLite file -- this is
our own service's state, not something written back to the IoT database.
Every row carries its iddev; uniqueness is per device.
"""
import sqlite3
from contextlib import contextmanager

import pandas as pd

from core import sqlite_migrate

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    iddev INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    made_at TEXT NOT NULL,
    target_time TEXT NOT NULL,
    horizon_minutes INTEGER NOT NULL,
    predicted_temperature REAL NOT NULL,
    UNIQUE(iddev, model_name, made_at)
)
"""
LEGACY_COLS = ["id", "model_name", "made_at", "target_time", "horizon_minutes", "predicted_temperature"]


@contextmanager
def connect():
    config.PRED_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.PRED_DB_PATH)
    try:
        sqlite_migrate.add_iddev(conn, "predictions", SCHEMA, LEGACY_COLS)
        conn.execute(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def log_prediction(iddev, model_name, made_at, target_time, horizon_minutes, predicted_temperature):
    with connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO predictions "
            "(iddev, model_name, made_at, target_time, horizon_minutes, predicted_temperature) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (int(iddev), model_name, made_at.isoformat(), target_time.isoformat(),
             horizon_minutes, float(predicted_temperature)),
        )


def get_predictions(iddev, since=None):
    with connect() as conn:
        if since is not None:
            return pd.read_sql(
                "SELECT * FROM predictions WHERE iddev = ? AND target_time >= ? ORDER BY target_time",
                conn, params=(int(iddev), since.isoformat()), parse_dates=["made_at", "target_time"],
            )
        return pd.read_sql(
            "SELECT * FROM predictions WHERE iddev = ? ORDER BY target_time",
            conn, params=(int(iddev),), parse_dates=["made_at", "target_time"],
        )
