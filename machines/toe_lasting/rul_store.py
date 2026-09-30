"""Permanent local log of RUL predictions: survives the dashboard's fetch window sliding past old rows, so a
triggered episode's history stays visible across Streamlit reruns.
Every row carries its iddev; uniqueness is per device. Each row also records which model file produced
it (model_version) and the exact feature values that model saw (features, JSON), so a jump in the RUL
series can be traced to the inputs behind it. Rows written before those columns existed hold NULL.
"""
import json
import sqlite3
from contextlib import contextmanager

import pandas as pd

from core import sqlite_migrate

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS rul_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    iddev INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    sensor TEXT NOT NULL,
    subtype TEXT NOT NULL,
    current_value REAL NOT NULL,
    band_progress REAL NOT NULL,
    predicted_rul_minutes REAL NOT NULL,
    model_version TEXT,
    features TEXT,
    UNIQUE(iddev, created_at, sensor, subtype)
)
"""
LEGACY_COLS = ["id", "created_at", "sensor", "subtype", "current_value", "band_progress", "predicted_rul_minutes"]
ADDED_COLS = {"model_version": "TEXT", "features": "TEXT"}   # for tables created before they existed


@contextmanager
def connect():
    config.RUL_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.RUL_DB_PATH)
    try:
        sqlite_migrate.add_iddev(conn, "rul_predictions", SCHEMA, LEGACY_COLS)
        conn.execute(SCHEMA)
        sqlite_migrate.add_columns(conn, "rul_predictions", ADDED_COLS)
        yield conn
        conn.commit()
    finally:
        conn.close()


def log_prediction(iddev, created_at, sensor, subtype, current_value, band_progress, predicted_rul_minutes,
                   model_version=None, features=None):
    """`features`: {name: value} of the model's numeric inputs for this reading (stored as JSON).
    INSERT OR IGNORE keeps the first prediction logged for a reading, with the model that made it."""
    features_json = None if features is None else json.dumps({k: float(v) for k, v in features.items()})
    with connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO rul_predictions "
            "(iddev, created_at, sensor, subtype, current_value, band_progress, predicted_rul_minutes, "
            "model_version, features) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (int(iddev), created_at.isoformat(), sensor, subtype, float(current_value),
             float(band_progress), float(predicted_rul_minutes), model_version, features_json),
        )


def get_predictions(iddev, sensor=None, subtype=None, since=None, limit=None):
    clauses, params = ["iddev = ?"], [int(iddev)]
    if sensor is not None:
        clauses.append("sensor = ?")
        params.append(sensor)
    if subtype is not None:
        clauses.append("subtype = ?")
        params.append(subtype)
    if since is not None:
        clauses.append("created_at >= ?")
        params.append(since.isoformat())
    query = f"SELECT * FROM rul_predictions WHERE {' AND '.join(clauses)} ORDER BY created_at"
    if limit is not None:
        query += f" DESC LIMIT {int(limit)}"
    with connect() as conn:
        df = pd.read_sql(query, conn, params=params, parse_dates=["created_at"])
    return df.sort_values("created_at") if limit is not None else df
