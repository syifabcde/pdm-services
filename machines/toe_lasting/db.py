"""Toe-lasting's view of core.db: binds its table and columns so callers only pass the device."""
from sqlalchemy import text

from core import db as core_db

from . import config


def get_engine():
    return core_db.get_engine()


def fetch_recent(engine, iddev, limit=500):
    return core_db.fetch_recent(engine, config.DB_TABLE, config.DB_COLUMNS, iddev, limit)


def fetch_range(engine, iddev, start, end):
    return core_db.fetch_range(engine, config.DB_TABLE, config.DB_COLUMNS, iddev, start, end)


def fetch_device_location(engine, iddev):
    """(building, cell) for one device from pdm_tl_device -- display/alert text only, not used
    in any feature/model computation. Uses dev_building_code/dev_cell_code, NOT the plain
    dev_building/dev_cell columns -- confirmed against live data (2026-09-30) that the latter are
    NULL/empty for every device while the _code columns hold the real values (e.g. "D100",
    "CSAS01"), matching what pdmlora.js's own device query and the old anomaly-alert message
    ("Gedung D100 | Cell CSAS01") both already display. (None, None) if the device has no row."""
    query = text("SELECT dev_building_code, dev_cell_code FROM pdm_tl_device WHERE iddev = :iddev")
    with engine.connect() as conn:
        row = conn.execute(query, {"iddev": iddev}).fetchone()
    return (row.dev_building_code, row.dev_cell_code) if row else (None, None)
