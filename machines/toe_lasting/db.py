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
    in any feature/model computation. (None, None) if the device has no row there."""
    query = text("SELECT dev_building, dev_cell FROM pdm_tl_device WHERE iddev = :iddev")
    with engine.connect() as conn:
        row = conn.execute(query, {"iddev": iddev}).fetchone()
    return (row.dev_building, row.dev_cell) if row else (None, None)
