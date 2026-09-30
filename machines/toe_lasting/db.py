"""Toe-lasting's view of core.db: binds its table and columns so callers only pass the device."""
import re

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
    in any feature/model computation.

    `building` is dev_building_code (e.g. "D100") -- confirmed against live data (2026-09-30)
    that the plain dev_building column is NULL for every device.

    `cell` is deliberately NOT dev_cell_code (an internal code like "CSAS10", not human-friendly)
    -- it's the trailing number in dev_name (e.g. "TOE LASTING CELL 10" -> "10"), per user request
    2026-09-30 to drop the "Toe Lasting Cell" wording and keep just the number. None if dev_name
    doesn't end in a number.

    (None, None) if the device has no row."""
    query = text("SELECT dev_name, dev_building_code FROM pdm_tl_device WHERE iddev = :iddev")
    with engine.connect() as conn:
        row = conn.execute(query, {"iddev": iddev}).fetchone()
    if not row:
        return None, None
    match = re.search(r"(\d+)\s*$", row.dev_name or "")
    return row.dev_building_code, (match.group(1) if match else None)
