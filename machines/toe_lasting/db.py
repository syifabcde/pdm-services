"""Toe-lasting's view of core.db: binds its table and columns so callers only pass the device."""
from core import db as core_db

from . import config


def get_engine():
    return core_db.get_engine()


def fetch_recent(engine, iddev, limit=500):
    return core_db.fetch_recent(engine, config.DB_TABLE, config.DB_COLUMNS, iddev, limit)


def fetch_range(engine, iddev, start, end):
    return core_db.fetch_range(engine, config.DB_TABLE, config.DB_COLUMNS, iddev, start, end)
