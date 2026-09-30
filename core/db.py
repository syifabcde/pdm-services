"""
Read-only access to the IoT Postgres log. Machine-agnostic: the caller passes the
table, the columns and the device, all of which come from that machine's machine.toml.
"""
import re

import pandas as pd
from sqlalchemy import create_engine, text

from core import settings

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _ident(name):
    # table/column names come from machine.toml (trusted), but they are interpolated
    # into SQL, so refuse anything that isn't a plain identifier.
    if not _IDENT.match(name):
        raise ValueError(f"not a valid SQL identifier: {name!r}")
    return name


def get_engine():
    return create_engine(settings.db_url())


def fetch_recent(engine, table, columns, iddev, limit=500):
    """Fetch the most recent `limit` rows for a device, oldest-first.

    Ordering by created_at DESC + LIMIT then re-sorting ascending (instead of
    a NOW() - INTERVAL window) sidesteps DB-vs-app timezone mismatches --
    created_at in the source data has no tz marker, so a naive interval
    comparison could silently drift.
    """
    cols = ", ".join(_ident(c) for c in columns)
    query = text(
        f"""
        SELECT {cols}
        FROM {_ident(table)}
        WHERE iddev = :iddev
        ORDER BY created_at DESC
        LIMIT :limit
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"iddev": iddev, "limit": limit})
    return df.sort_values("created_at").reset_index(drop=True)


def fetch_range(engine, table, columns, iddev, start, end):
    """Fetch all rows for a device within [start, end), oldest-first --
    read-only, for backtesting a pipeline against a specific historical
    window instead of just 'the most recent N rows'."""
    cols = ", ".join(_ident(c) for c in columns)
    query = text(
        f"""
        SELECT {cols}
        FROM {_ident(table)}
        WHERE iddev = :iddev AND created_at >= :start AND created_at < :end
        ORDER BY created_at
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"iddev": iddev, "start": start, "end": end})
    return df.reset_index(drop=True)
