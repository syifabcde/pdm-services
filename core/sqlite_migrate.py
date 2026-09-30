"""
One-time schema migration for the sqlite audit logs: add an `iddev` column and make
the UNIQUE constraint per device.

Why the constraint matters, not just the column: the stores write with
INSERT OR IGNORE, and the old schemas were UNIQUE on created_at (or created_at +
sensor/subtype) alone. Two devices polled at the same instant would then collide and
the second row would be dropped without any error.

Every row written before 2026-09-24 came from the single-device service, which ran with
DEVICE_ID=2 (confirmed against the data: the legacy anomaly_scores' mean temperature was
110.6 C, iddev2's regime), so legacy rows are backfilled as iddev2.

add_columns() is the additive counterpart: new nullable columns on a table that already exists.
"""
import sqlite3

LEGACY_IDDEV = 2


def add_iddev(conn, table, new_schema_sql, copy_cols, legacy_iddev=LEGACY_IDDEV):
    """Rebuild `table` from `new_schema_sql` (which must declare an `iddev` column) if it
    still has the old shape. Returns True if it migrated, False if there was nothing to do
    (fresh database, or already migrated). Atomic: on any failure the old table is untouched."""
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
    if not cols or "iddev" in cols:
        return False

    col_list = ", ".join(copy_cols)
    conn.execute("BEGIN")
    try:
        conn.execute(f"ALTER TABLE {table} RENAME TO {table}_pre_iddev")
        conn.execute(new_schema_sql)
        conn.execute(
            f"INSERT INTO {table} (iddev, {col_list}) SELECT ?, {col_list} FROM {table}_pre_iddev",
            (legacy_iddev,),
        )
        n_old = conn.execute(f"SELECT COUNT(*) FROM {table}_pre_iddev").fetchone()[0]
        n_new = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if n_old != n_new:
            raise RuntimeError(f"{table}: migration copied {n_new} of {n_old} rows")
        conn.execute(f"DROP TABLE {table}_pre_iddev")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return True


def add_columns(conn, table, columns):
    """Add whichever of `columns` ({name: sqlite type}) `table` lacks. Existing rows get NULL, which
    reads as "unknown" (written before the column existed) rather than a made-up value. Idempotent,
    and safe if two connections race to add the same column. Returns the names actually added."""
    have = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    added = []
    for name, sql_type in columns.items():
        if name in have:
            continue
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")
        except sqlite3.OperationalError as e:
            if "duplicate column" not in str(e).lower():
                raise
            continue
        added.append(name)
    return added
