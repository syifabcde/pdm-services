"""
Advisory check: has a device's real temperature likely drifted away from its
configured fixed band (config/bands.toml's [iddevN.temperature])? E.g. after a
recipe/glue change moves the heating setpoint (see project memory "toe-lasting
adaptive band prototype" / "toe-lasting iddev1 vs iddev2 temp regime").

This does NOT change RUL scoring -- live scoring (rul_features.py) keeps using the
fixed band exactly as before, unchanged by anything here. It only raises a flag for a
human (engineering) to review and, if the drift is confirmed real, manually update
bands.toml -- then retrain (toe-lasting/pipeline/feature_engineering.py --all +
save_final_rul_model.py) so training and live stay consistent, same as any other
bands.toml edit.

Why advisory-only, not automatic: an earlier attempt at automatically re-deriving
band_progress from the live operating point (a session-relative "adaptive" band) was
built, deployed, and reverted the same day -- on real injected failure episodes it
produced a SILENT failure mode (a genuine overheat's band_progress stayed at exactly
0.000 for an entire ~12h precursor, because the same rolling mechanism meant to notice
drift also gets dragged by an active failure ramp; see that memory for the full
writeup). A once-in-a-while advisory check carries none of that risk: it only has to
be right eventually, not react within minutes, and nothing about scoring moves until a
human confirms.

Method: over a multi-day lookback window, what fraction of ON readings (vibration AND
pressure both below the noise floor = off, same rule as rul_features.add_session_id)
fall OUTSIDE the 'normal' status tier. Calibrated against real data (2026-09-28):
genuinely healthy operation sits under 10% for both iddev1 and iddev2 (0.8% / 8.6% over
their full real history); the one real recipe-driven drift on record (iddev1,
2026-09-24) shows 77%; iddev3/iddev4's already-known band mismatch (project memory
"toe-lasting iddev3/4 raw data assessment") shows 90-98% -- wide margin on both sides
of the 30% threshold below.
"""
import sqlite3
from contextlib import contextmanager

import pandas as pd

from core import notify

from . import config
from .rul_bands import LABEL_FN
from .rul_features import PRESSURE_OFF_THRESHOLD, VIBRATION_OFF_THRESHOLD

LOOKBACK_DAYS = 3
MIN_ON_READINGS = 20        # below this, there's not enough recent on-time data to judge either way
DRIFT_FRACTION_THRESHOLD = 0.30   # healthy real history sits <10%; the one real drift case sits at 77%

# raw DB reads have NO cleaning applied (db.fetch_recent() is a plain passthrough, unlike
# toe-lasting/pipeline/clean_log.py's implausible-row filter on the training side) -- found live
# 2026-09-28: iddev2 readings spiking to 722-982C get misclassified as genuine 'danger'-tier
# readings, which on their own pushed a healthy device (median 116C) over the drift threshold.
# 700 is comfortably above both devices' 'danger'-band sampling ceiling (critical*1.5 = 600, see
# inject_failures.band_bounds()) and comfortably below the observed glitch values.
IMPLAUSIBLE_TEMP_MAX = 700


def check_temperature_drift(raw, iddev, now=None):
    """raw: recent rows for ONE device (db.fetch_recent/fetch_range), covering at
    least LOOKBACK_DAYS -- caller's responsibility, this function doesn't fetch.

    Returns a dict, never raises:
        drifted            bool -- True if this device's real temperature looks like
                                    it's settled on a different regime than bands.toml
                                    assumes
        frac_not_normal    float or None -- fraction of recent on-readings outside the
                                    'normal' tier (None if not enough data)
        n_on_readings       int
        current_median      float or None -- recent on-time temperature median, for
                                    display (what a human would compare against
                                    bands.toml's standby/normal when deciding a new value)
        reason              str -- human-readable, for display when not drifted
    """
    now = pd.Timestamp.now() if now is None else now
    cutoff = now - pd.Timedelta(days=LOOKBACK_DAYS)
    recent = raw[raw["created_at"] >= cutoff]

    recent = recent[recent["temperature"] <= IMPLAUSIBLE_TEMP_MAX]
    is_off = (recent["vibration"].fillna(0) < VIBRATION_OFF_THRESHOLD) & (recent["pressure"].fillna(0) < PRESSURE_OFF_THRESHOLD)
    on = recent.loc[~is_off]
    if len(on) < MIN_ON_READINGS:
        return {"drifted": False, "frac_not_normal": None, "n_on_readings": len(on),
                "current_median": None, "reason": "belum cukup data on-time beberapa hari terakhir"}

    status = pd.Series(LABEL_FN["temperature"](on["temperature"].values, iddev), index=on.index)
    valid = status[status != "off"]
    if len(valid) < MIN_ON_READINGS:
        return {"drifted": False, "frac_not_normal": None, "n_on_readings": len(valid),
                "current_median": None, "reason": "belum cukup data on-time beberapa hari terakhir"}

    frac_not_normal = float((valid != "normal").mean())
    return {
        "drifted": frac_not_normal >= DRIFT_FRACTION_THRESHOLD,
        "frac_not_normal": frac_not_normal,
        "n_on_readings": int(len(valid)),
        "current_median": float(on.loc[valid.index, "temperature"].median()),
        "reason": "ok",
    }


# ============================================================================
# Persistence + Telegram debounce. Keeps just the LAST known state per (iddev, sensor)
# -- enough to tell "just started drifting" / "still drifting" / "just cleared" apart,
# which is all a debounced notification needs. Not a history log (rul_store.py already
# has that pattern for RUL predictions if a full audit trail is ever wanted here too).
# ============================================================================

SCHEMA = """
CREATE TABLE IF NOT EXISTS band_drift_state (
    iddev INTEGER NOT NULL,
    sensor TEXT NOT NULL DEFAULT 'temperature',
    drifted INTEGER NOT NULL,
    first_detected_at TEXT,
    last_checked_at TEXT NOT NULL,
    last_notified_at TEXT,
    PRIMARY KEY (iddev, sensor)
)
"""


@contextmanager
def connect():
    config.DRIFT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DRIFT_DB_PATH)
    try:
        conn.execute(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _format_message(iddev, result):
    # Plain text -- notify.send_telegram() sends with no parse_mode (see its docstring),
    # so no *bold*/_italic_ markup here; it would just show up as literal asterisks/underscores.
    return (
        f"⚠️ iddev{iddev}: suhu kemungkinan sudah bergeser dari band yang "
        f"dikonfigurasi di config/bands.toml.\n"
        f"{result['frac_not_normal']:.0%} dari {result['n_on_readings']} bacaan on-time "
        f"{LOOKBACK_DAYS} hari terakhir di luar tier 'normal' "
        f"(median suhu saat ini ≈ {result['current_median']:.1f}°C).\n"
        f"Konfirmasi ke engineering sebelum mengubah band secara manual -- RUL masih "
        f"memakai band lama sampai dikonfirmasi."
    )


def _format_clear_message(iddev):
    return f"✅ iddev{iddev}: suhu sudah kembali sesuai band yang dikonfigurasi."


def update_drift_state_and_notify(iddev, result, sensor="temperature", now=None):
    """Persists `result` (from check_temperature_drift) and sends exactly ONE Telegram
    message per drift EPISODE -- when `drifted` flips False->True (not on every check
    while it stays True) and once more when it clears True->False. Returns the message
    actually sent, or None if nothing was sent this call. Never raises: a notification
    failure is swallowed inside notify.send_telegram, and this function's own sqlite
    write is the caller's problem only if it also wants that to be fatal (it isn't
    wrapped here because unlike a notification, silently losing the local state IS worth
    surfacing -- it would otherwise mean every future check re-detects "just started").
    """
    now = pd.Timestamp.now() if now is None else now
    now_iso = now.isoformat()

    with connect() as conn:
        row = conn.execute(
            "SELECT drifted, first_detected_at FROM band_drift_state WHERE iddev=? AND sensor=?",
            (iddev, sensor),
        ).fetchone()
        was_drifted = bool(row[0]) if row else False
        first_detected_at = row[1] if row else None
        now_drifted = bool(result["drifted"])

        message = None
        notified_at = None
        if now_drifted and not was_drifted:
            first_detected_at = now_iso
            message = _format_message(iddev, result)
        elif not now_drifted and was_drifted:
            first_detected_at = None
            message = _format_clear_message(iddev)
        # else: no state transition -- still drifted or still fine, don't re-notify

        if message is not None:
            notify.send_telegram(message)
            notified_at = now_iso

        if row is None:
            conn.execute(
                "INSERT INTO band_drift_state (iddev, sensor, drifted, first_detected_at, last_checked_at, last_notified_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (iddev, sensor, int(now_drifted), first_detected_at, now_iso, notified_at),
            )
        else:
            if notified_at is not None:
                conn.execute(
                    "UPDATE band_drift_state SET drifted=?, first_detected_at=?, last_checked_at=?, last_notified_at=? "
                    "WHERE iddev=? AND sensor=?",
                    (int(now_drifted), first_detected_at, now_iso, notified_at, iddev, sensor),
                )
            else:
                conn.execute(
                    "UPDATE band_drift_state SET drifted=?, first_detected_at=?, last_checked_at=? "
                    "WHERE iddev=? AND sensor=?",
                    (int(now_drifted), first_detected_at, now_iso, iddev, sensor),
                )

    return message
