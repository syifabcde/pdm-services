"""
Telegram alert for a subtype crossing out of its normal band with a usable RUL prediction --
the moment a status card turns red in dashboard.py's render_status_cards().

Deliberately does NOT fire for the other "triggered" branch (sustained out-of-band, but not
enough session history yet for the rolling features a model needs) -- there's no actual
value/RUL number to show yet, so a notification there would just be noise with nothing
actionable in it. Only the branch with a usable row reaches notify_if_new_trigger().

Debounced the same way as band_drift_watch.py: exactly ONE message per trigger EPISODE (the
not-triggered -> triggered transition), silence on every later reading while it stays
triggered, and a silent (no-message) reset back to armed once the subtype is confirmed no
longer triggered -- a "back to normal" Telegram message is a deliberate follow-up, not built
yet (kept out on purpose per user request 2026-09-29; see project memory "toe-lasting rul
trigger telegram template").
"""
import sqlite3
from contextlib import contextmanager

import pandas as pd

from core import notify

from . import config
from .rul_features import FAILURE_DESCRIPTIONS

# Display unit per sensor -- matches what PRB-MULTICOMPANY's dashboard already shows
# (temperature in C, vibration in mm/s, pressure in kg/cm2).
UNITS = {"temperature": "°C", "vibration": "mm/s", "pressure": "kg/cm²"}

# ============================================================================
# Persistence + Telegram debounce. Keeps just "have I already sent a message for the trigger
# episode this subtype is currently in", per (iddev, sensor, subtype) -- same spirit as
# band_drift_watch.py's band_drift_state, one level more granular (subtype, not just sensor)
# since two subtypes on the same sensor (e.g. temperature's heater_burnout/thermocouple_failure)
# trigger independently of each other.
# ============================================================================

SCHEMA = """
CREATE TABLE IF NOT EXISTS rul_trigger_state (
    iddev INTEGER NOT NULL,
    sensor TEXT NOT NULL,
    subtype TEXT NOT NULL,
    notified INTEGER NOT NULL,
    first_notified_at TEXT,
    last_checked_at TEXT NOT NULL,
    PRIMARY KEY (iddev, sensor, subtype)
)
"""


@contextmanager
def connect():
    config.RUL_TRIGGER_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.RUL_TRIGGER_DB_PATH)
    try:
        conn.execute(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _format_message(sensor, subtype, created_at, current_value, band_progress, smooth, building=None, cell=None):
    """
    Formats a Telegram message for a new RUL trigger.

    `building`/`cell`: physical location (e.g. resolved from pdm_tl_device) -- both optional
    because pdm-services has no device->location lookup wired in yet (only Laravel/
    PRB-MULTICOMPANY's DB queries join that table today, and wiring it up here is deliberately
    deferred, per user 2026-09-29). Shown as "N/A" (not omitted) when missing.
    `Machine` (config.LABEL, "Toe-Lasting") is always shown -- this module only ever alerts for
    this one machine type, so it's a fixed value, not a param.
    """
    key = (sensor, subtype)
    desc = FAILURE_DESCRIPTIONS[key]
    unit = UNITS[sensor]

    spread = ""
    if smooth["n_used"] > 1:
        spread = (f", kisaran {smooth['eta_low'].strftime('%H:%M')}-{smooth['eta_high'].strftime('%H:%M')} "
                  f"dari {int(smooth['n_used'])} bacaan terakhir")

    return (    
        f"🚨 *PDM ALERT* 🚨\n"
        f"• Gedung `{building or 'N/A'}`\n"
        f"• Cell `{cell or 'N/A'}`\n"
        f"• Machine `{config.LABEL}`\n"
        f"• Sensor `{sensor}`\n\n"
        f"⚠️ *INDIKASI*\n"
        f"{desc}\n\n"
        f"📈 *KONDISI SEKARANG*\n"
        f"• Nilai aktual `{current_value:.1f}{unit}`\n"
        f"• Progress `{band_progress:.0%}` menuju batas kerusakan\n\n"
        f"⏳ *PREDIKSI KERUSAKAN*\n"
        f"• Sisa waktu `±{smooth['rul_smooth_min']:.0f} menit`\n"
        f"• Estimasi `{smooth['eta_hat'].strftime('%H:%M')}`{spread}\n\n"
        f"📅 *WAKTU TRIGGER* `{created_at.strftime('%d/%m/%Y %H:%M:%S')}`\n\n"
    )


def notify_if_new_trigger(iddev, sensor, subtype, created_at, current_value, band_progress, smooth,
                           building=None, cell=None, now=None):
    """Sends exactly ONE Telegram message per trigger EPISODE for (iddev, sensor, subtype).

    `iddev` is still required here -- it's the debounce state's key -- but is NOT passed into
    the message text itself anymore (see _format_message's docstring: the user asked for the raw
    device id to not appear in the alert at all, replaced by building/cell instead).

    Only call this from the branch that already has a usable row + smoothed RUL (rul_smooth.py's
    output for the latest reading) -- see module docstring for why the "triggered but no row yet"
    branch should never reach this function at all. Call reset_if_cleared() separately for every
    subtype confirmed NOT triggered, so the debounce re-arms for that subtype's next episode.

    Never raises: notify.send_telegram() is itself best-effort, and an sqlite hiccup here is left
    unguarded on purpose (same reasoning as band_drift_watch.update_drift_state_and_notify) --
    silently losing this state would mean the next check re-treats an ongoing episode as new.
    """
    now = pd.Timestamp.now() if now is None else now
    now_iso = now.isoformat()

    with connect() as conn:
        row = conn.execute(
            "SELECT notified FROM rul_trigger_state WHERE iddev=? AND sensor=? AND subtype=?",
            (iddev, sensor, subtype),
        ).fetchone()
        already_notified = bool(row[0]) if row else False

        if already_notified:
            conn.execute(
                "UPDATE rul_trigger_state SET last_checked_at=? WHERE iddev=? AND sensor=? AND subtype=?",
                (now_iso, iddev, sensor, subtype),
            )
            return None

        message = _format_message(sensor, subtype, created_at, current_value, band_progress, smooth, building, cell)
        notify.send_telegram(message, parse_mode="Markdown")

        if row is None:
            conn.execute(
                "INSERT INTO rul_trigger_state "
                "(iddev, sensor, subtype, notified, first_notified_at, last_checked_at) "
                "VALUES (?, ?, ?, 1, ?, ?)",
                (iddev, sensor, subtype, now_iso, now_iso),
            )
        else:
            conn.execute(
                "UPDATE rul_trigger_state SET notified=1, first_notified_at=?, last_checked_at=? "
                "WHERE iddev=? AND sensor=? AND subtype=?",
                (now_iso, now_iso, iddev, sensor, subtype),
            )
    return message


def reset_if_cleared(iddev, sensor, subtype, now=None):
    """Call once per check for every subtype that is confirmed NOT triggered right now (the
    dashboard's "Normal" branch). Silently re-arms the debounce -- no Telegram message sent here
    (a "back to normal" notification is a deliberate follow-up, not built yet) -- so the next time
    this subtype trips, notify_if_new_trigger() fires again instead of staying silent forever.
    A no-op (0 rows affected) if this subtype has never been triggered before; harmless."""
    now_iso = (pd.Timestamp.now() if now is None else now).isoformat()
    with connect() as conn:
        conn.execute(
            "UPDATE rul_trigger_state SET notified=0, first_notified_at=NULL, last_checked_at=? "
            "WHERE iddev=? AND sensor=? AND subtype=?",
            (now_iso, iddev, sensor, subtype),
        )
