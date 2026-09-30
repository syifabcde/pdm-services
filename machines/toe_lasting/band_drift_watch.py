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
from .rul_bands import LABEL_FN, get_bands
from .rul_features import PRESSURE_OFF_THRESHOLD, VIBRATION_OFF_THRESHOLD

SENSORS = ("temperature", "vibration", "pressure")
LOOKBACK_DAYS = 3
MIN_ON_READINGS = 20        # below this, there's not enough recent on-time data to judge either way
# Per-sensor fraction of on-readings outside the 'normal' tier that counts as drift. Temperature's
# 30% is calibrated (healthy real history <10%, the one real drift case 77%). Backtested
# 2026-09-30 on rolling 3-day windows of iddev1/iddev2 cleaned history: healthy vibration
# stays <=1% (30% is fine), healthy pressure runs up to 25% (iddev1 median 16%), so it gets a
# higher 40% to keep margin. Neither has a real drift case on record to validate detection
# against -- only iddev3/4's known band mismatch (vibration 40-94%, pressure iddev3 100%).
DRIFT_FRACTION_THRESHOLD = {"temperature": 0.30, "vibration": 0.30, "pressure": 0.40}

# display unit + label per sensor (units match rul_notify.UNITS)
UNITS = {"temperature": "°C", "vibration": "mm/s", "pressure": "kg/cm²"}
SENSOR_LABEL = {"temperature": "Suhu", "vibration": "Vibrasi", "pressure": "Tekanan"}

# raw DB reads have NO cleaning applied (db.fetch_recent() is a plain passthrough, unlike
# toe-lasting/pipeline/clean_log.py's implausible-row filter on the training side) -- found live
# 2026-09-28: iddev2 readings spiking to 722-982C get misclassified as genuine 'danger'-tier
# readings, which on their own pushed a healthy device (median 116C) over the drift threshold.
# 700 is comfortably above both devices' 'danger'-band sampling ceiling (critical*1.5 = 600, see
# inject_failures.band_bounds()) and comfortably below the observed glitch values.
IMPLAUSIBLE_TEMP_MAX = 700
# Only temperature has a known glitch ceiling so far; vibration/pressure get no upper filter
# (the label functions already map NaN/0 to 'off').
IMPLAUSIBLE_MAX = {"temperature": IMPLAUSIBLE_TEMP_MAX}


def check_band_drift(raw, iddev, sensor, now=None):
    """raw: recent rows for ONE device (db.fetch_recent/fetch_range), covering at
    least LOOKBACK_DAYS -- caller's responsibility, this function doesn't fetch.
    sensor: one of SENSORS -- which sensor's band to check.

    Returns a dict, never raises:
        drifted            bool -- True if this sensor's real values look like they've
                                    settled on a different regime than bands.toml assumes
        frac_not_normal    float or None -- fraction of recent on-readings outside the
                                    'normal' tier (None if not enough data)
        n_on_readings       int
        current_median      float or None -- recent on-time median of this sensor, for
                                    display (what a human would compare against
                                    bands.toml's tiers when deciding a new value)
        reason              str -- human-readable, for display when not drifted
    """
    now = pd.Timestamp.now() if now is None else now
    cutoff = now - pd.Timedelta(days=LOOKBACK_DAYS)
    recent = raw[raw["created_at"] >= cutoff]

    max_ok = IMPLAUSIBLE_MAX.get(sensor)
    if max_ok is not None:
        recent = recent[recent[sensor] <= max_ok]
    is_off = (recent["vibration"].fillna(0) < VIBRATION_OFF_THRESHOLD) & (recent["pressure"].fillna(0) < PRESSURE_OFF_THRESHOLD)
    on = recent.loc[~is_off]
    if len(on) < MIN_ON_READINGS:
        return {"drifted": False, "frac_not_normal": None, "n_on_readings": len(on),
                "current_median": None, "reason": "belum cukup data on-time beberapa hari terakhir"}

    status = pd.Series(LABEL_FN[sensor](on[sensor].values, iddev), index=on.index)
    valid = status[status != "off"]
    if len(valid) < MIN_ON_READINGS:
        return {"drifted": False, "frac_not_normal": None, "n_on_readings": len(valid),
                "current_median": None, "reason": "belum cukup data on-time beberapa hari terakhir"}

    frac_not_normal = float((valid != "normal").mean())
    return {
        "drifted": frac_not_normal >= DRIFT_FRACTION_THRESHOLD[sensor],
        "frac_not_normal": frac_not_normal,
        "n_on_readings": int(len(valid)),
        "current_median": float(on.loc[valid.index, sensor].median()),
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


# Each sensor's ladder, and the two tiers bounding its 'normal' band (whose center the shift
# anchors on). Temperature/vibration rise standby<normal<warning<critical, 'normal' = (standby,
# normal]. Pressure falls critical<warning<normal with no standby, 'normal' = (warning, normal].
_TIERS = {
    "temperature": (("standby", "normal", "warning", "critical"), ("standby", "normal")),
    "vibration": (("standby", "normal", "warning", "critical"), ("standby", "normal")),
    "pressure": (("critical", "warning", "normal"), ("warning", "normal")),
}


def suggest_shifted_bands(iddev, current_median, sensor):
    """Naive heuristic ONLY, for the drift alert text -- NOT a calibrated replacement for
    bands.toml. Shifts every tier of `sensor`'s ladder by the same delta, so the new median
    sits where the old median used to sit relative to the band (the center of the old 'normal'
    tier) -- preserves each tier's width, just recenters the whole ladder. The real calibration
    (see bands.toml's own header) uses actual healthy-history percentiles over a real window,
    not a single current median, so this is only ever a rough starting point for engineering
    to sanity-check, never auto-applied."""
    bands = get_bands(iddev)[sensor]
    tiers, (lo, hi) = _TIERS[sensor]
    center = (bands[lo] + bands[hi]) / 2
    delta = current_median - center
    return {tier: bands[tier] + delta for tier in tiers}


def _format_message(iddev, result, sensor, building, cell):
    # Plain text -- notify.send_telegram() sends with no parse_mode (see its docstring),
    # so no *bold*/_italic_ markup here; it would just show up as literal asterisks/underscores.
    s = suggest_shifted_bands(iddev, result["current_median"], sensor)
    label, unit = SENSOR_LABEL[sensor], UNITS[sensor]
    dec = 2 if sensor == "vibration" else 0
    tiers = "\n".join(f"• {tier} ≈ {v:.{dec}f}" for tier, v in s.items())
    return (
        f"⚠️ PDM WARNING — PERGESERAN THRESHOLD\n\n"
        f"• Gedung {building or 'N/A'}\n"
        f"• Cell {cell or 'N/A'}\n"
        f"• Mesin {config.LABEL}\n"
        f"• Sensor {sensor}\n\n"
        f"📊 TEMUAN\n"
        f"• Bacaan di luar normal : {result['frac_not_normal']:.0%}\n"
        f"• Total bacaan : {result['n_on_readings']}\n"
        f"• Periode pemantauan : {LOOKBACK_DAYS} hari terakhir\n"
        f"• Median {label.lower()} saat ini : {result['current_median']:.{max(dec, 1)}f}{unit}\n\n"
        f"⚠️ INDIKASI\n"
        f"{label} terindikasi bergeser dari threshold yang dikonfigurasi.\n\n"
        f"💡 REKOMENDASI GESER THRESHOLD\n"
        f"{tiers}\n\n"
        f"🛠️ TINDAKAN\n"
        f"Konfirmasi usulan di atas ke tim Engineering sebelum mengubah threshold secara manual.\n\n"
        f"ℹ️ CATATAN\n"
        f"PDM masih menggunakan threshold lama sampai konfigurasi dikonfirmasi."
    )


def _format_clear_message(iddev, sensor):
    return f"✅ iddev{iddev}: {SENSOR_LABEL[sensor].lower()} sudah kembali sesuai band yang dikonfigurasi."


def update_drift_state_and_notify(iddev, result, sensor, building, cell, now=None):
    """Persists `result` (from check_band_drift) and sends exactly ONE Telegram
    message per drift EPISODE -- when `drifted` flips False->True (not on every check
    while it stays True) and once more when it clears True->False. Returns the message
    actually sent, or None if nothing was sent this call. A transition whose Telegram send
    fails is NOT persisted (None is returned) so the next check retries it. Never raises: a
    notification failure is swallowed inside notify.send_telegram, and this function's own sqlite
    write is the caller's problem only if it also wants that to be fatal (it isn't
    wrapped here because unlike a notification, silently losing the local state IS worth
    surfacing -- it would otherwise mean every future check re-detects "just started").

    `building`/`cell`: passed straight through to _format_message (see machines.toe_lasting.db.
    fetch_device_location: dev_building_code and the number in dev_name from pdm_tl_device) --
    required; _format_message still shows "N/A" if the DB itself returns None for either.
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
            message = _format_message(iddev, result, sensor=sensor, building=building, cell=cell)
        elif not now_drifted and was_drifted:
            first_detected_at = None
            message = _format_clear_message(iddev, sensor)
        # else: no state transition -- still drifted or still fine, don't re-notify

        if message is not None:
            if not notify.send_telegram(message):
                # not delivered: keep the previous drifted/first_detected_at so the same transition is
                # detected again on the next check and retried, instead of being lost for good
                if row is not None:
                    conn.execute(
                        "UPDATE band_drift_state SET last_checked_at=? WHERE iddev=? AND sensor=?",
                        (now_iso, iddev, sensor),
                    )
                return None
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
