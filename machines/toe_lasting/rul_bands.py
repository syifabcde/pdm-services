"""
Official band thresholds for RUL feature computation and episode-start
triggering -- per device (iddev), read from toe-lasting/config/bands.toml.

Toe-lasting has up to 10 physical machines, and different devices can run
genuinely different sensor regimes: iddev1's healthy temperature (median
~254C) and iddev2's (median ~114C) don't even overlap (see project memory
"toe-lasting iddev1 raw log data quality" / "toe-lasting iddev1 vs iddev2
temp regime"). This module used to hold ONE global dict (iddev2's numbers)
applied to every device -- confirmed 2026-09-23 that this misclassifies
~99.9% of iddev1's genuinely healthy readings as "warning".

bands.toml is the single source of truth, shared with the training pipeline
(toe-lasting/pipeline/bands.py) and checked against shared/docs/threshold.md
by toe-lasting/pipeline/check_thresholds.py. This module keeps no numbers of
its own. The old thresholds.py (forecast-crossing helper whose code had drifted
from its own docstring) was deleted 2026-09-24; nothing imported it.
"""
import tomllib

import numpy as np
import pandas as pd

from . import config

BANDS_FILE = config.DATA_DIR / "config" / "bands.toml"

with open(BANDS_FILE, "rb") as _f:
    _CFG = tomllib.load(_f)

# {iddev: {sensor: {tier: upper_bound}}}; see bands.toml's header for the ladder.
BANDS = {int(k[len("iddev"):]): v for k, v in _CFG.items() if k.startswith("iddev")}

# Devices with no bands.toml entry (iddev3-10 have no operating history yet) fall
# back to this device's band -- a placeholder until each gets its own calibration.
DEFAULT_IDDEV = _CFG["default_iddev"]


def get_bands(iddev):
    return BANDS.get(iddev, BANDS[DEFAULT_IDDEV])


# (near_band, far_band) per (sensor, subtype): which band tier is the
# boundary where "concern" starts, and which is the value at full failure.
# far=0 is a literal value (not a band-tier lookup), for failure modes that
# fall toward zero rather than toward another named tier. Same formula
# (val - near) / (far - near) works for both rising and falling failure
# modes -- resolved against a specific device's BANDS by get_band_edges().
#
# UPDATED 2026-09-22 to the strict-PDM-Category taxonomy (see
# toe-lasting/pipeline/README.md) -- replaces the old 4-subtype
# free-text-reclassification taxonomy entirely. Must exactly match
# toe-lasting/pipeline/bands.py's EDGE_BANDS (the training-time twin of this
# mapping) -- a model trained on one definition and scored with another is wrong.
_EDGE_BANDS = {
    ("vibration", "bearing_seizure"): ("normal", "critical"),
    ("vibration", "motor_winding_failure"): ("normal", "critical"),
    ("temperature", "heater_burnout"): ("normal", "critical"),
    ("temperature", "thermocouple_failure"): ("standby", 0),
    ("pressure", "pump_failure"): ("warning", 0),
}


def get_band_edges(iddev, sensor, subtype):
    sensor_bands = get_bands(iddev)[sensor]
    near_key, far_key = _EDGE_BANDS[(sensor, subtype)]
    near_edge = sensor_bands[near_key]
    far_edge = far_key if isinstance(far_key, (int, float)) else sensor_bands[far_key]
    return near_edge, far_edge


# which direction each modeled subtype's precursor departs from "normal" --
# used to decide, from a single current reading, which subtype (if any)
# could plausibly be in progress for a sensor shared by more than one
# subtype (e.g. vibration has both bearing_seizure and motor_winding_failure,
# both rising -- ambiguous which one is in progress from vibration alone,
# same limitation the old taxonomy had between its two rising subtypes).
# Direction is structural (which tier a failure mode moves toward), not a
# band value, so it doesn't vary per device. Must match
# toe-lasting/pipeline/train_rul_models.py's DIRECTION.
DIRECTION = {
    ("vibration", "bearing_seizure"): "rise",
    ("vibration", "motor_winding_failure"): "rise",
    ("temperature", "heater_burnout"): "rise",
    ("temperature", "thermocouple_failure"): "fall",
    ("pressure", "pump_failure"): "fall",
}


def label_vibration(s, iddev):
    b = get_bands(iddev)["vibration"]
    return np.select(
        [pd.isna(s) | (s == 0), s < b["standby"], s <= b["normal"], s <= b["warning"], s <= b["critical"]],
        ["off", "standby", "normal", "warning", "critical"], default="danger",
    )


def label_temperature(s, iddev):
    b = get_bands(iddev)["temperature"]
    return np.select(
        [pd.isna(s) | (s == 0), s < b["standby"], s <= b["normal"], s <= b["warning"], s <= b["critical"]],
        ["off", "standby", "normal", "warning", "critical"], default="danger",
    )


def label_pressure(s, iddev):
    b = get_bands(iddev)["pressure"]
    return np.select(
        [pd.isna(s) | (s == 0), s < b["critical"], s <= b["warning"], s <= b["normal"]],
        ["off", "critical", "warning", "normal"], default="danger",
    )


LABEL_FN = {"vibration": label_vibration, "temperature": label_temperature, "pressure": label_pressure}


def band_progress(value, iddev, sensor, subtype):
    """0 at the normal/warning boundary, 1 at the failure extreme, clipped
    both sides -- see get_band_edges() docstring above."""
    near_edge, far_edge = get_band_edges(iddev, sensor, subtype)
    return float(np.clip((value - near_edge) / (far_edge - near_edge), 0, 1))


def has_left_normal_toward_failure(value, iddev, sensor, subtype):
    """True once `value` has crossed out of the official normal band in the
    direction this subtype's precursor moves -- the live trigger point for
    starting to score this subtype. Does NOT match training's definition of
    'episode start' (a real normal-status reading, before any visible
    deviation) -- that point has no observable signal and can't be detected
    live. This is the closest practical proxy: it fires as soon as there's
    *some* evidence, not from t=0 of the true (unknowable) precursor. See
    project memory for the full reasoning."""
    return band_progress(value, iddev, sensor, subtype) > 0


# ============================================================================
# Adaptive (session-relative) band_progress -- TEMPERATURE subtypes only.
# A recipe (glue/shoe model) change moves the heating setpoint, which makes the
# fixed edges above stale -- confirmed live 2026-09-28 (iddev1's real temperature
# dropped 254->163C on 2026-09-24 and thermocouple_failure has been falsely
# TRIGGERED ever since, since standby=220 no longer describes its current
# normal range). Vibration/pressure don't need this (confirmed empirically,
# project memory "toe-lasting adaptive band prototype") and stay on the fixed
# band_progress() above. This is the live-side twin of
# toe-lasting/pipeline/bands.py's copy of the same -- keep both in sync.
# ============================================================================

# DISABLED 2026-09-28 -- kept EMPTY on purpose, do not re-enable without a redesign.
# Must stay in sync with toe-lasting/pipeline/bands.py's copy -- see its comment for the
# full finding (session-scoped rolling baseline gets contaminated by the very precursor
# it's meant to detect: verified on real injected episodes, not just healthy-history
# backtests). heater_burnout in particular showed a SILENT failure mode -- band_progress
# stuck at 0.000 for an entire real overheat ramp up to 20 min before the crash -- worse
# than a false alarm because nothing on the dashboard looks wrong.
ADAPTIVE_SUBTYPES = set()


def is_adaptive(sensor, subtype):
    return (sensor, subtype) in ADAPTIVE_SUBTYPES


def get_adaptive_config(iddev):
    """[iddevN.temperature_adaptive] from bands.toml. No fallback to default_iddev on
    purpose -- k's don't transfer between devices (confirmed empirically: iddev1 vs
    iddev2 differ ~7-10x, not just recipe-dependent). Raises for a device that hasn't
    been calibrated yet, rather than silently borrowing another device's numbers."""
    key = f"iddev{iddev}"
    if key not in _CFG or "temperature_adaptive" not in _CFG[key]:
        raise KeyError(
            f"No [iddev{iddev}.temperature_adaptive] calibration in bands.toml -- "
            f"adaptive band_progress for temperature subtypes needs its own per-device "
            f"calibration, it cannot fall back to another device's."
        )
    return _CFG[key]["temperature_adaptive"]


def adaptive_edges(baseline, spread, iddev, sensor, subtype):
    """near/far edges for an adaptive subtype, resolved against THIS session's current
    baseline/spread instead of a fixed bands.toml number -- same (near, far) shape as
    get_band_edges()'s fixed lookup above. baseline/spread may be scalars or same-length
    arrays/Series (vectorizes over a whole session)."""
    cfg = get_adaptive_config(iddev)
    if sensor == "temperature" and subtype == "heater_burnout":
        return baseline + cfg["k_near"] * spread, baseline + cfg["k_far"] * spread
    if sensor == "temperature" and subtype == "thermocouple_failure":
        return baseline - cfg["k_near_thermo"] * spread, 0.0
    raise ValueError(f"{(sensor, subtype)} is not an adaptive subtype -- use get_band_edges() instead")
