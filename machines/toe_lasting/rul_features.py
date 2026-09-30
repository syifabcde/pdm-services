"""
Live feature computation for the RUL models in toe-lasting/models/rul/.

Mirrors the training-time feature logic in
toe-lasting/pipeline/feature_engineering.py's compute_episode_features()
as closely as possible, so a live feature vector matches what the model was
actually trained on. Two differences, both deliberate:

  1. Training computed rolling/slope features PER INJECTED EPISODE (rows
     tagged is_synthetic=True for one failure occurrence). Live has no such
     concept -- there's no "episode" until a failure is already suspected.
     The natural live analog is PER SESSION (one continuous on-time run,
     same session_break_min boundary used elsewhere in this app) instead:
     by the time a live reading's band_progress leaves 0 (see below), the
     current session will already hold many rolling-window's worth of
     history, so this substitution doesn't change what the rolling/slope
     features actually see.
  2. `time_since_episode_start_min` is DELIBERATELY excluded. Its t=0 in
     training is a real 'normal'-status reading chosen by the injector --
     indistinguishable from any other ordinary reading, with no observable
     marker. A live system has no way to know, in the moment, which normal
     reading is secretly about to start a multi-hour precursor, so this
     feature cannot be computed consistently at inference time (see project
     memory "toe-lasting synthetic 2024-2026 build" for the full reasoning
     and the head-to-head accuracy comparison with/without it). The models
     in toe-lasting/models/rul/ were trained WITHOUT it for this reason.
"""
import numpy as np
import pandas as pd

from . import config
from .rul_bands import DIRECTION, LABEL_FN, adaptive_edges, get_adaptive_config, get_band_edges, is_adaptive


# Everything here that depends on how often a device polls is read PER DEVICE from
# toe-lasting/config/devices.toml (the same file the training pipeline uses): iddev1 polls
# every 5 min, iddev2 every 2 min. Times stay in minutes; only the number of readings inside
# them differs between devices.
def roll_windows(iddev):
    """(short, long) rolling windows in minutes -- the ones the device's model was trained on."""
    return tuple(config.device(iddev)["roll_windows_min"])


def session_break_minutes(iddev):
    """A gap longer than this ends a session (rolling features restart)."""
    return config.device(iddev)["session_break_min"]


def data_age_minutes(raw, now=None):
    """Minutes since the newest reading in `raw`. `now` is naive local time: the IoT DB stamps
    created_at in local time (Asia/Jakarta, timestamp without tz) and the dashboard host runs on
    the same clock. Kept out of latest_rul_inputs() on purpose -- that one is also used to replay
    historical windows (rul_backtest), where 'now' is meaningless."""
    now = pd.Timestamp.now() if now is None else now
    return (now - raw["created_at"].max()).total_seconds() / 60


def is_stale(age_min, iddev):
    """True once the device has been silent for longer than a session break. By the model's own
    definition the last session is then over, so its 'triggered' state and RUL describe a machine
    that is no longer reporting (and 'Normal' would just be its last reading before it went quiet)."""
    return age_min > session_break_minutes(iddev)


def trigger_readings(iddev):
    """Consecutive out-of-band readings required before a subtype is scored."""
    return config.device(iddev)["trigger_readings"]


def trigger_minutes(iddev):
    return trigger_readings(iddev) * config.device(iddev)["cadence_s"] / 60


def feature_cols(iddev):
    """Model input columns for this device; names carry the real windows. Must equal
    toe-lasting/pipeline/feature_engineering.feature_cols() -- rul_predict.load_models()
    checks the saved model against this."""
    short, long_ = roll_windows(iddev)
    return [
        "delta_prev",
        f"roll_mean_{short}min", f"roll_std_{short}min", f"roll_mean_{long_}min", f"roll_std_{long_}min",
        f"slope_{long_}min_per_sec", "band_progress", "value",
    ]


# 'off' = vibration AND pressure both below a small noise-floor threshold --
# NOT literal ==0 (features.py's/anomaly.py's stricter rule), and NOT the
# official 'standby' (vibration <0.17) or 'critical' (pressure <35) band
# edges either, since those are real ON statuses / genuine failure signals,
# not off. Calibrated against real shutdown/warmup transition noise found
# via rul_backtest.py: vibration lingers at 0.01-0.07 (gap-checked against
# the full real vibration distribution -- next real cluster starts at 0.12,
# nothing in between) while the machine spins down/up; pressure similarly
# drifts through a few kg/cm2 before settling. Using the full standby/
# critical band edges here would misclassify genuine standby activity (or,
# for pressure, an actual leak-precursor reading) as 'off'. See project
# memory "toe-lasting raw log data quality" for the specific false-trigger
# cases (2026-08-04 shift-end, 2026-08-07/12/14 morning warmup) this fixes.
VIBRATION_OFF_THRESHOLD = 0.07
PRESSURE_OFF_THRESHOLD = 15

# a single out-of-band reading is overwhelmingly sensor noise, not a real
# precursor -- backtested against real iddev2 history (pdm_tl_log_iddev2_
# cleaned.csv): requiring 1 reading gives 606 (heating_failure_dropout) and
# 386 (gradual_leak_then_rupture) separate "trigger episodes" over ~5
# months, median duration ONE reading (2min) each -- textbook alert fatigue.
# Requiring 5 consecutive readings (~10min) cuts that to 34 and 8
# respectively, while real precursors (hours long) are unaffected.
# The count is per device (trigger_readings in devices.toml). iddev1's 5-min cadence makes
# 5 readings = 25 min, kept on purpose: measured on its real healthy window, 2 readings gave
# 82 false pump_failure episodes in 2 months, 5 gave 0.

# UPDATED 2026-09-22 to the strict-PDM-Category taxonomy -- replaces the old
# 4-subtype list entirely (see toe-lasting/pipeline/README.md). Must
# match toe-lasting/pipeline/feature_engineering.py's GRADUAL_SUBTYPES
# and the `subtypes` list saved inside toe-lasting/models/rul/iddevN/xgb_pooled_v2.joblib.
SUBTYPES = [
    ("vibration", "bearing_seizure"),
    ("vibration", "motor_winding_failure"),
    ("temperature", "heater_burnout"),
    ("temperature", "thermocouple_failure"),
    ("pressure", "pump_failure"),
]

# official PDM Skenario.xlsx part name per subtype -- for display only (e.g.
# dashboard.py's subtype legend), not used in any feature/model computation.
PART_NAMES = {
    ("vibration", "bearing_seizure"): "Bearing",
    ("vibration", "motor_winding_failure"): "Motor",
    ("temperature", "heater_burnout"): "Heater/Nirlamp",
    ("temperature", "thermocouple_failure"): "Thermocouple",
    ("pressure", "pump_failure"): "Pompa",
}

# one-line description of the failure mode itself, in plain terms -- display
# only (dashboard.py's subtype legend), see toe-lasting/config/
# injection_parameters.csv for the full pattern_shape/confidence notes.
FAILURE_DESCRIPTIONS = {
    ("vibration", "bearing_seizure"): "Bearing aus bertahap lalu macet",
    ("vibration", "motor_winding_failure"): "Motor terbakar (winding)",
    ("temperature", "heater_burnout"): "Heater terbakar akibat korslet",
    ("temperature", "thermocouple_failure"): "Thermocouple gagal memanas",
    ("pressure", "pump_failure"): "Pompa terbakar sehingga tekanan tidak stabil",
}

# categories the pooled model (xgb_pooled_v2.joblib) was trained with -- used
# to tag every feature row with matching pandas Categorical dtype for the
# 'sensor'/'subtype' columns below. Order/membership must match the bundle's
# saved sensor_categories/subtype_categories exactly, or XGBoost's
# categorical integer encoding at inference won't line up with training.
ALL_SENSORS = sorted({sensor for sensor, _ in SUBTYPES})
ALL_SUBTYPES = sorted({subtype for _, subtype in SUBTYPES})

def add_session_id(df, session_break_min):
    """Keep only 'on' rows (vibration AND pressure both below their
    noise-floor threshold = off -- see VIBRATION_OFF_THRESHOLD/
    PRESSURE_OFF_THRESHOLD above) and assign a session_id per contiguous
    run.
    """
    is_off = (df["vibration"].fillna(0) < VIBRATION_OFF_THRESHOLD) & (df["pressure"].fillna(0) < PRESSURE_OFF_THRESHOLD)
    df_on = df[~is_off].sort_values("created_at").copy().reset_index(drop=True)

    gap_min = df_on["created_at"].diff().dt.total_seconds() / 60
    df_on["session_id"] = (gap_min > session_break_min).cumsum()
    return df_on


def _trailing_slope(times, values, idx, window_min):
    end_t = times.iat[idx]
    start_t = end_t - pd.Timedelta(minutes=window_min)
    mask = (times > start_t) & (times <= end_t)
    sub_t, sub_v = times[mask], values[mask]
    if len(sub_t) < 3:
        return np.nan
    x = (sub_t - sub_t.iat[0]).dt.total_seconds().values
    y = sub_v.values
    if np.ptp(x) == 0:
        return 0.0
    slope, _ = np.polyfit(x, y, 1)
    return slope


def compute_session_features(session_df, sensor, subtype, iddev):
    """session_df: one session's rows (single session_id), sorted by
    created_at, for ONE sensor. Returns session_df with feature columns
    added, computed the same way as the training pipeline's
    compute_episode_features() (minus time_since_episode_start_min).

    `iddev` selects which device's BANDS (app.rul_bands) to score
    band_progress against -- see project memory "toe-lasting AI early
    warning" for why this can't be a single global band across devices.

    Also tags every row with 'sensor'/'subtype' as pandas Categorical
    (categories = ALL_SENSORS/ALL_SUBTYPES) -- the deployed model
    (xgb_pooled_v2.joblib) is a single POOLED model across all 5 subtypes,
    so it needs sensor/subtype as explicit categorical input features to
    know which subtype's pattern it's scoring. A plain .astype('category')
    on a single-subtype slice would only ever see ONE category value and
    encode it inconsistently with what the model saw at training time --
    the fixed category list here is what keeps the encoding aligned."""
    g = session_df.sort_values("created_at").reset_index(drop=True)
    val = g[sensor]

    g["delta_prev"] = val.diff()
    if is_adaptive(sensor, subtype):
        # session-relative edges -- baseline/spread are THIS session's own trailing
        # rolling median/MAD (session_df already IS one session's rows, so no separate
        # session-splitting step is needed here, unlike the training-side twin in
        # toe-lasting/pipeline/bands.py's add_session_baseline(), which has to compute
        # this across a whole multi-session replica first).
        cfg = get_adaptive_config(iddev)
        win_min, min_periods = cfg["win_min"], cfg["min_periods"]
        s = pd.Series(val.values, index=g["created_at"])
        roll = s.rolling(f"{win_min}min", min_periods=min_periods)
        baseline = roll.median()
        spread = (s - baseline).abs().rolling(f"{win_min}min", min_periods=min_periods).median() * 1.4826
        g["baseline"], g["spread"] = baseline.values, spread.values
        near_edge, far_edge = adaptive_edges(g["baseline"], g["spread"], iddev, sensor, subtype)
    else:
        near_edge, far_edge = get_band_edges(iddev, sensor, subtype)
    g["band_progress"] = np.clip((val - near_edge) / (far_edge - near_edge), 0, 1)
    g["sensor"] = pd.Categorical([sensor] * len(g), categories=ALL_SENSORS)
    g["subtype"] = pd.Categorical([subtype] * len(g), categories=ALL_SUBTYPES)

    s = pd.Series(val.values, index=g["created_at"])
    windows = roll_windows(iddev)
    for w in windows:
        roll = s.rolling(f"{w}min", min_periods=2)
        g[f"roll_mean_{w}min"] = roll.mean().values
        g[f"roll_std_{w}min"] = roll.std().values

    g[f"slope_{windows[-1]}min_per_sec"] = [
        _trailing_slope(g["created_at"], val, i, windows[-1]) for i in range(len(g))
    ]
    g["value"] = val
    return g


def latest_status(raw, sensor, iddev):
    """Official status (off/standby/normal/warning/critical/danger) of the
    most recent reading for `sensor`, using `iddev`'s corrected bands
    (app.rul_bands, i.e. config/bands.toml) -- not another device's band."""
    if raw.empty:
        return None
    latest_val = raw.sort_values("created_at")[sensor].iloc[-1]
    return LABEL_FN[sensor]([latest_val], iddev)[0]


def latest_rul_inputs(raw, iddev, session_break_min=None):
    """For every modeled (sensor, subtype), return the latest feature row
    ready for model.predict() if that subtype's failure direction is
    currently triggered (band_progress > 0) and there's enough session
    history -- else None for that subtype, with the reason.

    `iddev` selects which device's BANDS (app.rul_bands) to score against --
    `raw` is expected to already be scoped to that one device.

    Returns: {(sensor, subtype): {"row": DataFrame(1 row) | None,
                                   "triggered": bool, "reason": str}}
    """
    session_break_min = session_break_minutes(iddev) if session_break_min is None else session_break_min
    n_trigger = trigger_readings(iddev)
    df_on = add_session_id(raw, session_break_min)
    results = {}

    for sensor, subtype in SUBTYPES:
        key = (sensor, subtype)
        if df_on.empty:
            results[key] = {"row": None, "triggered": False, "reason": "no on-session data"}
            continue

        latest_session_id = df_on["session_id"].max()
        session_df = df_on[df_on["session_id"] == latest_session_id].sort_values("created_at")
        # compute_session_features() up front (not just band_progress via a scalar call)
        # because an adaptive subtype's band_progress needs the session's OWN rolling
        # baseline/spread, not just the single value -- there's no meaningful scalar
        # shortcut for it the way there was for a fixed band.
        feat = compute_session_features(session_df, sensor, subtype, iddev)
        bp_recent = feat["band_progress"].tail(n_trigger)

        if len(bp_recent) < n_trigger or bp_recent.isna().any() or not (bp_recent > 0).all():
            results[key] = {"row": None, "triggered": False,
                             "reason": f"{sensor} not sustained out of normal band toward "
                                       f"{subtype} for {n_trigger} consecutive readings"}
            continue

        latest = feat.dropna(subset=feature_cols(iddev)).tail(1)
        if latest.empty:
            results[key] = {"row": None, "triggered": True,
                             "reason": "triggered but not enough session history for rolling features yet"}
            continue

        results[key] = {"row": latest, "triggered": True, "reason": "ok"}

    return results


def score_all_sessions(raw, bundles, iddev, session_break_min=None):
    """Backtest the RUL pipeline across the WHOLE history in `raw`, not just
    the latest session -- for every (sensor, subtype), find every point
    where the sustained-trigger rule (trigger_readings(iddev)) fires and
    there's enough rolling history, run the model, and return the result.
    Lets the live pipeline be tested against any historical window (real
    Postgres data or a local CSV, including the synthetic episodes) without
    waiting for a real-time trigger.

    `bundles`: {(sensor, subtype): joblib bundle} from rul_predict.load_models().
    `iddev` selects which device's BANDS (app.rul_bands) to score against --
    `raw` is expected to already be scoped to that one device.
    Returns: {(sensor, subtype): DataFrame} -- one row per triggered
    timestamp, with `predicted_rul_minutes` attached (empty DataFrame if
    that subtype never triggered in this window).
    """
    session_break_min = session_break_minutes(iddev) if session_break_min is None else session_break_min
    n_trigger = trigger_readings(iddev)
    df_on = add_session_id(raw, session_break_min)
    results = {}

    for sensor, subtype in SUBTYPES:
        key = (sensor, subtype)
        if df_on.empty:
            results[key] = pd.DataFrame()
            continue

        session_frames = []
        for _, session_df in df_on.groupby("session_id"):
            session_df = session_df.sort_values("created_at")
            # feat computed unconditionally now (was: cheap scalar band_progress check
            # first, full features only if sustained.any()) -- an adaptive subtype's
            # band_progress needs the session's own rolling baseline/spread, so there is
            # no scalar-only shortcut left to check sustainment before paying for it.
            feat = compute_session_features(session_df, sensor, subtype, iddev)
            triggered = (feat["band_progress"] > 0).fillna(False)
            sustained = (
                triggered.rolling(n_trigger, min_periods=n_trigger)
                .apply(lambda x: x.all(), raw=True)
                .fillna(0)
                .astype(bool)
            )
            if not sustained.any():
                continue

            feat = feat[sustained.values].dropna(subset=feature_cols(iddev))
            if feat.empty:
                continue

            bundle = bundles[key]
            feat = feat.copy()
            feat["predicted_rul_minutes"] = np.maximum(
                0.0, bundle["model"].predict(feat[bundle["feature_cols"]]) / 60
            )
            session_frames.append(feat)

        results[key] = pd.concat(session_frames, ignore_index=True) if session_frames else pd.DataFrame()

    return results
