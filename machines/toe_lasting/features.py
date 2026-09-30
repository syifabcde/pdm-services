import pandas as pd

SESSION_BREAK_MIN = 10
LAGS = [1, 2, 3, 5, 10, 15]
ROLL_WINDOW = 15


def add_session_id(df, session_break_min=SESSION_BREAK_MIN):
    """Keep only 'on' rows and assign a session_id per contiguous run.

    On/off is read from vibration & pressure, not temperature: the
    temperature sensor has thermal inertia and stays elevated well after the
    machine actually stops, so an off-detector based on temperature misses
    ~98% of real downtime and merges sessions across it, which corrupts the
    lag features (see temperature-tuning-rf.ipynb section 4).
    """
    is_off = (df["vibration"].fillna(0) == 0) & (df["pressure"].fillna(0) == 0)
    df_on = df[~is_off].sort_values("created_at").copy().reset_index(drop=True)

    gap_min = df_on["created_at"].diff().dt.total_seconds() / 60
    df_on["session_id"] = (gap_min > session_break_min).cumsum()
    return df_on


def build_features(df_on, lags=LAGS, roll_window=ROLL_WINDOW):
    """Mirrors build_supervised() in temperature-tuning-rf.ipynb, minus the
    target column -- this is for inference, not training."""
    rows = []
    for sess_id, g in df_on.groupby("session_id"):
        g = g.sort_values("created_at").reset_index(drop=True)
        feat = pd.DataFrame(index=g.index)
        feat["created_at"] = g["created_at"]
        feat["session_id"] = sess_id
        feat["elapsed_min"] = (g["created_at"] - g["created_at"].iloc[0]).dt.total_seconds() / 60
        feat["temperature"] = g["temperature"]
        for lag in lags:
            feat[f"temperature_lag{lag}"] = g["temperature"].shift(lag)
        feat["temperature_roll_mean"] = g["temperature"].rolling(roll_window, min_periods=3).mean()
        feat["temperature_roll_std"] = g["temperature"].rolling(roll_window, min_periods=3).std()
        rows.append(feat)
    return pd.concat(rows, ignore_index=True)


def latest_feature_row(df, feature_cols, session_break_min=SESSION_BREAK_MIN,
                        lags=LAGS, roll_window=ROLL_WINDOW):
    """Return the most recent feature row ready for model.predict(), or None
    if the current session doesn't have enough history yet (< max(lags)
    samples since the session started)."""
    df_on = add_session_id(df, session_break_min)
    if df_on.empty:
        return None

    feat = build_features(df_on, lags, roll_window)
    latest_session = feat["session_id"].max()
    latest = feat[feat["session_id"] == latest_session].dropna(subset=feature_cols)
    if latest.empty:
        return None
    return latest.iloc[[-1]]
