"""
Display-side smoothing of the RUL series.

The model scores every reading on its own, so consecutive predictions wander far more than the
5 minutes of countdown that really separate two readings: on held-out synthetic episodes and on the
real iddev1 series the failure time a prediction implies moves 30-60 min per reading. A countdown
should mostly just count down, so each prediction is turned into the failure time it implies
(reading time + RUL) and the MEDIAN of the last few is the failure time to show; the RUL shown is
that time minus the newest reading's time. The raw predictions are never altered -- rul_store keeps
them exactly as the model produced them, and everything here is computed from them on the fly.

What this is not: a better prediction. It steadies the number; it cannot fix a model that keeps
changing its mind for good (the display then follows after ~2 readings), and it says nothing about
whether the model is right. The (eta_low, eta_high) range is just the spread of the model's own
recent guesses, not a statistical confidence interval.
"""
import numpy as np
import pandas as pd

# Median over the last 4 readings (~20 min at iddev1's 5-min cadence). Picked on iddev1's held-out
# synthetic episodes: implied-failure-time jitter 31 -> 13.5 min/reading for <=1 min of MAE. Not
# tested on iddev2's 2-min cadence (no live data there yet), where 4 readings is only ~8 min.
SMOOTH_READINGS = 4

ADDED_COLS = ["rul_smooth_min", "eta_hat", "eta_low", "eta_high", "n_used"]


def smooth_rul(df, max_gap_min, n=SMOOTH_READINGS):
    """df: the predictions of ONE (iddev, sensor, subtype) -- columns created_at (datetime) and
    predicted_rul_minutes (raw model output), any others are kept. Returns a copy, oldest first, plus:
        rul_smooth_min   RUL to show (never below 0)
        eta_hat          failure time to show (= created_at + rul_smooth_min)
        eta_low/eta_high earliest / latest failure time among the readings pooled
        n_used           how many readings were pooled (fewer at the start of a run)
    A reading pools only with earlier readings of the same run: a gap longer than `max_gap_min`
    (the device's session break) starts a new run, so estimates never bridge a machine stop."""
    df = df.sort_values("created_at").reset_index(drop=True)
    if df.empty:
        return df.assign(**{c: pd.Series(dtype="float64") for c in ADDED_COLS})

    created = df["created_at"]
    eta = created + pd.to_timedelta(df["predicted_rul_minutes"], unit="min")
    gap = pd.Timedelta(minutes=max_gap_min)

    rul, low, high, used = [], [], [], []
    for i in range(len(df)):
        j = i
        while i - j + 1 < n and j > 0 and created[j] - created[j - 1] <= gap:
            j -= 1
        window = eta.iloc[j:i + 1]
        # the RUL each pooled estimate implies, measured from reading i; the median of these is
        # the median failure time minus created[i]
        rul.append(max(0.0, float(np.median((window - created[i]).dt.total_seconds())) / 60))
        low.append(window.min())
        high.append(window.max())
        used.append(i - j + 1)

    out = df.assign(rul_smooth_min=rul, eta_low=low, eta_high=high, n_used=used)
    out["eta_hat"] = out["created_at"] + pd.to_timedelta(out["rul_smooth_min"], unit="min")
    return out


def smooth_history(df, max_gap_min, n=SMOOTH_READINGS):
    """smooth_rul() applied separately to every (sensor, subtype) in a mixed predictions table."""
    if df.empty:
        return smooth_rul(df, max_gap_min, n)
    parts = [smooth_rul(g, max_gap_min, n) for _, g in df.groupby(["sensor", "subtype"])]
    return pd.concat(parts, ignore_index=True)
