"""
Toe-lasting's sections of the shared dashboard. The page itself (title, machine/device
selectors, auto-refresh) lives in pdm-services/dashboard.py; this module only draws what is
specific to this machine for ONE device, via render(iddev).
"""
import math
import os

import streamlit as st

from . import band_drift_watch, config, db, rul_features, rul_smooth, rul_store
from .rul_predict import load_models as load_rul_models, model_path, score_and_notify

FETCH_MINUTES = 1000   # same time span for every device (500 rows at iddev2's 2-min cadence, 200 at iddev1's 5-min)
CARD_COLUMNS = 3       # status cards per row


@st.cache_resource
def get_engine():
    return db.get_engine()


@st.cache_resource
def _load_rul_models(iddev, model_mtime):
    # model_mtime only exists to be part of the cache key: retraining replaces the .joblib, and the
    # next rerun then reloads it instead of serving the copy cached when the server started
    return load_rul_models(iddev)


def get_rul_models(iddev):
    path = model_path(iddev)
    # a missing file is left to load_models(), which raises FileNotFoundError with the helpful message
    return _load_rul_models(iddev, os.path.getmtime(path) if os.path.exists(path) else None)


# band_drift_watch is a slow, advisory signal (days-scale, not minutes) -- cached with a long TTL so
# it doesn't re-query LOOKBACK_DAYS of history on every ~1-minute rerun (see project memory "PDM
# dashboard ~1min rerun" for why reruns happen that often).
@st.cache_data(ttl=3600)
def _fetch_for_drift_check(iddev):
    engine = get_engine()
    minutes = band_drift_watch.LOOKBACK_DAYS * 24 * 60
    limit = math.ceil(minutes / (config.device(iddev)["cadence_s"] / 60))
    return db.fetch_recent(engine, iddev, limit=limit)


def render_drift_banner(iddev):
    """Advisory-only: never touches RUL scoring, just flags for a human to review. See
    band_drift_watch.py's docstring for why this exists as a separate, slow, human-in-the-loop
    check instead of feeding back into band_progress automatically."""
    try:
        raw = _fetch_for_drift_check(iddev)
    except Exception:
        return  # a drift-check hiccup should never take down the rest of the page
    if raw.empty:
        return
    result = band_drift_watch.check_temperature_drift(raw, iddev)
    try:
        # cheap + idempotent (only sends a Telegram message on a drifted<->ok transition) --
        # safe to call every rerun even though _fetch_for_drift_check's underlying data is
        # only an hour fresh at most.
        band_drift_watch.update_drift_state_and_notify(iddev, result)
    except Exception:
        pass  # persistence/notify hiccup should never block the banner itself from showing
    if result["drifted"]:
        st.error(
            f"⚠️ Suhu iddev{iddev} kemungkinan sudah bergeser dari band yang dikonfigurasi di "
            f"config/bands.toml: **{result['frac_not_normal']:.0%}** dari bacaan on-time "
            f"{band_drift_watch.LOOKBACK_DAYS} hari terakhir ({result['n_on_readings']} bacaan) "
            f"di luar tier 'normal' (median suhu saat ini ≈ {result['current_median']:.1f}°C). "
            "Kemungkinan recipe/setpoint berubah -- konfirmasi ke engineering sebelum mengubah "
            "band secara manual. RUL di bawah ini masih memakai band lama sampai dikonfirmasi."
        )


def fmt_age(minutes):
    if minutes < 90:
        return f"{minutes:.0f} menit"
    if minutes < 48 * 60:
        return f"{minutes / 60:.1f} jam"
    return f"{minutes / 1440:.0f} hari"


def fmt_time(ts, ref):
    """Clock time, with the date added when it isn't the same day as `ref`."""
    return ts.strftime("%H:%M") if ts.date() == ref.date() else ts.strftime("%d/%m %H:%M")


def render_status_cards(raw, iddev, bundles):
    results = {(r["sensor"], r["subtype"]): r for r in score_and_notify(raw, iddev, bundles)}

    # CARD_COLUMNS cards per row; the last row simply has empty slots left over
    n_rows = math.ceil(len(rul_features.SUBTYPES) / CARD_COLUMNS)
    cols = [col for _ in range(n_rows) for col in st.columns(CARD_COLUMNS)]
    for col, (sensor, subtype) in zip(cols, rul_features.SUBTYPES):
        key = (sensor, subtype)
        r = results[key]
        desc = rul_features.FAILURE_DESCRIPTIONS[key]
        label = f"**{sensor} / {subtype}**\n\n*{desc}*"

        if r["status"] == "normal":
            col.success(f"{label}\n\nNormal")
            continue

        if r["status"] == "triggered_no_data":
            # sustained out-of-band, but not enough session history yet for a real RUL number --
            # rul_notify's debounce state is deliberately left untouched for this branch (see
            # score_and_notify): this isn't "back to normal" (don't re-arm) and there's nothing
            # actionable to alert on yet.
            col.warning(f"{label}\n\nTriggered -- {r['reason']}")
            continue

        created_at = r["as_of"]
        if r["n_used"] > 1:
            span_min = r["n_used"] * config.device(iddev)["cadence_s"] / 60
            spread = (f"tebakan model dari {r['n_used']} bacaan terakhir (~{span_min:.0f} menit) berkisar "
                      f"{fmt_time(r['eta_low'], created_at)} sampai {fmt_time(r['eta_high'], created_at)}")
        else:
            spread = "baru 1 tebakan, belum ada kisaran"

        col.error(
            f"{label}\n\nTRIGGERED @ {created_at.strftime('%H:%M:%S')}\n\n"
            f"nilai={r['current_value']:.3f} | band_progress={r['band_progress']:.2f}\n\n"
            f"prediksi RUL ~ **{r['rul_smooth_minutes']:.0f} menit** "
            f"(diperkirakan rusak sekitar {fmt_time(r['eta'], created_at)})\n\n"
            f"{spread}"
        )


def render_rul_section(raw, iddev, age_min):
    st.subheader("RUL Prediction")
    st.caption(
        "Model : XGBoost gabungan (1 model, 5 subtype gradual) | trigger : keluar band normal, "
        f"bertahan >= {rul_features.trigger_readings(iddev)} bacaan berturut (~{rul_features.trigger_minutes(iddev):.0f} menit)"
    )

    try:
        bundles = get_rul_models(iddev)
    except FileNotFoundError as e:
        st.warning(f"RUL belum tersedia untuk iddev={iddev}: {e}")
        return

    if rul_features.is_stale(age_min, iddev):
        st.warning(
            f"Data terakhir device ini sudah **{fmt_age(age_min)} lalu** "
            f"(batas {rul_features.session_break_minutes(iddev)} menit): mesin berhenti atau logger tidak mengirim data. "
            "Status subtype dan RUL tidak ditampilkan karena angka lama bisa menyesatkan."
        )
    else:
        render_status_cards(raw, iddev, bundles)

    gap = rul_features.session_break_minutes(iddev)
    all_history = rul_smooth.smooth_history(rul_store.get_predictions(iddev, limit=50), gap)
    if not all_history.empty:
        st.subheader("Riwayat prediksi RUL (saat triggered)")
        st.dataframe(
            all_history.sort_values("created_at", ascending=False)[
                ["created_at", "sensor", "subtype", "current_value", "band_progress", "rul_smooth_min"]
            ].rename(columns={"rul_smooth_min": "rul_minutes"}).round({"rul_minutes": 1}),
            width='stretch',
        )


def render(iddev):
    engine = get_engine()

    limit = math.ceil(FETCH_MINUTES / (config.device(iddev)["cadence_s"] / 60))
    raw = db.fetch_recent(engine, iddev, limit=limit)
    if raw.empty:
        st.warning(f"Tidak ada data untuk iddev={iddev}")
        return

    latest = raw.iloc[-1]
    age_min = rul_features.data_age_minutes(raw)
    st.caption(
        f"Update terakhir dari DB: {latest['created_at']} ({fmt_age(age_min)} lalu) | "
        f"temperature={latest['temperature']:.2f} | vibration={latest['vibration']:.2f} | "
        f"pressure={latest['pressure']:.2f}"
    )

    render_drift_banner(iddev)
    render_rul_section(raw, iddev, age_min)
