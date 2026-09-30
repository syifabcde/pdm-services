"""
Manual test helper for machines/toe_lasting/rul_notify.py's trigger alert template --
so the template can be iterated on and checked without going through the real dashboard/DB,
and without me (Claude) running the send. Bypasses notify_if_new_trigger()'s debounce on
purpose: this is for eyeballing the template repeatedly, not for exercising the debounce logic.

Run from pdm-services/ :
    venv\\Scripts\\python.exe preview_rul_alert.py            # just prints the message (no send)
    venv\\Scripts\\python.exe preview_rul_alert.py --send     # also sends it to the Telegram dev/test bot

Edit the dummy values below to try different scenarios (sensor/subtype, building/cell, the
numbers) -- every run re-imports rul_notify fresh, so any edit you've made to
_format_message() in rul_notify.py is picked up immediately, no restart needed.
"""
import argparse
import sys

if sys.platform == "win32":
    # emoji in the template crash a plain Windows console (cp1252) otherwise -- see chat
    # 2026-09-29 for the exact traceback this avoids.
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

from core import notify
from machines.toe_lasting import rul_notify

# ---- edit these to try different scenarios ----
SENSOR = "temperature"                  # "temperature" | "vibration" | "pressure"
SUBTYPE = "thermocouple_failure"        # must be a real (sensor, subtype) pair -- see
                                         # rul_features.SUBTYPES for the other 4
BUILDING = "D1"                         # or None to see the "belum ada mapping" case
CELL = "01"                             # or None
CURRENT_VALUE = 78.4
BAND_PROGRESS = 0.62                    # 0-1
RUL_MINUTES = 128.0
N_USED = 5                              # >1 shows the "kisaran ..." spread text; 1 hides it
CREATED_AT = pd.Timestamp("2026-09-29 14:32:10")
# -------------------------------------------------


def build_smooth():
    return pd.Series({
        "rul_smooth_min": RUL_MINUTES,
        "eta_hat": CREATED_AT + pd.Timedelta(minutes=RUL_MINUTES),
        "eta_low": CREATED_AT + pd.Timedelta(minutes=max(0, RUL_MINUTES - 13)),
        "eta_high": CREATED_AT + pd.Timedelta(minutes=RUL_MINUTES + 17),
        "n_used": N_USED,
    })


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--send", action="store_true", help="actually send to Telegram, not just print")
    args = ap.parse_args()

    msg = rul_notify._format_message(
        SENSOR, SUBTYPE, CREATED_AT, CURRENT_VALUE, BAND_PROGRESS, build_smooth(),
        building=BUILDING, cell=CELL,
    )
    print(msg)

    if args.send:
        ok = notify.send_telegram(msg, parse_mode="Markdown")
        print(f"\n--- terkirim: {ok} ---")
    else:
        print("\n--- belum dikirim -- tambah --send buat kirim ke Telegram beneran ---")
