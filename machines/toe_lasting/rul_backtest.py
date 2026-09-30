"""
Backtest/replay the RUL pipeline against a historical window -- either a
slice of the real Postgres log (read-only, no writes) or a local CSV
(cleaned real history, or the synthetic failure episodes) -- without
waiting for a real-time trigger to happen on its own.

Usage:
    python -m machines.toe_lasting.rul_backtest --iddev 2 --days 30
    python -m machines.toe_lasting.rul_backtest --iddev 2 --start 2026-08-01 --end 2026-09-01
    python -m machines.toe_lasting.rul_backtest --iddev 2 --csv "D:\\syifa\\pdm\\toe-lasting\\data\\cleaned\\iddev2\\pdm_tl_log_iddev2_cleaned.csv"
"""
import argparse

import pandas as pd

from . import db
from .rul_features import SUBTYPES, score_all_sessions
from .rul_predict import load_models


def run(raw, iddev):
    if raw.empty:
        print("Tidak ada baris di rentang ini.")
        return

    print(f"Replay {len(raw)} baris, {raw['created_at'].min()} -> {raw['created_at'].max()}\n")
    bundles = load_models(iddev)
    results = score_all_sessions(raw, bundles, iddev)

    for sensor, subtype in SUBTYPES:
        df = results[(sensor, subtype)]
        label = f"{sensor}/{subtype}"
        if df.empty:
            print(f"[{label}] tidak pernah triggered di rentang ini")
            continue
        print(f"[{label}] {len(df)} bacaan triggered")
        print(
            df[["created_at", "value", "band_progress", "predicted_rul_minutes"]]
            .rename(columns={"created_at": "time"})
            .to_string(index=False)
        )
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iddev", type=int, required=True, help="device yang di-backtest")
    ap.add_argument("--days", type=int, help="N hari terakhir dari Postgres")
    ap.add_argument("--start", type=str, help="tanggal mulai (YYYY-MM-DD), dipakai bareng --end")
    ap.add_argument("--end", type=str, help="tanggal akhir (YYYY-MM-DD), dipakai bareng --start")
    ap.add_argument("--csv", type=str, help="CSV lokal (skema sama) sebagai ganti Postgres")
    args = ap.parse_args()

    if args.csv:
        raw = pd.read_csv(args.csv)
        raw["created_at"] = pd.to_datetime(raw["created_at"])
        if "iddev" in raw.columns:
            raw = raw[raw["iddev"] == args.iddev]
        run(raw.sort_values("created_at").reset_index(drop=True), args.iddev)
        return

    engine = db.get_engine()
    if args.days:
        end = pd.Timestamp.now()
        start = end - pd.Timedelta(days=args.days)
    elif args.start and args.end:
        start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    else:
        raise SystemExit("Tentukan --days N, atau --start/--end, atau --csv")

    raw = db.fetch_range(engine, args.iddev, start, end)
    run(raw, args.iddev)


if __name__ == "__main__":
    main()
