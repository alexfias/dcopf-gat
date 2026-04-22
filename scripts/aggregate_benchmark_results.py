from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate per-setting benchmark CSVs")
    p.add_argument("--inputs", nargs="+", type=Path, required=True, help="Input CSV files")
    p.add_argument("--output", type=Path, required=True, help="Aggregated CSV output")
    return p.parse_args()


def main():
    args = parse_args()
    frames = [pd.read_csv(path) for path in args.inputs]
    df = pd.concat(frames, ignore_index=True)
    sort_cols = [c for c in ["window_mode", "arch", "window"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved aggregated CSV to {args.output}")


if __name__ == "__main__":
    main()
