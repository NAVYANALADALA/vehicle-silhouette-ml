"""Build data/vehicle_silhouettes.csv from the UCI Statlog (Vehicle Silhouettes)
data: https://archive.ics.uci.edu/dataset/149/statlog+vehicle+silhouettes

Inputs (first one found is used):
  data/raw/xaa.dat ... xai.dat  - the original UCI files (18 integers + class name per line)
  data/vehicle.tsv.gz           - the same 846 records from the PMLB project, where the
                                  label is coded 1-4; codes are mapped back to names using
                                  the UCI class counts, which are all different.
The result is checked against the UCI shape (846 x 19) and class counts.

Usage:  python prepare_dataset.py [--source auto|uci|pmlb]
"""
import argparse
import sys

import pandas as pd

from project_config import (CLEAN_DATA_FILE, DATA_DIR, EXPECTED_CLASS_COUNTS,
                            FEATURES, TARGET)

RAW_DIR = DATA_DIR / "raw"
PMLB_FILE = DATA_DIR / "vehicle.tsv.gz"


def load_from_uci(raw_dir):
    """Read and concatenate the nine original UCI .dat files."""
    files = sorted(raw_dir.glob("xa?.dat"))
    if not files:
        raise FileNotFoundError(f"No xa?.dat files found in {raw_dir}")
    frames = [
        pd.read_csv(f, sep=r"\s+", header=None, names=FEATURES + [TARGET])
        for f in files
    ]
    df = pd.concat(frames, ignore_index=True)
    df[TARGET] = df[TARGET].str.strip().str.lower()
    print(f"Loaded {len(files)} UCI files -> {df.shape}")
    return df


def load_from_pmlb(path):
    """Read the PMLB copy and restore the original class names."""
    df = pd.read_csv(path, sep="\t")
    df.columns = FEATURES + [TARGET]            # same column order as UCI
    counts = df[TARGET].value_counts()
    # map each numeric code to the UCI class with the same (unique) count
    count_to_name = {v: k for k, v in EXPECTED_CLASS_COUNTS.items()}
    code_to_name = {code: count_to_name[n] for code, n in counts.items()}
    df[TARGET] = df[TARGET].map(code_to_name)
    print(f"Loaded PMLB mirror -> {df.shape}; label mapping {code_to_name}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Build the clean vehicle CSV")
    parser.add_argument("--source", choices=["auto", "uci", "pmlb"], default="auto")
    args = parser.parse_args()

    if args.source == "uci" or (args.source == "auto" and any(RAW_DIR.glob("xa?.dat"))):
        df = load_from_uci(RAW_DIR)
    elif PMLB_FILE.exists():
        df = load_from_pmlb(PMLB_FILE)
    else:
        sys.exit("No data found. Put the UCI xa?.dat files in data/raw/ (see docstring).")

    # integrity checks against the UCI documentation
    assert df.shape == (846, 19), f"Unexpected shape {df.shape}"
    observed = df[TARGET].value_counts().to_dict()
    assert observed == EXPECTED_CLASS_COUNTS, f"Class counts differ: {observed}"
    df[FEATURES] = df[FEATURES].astype(int)     # all attributes are integers

    CLEAN_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_DATA_FILE, index=False)
    print(f"Saved {CLEAN_DATA_FILE.name}: {df.shape[0]} rows x {len(FEATURES)} features + target")
    print("Class counts:", observed)


if __name__ == "__main__":
    main()
