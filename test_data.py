"""Load and preprocess the SPOT master table, then print a short summary."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from load import load_source_tables
from preprocess import preprocess_master_frame


def main() -> None:
    tables = load_source_tables()
    df = preprocess_master_frame(tables)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    print(df.shape)
    print(df.head())
    summary = df.describe(include="all").T
    summary.insert(0, "dtype", df.dtypes.astype(str))
    print(summary)


if __name__ == "__main__":
    main()
