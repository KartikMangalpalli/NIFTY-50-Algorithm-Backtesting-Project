"""
data_loader.py
==============
Data loading and preprocessing module for the NIFTY 50 backtesting project.

Responsibilities
----------------
- Load the Excel dataset with robust date parsing.
- Validate OHLCV integrity (no negatives, HIGH >= LOW, etc.).
- Add derived columns (daily return, log return, true range).
- Provide a chronological train / validation / test splitter.
- Expose a normalised equity-index helper (rebased to 1 on the first date).

All functions are pure (no side-effects on the caller's DataFrame) except
where noted.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Default data path (relative to this file's location)
# ---------------------------------------------------------------------------
_MODULE_DIR = Path(__file__).parent
_DEFAULT_DATA_PATH = _MODULE_DIR.parent / "data" / "NIFTY_50_Historical_01012016_to_01012026.xlsx"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_data(filepath: str | Path | None = None) -> pd.DataFrame:
    """Load, clean and enrich the NIFTY 50 Excel dataset.

    Parameters
    ----------
    filepath : str or Path, optional
        Path to the Excel file. Defaults to the bundled dataset.

    Returns
    -------
    pd.DataFrame
        Sorted, indexed DataFrame with the following columns:

        Original
        ~~~~~~~~
        OPEN, HIGH, LOW, CLOSE, SHARES_TRADED, TURNOVER_CR

        Derived
        ~~~~~~~
        PREV_CLOSE   – prior trading-day close (NaN on first row)
        DAILY_RETURN – simple daily return  = CLOSE / PREV_CLOSE – 1
        LOG_RETURN   – log return           = ln(CLOSE / PREV_CLOSE)
        TRUE_RANGE   – max(HIGH-LOW, |HIGH-PREV_CLOSE|, |LOW-PREV_CLOSE|)
        TYPICAL_PRICE– (HIGH + LOW + CLOSE) / 3
        NEXT_OPEN    – next trading-day open (NaN on last row)
                       Used for realistic execution on the following day.

    Index
    ~~~~~
        DATE (DatetimeIndex, UTC-naive)
    """
    path = Path(filepath) if filepath else _DEFAULT_DATA_PATH

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    raw = pd.read_excel(path, engine="openpyxl")

    # ---- Rename columns to safe identifiers ----------------------------------
    raw.columns = raw.columns.str.strip()
    col_map = {
        "DATE": "DATE",
        "OPEN": "OPEN",
        "HIGH": "HIGH",
        "LOW": "LOW",
        "CLOSE": "CLOSE",
        "SHARES TRADED": "SHARES_TRADED",
        "TURNOVER (INR CR)": "TURNOVER_CR",
    }
    missing = [c for c in col_map if c not in raw.columns]
    if missing:
        raise ValueError(f"Expected columns not found: {missing}")

    df = raw[list(col_map.keys())].rename(columns=col_map).copy()

    # ---- Parse dates ---------------------------------------------------------
    df["DATE"] = pd.to_datetime(df["DATE"], dayfirst=True, errors="coerce")
    n_bad_dates = df["DATE"].isna().sum()
    if n_bad_dates:
        warnings.warn(f"{n_bad_dates} rows had unparseable dates and will be dropped.")
        df = df.dropna(subset=["DATE"])

    df = df.sort_values("DATE").reset_index(drop=True)
    df = df.set_index("DATE")

    # ---- Duplicate-date guard ------------------------------------------------
    dupes = df.index.duplicated().sum()
    if dupes:
        warnings.warn(f"{dupes} duplicate date(s) found; keeping last occurrence.")
        df = df[~df.index.duplicated(keep="last")]

    # ---- Basic OHLCV sanity --------------------------------------------------
    _validate_ohlcv(df)

    # ---- Derived columns -----------------------------------------------------
    df["PREV_CLOSE"]     = df["CLOSE"].shift(1)
    df["DAILY_RETURN"]   = df["CLOSE"] / df["PREV_CLOSE"] - 1
    df["LOG_RETURN"]     = np.log(df["CLOSE"] / df["PREV_CLOSE"])
    df["TRUE_RANGE"]     = _true_range(df)
    df["TYPICAL_PRICE"]  = (df["HIGH"] + df["LOW"] + df["CLOSE"]) / 3

    # Next-open: shift OPEN backwards so row[t] holds the open of t+1.
    # This is the realistic execution price for a signal generated at close of t.
    df["NEXT_OPEN"] = df["OPEN"].shift(-1)

    return df


def split_data(
    df: pd.DataFrame,
    train_end: str = "2021-12-31",
    val_end: str = "2023-06-30",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train / validation / test split.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset as returned by :func:`load_data`.
    train_end : str
        Last date (inclusive) of the training set.
    val_end : str
        Last date (inclusive) of the validation set.
        Test set = everything after val_end.

    Returns
    -------
    tuple of (train, val, test) DataFrames

    Notes
    -----
    Approx split (2016-01-01 to 2026-01-01):
        Train : 2016–2021  (~1495 days, ~60%)
        Val   : 2022–mid-2023 (~380 days, ~15%)
        Test  : mid-2023–2026 (~600 days, ~24%)
    """
    train = df.loc[: pd.Timestamp(train_end)]
    val   = df.loc[pd.Timestamp(train_end) + pd.Timedelta(days=1) : pd.Timestamp(val_end)]
    test  = df.loc[pd.Timestamp(val_end)   + pd.Timedelta(days=1) :]
    return train, val, test


def normalise_prices(df: pd.DataFrame, base_col: str = "CLOSE") -> pd.Series:
    """Rebase a price series to 1.0 on its first observation.

    Useful for plotting multiple strategies on the same equity-curve chart.
    """
    s = df[base_col].dropna()
    return s / s.iloc[0]


def describe_dataset(df: pd.DataFrame) -> None:
    """Print a human-readable summary of the loaded dataset."""
    print("=" * 60)
    print("NIFTY 50 Historical Dataset – Summary")
    print("=" * 60)
    print(f"Date range   : {df.index.min().date()}  →  {df.index.max().date()}")
    print(f"Trading days : {len(df):,}")
    print(f"Columns      : {list(df.columns)}")
    print()
    print("OHLC price range")
    for col in ["OPEN", "HIGH", "LOW", "CLOSE"]:
        print(f"  {col:5s}: min={df[col].min():,.2f}   max={df[col].max():,.2f}")
    total_ret = (df["CLOSE"].iloc[-1] / df["CLOSE"].iloc[0] - 1) * 100
    ann_ret   = ((df["CLOSE"].iloc[-1] / df["CLOSE"].iloc[0]) ** (252 / len(df)) - 1) * 100
    print()
    print(f"Total return (buy & hold CLOSE): {total_ret:.2f}%")
    print(f"Ann. return  (buy & hold CLOSE): {ann_ret:.2f}%")
    print()
    print("Missing values:")
    print(df.isnull().sum())
    print("=" * 60)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_ohlcv(df: pd.DataFrame) -> None:
    errors = []
    for col in ["OPEN", "HIGH", "LOW", "CLOSE"]:
        n = (df[col] <= 0).sum()
        if n:
            errors.append(f"{col}: {n} non-positive values")
    if (df["HIGH"] < df["LOW"]).sum():
        errors.append("HIGH < LOW in some rows")
    if (df["CLOSE"] > df["HIGH"]).sum():
        errors.append("CLOSE > HIGH in some rows")
    if (df["CLOSE"] < df["LOW"]).sum():
        errors.append("CLOSE < LOW in some rows")
    if errors:
        raise ValueError("OHLCV validation failed:\n  " + "\n  ".join(errors))


def _true_range(df: pd.DataFrame) -> pd.Series:
    hl  = df["HIGH"] - df["LOW"]
    hpc = (df["HIGH"] - df["PREV_CLOSE"]).abs()
    lpc = (df["LOW"]  - df["PREV_CLOSE"]).abs()
    return pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
