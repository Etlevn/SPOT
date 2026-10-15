"""Build the SPOT master table from loaded source frames.

This module cannot be run as a script.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# One row is one USD security on one price date.
# Holding columns describe the latest SEC snapshot whose filing date is on or
# before that price date. A missing weight means the security was not in that
# published snapshot. Non-USD listings and issuer holdings without a verified
# publication time are left out.
MASTER_COLUMNS = [
    "date",
    "security_id",
    "issuer_ticker",
    "security_name",
    "isins",
    "exchange",
    "currency",
    "close",
    "adjusted_close",
    "daily_return",
    "return_basis",
    "price_source",
    "holdings_as_of",
    "publication_at",
    "holdings_source",
    "source_record_id",
    "holding_name",
    "cusip",
    "holding_isin",
    "reported_weight_pct",
    "holding_country",
    "in_published_holdings",
    "aiq_close",
    "aiq_adjusted_close",
    "aiq_daily_return",
    "quality_flag",
]


def preprocess_master_frame(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge identifiers, USD prices, SEC holdings, and the AIQ benchmark."""
    _require_tables(tables)
    crosswalk = _security_crosswalk(tables["security_crosswalk"])
    isin_map = _isin_to_security(crosswalk)
    holdings = _sec_holdings(tables["sec_holdings"], isin_map, crosswalk)
    prices = _price_panel(tables, crosswalk, holdings)
    master = _attach_point_in_time_holdings(prices, holdings)
    master = _attach_benchmark(master, tables["benchmark_prices"])
    master["quality_flag"] = _quality_flags(master)
    master = master[MASTER_COLUMNS].sort_values(["security_id", "date"], kind="mergesort")
    master = master.reset_index(drop=True)
    _validate_master(master)
    return master


def _require_tables(tables: dict[str, pd.DataFrame]) -> None:
    required = {
        "sec_holdings",
        "security_crosswalk",
        "benchmark_prices",
        "yahoo_prices",
        "wrds_prices",
    }
    missing = required.difference(tables)
    if missing:
        raise KeyError(f"Missing source tables: {sorted(missing)}")


def _security_crosswalk(coverage: pd.DataFrame) -> pd.DataFrame:
    frame = coverage.rename(
        columns={
            "symbol": "security_id",
            "name": "security_name",
            "isins": "isins",
        }
    )
    keep = [
        "security_id",
        "issuer_ticker",
        "security_name",
        "isins",
        "exchange",
        "currency",
        "status",
    ]
    _require_columns(frame, keep, "Yahoo coverage")
    frame = frame[keep].copy()
    frame["security_id"] = _clean_text(frame["security_id"])
    frame = frame[frame["security_id"].notna() & frame["security_id"].ne("AIQ")]
    frame["issuer_ticker"] = _clean_text(frame["issuer_ticker"])
    frame["security_name"] = _clean_text(frame["security_name"])
    frame["isins"] = _clean_text(frame["isins"])
    frame["exchange"] = _clean_text(frame["exchange"])
    frame["currency"] = _clean_text(frame["currency"])
    frame["status"] = _clean_text(frame["status"])
    if frame["security_id"].duplicated().any():
        duplicated = frame.loc[frame["security_id"].duplicated(), "security_id"].unique()
        raise ValueError(f"Duplicate security ids in the crosswalk: {duplicated[:10]}")
    return frame.reset_index(drop=True)


def _isin_to_security(crosswalk: pd.DataFrame) -> pd.DataFrame:
    exploded = crosswalk[["security_id", "isins"]].dropna(subset=["isins"]).copy()
    exploded["isin"] = exploded["isins"].str.split(";")
    exploded = exploded.explode("isin")
    exploded["isin"] = exploded["isin"].str.strip()
    exploded = exploded[exploded["isin"].ne("")]
    collisions = exploded.groupby("isin")["security_id"].nunique()
    collisions = collisions[collisions > 1]
    if not collisions.empty:
        raise ValueError(
            "An ISIN maps to more than one security id: "
            f"{collisions.index.tolist()[:10]}"
        )
    return exploded[["isin", "security_id"]].drop_duplicates()


def _sec_holdings(
    raw: pd.DataFrame,
    isin_map: pd.DataFrame,
    crosswalk: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(
        raw,
        ["asOf", "filedDate", "cusip", "isin", "name", "country", "pctOfNetAssets", "source_url"],
        "SEC holdings",
    )
    frame = pd.DataFrame(
        {
            "holdings_as_of": _to_date(raw["asOf"]),
            "publication_at": _to_date(raw["filedDate"]),
            "cusip": _clean_text(raw["cusip"]),
            "isin": _clean_text(raw["isin"]),
            "holding_name": _clean_text(raw["name"]),
            "holding_country": _clean_text(raw["country"]),
            "reported_weight_pct": pd.to_numeric(raw["pctOfNetAssets"], errors="coerce"),
            "source_record_id": _clean_text(raw["source_url"]),
        }
    )
    frame = frame[frame["isin"].notna()]
    frame = frame.merge(isin_map, on="isin", how="inner")
    frame = frame.merge(
        crosswalk[["security_id", "currency", "status"]],
        on="security_id",
        how="left",
    )
    usd_listed = frame["currency"].eq("USD")
    quote_unknown = frame["currency"].isna()
    frame = frame[usd_listed | quote_unknown].drop(columns=["currency", "status"])
    if frame.duplicated(["holdings_as_of", "isin"]).any():
        raise ValueError("SEC holdings repeat the same ISIN inside one snapshot.")
    return frame.reset_index(drop=True)


def _price_panel(
    tables: dict[str, pd.DataFrame],
    crosswalk: pd.DataFrame,
    holdings: pd.DataFrame,
) -> pd.DataFrame:
    linked_ids = set(holdings["security_id"].unique())
    identity = crosswalk[crosswalk["security_id"].isin(linked_ids)].copy()
    yahoo_ids = set(
        identity.loc[
            identity["currency"].eq("USD") & identity["status"].eq("available"),
            "security_id",
        ]
    )
    wrds_ids = set(
        identity.loc[
            identity["status"].ne("available")
            & (identity["currency"].eq("USD") | identity["currency"].isna()),
            "security_id",
        ]
    )

    yahoo = _yahoo_prices(tables["yahoo_prices"], yahoo_ids)
    wrds = _wrds_prices(tables["wrds_prices"], wrds_ids)
    prices = pd.concat([yahoo, wrds], ignore_index=True)
    if prices.empty:
        raise ValueError("No USD prices matched the SEC holdings crosswalk.")
    if prices.duplicated(["security_id", "date"]).any():
        raise ValueError("Price panel has duplicate security-date rows.")

    prices = prices.merge(
        identity[
            ["security_id", "issuer_ticker", "security_name", "isins", "exchange", "currency"]
        ],
        on="security_id",
        how="left",
        suffixes=("", "_crosswalk"),
    )
    prices["exchange"] = prices["exchange"].fillna(prices["exchange_crosswalk"])
    prices = prices.drop(columns=["exchange_crosswalk"])
    prices["currency"] = "USD"
    return prices


def _yahoo_prices(raw: pd.DataFrame, security_ids: set[str]) -> pd.DataFrame:
    _require_columns(
        raw,
        ["date", "ticker", "exchange", "currency", "close", "adjusted_close", "daily_return"],
        "Yahoo prices",
    )
    frame = pd.DataFrame(
        {
            "date": _to_date(raw["date"]),
            "security_id": _clean_text(raw["ticker"]),
            "exchange": _clean_text(raw["exchange"]),
            "currency": _clean_text(raw["currency"]),
            "close": pd.to_numeric(raw["close"], errors="coerce"),
            "adjusted_close": pd.to_numeric(raw["adjusted_close"], errors="coerce"),
            "daily_return": pd.to_numeric(raw["daily_return"], errors="coerce"),
        }
    )
    frame = frame[frame["security_id"].isin(security_ids) & frame["currency"].eq("USD")]
    frame["return_basis"] = "adjusted_close"
    frame["price_source"] = "yahoo"
    return frame.drop(columns=["currency"])


def _wrds_prices(raw: pd.DataFrame, security_ids: set[str]) -> pd.DataFrame:
    _require_columns(
        raw,
        ["tic", "datadate", "curcdd", "ajexdi", "prccd", "trfd"],
        "WRDS prices",
    )
    frame = pd.DataFrame(
        {
            "date": _to_date(raw["datadate"]),
            "security_id": _clean_text(raw["tic"]),
            "currency": _clean_text(raw["curcdd"]),
            "ajexdi": pd.to_numeric(raw["ajexdi"], errors="coerce"),
            "close": pd.to_numeric(raw["prccd"], errors="coerce"),
            "trfd": pd.to_numeric(raw["trfd"], errors="coerce"),
        }
    )
    frame = frame[frame["security_id"].isin(security_ids) & frame["currency"].eq("USD")]
    frame = frame.sort_values(["security_id", "date"], kind="mergesort")
    valid_factor = frame["ajexdi"].notna() & frame["ajexdi"].ne(0) & frame["close"].notna()
    frame["adjusted_close"] = pd.NA
    frame.loc[valid_factor, "adjusted_close"] = (
        frame.loc[valid_factor, "close"] / frame.loc[valid_factor, "ajexdi"]
    )
    frame["adjusted_close"] = pd.to_numeric(frame["adjusted_close"], errors="coerce")
    grouped = frame.groupby("security_id", sort=False)
    previous_level = grouped["adjusted_close"].shift(1)
    previous_trfd = grouped["trfd"].shift(1)
    price_return = frame["adjusted_close"] / previous_level - 1
    total_level = frame["adjusted_close"] * frame["trfd"]
    previous_total = previous_level * previous_trfd
    total_return = total_level / previous_total - 1
    has_total = frame["trfd"].notna() & previous_trfd.notna() & previous_total.ne(0)
    frame["daily_return"] = price_return
    frame.loc[has_total, "daily_return"] = total_return.loc[has_total]
    frame["return_basis"] = "wrds_price_return"
    frame.loc[frame["trfd"].notna(), "return_basis"] = "wrds_total_return"
    frame["price_source"] = "wrds_compustat"
    frame["exchange"] = pd.NA
    return frame[
        [
            "date",
            "security_id",
            "exchange",
            "close",
            "adjusted_close",
            "daily_return",
            "return_basis",
            "price_source",
        ]
    ]


def _attach_point_in_time_holdings(prices: pd.DataFrame, holdings: pd.DataFrame) -> pd.DataFrame:
    snapshots = (
        holdings[["holdings_as_of", "publication_at", "source_record_id"]]
        .drop_duplicates(["holdings_as_of", "publication_at"])
        .sort_values("publication_at", kind="mergesort")
    )
    dates = prices[["date"]].drop_duplicates().sort_values("date", kind="mergesort")
    dated = pd.merge_asof(
        dates,
        snapshots,
        left_on="date",
        right_on="publication_at",
        direction="backward",
    )
    held = _aggregate_holdings(holdings)
    frame = prices.merge(dated, on="date", how="left")
    frame = frame.merge(
        held,
        on=["holdings_as_of", "publication_at", "security_id"],
        how="left",
        suffixes=("", "_holding"),
    )
    if "source_record_id_holding" in frame.columns:
        frame["source_record_id"] = frame["source_record_id_holding"].fillna(
            frame["source_record_id"]
        )
        frame = frame.drop(columns=["source_record_id_holding"])
    frame["holdings_source"] = pd.NA
    frame.loc[frame["publication_at"].notna(), "holdings_source"] = "sec_derived"
    frame["in_published_holdings"] = frame["reported_weight_pct"].notna()
    return frame


def _aggregate_holdings(holdings: pd.DataFrame) -> pd.DataFrame:
    return (
        holdings.groupby(["holdings_as_of", "publication_at", "security_id"], as_index=False)
        .agg(
            reported_weight_pct=("reported_weight_pct", "sum"),
            holding_isin=("isin", _join_unique),
            cusip=("cusip", _join_unique),
            holding_name=("holding_name", "first"),
            holding_country=("holding_country", "first"),
            source_record_id_holding=("source_record_id", "first"),
        )
    )


def _attach_benchmark(master: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        benchmark,
        ["date", "close", "adjusted_close", "daily_return", "currency"],
        "AIQ prices",
    )
    prices = pd.DataFrame(
        {
            "date": _to_date(benchmark["date"]),
            "aiq_close": pd.to_numeric(benchmark["close"], errors="coerce"),
            "aiq_adjusted_close": pd.to_numeric(benchmark["adjusted_close"], errors="coerce"),
            "aiq_daily_return": pd.to_numeric(benchmark["daily_return"], errors="coerce"),
            "benchmark_currency": _clean_text(benchmark["currency"]),
        }
    )
    if not prices["benchmark_currency"].eq("USD").all():
        raise ValueError("AIQ benchmark prices are not all quoted in USD.")
    prices = prices.drop(columns=["benchmark_currency"])
    if prices["date"].duplicated().any():
        raise ValueError("AIQ benchmark prices repeat a date.")
    return master.merge(prices, on="date", how="left")


def _quality_flags(master: pd.DataFrame) -> pd.Series:
    first_date = master.groupby("security_id", sort=False)["date"].transform("min")
    benchmark_start = master["date"].min()
    masks = [
        (master["adjusted_close"].isna(), "missing_adjusted_close"),
        (
            master["daily_return"].isna() & master["date"].ne(first_date),
            "missing_daily_return",
        ),
        (
            master["return_basis"].eq("wrds_price_return") & master["daily_return"].notna(),
            "wrds_price_return_without_total_return_factor",
        ),
        (
            master["aiq_daily_return"].isna() & master["date"].ne(benchmark_start),
            "missing_benchmark_return",
        ),
    ]
    flags = np.full(len(master), "", dtype=object)
    for mask, label in masks:
        selected = mask.to_numpy()
        empty = flags == ""
        flags = np.where(
            selected,
            np.where(empty, label, flags + ";" + label),
            flags,
        )
    return pd.Series(flags, index=master.index, dtype="string")


def _validate_master(master: pd.DataFrame) -> None:
    if master.empty:
        raise ValueError("Master table is empty.")
    if master.duplicated(["security_id", "date"]).any():
        raise ValueError("Master table has duplicate security-date rows.")
    if not master["currency"].eq("USD").all():
        raise ValueError("Master table contains a non-USD row.")
    numeric = ["close", "adjusted_close", "daily_return", "reported_weight_pct", "aiq_daily_return"]
    for column in numeric:
        if np.isinf(pd.to_numeric(master[column], errors="coerce")).any():
            raise ValueError(f"Column {column} contains an infinite value.")


def _require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{label} is missing columns: {missing}")


def _clean_text(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip()
    return text.mask(text.isna() | text.isin(["", "N/A", "None", "nan", "<NA>"]))


def _to_date(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.normalize()


def _join_unique(values: pd.Series) -> object:
    cleaned: list[str] = []
    for value in values.dropna().unique():
        text = str(value).strip()
        if text and text not in {"N/A", "nan", "None", "<NA>"}:
            cleaned.append(text)
    if not cleaned:
        return pd.NA
    return ";".join(cleaned)


if __name__ == "__main__":
    raise SystemExit(
        "data.preprocess cannot be executed directly. "
        "Call it from test_data.py or main."
    )
