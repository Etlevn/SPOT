"""Quarterly holdings, cleaning, and model-input checks.

This module cannot be run as a script.
"""

from __future__ import annotations

import math

if __name__ == "__main__":
    raise SystemExit(
        "quarters cannot be executed directly. Call it from check.py or main."
    )

import pandas as pd
from tqdm.auto import tqdm

from load import _SHEETS
from preprocess import (
    _clean_text,
    _isin_to_security,
    _sec_holdings,
    _security_crosswalk,
    _to_date,
)

QUARTER_STARTS = pd.to_datetime(
    [
        "2023-10-01",
        "2024-01-01",
        "2024-04-01",
        "2024-07-01",
        "2024-10-01",
        "2025-01-01",
        "2025-04-01",
        "2025-07-01",
        "2025-10-01",
        "2026-01-01",
        "2026-04-01",
        "2026-07-01",
    ]
)
WINDOWS_MONTHS = (3, 6, 12)
BASELINE_ALPHA = 0.10


def summarize_quarters(
    tables: dict[str, pd.DataFrame],
    master: pd.DataFrame,
) -> dict[str, object]:
    """Build the quarterly audit, parameter tables, and model-input matrices.

    A name is eligible when it is in the SEC snapshot already filed by the
    decision date and its daily return is present on every AIQ trading day in
    the preceding twelve months. The three- and six-month windows use that
    same eligible set.
    """
    calendar = _benchmark_calendar(tables["benchmark_prices"])
    decisions = [_first_trading_day(calendar, start) for start in QUARTER_STARTS]
    returns = master.pivot(index="date", columns="security_id", values="daily_return")
    benchmark = (
        master.drop_duplicates("date")
        .set_index("date")["aiq_daily_return"]
        .reindex(calendar)
    )
    disclosed = _disclosed_holdings(tables)
    priced = set(master["security_id"].unique())

    holdings_rows: list[dict] = []
    parameter_rows: list[dict] = []
    input_rows: list[dict] = []
    panels: list[dict[str, object]] = []
    with tqdm(total=len(decisions), position=0, leave=False, unit="") as progress:
        for decision in decisions:
            day = master.loc[master["date"].eq(decision)]
            if day.empty:
                raise ValueError(f"No master rows on decision date {decision.date()}.")
            as_of = _one_value(day["holdings_as_of"], "holdings_as_of", decision)
            filed = _one_value(day["publication_at"], "publication_at", decision)
            held = set(day.loc[day["in_published_holdings"], "security_id"])
            history = _window(calendar, decision, 12)
            complete = returns.reindex(index=history, columns=sorted(held)).notna().all(axis=0)
            eligible = set(complete.index[complete])
            quarter = f"{decision.year}Q{decision.quarter}"
            holdings_rows.append(
                {
                    "quarter": quarter,
                    "decision_date": decision.date(),
                    "holdings_as_of": pd.Timestamp(as_of).date(),
                    "publication_at": pd.Timestamp(filed).date(),
                    "public_before_rebalance": pd.Timestamp(filed) <= decision,
                }
            )
            parameter_rows.append(
                _parameter_row(
                    quarter,
                    decision,
                    calendar,
                    disclosed,
                    as_of,
                    filed,
                    priced,
                    eligible,
                )
            )
            securities = sorted(eligible)
            for months in WINDOWS_MONTHS:
                dates = _window(calendar, decision, months)
                block = returns.reindex(index=dates, columns=securities).copy()
                target = benchmark.reindex(dates).copy()
                block.index.name = "date"
                target.index.name = "date"
                target.name = "b"
                panels.append({"quarter": quarter, "months": months, "R": block, "b": target})
                realized = pd.Series(block.to_numpy(dtype="float64").ravel())
                input_rows.append(
                    {
                        "quarter": quarter,
                        "months": months,
                        "rows": len(dates),
                        "cols": len(eligible),
                        "finite": bool(block.notna().all().all() and target.notna().all()),
                        "r_missing": int(block.isna().sum().sum()),
                        "r_mean": float(realized.mean()),
                        "r_std": float(realized.std(ddof=1)),
                        "r_min": float(realized.min()),
                        "r_max": float(realized.max()),
                        "b_missing": int(target.isna().sum()),
                        "b_mean": float(target.mean()),
                        "b_std": float(target.std(ddof=1)),
                        "b_min": float(target.min()),
                        "b_max": float(target.max()),
                    }
                )
            progress.update(1)

    holdings = pd.DataFrame(holdings_rows)
    parameters = pd.DataFrame(parameter_rows)
    return {
        "sources": _source_coverage(tables),
        "unread": _unread_sheets(),
        "gaps": _remaining_gaps(tables, master),
        "holdings": holdings,
        "mapping": _identifier_mapping(tables, master),
        "missing": _missing_values(master),
        "currencies": _currency_coverage(tables, master),
        "returns": _return_construction(master),
        "alignment": _trading_day_alignment(tables, master),
        "parameters": parameters,
        "inputs": pd.DataFrame(input_rows),
        "panels": panels,
        "aiq_returns": _aiq_return_stats(tables),
        "coverage": _security_coverage(master),
    }


def _parameter_row(
    quarter: str,
    decision: pd.Timestamp,
    calendar: pd.DatetimeIndex,
    disclosed: pd.DataFrame,
    as_of: pd.Timestamp,
    filed: pd.Timestamp,
    priced: set[str],
    eligible: set[str],
) -> dict:
    snapshot = disclosed.loc[
        disclosed["holdings_as_of"].eq(as_of) & disclosed["publication_at"].eq(filed)
    ].copy()
    snapshot["reason"] = snapshot.apply(
        lambda row: _exclusion_reason(row, priced, eligible),
        axis=1,
    )
    row = {
        "quarter": quarter,
        "Nq": len(eligible),
        "Kq": math.floor(BASELINE_ALPHA * len(eligible)),
        "kq_at_least_one": math.floor(BASELINE_ALPHA * len(eligible)) >= 1,
        "T_3": len(_window(calendar, decision, 3)),
        "T_6": len(_window(calendar, decision, 6)),
        "T_12": len(_window(calendar, decision, 12)),
    }
    excluded = snapshot.loc[snapshot["reason"].ne("eligible")]
    row["n_excluded"] = int(
        excluded.loc[excluded["security_id"].isna()].shape[0]
        + excluded.loc[excluded["security_id"].notna(), "security_id"].nunique()
    )
    row["excluded_weight_pct"] = float(excluded["weight"].sum())
    for reason in ("non_usd", "unmapped", "no_price", "short_history"):
        group = snapshot.loc[snapshot["reason"].eq(reason)]
        if reason == "unmapped":
            row[f"n_{reason}"] = int(len(group))
        else:
            row[f"n_{reason}"] = int(group["security_id"].nunique())
        row[f"w_{reason}"] = float(group["weight"].sum())
    return row


def _exclusion_reason(row: pd.Series, priced: set[str], eligible: set[str]) -> str:
    security_id = row["security_id"]
    if pd.isna(security_id):
        return "unmapped"
    currency = row["currency"]
    if pd.notna(currency) and currency != "USD":
        return "non_usd"
    if security_id not in priced:
        return "no_price"
    if security_id not in eligible:
        return "short_history"
    return "eligible"


def _disclosed_holdings(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    raw = tables["sec_holdings"]
    crosswalk = _security_crosswalk(tables["security_crosswalk"])
    isin_map = _isin_to_security(crosswalk)
    frame = pd.DataFrame(
        {
            "holdings_as_of": _to_date(raw["asOf"]),
            "publication_at": _to_date(raw["filedDate"]),
            "isin": _clean_text(raw["isin"]),
            "weight": pd.to_numeric(raw["pctOfNetAssets"], errors="coerce"),
        }
    )
    frame = frame.merge(isin_map, on="isin", how="left")
    return frame.merge(crosswalk[["security_id", "currency"]], on="security_id", how="left")


def _benchmark_calendar(benchmark: pd.DataFrame) -> pd.DatetimeIndex:
    dates = _to_date(benchmark["date"]).dropna().drop_duplicates().sort_values()
    return pd.DatetimeIndex(dates)


def _first_trading_day(calendar: pd.DatetimeIndex, start: pd.Timestamp) -> pd.Timestamp:
    later = calendar[calendar >= start]
    if len(later) == 0:
        raise ValueError(f"No benchmark trading day on or after {start.date()}.")
    return pd.Timestamp(later[0])


def _window(calendar: pd.DatetimeIndex, decision: pd.Timestamp, months: int) -> pd.DatetimeIndex:
    start = decision - pd.DateOffset(months=months)
    return calendar[(calendar >= start) & (calendar < decision)]


def _source_coverage(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for key, frame in tables.items():
        workbook, sheet = _SHEETS[key]
        start, end = _date_span(frame, _DATE_COLUMNS[key])
        rows.append(
            {
                "input": key,
                "workbook": workbook,
                "sheet": sheet,
                "rows": len(frame),
                "start": start,
                "end": end,
                "n_ids": _n_ids(frame, key),
                "detail": _source_detail(frame, key),
            }
        )
    return pd.DataFrame(rows)


def _unread_sheets() -> pd.DataFrame:
    holdings_workbook = _SHEETS["sec_holdings"][0]
    wrds_workbook = _SHEETS["wrds_prices"][0]
    return pd.DataFrame(
        [
            {
                "workbook": holdings_workbook,
                "sheet": "Issuer holdings",
                "reason": "No verified publication date. SEC filing dates are used instead.",
            },
            {
                "workbook": wrds_workbook,
                "sheet": "Compustat Global",
                "reason": "Non-USD prices. Those listings are excluded instead of converted.",
            },
        ]
    )


def _remaining_gaps(tables: dict[str, pd.DataFrame], master: pd.DataFrame) -> pd.DataFrame:
    crosswalk = _security_crosswalk(tables["security_crosswalk"])
    isin_map = _isin_to_security(crosswalk)
    held_ids = set(_sec_holdings(tables["sec_holdings"], isin_map, crosswalk)["security_id"])
    priced = set(master["security_id"])
    absent = crosswalk.loc[~crosswalk["security_id"].isin(priced)].copy()
    absent["reason"] = absent.apply(lambda row: _absent_reason(row, held_ids), axis=1)
    absent = absent.sort_values(["reason", "security_id"], kind="mergesort")
    return absent[["security_id", "issuer_ticker", "currency", "status", "reason"]].reset_index(
        drop=True
    )


def _identifier_mapping(tables: dict[str, pd.DataFrame], master: pd.DataFrame) -> pd.DataFrame:
    crosswalk = _security_crosswalk(tables["security_crosswalk"])
    isin_map = _isin_to_security(crosswalk)
    isin = _clean_text(tables["sec_holdings"]["isin"])
    sec = pd.DataFrame({"isin": isin}).dropna()
    mapped = sec["isin"].isin(set(isin_map["isin"]))
    return _metric_frame(
        [
            ("crosswalk_securities", int(len(crosswalk))),
            ("crosswalk_isins", int(len(isin_map))),
            ("sec_rows_with_isin", int(len(sec))),
            ("sec_rows_mapped", int(mapped.sum())),
            ("sec_rows_unmapped", int((~mapped).sum())),
            ("unmapped_isins", int(sec.loc[~mapped, "isin"].nunique())),
            ("master_securities", int(master["security_id"].nunique())),
            ("master_rows", int(len(master))),
            ("issuer_ticker_missing_securities", int(master.loc[master["issuer_ticker"].isna(), "security_id"].nunique())),
        ]
    )


def _missing_values(master: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "close",
        "adjusted_close",
        "daily_return",
        "aiq_close",
        "aiq_adjusted_close",
        "aiq_daily_return",
        "issuer_ticker",
        "exchange",
        "cusip",
        "holding_isin",
        "reported_weight_pct",
        "holding_country",
    ]
    rows = []
    for column in columns:
        null_rows = int(master[column].isna().sum())
        rows.append(
            {
                "column": column,
                "null_rows": null_rows,
                "share_of_rows": null_rows / len(master),
            }
        )
    return pd.DataFrame(rows)


def _currency_coverage(tables: dict[str, pd.DataFrame], master: pd.DataFrame) -> pd.DataFrame:
    crosswalk = _security_crosswalk(tables["security_crosswalk"])
    priced = set(master["security_id"])
    frame = crosswalk.copy()
    frame["currency"] = frame["currency"].fillna("unknown")
    frame["in_master"] = frame["security_id"].isin(priced)
    rows = []
    for currency, group in frame.groupby("currency", dropna=False, sort=True):
        rows.append(
            {
                "currency": currency,
                "crosswalk_securities": int(len(group)),
                "in_master": int(group["in_master"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _return_construction(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for basis, group in master.groupby("return_basis", sort=True):
        rows.append(
            {
                "item": "return_basis",
                "label": basis,
                "rows": int(len(group)),
                "securities": int(group["security_id"].nunique()),
            }
        )
    flagged = master.loc[master["quality_flag"].fillna("").ne("")]
    if flagged.empty:
        rows.append({"item": "quality_flag", "label": "none", "rows": 0, "securities": 0})
    else:
        for label, group in flagged.groupby("quality_flag", sort=True):
            rows.append(
                {
                    "item": "quality_flag",
                    "label": label,
                    "rows": int(len(group)),
                    "securities": int(group["security_id"].nunique()),
                }
            )
    first_date = master.groupby("security_id", sort=False)["date"].transform("min")
    later_null = master["daily_return"].isna() & master["date"].ne(first_date)
    rows.append(
        {
            "item": "rule",
            "label": "null_return_after_first_day",
            "rows": int(later_null.sum()),
            "securities": int(master.loc[later_null, "security_id"].nunique()),
        }
    )
    return pd.DataFrame(rows)


def _trading_day_alignment(tables: dict[str, pd.DataFrame], master: pd.DataFrame) -> pd.DataFrame:
    calendar = _benchmark_calendar(tables["benchmark_prices"])
    master_dates = pd.DatetimeIndex(master["date"].drop_duplicates())
    yahoo_dates = _to_date(tables["yahoo_prices"]["date"])
    return _metric_frame(
        [
            ("aiq_trading_days", int(len(calendar))),
            ("master_dates", int(len(master_dates))),
            ("master_dates_outside_aiq", int(len(master_dates.difference(calendar)))),
            ("aiq_dates_missing_from_master", int(len(calendar.difference(master_dates)))),
            ("master_rows_outside_aiq", int((~master["date"].isin(calendar)).sum())),
            ("yahoo_rows_outside_aiq", int((yahoo_dates.notna() & ~yahoo_dates.isin(calendar)).sum())),
        ]
    )


def _aiq_return_stats(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frame = tables["benchmark_prices"]
    dates = _to_date(frame["date"])
    returns = pd.to_numeric(frame["daily_return"], errors="coerce")
    observed = returns.dropna()
    return pd.DataFrame(
        [
            {
                "start": dates.min().date(),
                "end": dates.max().date(),
                "n_prices": int(len(frame)),
                "n_returns": int(observed.shape[0]),
                "null_returns": int(returns.isna().sum()),
                "mean": float(observed.mean()),
                "std": float(observed.std(ddof=1)),
                "min": float(observed.min()),
                "median": float(observed.median()),
                "max": float(observed.max()),
            }
        ]
    )


def _security_coverage(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, group in master.groupby("price_source", sort=True):
        rows.append(
            {
                "price_source": source,
                "securities": int(group["security_id"].nunique()),
                "rows": int(len(group)),
                "start": pd.Timestamp(group["date"].min()).date(),
                "end": pd.Timestamp(group["date"].max()).date(),
                "null_daily_returns": int(group["daily_return"].isna().sum()),
            }
        )
    return pd.DataFrame(rows)


_DATE_COLUMNS = {
    "sec_holdings": "asOf",
    "security_crosswalk": None,
    "benchmark_prices": "date",
    "yahoo_prices": "date",
    "wrds_prices": "datadate",
}


def _date_span(frame: pd.DataFrame, column: str | None) -> tuple[object, object]:
    if column is None or column not in frame.columns:
        return pd.NA, pd.NA
    dates = _to_date(frame[column]).dropna()
    if dates.empty:
        return pd.NA, pd.NA
    return dates.min().date(), dates.max().date()


def _n_ids(frame: pd.DataFrame, key: str) -> int:
    column = {
        "sec_holdings": "isin",
        "security_crosswalk": "symbol",
        "benchmark_prices": None,
        "yahoo_prices": "ticker",
        "wrds_prices": "tic",
    }[key]
    if column is None:
        return 1
    return int(_clean_text(frame[column]).nunique(dropna=True))


def _source_detail(frame: pd.DataFrame, key: str) -> str:
    if key == "sec_holdings":
        pairs = pd.DataFrame(
            {
                "as_of": _to_date(frame["asOf"]),
                "filed": _to_date(frame["filedDate"]),
            }
        ).drop_duplicates()
        filed_after = bool((pairs["filed"] >= pairs["as_of"]).all())
        state = "every filing date is on or after the as-of date" if filed_after else "a filing date precedes its as-of date"
        urls = int(_clean_text(frame["source_url"]).nunique(dropna=True))
        return f"{len(pairs)} snapshots; {state}; {urls} source url"
    if key == "security_crosswalk":
        status = _clean_text(frame["status"]).fillna("unknown").value_counts()
        listed = ", ".join(f"{label} {int(count)}" for label, count in status.items())
        return f"status: {listed}"
    if key == "benchmark_prices":
        currency = _clean_text(frame["currency"]).dropna().unique()
        quoted = ", ".join(str(value) for value in currency) or "unknown"
        null_prices = int(pd.to_numeric(frame["adjusted_close"], errors="coerce").isna().sum())
        return f"currency {quoted}; null adjusted closes {null_prices}"
    if key == "yahoo_prices":
        return _id_currency_detail(frame, "ticker", "currency")
    if key == "wrds_prices":
        pairs = pd.DataFrame(
            {
                "security_id": _clean_text(frame["tic"]),
                "currency": _clean_text(frame["curcdd"]),
            }
        )
        quoted = pairs.groupby("security_id")["currency"].apply(
            lambda values: "USD" if values.eq("USD").any() else "unknown"
        )
        blank = int(pairs.loc[pairs["currency"].isna(), "security_id"].nunique())
        counts = quoted.value_counts()
        listed = ", ".join(f"{currency} {int(count)}" for currency, count in counts.items())
        return f"{listed}; {blank} tickers have rows with a blank currency"
    return ""


def _id_currency_detail(frame: pd.DataFrame, id_column: str, currency_column: str) -> str:
    pairs = pd.DataFrame(
        {
            "security_id": _clean_text(frame[id_column]),
            "currency": _clean_text(frame[currency_column]).fillna("unknown"),
        }
    ).drop_duplicates()
    counts = pairs.groupby("currency", sort=True).size()
    return ", ".join(f"{currency} {int(count)}" for currency, count in counts.items())


def _absent_reason(row: pd.Series, held_ids: set[str]) -> str:
    if pd.notna(row["currency"]) and row["currency"] != "USD":
        return "non_usd"
    if row["security_id"] not in held_ids:
        return "not_in_sec_usd_holdings"
    if row["status"] != "available":
        return "no_usd_price"
    return "not_in_master"


def _metric_frame(pairs: list[tuple[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(pairs, columns=["metric", "value"])


def _one_value(values: pd.Series, label: str, decision: pd.Timestamp) -> pd.Timestamp:
    unique = values.dropna().drop_duplicates()
    if len(unique) != 1:
        raise ValueError(f"{label} is not unique on {decision.date()}.")
    return pd.Timestamp(unique.iloc[0])
