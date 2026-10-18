"""Out-of-sample backtest from a finished solve run."""

from __future__ import annotations

# ============================== [CONFIG] ==============================


# Folder name under output/solve/, or a path to that folder.
# Empty string selects the latest run that contains weights.csv.
# Every window_months value present in that run's weights.csv is evaluated.
SOLVE_RUN = "261017_214151"

# Selection ratios to evaluate from the solve run.
# () uses every alpha present in weights.csv (or the run snapshot).
# (0.10,) evaluates one ratio; (0.10, 0.30) evaluates a subset.
EVAL_ALPHAS: tuple[float, ...] = ()

# Drop the final decision date as a formation date. Its decision day is still
# used as the end of the previous holding period.
EXCLUDE_FINAL_QUARTER = True

# ======================================================================

import json
import math
import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from load import load_source_tables
from preprocess import preprocess_master_frame
from quarters import summarize_quarters

PROJECT_ROOT = Path(__file__).resolve().parent
SOLVE_ROOT = PROJECT_ROOT / "output" / "solve"
EVAL_ROOT = PROJECT_ROOT / "output" / "eval"

STRATEGY_MIQP = "miqp"
STRATEGY_EW = "ew"
STRATEGY_TOPK_EW = "topk_ew"
STRATEGY_TOPK_DW = "topk_dw"
BASELINE_STRATEGIES = (STRATEGY_EW, STRATEGY_TOPK_EW, STRATEGY_TOPK_DW)
STRATEGY_ORDER = (STRATEGY_MIQP, STRATEGY_EW, STRATEGY_TOPK_EW, STRATEGY_TOPK_DW)


def main() -> None:
    solve_dir = _resolve_solve_dir(SOLVE_RUN)
    weights = _load_weights(solve_dir / "weights.csv", solve_dir)
    alphas = _eval_alphas(weights, solve_dir)
    tables = load_source_tables()
    master = preprocess_master_frame(tables)
    report = summarize_quarters(tables, master)
    decisions = _decision_table(report["holdings"])
    calendar = _full_calendar(master, decisions)
    returns, benchmark = _return_panels(master, calendar)

    formation = decisions.iloc[:-1] if EXCLUDE_FINAL_QUARTER else decisions
    if formation.empty:
        raise ValueError("No formation quarters remain after applying the eval filters.")

    eval_dir = _eval_dir_for_solve(solve_dir)
    windows = sorted(int(value) for value in weights["window_months"].unique())
    snapshot = {
        "solve_run": solve_dir.name,
        "solve_dir": str(solve_dir.relative_to(PROJECT_ROOT)),
        "eval_dir": str(eval_dir.relative_to(PROJECT_ROOT)),
        "window_months": windows,
        "alphas": alphas,
        "exclude_final_quarter": EXCLUDE_FINAL_QUARTER,
        "strategies": [STRATEGY_MIQP, *BASELINE_STRATEGIES],
        "best_path_rule": "Within each alpha, choose the MIQP window with the lowest RMSE.",
        "execution": (
            "Weights are known before the decision-date open; "
            "the holding period is [decision_q, decision_{q+1})."
        ),
        "formation_quarters": formation["quarter"].tolist(),
    }
    (eval_dir / "eval.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    _write_excluded_weights(report["parameters"], decisions["quarter"].tolist(), eval_dir / "excluded_weights.csv")

    daily_rows: list[pd.DataFrame] = []
    quarterly_rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []
    targets: dict[tuple[float, str, int | None, str], pd.Series] = {}
    end_weights: dict[tuple[float, str, int | None, str], pd.Series] = {}

    jobs: list[tuple[float, object, str, int | None, pd.Series]] = []
    for alpha in alphas:
        alpha_weights = weights.loc[weights["alpha"].eq(alpha)]
        baselines = {
            row.quarter: _baseline_weights(master, report, row.decision_date, row.quarter, alpha)
            for row in formation.itertuples(index=False)
        }
        baseline_rows.extend(_baseline_weight_rows(baselines, alpha))
        for row in formation.itertuples(index=False):
            for months in windows:
                block = alpha_weights.loc[
                    alpha_weights["quarter"].eq(row.quarter) & alpha_weights["window_months"].eq(months)
                ]
                if block.empty:
                    continue
                jobs.append(
                    (
                        alpha,
                        row,
                        STRATEGY_MIQP,
                        months,
                        _formation_weights(alpha_weights, row.quarter, months),
                    )
                )
            for strategy in BASELINE_STRATEGIES:
                jobs.append((alpha, row, strategy, None, baselines[row.quarter][strategy]))

    with tqdm(total=len(jobs), position=0, leave=False, unit="") as progress:
        for alpha, row, strategy, months, target in jobs:
            nxt = _next_quarter(row.quarter, decisions)
            end = decisions.loc[decisions["quarter"].eq(nxt), "decision_date"].iloc[0]
            hold_dates = calendar[(calendar >= row.decision_date) & (calendar < end)]
            if len(hold_dates) == 0:
                raise ValueError(
                    f"Empty holding period for a{_alpha_tag(alpha)} {row.quarter} {strategy}."
                )
            daily, drifted = _buy_and_hold(target, returns, benchmark, hold_dates)
            daily.insert(0, "window_months", months)
            daily.insert(0, "strategy", strategy)
            daily.insert(0, "quarter", row.quarter)
            daily.insert(0, "alpha", alpha)
            daily_rows.append(daily)
            key = (alpha, strategy, months, row.quarter)
            targets[key] = target
            end_weights[key] = drifted
            metrics = _quarter_metrics(row.quarter, strategy, months, target, daily)
            metrics["alpha"] = alpha
            quarterly_rows.append(metrics)
            progress.update(1)

    daily = pd.concat(daily_rows, ignore_index=True)
    daily = _add_paths(daily)
    quarterly = _sort_strategy_frame(pd.DataFrame(quarterly_rows))
    quarterly = _attach_stability_metrics(quarterly, targets, end_weights, formation["quarter"].tolist())
    path_summary = _sort_strategy_frame(_path_metrics(daily, quarterly))
    pd.DataFrame(baseline_rows).to_csv(eval_dir / "baseline_weights.csv", index=False)
    daily.to_csv(eval_dir / "daily_returns.csv", index=False)
    quarterly.to_csv(eval_dir / "quarterly_metrics.csv", index=False)
    path_summary.to_csv(eval_dir / "window_metrics.csv", index=False)

    for alpha in alphas:
        alpha_daily = daily.loc[daily["alpha"].eq(alpha)].copy()
        alpha_summary = path_summary.loc[path_summary["alpha"].eq(alpha)].copy()
        figure_name = "paths.png" if len(alphas) == 1 else f"paths_a{_alpha_tag(alpha)}.png"
        _write_path_figure(alpha_daily, alpha_summary, decisions, eval_dir / figure_name)

    if len(alphas) > 1:
        ranked = _rank_window_metrics(path_summary)
        ranked.to_csv(eval_dir / "window_metrics_ranked.csv", index=False)
        best = _best_paths_by_alpha(path_summary)
        best.to_csv(eval_dir / "best_by_alpha.csv", index=False)
        _write_best_alpha_figure(daily, best, decisions, eval_dir / "paths_best.png")
        tqdm.write(ranked.to_string(index=False))
    else:
        tqdm.write(path_summary.to_string(index=False))


def _resolve_solve_dir(solve_run: str) -> Path:
    text = solve_run.strip()
    if text:
        path = Path(text)
        if not path.is_absolute():
            path = SOLVE_ROOT / text if path.parent == Path(".") else PROJECT_ROOT / path
        if not path.is_dir():
            raise FileNotFoundError(f"Solve run directory not found: {path}")
        if not (path / "weights.csv").is_file():
            raise FileNotFoundError(f"Missing weights.csv in {path}")
        return path.resolve()

    candidates = sorted(
        (path for path in SOLVE_ROOT.glob("*/weights.csv") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No weights.csv under {SOLVE_ROOT}")
    return candidates[0].parent.resolve()


def _load_weights(path: Path, solve_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"quarter", "window_months", "security_id", "weight"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"weights.csv is missing columns: {sorted(missing)}")
    frame["window_months"] = frame["window_months"].astype(int)
    frame["security_id"] = frame["security_id"].astype(str)
    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce")
    if frame["weight"].isna().any():
        raise ValueError("weights.csv contains non-numeric weights.")
    if "alpha" not in frame.columns:
        frame["alpha"] = _fallback_alpha(solve_dir)
    frame["alpha"] = pd.to_numeric(frame["alpha"], errors="coerce")
    if frame["alpha"].isna().any():
        raise ValueError("weights.csv contains non-numeric alpha values.")
    return frame


def _fallback_alpha(solve_dir: Path) -> float:
    summary_path = solve_dir / "summary.csv"
    if summary_path.is_file():
        summary = pd.read_csv(summary_path)
        if "alpha" in summary.columns and not summary.empty:
            return float(summary["alpha"].iloc[0])
    run_path = solve_dir / "run.json"
    if run_path.is_file():
        payload = json.loads(run_path.read_text())
        if "alphas" in payload and payload["alphas"]:
            return float(payload["alphas"][0])
        if "alpha" in payload:
            return float(payload["alpha"])
    raise ValueError(f"Could not infer alpha for {solve_dir}")


def _solve_alphas(weights: pd.DataFrame, solve_dir: Path) -> list[float]:
    values = sorted({float(value) for value in weights["alpha"].unique()})
    if values:
        return values
    run_path = solve_dir / "run.json"
    if run_path.is_file():
        payload = json.loads(run_path.read_text())
        if "alphas" in payload and payload["alphas"]:
            return [float(value) for value in payload["alphas"]]
    return [_fallback_alpha(solve_dir)]


def _eval_alphas(weights: pd.DataFrame, solve_dir: Path) -> list[float]:
    available = _solve_alphas(weights, solve_dir)
    if not EVAL_ALPHAS:
        return available
    wanted = [float(value) for value in EVAL_ALPHAS]
    missing = [value for value in wanted if not any(math.isclose(value, have, abs_tol=1e-12) for have in available)]
    if missing:
        raise ValueError(f"EVAL_ALPHAS not present in solve run: {missing}; available={available}")
    return wanted


def _decision_table(holdings: pd.DataFrame) -> pd.DataFrame:
    frame = holdings.loc[:, ["quarter", "decision_date"]].copy()
    frame["decision_date"] = pd.to_datetime(frame["decision_date"])
    return frame.sort_values("decision_date", kind="mergesort").reset_index(drop=True)


def _full_calendar(master: pd.DataFrame, decisions: pd.DataFrame) -> pd.DatetimeIndex:
    start = decisions["decision_date"].min()
    end = decisions["decision_date"].max()
    dates = pd.DatetimeIndex(pd.to_datetime(master["date"].drop_duplicates())).sort_values()
    return dates[(dates >= start) & (dates <= end)]


def _return_panels(
    master: pd.DataFrame, calendar: pd.DatetimeIndex
) -> tuple[pd.DataFrame, pd.Series]:
    returns = master.pivot(index="date", columns="security_id", values="daily_return")
    returns.index = pd.DatetimeIndex(returns.index)
    returns = returns.reindex(calendar)
    benchmark = (
        master.drop_duplicates("date")
        .set_index("date")["aiq_daily_return"]
        .reindex(calendar)
    )
    benchmark.index = pd.DatetimeIndex(benchmark.index)
    benchmark.name = "benchmark_return"
    return returns, benchmark


def _formation_weights(weights: pd.DataFrame, quarter: str, months: int) -> pd.Series:
    block = weights.loc[
        weights["quarter"].eq(quarter) & weights["window_months"].eq(months),
        ["security_id", "weight"],
    ]
    if block.empty:
        raise ValueError(f"No weights for {quarter} L{months}.")
    series = block.groupby("security_id", sort=True)["weight"].sum()
    total = float(series.sum())
    if abs(total - 1.0) > 1e-4:
        raise ValueError(f"Weights for {quarter} L{months} sum to {total}, not 1.")
    return series


def _eligible_ids(report: dict[str, object], quarter: str) -> list[str]:
    for panel in report["panels"]:
        if panel["quarter"] == quarter:
            return [str(security) for security in panel["R"].columns]
    raise ValueError(f"No prepared panel for {quarter}.")


def _baseline_weights(
    master: pd.DataFrame,
    report: dict[str, object],
    decision: pd.Timestamp,
    quarter: str,
    alpha: float,
) -> dict[str, pd.Series]:
    eligible = _eligible_ids(report, quarter)
    day = master.loc[master["date"].eq(pd.Timestamp(decision))].copy()
    if day.empty:
        raise ValueError(f"No master rows on decision date {pd.Timestamp(decision).date()}.")
    disclosed = (
        day.loc[day["security_id"].isin(eligible), ["security_id", "reported_weight_pct"]]
        .groupby("security_id", sort=False)["reported_weight_pct"]
        .sum()
        .reindex(eligible)
        .fillna(0.0)
    )
    cardinality = math.floor(alpha * len(eligible))
    if cardinality < 1:
        raise ValueError(f"K=0 for {quarter}; Top-K baselines are undefined.")
    top = disclosed.sort_values(ascending=False).head(cardinality)
    if float(top.sum()) <= 0:
        raise ValueError(f"Top-K disclosed weights are all zero for {quarter}.")
    topk_ew = pd.Series(1.0 / cardinality, index=top.index, dtype="float64")
    topk_dw = (top / top.sum()).astype("float64")
    universe = pd.Series(1.0 / len(eligible), index=eligible, dtype="float64")
    return {
        STRATEGY_EW: universe,
        STRATEGY_TOPK_EW: topk_ew,
        STRATEGY_TOPK_DW: topk_dw,
    }


def _baseline_weight_rows(baselines: dict[str, dict[str, pd.Series]], alpha: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for quarter, strategies in baselines.items():
        for strategy, weights in strategies.items():
            for position, (security_id, weight) in enumerate(weights.items(), start=1):
                rows.append(
                    {
                        "alpha": alpha,
                        "quarter": quarter,
                        "strategy": strategy,
                        "position": position,
                        "security_id": security_id,
                        "weight": float(weight),
                    }
                )
    return rows


def _write_excluded_weights(parameters: pd.DataFrame, quarters: list[str], path: Path) -> None:
    if parameters.empty:
        raise ValueError("Quarterly parameters are empty; excluded weights cannot be written.")
    keep = [
        "quarter",
        "Nq",
        "Kq",
        "n_excluded",
        "excluded_weight_pct",
        "n_non_usd",
        "w_non_usd",
        "n_unmapped",
        "w_unmapped",
        "n_no_price",
        "w_no_price",
        "n_short_history",
        "w_short_history",
    ]
    missing = [name for name in keep if name not in parameters.columns]
    if missing:
        raise KeyError(f"parameters is missing columns: {missing}")
    frame = parameters.loc[parameters["quarter"].isin(quarters), keep].copy()
    frame = frame.sort_values("quarter", kind="mergesort").reset_index(drop=True)
    frame.to_csv(path, index=False)


def _buy_and_hold(
    weights: pd.Series,
    returns: pd.DataFrame,
    benchmark: pd.Series,
    dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.Series]:
    held = weights[weights.abs() > 0]
    if held.empty:
        raise ValueError("Formation weights are all zero.")
    level = held.astype("float64").copy()
    rows: list[dict[str, object]] = []
    for date in dates:
        day = returns.loc[date, level.index]
        asset_return = day.astype("float64").fillna(0.0)
        portfolio_return = float((level * asset_return).sum() / level.sum())
        bench = benchmark.loc[date]
        bench_return = float(bench) if pd.notna(bench) else float("nan")
        active = portfolio_return - bench_return if pd.notna(bench_return) else float("nan")
        level = level * (1.0 + asset_return)
        rows.append(
            {
                "date": pd.Timestamp(date).date(),
                "portfolio_return": portfolio_return,
                "benchmark_return": bench_return,
                "active_return": active,
            }
        )
    total = float(level.sum())
    if total <= 0:
        raise ValueError("Buy-and-hold levels sum to zero at period end.")
    drifted = (level / total).astype("float64")
    drifted.name = "weight"
    return pd.DataFrame(rows), drifted


def _quarter_metrics(
    quarter: str,
    strategy: str,
    months: int | None,
    weights: pd.Series,
    daily: pd.DataFrame,
) -> dict[str, object]:
    active = daily["active_return"].dropna()
    portfolio = daily["portfolio_return"]
    benchmark = daily["benchmark_return"]
    hhi = float((weights.to_numpy(dtype="float64") ** 2).sum())
    positive = int((weights > 0).sum())
    return {
        "quarter": quarter,
        "strategy": strategy,
        "window_months": months,
        "n_days": int(len(daily)),
        "rmse": float((active.to_numpy(dtype="float64") ** 2).mean() ** 0.5) if len(active) else float("nan"),
        "tracking_error": float(active.std(ddof=1)) if len(active) > 1 else float("nan"),
        "return_corr": _return_corr(portfolio, benchmark),
        "mean_active_return": float(active.mean()) if len(active) else float("nan"),
        "cum_portfolio_return": float((1.0 + portfolio).prod() - 1.0),
        "cum_benchmark_return": float((1.0 + benchmark.fillna(0.0)).prod() - 1.0),
        "cum_active_return": float((1.0 + portfolio).prod() - (1.0 + benchmark.fillna(0.0)).prod()),
        "n_positive_weights": positive,
        "max_weight": float(weights.max()),
        "hhi": hhi,
        "effective_holdings": (1.0 / hhi) if hhi > 0 else float("nan"),
        "turnover": float("nan"),
        "holding_overlap": float("nan"),
    }


def _attach_stability_metrics(
    quarterly: pd.DataFrame,
    targets: dict[tuple[float, str, int | None, str], pd.Series],
    end_weights: dict[tuple[float, str, int | None, str], pd.Series],
    formation_quarters: list[str],
) -> pd.DataFrame:
    frame = quarterly.copy()
    previous = {quarter: formation_quarters[index - 1] for index, quarter in enumerate(formation_quarters) if index > 0}
    turnovers: list[float] = []
    overlaps: list[float] = []
    for row in frame.itertuples(index=False):
        months = None if pd.isna(row.window_months) else int(row.window_months)
        alpha = float(row.alpha)
        prior = previous.get(str(row.quarter))
        if prior is None:
            turnovers.append(float("nan"))
            overlaps.append(float("nan"))
            continue
        prior_key = (alpha, str(row.strategy), months, prior)
        current_key = (alpha, str(row.strategy), months, str(row.quarter))
        if prior_key not in end_weights or current_key not in targets:
            turnovers.append(float("nan"))
            overlaps.append(float("nan"))
            continue
        turnovers.append(_turnover(end_weights[prior_key], targets[current_key]))
        overlaps.append(_holding_overlap(targets[prior_key], targets[current_key]))
    frame["turnover"] = turnovers
    frame["holding_overlap"] = overlaps
    return frame


def _turnover(pre_rebalance: pd.Series, new_target: pd.Series) -> float:
    ids = sorted(set(pre_rebalance.index.astype(str)) | set(new_target.index.astype(str)))
    left = pre_rebalance.astype("float64").reindex(ids).fillna(0.0)
    right = new_target.astype("float64").reindex(ids).fillna(0.0)
    return 0.5 * float((left - right).abs().sum())


def _holding_overlap(previous: pd.Series, current: pd.Series) -> float:
    left = set(previous.loc[previous.astype("float64") > 0].index.astype(str))
    right = set(current.loc[current.astype("float64") > 0].index.astype(str))
    union = left | right
    if not union:
        return float("nan")
    return float(len(left & right) / len(union))


def _add_paths(daily: pd.DataFrame) -> pd.DataFrame:
    frame = _sort_strategy_frame(daily)
    pieces: list[pd.DataFrame] = []
    grouped = frame.groupby(["alpha", "strategy", "window_months"], dropna=False, sort=False)
    for (_, _, _), block in grouped:
        block = block.copy()
        block["portfolio_cum"] = (1.0 + block["portfolio_return"]).cumprod()
        block["benchmark_cum"] = (1.0 + block["benchmark_return"].fillna(0.0)).cumprod()
        block["active_cum"] = block["portfolio_cum"] / block["benchmark_cum"]
        pieces.append(block)
    return pd.concat(pieces, ignore_index=True)


def _path_metrics(daily: pd.DataFrame, quarterly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ordered = _sort_strategy_frame(daily)
    grouped = ordered.groupby(["alpha", "strategy", "window_months"], dropna=False, sort=False)
    for (alpha, strategy, months), block in grouped:
        active = block["active_return"].dropna()
        month_key = None if pd.isna(months) else int(months)
        stability = quarterly.loc[quarterly["alpha"].eq(alpha) & quarterly["strategy"].eq(strategy)]
        if month_key is None:
            stability = stability.loc[stability["window_months"].isna()]
        else:
            stability = stability.loc[stability["window_months"].eq(month_key)]
        rows.append(
            {
                "alpha": float(alpha),
                "strategy": strategy,
                "window_months": month_key,
                "n_quarters": int(block["quarter"].nunique()),
                "n_days": int(len(block)),
                "rmse": float((active.to_numpy(dtype="float64") ** 2).mean() ** 0.5) if len(active) else float("nan"),
                "tracking_error": float(active.std(ddof=1)) if len(active) > 1 else float("nan"),
                "return_corr": _return_corr(block["portfolio_return"], block["benchmark_return"]),
                "mean_active_return": float(active.mean()) if len(active) else float("nan"),
                "cum_portfolio_return": float(block["portfolio_cum"].iloc[-1] - 1.0),
                "cum_benchmark_return": float(block["benchmark_cum"].iloc[-1] - 1.0),
                "cum_active_return": float(block["portfolio_cum"].iloc[-1] - block["benchmark_cum"].iloc[-1]),
                "mean_turnover": float(stability["turnover"].mean(skipna=True)),
                "mean_holding_overlap": float(stability["holding_overlap"].mean(skipna=True)),
            }
        )
    return pd.DataFrame(rows)


def _rank_window_metrics(path_summary: pd.DataFrame) -> pd.DataFrame:
    """Rank every alpha/strategy/window path by tracking quality.

    Lower RMSE ranks first.
    """
    frame = path_summary.copy()
    frame = frame.dropna(subset=["rmse"])
    if frame.empty:
        raise ValueError("No finite RMSE rows available to rank window metrics.")
    frame = frame.sort_values("rmse", ascending=True, kind="mergesort").reset_index(drop=True)
    frame.insert(0, "rank", range(1, len(frame) + 1))
    return frame


def _best_paths_by_alpha(path_summary: pd.DataFrame) -> pd.DataFrame:
    frame = path_summary.loc[path_summary["strategy"].eq(STRATEGY_MIQP)].copy()
    frame = frame.dropna(subset=["rmse"])
    if frame.empty:
        raise ValueError("No finite MIQP RMSE rows available to choose best paths.")
    frame = frame.sort_values(["alpha", "rmse"], ascending=[True, True], kind="mergesort")
    best = frame.groupby("alpha", sort=True, as_index=False).head(1).reset_index(drop=True)
    return _sort_strategy_frame(best)


def _return_corr(portfolio: pd.Series, benchmark: pd.Series) -> float:
    paired = pd.DataFrame({"portfolio": portfolio, "benchmark": benchmark}).dropna()
    if len(paired) < 2:
        return float("nan")
    if float(paired["portfolio"].std(ddof=1)) == 0 or float(paired["benchmark"].std(ddof=1)) == 0:
        return float("nan")
    return float(paired["portfolio"].corr(paired["benchmark"]))


def _sort_strategy_frame(frame: pd.DataFrame) -> pd.DataFrame:
    order = {name: index for index, name in enumerate(STRATEGY_ORDER)}
    out = frame.copy()
    out["_strategy_order"] = out["strategy"].map(order).fillna(len(STRATEGY_ORDER))
    sort_columns = []
    if "alpha" in out.columns:
        sort_columns.append("alpha")
    sort_columns.append("_strategy_order")
    if "window_months" in out.columns:
        sort_columns.append("window_months")
    if "quarter" in out.columns:
        sort_columns.append("quarter")
    if "date" in out.columns:
        sort_columns.append("date")
    out = out.sort_values(sort_columns, kind="mergesort")
    return out.drop(columns="_strategy_order")


def _next_quarter(quarter: str, decisions: pd.DataFrame) -> str:
    labels = list(decisions["quarter"])
    try:
        index = labels.index(quarter)
    except ValueError as error:
        raise ValueError(f"Unknown quarter {quarter}.") from error
    if index + 1 >= len(labels):
        raise ValueError(f"Quarter {quarter} has no successor in the decision calendar.")
    return labels[index + 1]


def _eval_dir_for_solve(solve_dir: Path) -> Path:
    eval_dir = EVAL_ROOT / solve_dir.name
    eval_dir.mkdir(parents=True, exist_ok=True)
    return eval_dir


def _alpha_tag(alpha: float) -> str:
    return f"{int(round(float(alpha) * 100)):02d}"


def _write_path_figure(
    daily: pd.DataFrame,
    path_summary: pd.DataFrame,
    decisions: pd.DataFrame,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    miqp_colors = {3: "#ff7f0e", 6: "#2ca02c", 9: "#17becf", 12: "#9467bd"}
    baseline_colors = {
        STRATEGY_EW: "#d62728",
        STRATEGY_TOPK_EW: "#1f77b4",
        STRATEGY_TOPK_DW: "#9ecae1",
    }
    # Layering: AIQ, then baselines (topk_dw under topk_ew), then MIQP on top.
    baseline_zorder = {
        STRATEGY_TOPK_DW: 2,
        STRATEGY_TOPK_EW: 3,
        STRATEGY_EW: 3,
    }
    metrics = _sort_strategy_frame(path_summary).copy()
    metrics["_months_key"] = metrics["window_months"].apply(
        lambda value: None if pd.isna(value) else int(value)
    )
    figure, axis = plt.subplots(figsize=(10, 5.5), dpi=150)

    aiq = (
        daily.groupby("date", as_index=False)["benchmark_return"]
        .first()
        .sort_values("date", kind="mergesort")
    )
    aiq["date"] = pd.to_datetime(aiq["date"])
    aiq["benchmark_cum"] = (1.0 + aiq["benchmark_return"].fillna(0.0)).cumprod()

    x_min = aiq["date"].min()
    x_max = aiq["date"].max()
    for decision in decisions["decision_date"]:
        stamp = pd.Timestamp(decision)
        if x_min <= stamp <= x_max:
            axis.axvline(stamp, color="#d0d0d0", linewidth=0.8, zorder=0)

    axis.plot(
        aiq["date"],
        aiq["benchmark_cum"],
        color="#000000",
        linewidth=1.0,
        label="AIQ",
        zorder=1,
    )

    ordered = _sort_strategy_frame(daily)
    plot_keys = list(ordered.groupby(["strategy", "window_months"], dropna=False, sort=False).groups)
    baseline_keys = [(strategy, months) for strategy, months in plot_keys if strategy != STRATEGY_MIQP]
    miqp_keys = [(strategy, months) for strategy, months in plot_keys if strategy == STRATEGY_MIQP]

    def _plot_strategy(strategy: object, months: object, color: str, zorder: int) -> None:
        block = ordered.loc[
            (ordered["strategy"] == strategy)
            & (
                ordered["window_months"].isna()
                if pd.isna(months)
                else ordered["window_months"].eq(months)
            )
        ].sort_values("date", kind="mergesort")
        block = block.copy()
        block["date"] = pd.to_datetime(block["date"])
        month_key = None if pd.isna(months) else int(months)
        row = metrics.loc[
            (metrics["strategy"] == strategy)
            & (
                metrics["_months_key"].isna()
                if month_key is None
                else metrics["_months_key"].eq(month_key)
            )
        ].iloc[0]
        label = (
            f"{_strategy_label(str(strategy), month_key)}  "
            f"RMSE={_format_metric(float(row['rmse']))}  "
            f"corr={_format_metric(float(row['return_corr']))}"
        )
        axis.plot(
            block["date"],
            block["portfolio_cum"],
            color=color,
            linewidth=1.0,
            label=label,
            zorder=zorder,
        )

    for strategy, months in baseline_keys:
        _plot_strategy(
            strategy,
            months,
            baseline_colors.get(str(strategy), "#8c564b"),
            baseline_zorder.get(str(strategy), 3),
        )
    for strategy, months in miqp_keys:
        month_key = None if pd.isna(months) else int(months)
        _plot_strategy(
            strategy,
            months,
            miqp_colors.get(month_key or 0, "#8c564b"),
            4,
        )

    _style_path_axis(axis)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _write_best_alpha_figure(
    daily: pd.DataFrame,
    best: pd.DataFrame,
    decisions: pd.DataFrame,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    figure, axis = plt.subplots(figsize=(10, 5.5), dpi=150)

    aiq = (
        daily.groupby("date", as_index=False)["benchmark_return"]
        .first()
        .sort_values("date", kind="mergesort")
    )
    aiq["date"] = pd.to_datetime(aiq["date"])
    aiq["benchmark_cum"] = (1.0 + aiq["benchmark_return"].fillna(0.0)).cumprod()

    x_min = aiq["date"].min()
    x_max = aiq["date"].max()
    for decision in decisions["decision_date"]:
        stamp = pd.Timestamp(decision)
        if x_min <= stamp <= x_max:
            axis.axvline(stamp, color="#d0d0d0", linewidth=0.8, zorder=0)

    axis.plot(
        aiq["date"],
        aiq["benchmark_cum"],
        color="#000000",
        linewidth=1.0,
        label="AIQ",
        zorder=1,
    )

    rows = best.reset_index(drop=True)
    baseline_rows = rows.loc[rows["strategy"] != STRATEGY_MIQP]
    miqp_rows = rows.loc[rows["strategy"] == STRATEGY_MIQP]
    plot_rows = pd.concat([baseline_rows, miqp_rows], ignore_index=True)

    for index, row in plot_rows.iterrows():
        alpha = float(row["alpha"])
        strategy = str(row["strategy"])
        months = row["window_months"]
        month_key = None if pd.isna(months) else int(months)
        block = daily.loc[
            daily["alpha"].eq(alpha)
            & daily["strategy"].eq(strategy)
            & (
                daily["window_months"].isna()
                if month_key is None
                else daily["window_months"].eq(month_key)
            )
        ].sort_values("date", kind="mergesort")
        block = block.copy()
        block["date"] = pd.to_datetime(block["date"])
        label = (
            f"a{_alpha_tag(alpha)} {_strategy_label(strategy, month_key)}  "
            f"RMSE={_format_metric(float(row['rmse']))}  "
            f"corr={_format_metric(float(row['return_corr']))}"
        )
        zorder = 4 if strategy == STRATEGY_MIQP else 3
        axis.plot(
            block["date"],
            block["portfolio_cum"],
            color=colors[int(index) % len(colors)],
            linewidth=1.0,
            label=label,
            zorder=zorder,
        )

    _style_path_axis(axis)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _style_path_axis(axis) -> None:
    axis.set_title("")
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.tick_params(labelbottom=False, labelleft=False, length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.grid(False)
    legend = axis.legend(frameon=False, loc="upper left")
    for text in legend.get_texts():
        text.set_fontsize(8)


def _strategy_label(strategy: str, months: int | None) -> str:
    if strategy == STRATEGY_MIQP:
        return f"L{months}"
    if strategy == STRATEGY_EW:
        return "ew"
    if strategy == STRATEGY_TOPK_EW:
        return "topk_ew"
    if strategy == STRATEGY_TOPK_DW:
        return "topk_dw"
    return strategy


def _format_metric(value: float) -> str:
    if pd.isna(value):
        return ""
    if abs(value) >= 0.01 or value == 0:
        return f"{value:.4f}"
    return f"{value:.6f}"


if __name__ == "__main__":
    main()
