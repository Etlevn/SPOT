"""Run the baseline SPOT solves from the project configuration."""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config import CONFIG
from load import load_source_tables
from preprocess import preprocess_master_frame
from quarters import summarize_quarters
from model import Solution, solve_miqp

OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "solve"


def main() -> None:
    tables = load_source_tables()
    master = preprocess_master_frame(tables)
    report = summarize_quarters(tables, master)
    panels = sorted(
        _selected_panels(report["panels"]),
        key=lambda panel: (str(panel["quarter"]), int(panel["months"])),
    )
    run_id, run_dir = _allocate_run_dir()
    snapshot = _run_snapshot()
    _write_snapshot(run_dir / "run.json", snapshot)
    tqdm.write(run_id)
    with tqdm(total=len(panels), position=0, leave=False, unit="") as progress:
        for panel in panels:
            solution = solve_miqp(
                panel,
                alpha=CONFIG.alpha,
                ell=CONFIG.ell,
                u=CONFIG.u,
                feasibility_tolerance=CONFIG.feasibility_tolerance,
                time_limit_seconds=CONFIG.time_limit_seconds,
                mip_gap=CONFIG.mip_gap,
                solver_log=CONFIG.solver_log,
                solver=CONFIG.solver,
            )
            if " " not in str(snapshot["solver"]):
                snapshot["solver"] = f"{solution.solver} {solution.solver_version}"
                _write_snapshot(run_dir / "run.json", snapshot)
            _print_solution(solution)
            _append_solution(solution, run_id, run_dir)
            progress.update(1)


def _print_solution(solution: Solution) -> None:
    label = f"{solution.quarter[:4]}_{solution.quarter[4:]}_L{solution.window_months}"
    hhi, effective_holdings = _concentration(solution.weights)
    solve = pd.DataFrame(
        [
            {
                "solver": solution.solver,
                "solver_version": solution.solver_version,
                "quarter": solution.quarter,
                "window_months": solution.window_months,
                "status": solution.status,
                "N": solution.n,
                "K": solution.k,
                "mean_squared_error": _format_optional(solution.mean_squared_error),
                "hhi": _format_optional(hhi),
                "effective_holdings": _format_optional(effective_holdings),
                "mip_gap": _format_optional(solution.mip_gap),
                "best_bound": _format_optional(solution.best_bound),
                "runtime_seconds": _format_optional(solution.runtime_seconds),
            }
        ]
    )
    checks = pd.DataFrame([_check_cells(solution.checks, formatted=True)])
    turned_on = solution.activated >= 0.5
    holdings = pd.DataFrame({"weight": solution.weights, "activated": solution.activated})
    holdings = holdings.loc[turned_on | holdings["weight"].abs().gt(CONFIG.feasibility_tolerance)]
    holdings = holdings.sort_values("weight", ascending=False)
    holdings["weight"] = holdings["weight"].map(_format_float)
    holdings["activated"] = holdings["activated"].map(_format_float)
    tqdm.write(f"{'=' * 30} {label} {'=' * 30}")
    tqdm.write("")
    tqdm.write("Solve")
    tqdm.write(solve.to_string(index=False))
    tqdm.write("")
    tqdm.write("Constraint checks")
    tqdm.write(checks.to_string(index=False))
    tqdm.write("")
    tqdm.write("Selected holdings")
    if holdings.empty:
        tqdm.write("No holdings in the solution.")
    else:
        tqdm.write(holdings.T.to_string())
    tqdm.write("")


def _selected_panels(panels: list[dict[str, object]]) -> list[dict[str, object]]:
    quarters = set(CONFIG.quarters)
    windows = set(CONFIG.window_months)
    selected = [
        panel
        for panel in panels
        if panel["months"] in windows and (not quarters or panel["quarter"] in quarters)
    ]
    if not selected:
        raise ValueError("No prepared window matches the quarters and windows in config.")
    return selected


def _allocate_run_dir() -> tuple[str, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    while True:
        run_id = datetime.now().strftime("%y%m%d_%H%M%S")
        run_dir = OUTPUT_DIR / run_id
        try:
            run_dir.mkdir(parents=True)
        except FileExistsError:
            time.sleep(1)
            continue
        return run_id, run_dir


def _run_snapshot() -> dict[str, object]:
    config = asdict(CONFIG)
    windows = list(config["window_months"])
    quarters = list(config["quarters"])
    snapshot: dict[str, object] = {
        "python": platform.python_version(),
        "solver": config["solver"],
        "windows": windows,
        "alpha": config["alpha"],
        "ell": config["ell"],
        "u": config["u"],
        "time_limit_seconds": config["time_limit_seconds"],
        "mip_gap": config["mip_gap"],
    }
    if quarters:
        snapshot["quarters"] = quarters
    return snapshot


def _write_snapshot(path: Path, snapshot: dict[str, object]) -> None:
    lines = ["{"]
    items = list(snapshot.items())
    for index, (key, value) in enumerate(items):
        comma = "," if index < len(items) - 1 else ""
        rendered = json.dumps(value, separators=(", ", ": "))
        lines.append(f"  {json.dumps(key)}: {rendered}{comma}")
    lines.append("}")
    path.write_text("\n".join(lines))


def _append_solution(solution: Solution, run_id: str, run_dir: Path) -> None:
    quarter = solution.quarter
    months = solution.window_months
    hhi, effective_holdings = _concentration(solution.weights)
    summary = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "solver": solution.solver,
                "solver_version": solution.solver_version,
                "quarter": quarter,
                "window_months": months,
                "status": solution.status,
                "N": solution.n,
                "K": solution.k,
                "alpha": CONFIG.alpha,
                "ell": CONFIG.ell,
                "u": CONFIG.u,
                "mean_squared_error": solution.mean_squared_error,
                "hhi": hhi,
                "effective_holdings": effective_holdings,
                "mip_gap": solution.mip_gap,
                "best_bound": solution.best_bound,
                "runtime_seconds": solution.runtime_seconds,
            }
        ]
    )
    checks = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "quarter": quarter,
                "window_months": months,
                **_check_cells(solution.checks, formatted=False),
            }
        ]
    )
    weights = pd.DataFrame(
        {
            "run_id": run_id,
            "quarter": quarter,
            "window_months": months,
            "position": range(1, len(solution.weights) + 1),
            "security_id": list(solution.weights.index),
            "weight": solution.weights.to_numpy(),
            "activated": solution.activated.reindex(solution.weights.index).to_numpy(),
        }
    )
    _write_sorted(summary, run_dir / "summary.csv", ["quarter", "window_months"])
    _write_sorted(checks, run_dir / "checks.csv", ["quarter", "window_months"])
    _write_sorted(weights, run_dir / "weights.csv", ["quarter", "window_months", "position"])


def _concentration(weights: pd.Series) -> tuple[float, float | None]:
    hhi = float((weights.to_numpy(dtype="float64") ** 2).sum())
    if hhi <= 0:
        return hhi, None
    return hhi, 1.0 / hhi


def _check_cells(checks: pd.DataFrame, formatted: bool) -> dict[str, str]:
    cells = {}
    for item in checks.itertuples(index=False):
        value = _format_float(float(item.value)) if formatted else item.value
        cells[item.check] = f"{value} ({item.passed})"
    return cells


def _write_sorted(frame: pd.DataFrame, path: Path, columns: list[str]) -> None:
    if path.exists():
        frame = pd.concat([pd.read_csv(path), frame], ignore_index=True)
    sort_columns = [column for column in columns if column in frame.columns]
    if "field" in frame.columns:
        frame["_field_order"] = frame["field"].map({"value": 0, "passed": 1})
        sort_columns.append("_field_order")
    frame = frame.sort_values(sort_columns, kind="mergesort").drop(columns="_field_order", errors="ignore")
    frame.to_csv(path, index=False)


def _format_optional(value: float | None) -> str:
    if value is None:
        return ""
    return _format_float(value)


def _format_float(value: float) -> str:
    if value == 0 or abs(value) >= 0.01:
        return f"{value:.4f}"
    return f"{value:.6f}"


if __name__ == "__main__":
    main()
