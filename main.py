"""Run the baseline SPOT solves from the project configuration."""

from __future__ import annotations

import json
import platform
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterator, TextIO

import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config import CONFIG, selected_alphas, selected_terminal_mode
from load import load_source_tables
from preprocess import preprocess_master_frame
from quarters import summarize_quarters
from model import Solution, solve_miqp

OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "solve"


def main() -> None:
    terminal_mode = selected_terminal_mode()
    tables = load_source_tables()
    master = preprocess_master_frame(tables)
    report = summarize_quarters(tables, master)
    panels = sorted(
        _selected_panels(report["panels"]),
        key=lambda panel: (str(panel["quarter"]), int(panel["months"])),
    )
    alphas = selected_alphas()
    run_id, run_dir = _allocate_run_dir()
    snapshot = _run_snapshot(alphas, terminal_mode)
    _write_snapshot(run_dir / "run.json", snapshot)
    jobs = len(alphas) * len(panels)
    with _terminal_display(jobs) as (progress, console):
        _console_write(run_id, console)
        for alpha in alphas:
            for panel in panels:
                label = _job_label(alpha, str(panel["quarter"]), int(panel["months"]))
                if terminal_mode == "solver":
                    _console_write(f"{'=' * 30} {label} {'=' * 30}", console)
                    progress.refresh()
                # Leave stdout/stderr alone so SCIP/Gurobi can stream their logs.
                solution = solve_miqp(
                    panel,
                    alpha=alpha,
                    ell=CONFIG.ell,
                    u=CONFIG.u,
                    feasibility_tolerance=CONFIG.feasibility_tolerance,
                    time_limit_seconds=CONFIG.time_limit_seconds,
                    mip_gap=CONFIG.mip_gap,
                    gap_stall_seconds=CONFIG.gap_stall_seconds,
                    gap_stall_min_drop=CONFIG.gap_stall_min_drop,
                    solver_log=terminal_mode == "solver",
                    solver=CONFIG.solver,
                )
                if " " not in str(snapshot["solver"]):
                    snapshot["solver"] = f"{solution.solver} {solution.solver_version}"
                    _write_snapshot(run_dir / "run.json", snapshot)
                if terminal_mode == "summary":
                    _print_solution(solution, alpha, console)
                _append_solution(solution, run_id, run_dir, alpha)
                progress.update(1)
                progress.refresh()


def _job_label(alpha: float, quarter: str, months: int) -> str:
    return f"a{_alpha_tag(alpha)} {quarter[:4]}_{quarter[4:]}_L{months}"


@contextmanager
def _terminal_display(jobs: int) -> Iterator[tuple[tqdm, TextIO]]:
    """Draw the progress bar on the real terminal, below scrolling solver logs."""
    console = _open_console_stream()
    owns_console = console is not sys.stderr and console is not sys.stdout
    try:
        with tqdm(
            total=jobs,
            position=0,
            leave=True,
            unit="",
            file=console,
            dynamic_ncols=True,
            mininterval=0.1,
        ) as progress:
            yield progress, console
    finally:
        if owns_console:
            console.close()


def _open_console_stream() -> TextIO:
    try:
        return open("/dev/tty", "w", encoding="utf-8", buffering=1)
    except OSError:
        return sys.stderr


def _console_write(text: str, console: TextIO) -> None:
    tqdm.write(text, file=console)


def _print_solution(solution: Solution, alpha: float, console: TextIO) -> None:
    label = _job_label(alpha, solution.quarter, solution.window_months)
    hhi, effective_holdings = _concentration(solution.weights)
    solve = pd.DataFrame(
        [
            {
                "solver": solution.solver,
                "solver_version": solution.solver_version,
                "alpha": alpha,
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
    _console_write(f"{'=' * 30} {label} {'=' * 30}", console)
    _console_write("", console)
    _console_write("Solve", console)
    _console_write(solve.to_string(index=False), console)
    _console_write("", console)
    _console_write("Constraint checks", console)
    _console_write(checks.to_string(index=False), console)
    _console_write("", console)
    _console_write("Selected holdings", console)
    if holdings.empty:
        _console_write("No holdings in the solution.", console)
    else:
        _console_write(holdings.T.to_string(), console)
    _console_write("", console)


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


def _run_snapshot(alphas: tuple[float, ...], terminal_mode: str) -> dict[str, object]:
    config = asdict(CONFIG)
    windows = list(config["window_months"])
    quarters = list(config["quarters"])
    snapshot: dict[str, object] = {
        "python": platform.python_version(),
        "solver": config["solver"],
        "windows": windows,
        "alphas": list(alphas),
        "ell": config["ell"],
        "u": config["u"],
        "time_limit_seconds": config["time_limit_seconds"],
        "mip_gap": config["mip_gap"],
        "gap_stall_seconds": config["gap_stall_seconds"],
        "gap_stall_min_drop": config["gap_stall_min_drop"],
        "terminal_mode": terminal_mode,
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


def _append_solution(solution: Solution, run_id: str, run_dir: Path, alpha: float) -> None:
    quarter = solution.quarter
    months = solution.window_months
    hhi, effective_holdings = _concentration(solution.weights)
    summary = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "solver": solution.solver,
                "solver_version": solution.solver_version,
                "alpha": alpha,
                "quarter": quarter,
                "window_months": months,
                "status": solution.status,
                "N": solution.n,
                "K": solution.k,
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
                "alpha": alpha,
                "quarter": quarter,
                "window_months": months,
                **_check_cells(solution.checks, formatted=False),
            }
        ]
    )
    weights = pd.DataFrame(
        {
            "run_id": run_id,
            "alpha": alpha,
            "quarter": quarter,
            "window_months": months,
            "position": range(1, len(solution.weights) + 1),
            "security_id": list(solution.weights.index),
            "weight": solution.weights.to_numpy(),
            "activated": solution.activated.reindex(solution.weights.index).to_numpy(),
        }
    )
    sort_keys = ["alpha", "quarter", "window_months"]
    _write_sorted(summary, run_dir / "summary.csv", sort_keys)
    _write_sorted(checks, run_dir / "checks.csv", sort_keys)
    _write_sorted(weights, run_dir / "weights.csv", [*sort_keys, "position"])


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


def _alpha_tag(alpha: float) -> str:
    return f"{int(round(alpha * 100)):02d}"


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
