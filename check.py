"""Check sources, cleaning, quarterly inputs, and descriptive statistics."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from data import load_source_tables, preprocess_master_frame, summarize_quarters

OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "check"
TABLE_NAMES = (
    "sources",
    "unread",
    "gaps",
    "holdings",
    "mapping",
    "missing",
    "currencies",
    "returns",
    "alignment",
    "parameters",
    "inputs",
    "aiq_returns",
    "coverage",
)


def main() -> None:
    tables = load_source_tables()
    master = preprocess_master_frame(tables)
    report = summarize_quarters(tables, master)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.float_format", _format_float)
    lines: list[str] = []
    for name in TABLE_NAMES:
        title = _TABLE_TITLES[name]
        _emit(lines, title)
        _emit(lines, report[name].to_string(index=False))
        _emit(lines, "")
    for panel in report["panels"]:
        _emit_panel(lines, panel)
    _write_report(report, lines)


def _emit_panel(lines: list[str], panel: dict[str, object]) -> None:
    label = f"{panel['quarter']} L{panel['months']}"
    returns = _with_dates(panel["R"])
    benchmark = _with_dates(panel["b"]).to_frame()
    securities = pd.DataFrame(
        {
            "position": range(1, returns.shape[1] + 1),
            "security_id": list(returns.columns),
        }
    )
    _emit(lines, f"Dates {label}")
    _emit(lines, returns.index.to_series().to_string(index=False))
    _emit(lines, "")
    _emit(lines, f"Securities {label}")
    _emit(lines, securities.to_string(index=False))
    _emit(lines, "")
    _emit(lines, f"R {label}")
    _emit(lines, returns.to_string())
    _emit(lines, "")
    _emit(lines, f"b {label}")
    _emit(lines, benchmark.to_string())
    _emit(lines, "")


def _write_report(report: dict[str, object], lines: list[str]) -> None:
    table_dir = OUTPUT_DIR / "tables"
    matrix_dir = OUTPUT_DIR / "model_inputs"
    table_dir.mkdir(parents=True, exist_ok=True)
    matrix_dir.mkdir(parents=True, exist_ok=True)
    for name in TABLE_NAMES:
        report[name].to_csv(table_dir / f"{name}.csv", index=False)
    report["parameters"].to_csv(OUTPUT_DIR / "quarterly_parameters.csv", index=False)
    for panel in report["panels"]:
        label = f"{panel['quarter']}_L{panel['months']}"
        returns = _with_dates(panel["R"])
        benchmark = _with_dates(panel["b"]).to_frame()
        securities = pd.DataFrame(
            {
                "position": range(1, returns.shape[1] + 1),
                "security_id": list(returns.columns),
            }
        )
        returns.to_csv(matrix_dir / f"{label}_R.csv")
        benchmark.to_csv(matrix_dir / f"{label}_b.csv")
        securities.to_csv(matrix_dir / f"{label}_securities.csv", index=False)
    (OUTPUT_DIR / "check_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _with_dates(values: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    dated = values.copy()
    dated.index = pd.Index(
        [pd.Timestamp(value).date() for value in dated.index],
        name="date",
    )
    return dated


def _emit(lines: list[str], text: str) -> None:
    print(text)
    lines.append(text)


def _format_float(value: float) -> str:
    if value == 0 or abs(value) >= 0.01:
        return f"{value:.4f}"
    return f"{value:.6f}"


_TABLE_TITLES = {
    "sources": "Data sources",
    "unread": "Sheets left unread",
    "gaps": "Names absent from the master table",
    "holdings": "Holdings",
    "mapping": "Identifier mapping",
    "missing": "Missing values",
    "currencies": "Currency coverage",
    "returns": "Return construction",
    "alignment": "Trading-day alignment",
    "parameters": "Parameters and exclusions",
    "inputs": "Model inputs",
    "aiq_returns": "AIQ returns",
    "coverage": "Security coverage",
}


if __name__ == "__main__":
    main()
