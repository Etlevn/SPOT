"""Read the SPOT Excel extracts. This module cannot be run as a script."""

from __future__ import annotations

import zipfile
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HOLDINGS_WORKBOOK = "raw_data_AIQ holdings and ETF prices.xlsx"
YAHOO_WORKBOOK = "raw_data_Yahoo_constituent_prices.xlsx"
WRDS_WORKBOOK = "raw_data_WRDS_supplemental_prices.xlsx"

# Issuer holdings have no verified publication time. Compustat Global is
# non-USD. Provenance sheets are notes, not model fields.
_SHEETS = {
    "sec_holdings": (HOLDINGS_WORKBOOK, "SEC derived holdings"),
    "security_crosswalk": (HOLDINGS_WORKBOOK, "Yahoo coverage"),
    "benchmark_prices": (HOLDINGS_WORKBOOK, "AIQ prices"),
    "yahoo_prices": (YAHOO_WORKBOOK, "Yahoo stock prices"),
    "wrds_prices": (WRDS_WORKBOOK, "Compustat NA"),
}

_SPREADSHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_OFFICE_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def _worksheet_row_counts(path: Path, sheet_names: list[str]) -> dict[str, int]:
    """Row counts for the progress total, without parsing cell values.

    Uses the worksheet dimension when Excel stored one. Sheets without it,
    such as the WRDS extract, are counted from their row tags.
    """
    with zipfile.ZipFile(path) as archive:
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        relationships = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {rel.attrib["Id"]: rel.attrib["Target"] for rel in relationships}
        paths: dict[str, str] = {}
        for sheet in workbook.findall(f"{{{_SPREADSHEET_NS}}}sheets/{{{_SPREADSHEET_NS}}}sheet"):
            target = targets[sheet.attrib[f"{{{_OFFICE_REL_NS}}}id"]].lstrip("/")
            if not target.startswith("xl/"):
                target = f"xl/{target}"
            paths[sheet.attrib["name"]] = target

        counts: dict[str, int] = {}
        for sheet_name in sheet_names:
            sheet_path = paths.get(sheet_name)
            if sheet_path is None:
                raise ValueError(f"Worksheet named {sheet_name!r} not found in {path.name}")
            counts[sheet_name] = _count_sheet_rows(archive, sheet_path)
    return counts


def _count_sheet_rows(archive: zipfile.ZipFile, sheet_path: str) -> int:
    dimension_tag = f"{{{_SPREADSHEET_NS}}}dimension"
    row_tag = f"{{{_SPREADSHEET_NS}}}row"
    counted = 0
    with archive.open(sheet_path) as handle:
        for _, element in ElementTree.iterparse(handle, events=("start",)):
            if element.tag == dimension_tag:
                ref = element.attrib.get("ref", "")
                row_number = "".join(character for character in ref.split(":")[-1] if character.isdigit())
                if row_number:
                    return int(row_number)
            elif element.tag == row_tag:
                counted += 1
            element.clear()
    return counted


def _read_sheet(workbook: pd.ExcelFile, sheet_name: str, progress: tqdm) -> tuple[pd.DataFrame, int]:
    """Read one sheet, advancing ``progress`` once per worksheet row.

    Row conversion matches ``pandas.io.excel._openpyxl.OpenpyxlReader.get_sheet_data``
    so the frame is the same one ``pandas.read_excel`` would return.
    """
    reader = workbook._reader
    consumed = 0

    def get_sheet_data(sheet, file_rows_needed: int | None = None):
        nonlocal consumed
        if reader.book.read_only:
            sheet.reset_dimensions()

        data: list[list] = []
        last_row_with_data = -1
        for row_number, row in enumerate(sheet.rows):
            converted_row = [reader._convert_cell(cell) for cell in row]
            while converted_row and converted_row[-1] == "":
                converted_row.pop()
            if converted_row:
                last_row_with_data = row_number
            data.append(converted_row)
            consumed += 1
            progress.update(1)
            if file_rows_needed is not None and len(data) >= file_rows_needed:
                break

        data = data[: last_row_with_data + 1]
        if data:
            max_width = max(len(data_row) for data_row in data)
            if min(len(data_row) for data_row in data) < max_width:
                empty_cell = [""]
                data = [
                    data_row + (max_width - len(data_row)) * empty_cell
                    for data_row in data
                ]
        return data

    reader.get_sheet_data = get_sheet_data
    frame = workbook.parse(sheet_name=sheet_name)
    return frame, consumed


def load_source_tables(data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Read the source sheets used to build the master table.

    Returns a dict of raw frames. Column names are the Excel headers.
    """
    root = Path(data_dir) if data_dir is not None else PROJECT_ROOT / "data"
    if not root.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {root}")

    sheets_by_path: dict[Path, list[tuple[str, str]]] = {}
    for key, (filename, sheet) in _SHEETS.items():
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing source workbook: {path}")
        sheets_by_path.setdefault(path, []).append((key, sheet))

    row_counts = {
        path: _worksheet_row_counts(path, [sheet_name for _, sheet_name in jobs])
        for path, jobs in sheets_by_path.items()
    }
    total_rows = sum(count for counts in row_counts.values() for count in counts.values())

    tables: dict[str, pd.DataFrame] = {}
    with tqdm(total=total_rows, position=0, leave=False, unit="") as progress:
        for path, jobs in sheets_by_path.items():
            workbook = pd.ExcelFile(path, engine="openpyxl")
            try:
                for key, sheet_name in jobs:
                    tables[key], consumed = _read_sheet(workbook, sheet_name, progress)
                    expected = row_counts[path][sheet_name]
                    if consumed != expected:
                        progress.total += consumed - expected
            finally:
                workbook.close()
    return tables


if __name__ == "__main__":
    raise SystemExit(
        "load cannot be executed directly. Call it from test_data.py or main."
    )
