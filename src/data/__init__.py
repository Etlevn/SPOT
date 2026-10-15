"""Data loading and preprocessing for the SPOT master table."""

if __name__ == "__main__":
    raise SystemExit(
        "data cannot be executed directly. Call it from test_data.py or main."
    )

from data.load import load_source_tables
from data.preprocess import preprocess_master_frame
from data.quarters import summarize_quarters

__all__ = ["load_source_tables", "preprocess_master_frame", "summarize_quarters"]
