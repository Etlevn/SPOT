"""Baseline MIQP for one SPOT decision."""

if __name__ == "__main__":
    raise SystemExit("model cannot be executed directly. Call it from main.py.")

from model.miqp import Solution, select_panel, solve_miqp

__all__ = ["Solution", "select_panel", "solve_miqp"]
