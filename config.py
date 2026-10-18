"""Global parameters for a SPOT run."""

from __future__ import annotations

from dataclasses import dataclass

# Selection ratios used when ``Config.alphas`` is empty.
ALL_ALPHAS: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40, 0.50)


@dataclass(frozen=True)
class Config:
    """Settings for one or more SPOT solves.

    The eligible universe and the return matrices come from the data
    pipeline. ``alphas`` sets the cardinality limits to solve. An empty
    ``alphas`` tuple runs every value in ``ALL_ALPHAS``. ``solver`` is
    ``scip`` until a full Gurobi license is available. Set it to
    ``gurobi`` to use Gurobi. An empty ``quarters`` tuple runs every
    prepared decision date.
    """

    solver: str = "scip"  # "gurobi", "scip", or "bonmin".
    quarters: tuple[str, ...] = ()  # () for every quarter, or ("2023Q4", "2024Q1").
    window_months: tuple[int, ...] = (3, 6, 9, 12)  # (12,) or (3, 6, 9, 12). A single value needs the comma. Only 3, 6, 9, and 12 are prepared.
    alphas: tuple[float, ...] = ()  # (0.10,) for one ratio; () for all of ALL_ALPHAS.
    ell: float = 0.0  # 0.0 is the minimum weight of an activated name.
    u: float = 1.0  # 1.0 is the maximum weight of one name.
    feasibility_tolerance: float = 1e-6  # 1e-6.
    time_limit_seconds: float | None = 600.0  # hard cap; None means no time limit.
    mip_gap: float | None = None  # stop when gap <= this; None disables.
    # Early-stop when relative gap drop stays below the threshold for this long.
    # Example: 60s with 0.05 means "if gap falls by less than 5% in 60s, stop".
    gap_stall_seconds: float | None = 60.0  # None disables stall stopping.
    gap_stall_min_drop: float = 0.05  # required relative drop, e.g. 0.05 = 5%.
    terminal_mode: str = "solver"  # "summary" or "solver".


TERMINAL_MODES = ("summary", "solver")

CONFIG = Config()


def selected_alphas(config: Config | None = None) -> tuple[float, ...]:
    """Return the selection ratios this config should solve."""
    settings = CONFIG if config is None else config
    if settings.alphas:
        return tuple(float(value) for value in settings.alphas)
    return ALL_ALPHAS


def selected_terminal_mode(config: Config | None = None) -> str:
    """Return a validated terminal display mode for ``main.py``."""
    settings = CONFIG if config is None else config
    mode = str(settings.terminal_mode).strip().lower()
    if mode not in TERMINAL_MODES:
        raise ValueError(f"terminal_mode must be one of {TERMINAL_MODES}, got {settings.terminal_mode!r}.")
    return mode
