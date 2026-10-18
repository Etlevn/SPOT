"""Global parameters for a SPOT run."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Settings for one baseline solve.

    The eligible universe and the return matrices come from the data
    pipeline. ``alpha`` only sets the cardinality limit for this solve.
    ``solver`` is ``scip`` until a full Gurobi license is available.
    Set it to ``gurobi`` to use Gurobi. An empty ``quarters``
    tuple runs every prepared decision date.
    """

    solver: str = "scip"  # "gurobi", "scip", or "bonmin".
    quarters: tuple[str, ...] = ()  # () for every quarter, or ("2023Q4", "2024Q1").
    window_months: tuple[int, ...] = (3,6,12)  # (12,) or (3, 6, 12). A single value needs the comma. Only 3, 6, and 12 are prepared.
    alpha: float = 0.10  # 0.10 means K = floor(0.10 * N).
    ell: float = 0.0  # 0.0 is the minimum weight of an activated name.
    u: float = 1.0  # 1.0 is the maximum weight of one name.
    feasibility_tolerance: float = 1e-6  # 1e-6.
    time_limit_seconds: float | None = 300.0  # 60.0, or None to search until proven optimal.
    mip_gap: float | None = None  # None, or 0.01 to stop at a 1% gap.
    solver_log: bool = False  # False, or True to print the solver log.


CONFIG = Config()
