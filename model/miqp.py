"""Solve and check the baseline tracking MIQP for one window."""

from __future__ import annotations

import logging
import math
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

if __name__ == "__main__":
    raise SystemExit("model.miqp cannot be executed directly. Call it from main.py.")

import pandas as pd

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError as error:
    gp = None
    GRB = None
    _GUROBI_IMPORT_ERROR = error
else:
    _GUROBI_IMPORT_ERROR = None


@dataclass(frozen=True)
class Solution:
    """One solved window and the checks made on that solution."""

    quarter: str
    window_months: int
    solver: str
    solver_version: str
    status: str
    mean_squared_error: float | None
    mip_gap: float | None
    best_bound: float | None
    runtime_seconds: float
    n: int
    k: int
    weights: pd.Series
    activated: pd.Series
    checks: pd.DataFrame


def select_panel(panels: list[dict[str, object]], quarter: str, window_months: int) -> dict[str, object]:
    """Return the single prepared window for this quarter and length."""
    matches = [
        panel
        for panel in panels
        if panel["quarter"] == quarter and panel["months"] == window_months
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one panel for {quarter} L{window_months}, found {len(matches)}.")
    return matches[0]


def solve_miqp(
    panel: dict[str, object],
    *,
    alpha: float,
    ell: float,
    u: float,
    feasibility_tolerance: float,
    time_limit_seconds: float | None,
    mip_gap: float | None,
    gap_stall_seconds: float | None,
    gap_stall_min_drop: float,
    solver_log: bool,
    solver: str,
) -> Solution:
    """Minimize mean squared tracking error for one eligible universe."""
    if not 0 <= ell <= u <= 1:
        raise ValueError(f"Position bounds must satisfy 0 <= ell <= u <= 1, got ell={ell}, u={u}.")
    if gap_stall_seconds is not None and gap_stall_seconds <= 0:
        raise ValueError(f"gap_stall_seconds must be positive when set, got {gap_stall_seconds}.")
    if not 0 <= gap_stall_min_drop < 1:
        raise ValueError(f"gap_stall_min_drop must satisfy 0 <= value < 1, got {gap_stall_min_drop}.")

    returns = panel["R"]
    benchmark = panel["b"]
    securities = [str(security) for security in returns.columns]
    dates = list(returns.index)
    observation_count = len(dates)
    universe_size = len(securities)
    if observation_count == 0 or universe_size == 0:
        raise ValueError("The selected window has no dates or no securities.")
    cardinality_limit = math.floor(alpha * universe_size)
    if solver == "gurobi":
        solved = _solve_gurobi(
            returns=returns,
            benchmark=benchmark,
            securities=securities,
            dates=dates,
            cardinality_limit=cardinality_limit,
            ell=ell,
            u=u,
            time_limit_seconds=time_limit_seconds,
            mip_gap=mip_gap,
            gap_stall_seconds=gap_stall_seconds,
            gap_stall_min_drop=gap_stall_min_drop,
            solver_log=solver_log,
            model_name=f"spot_{panel['quarter']}_L{panel['months']}",
        )
    elif solver == "scip":
        solved = _solve_scip(
            returns=returns,
            benchmark=benchmark,
            securities=securities,
            dates=dates,
            cardinality_limit=cardinality_limit,
            ell=ell,
            u=u,
            time_limit_seconds=time_limit_seconds,
            mip_gap=mip_gap,
            gap_stall_seconds=gap_stall_seconds,
            gap_stall_min_drop=gap_stall_min_drop,
            solver_log=solver_log,
        )
    elif solver == "bonmin":
        solved = _solve_bonmin(
            returns=returns,
            benchmark=benchmark,
            securities=securities,
            dates=dates,
            cardinality_limit=cardinality_limit,
            ell=ell,
            u=u,
            time_limit_seconds=time_limit_seconds,
            mip_gap=mip_gap,
            solver_log=solver_log,
        )
    else:
        raise ValueError(f"Unknown solver {solver!r}. Use 'scip', 'gurobi', or 'bonmin'.")

    weights = solved["weights"]
    activation = solved["activated"]
    objective = solved["objective"]
    checks = _check_solution(
        returns=returns,
        benchmark=benchmark,
        weights=weights,
        activated=activation,
        cardinality_limit=cardinality_limit,
        ell=ell,
        u=u,
        objective=objective,
        tolerance=feasibility_tolerance,
        has_solution=solved["has_solution"],
    )
    return Solution(
        quarter=str(panel["quarter"]),
        window_months=int(panel["months"]),
        solver=solver,
        solver_version=str(solved["solver_version"]),
        status=str(solved["status"]),
        mean_squared_error=objective,
        mip_gap=solved["mip_gap"],
        best_bound=solved["best_bound"],
        runtime_seconds=float(solved["runtime_seconds"]),
        n=universe_size,
        k=cardinality_limit,
        weights=weights,
        activated=activation,
        checks=checks,
    )


def _solve_gurobi(
    *,
    returns: pd.DataFrame,
    benchmark: pd.Series,
    securities: list[str],
    dates: list,
    cardinality_limit: int,
    ell: float,
    u: float,
    time_limit_seconds: float | None,
    mip_gap: float | None,
    gap_stall_seconds: float | None,
    gap_stall_min_drop: float,
    solver_log: bool,
    model_name: str,
) -> dict[str, object]:
    if gp is None or GRB is None:
        raise RuntimeError(
            "gurobipy is not installed. Gurobi 13 provides wheels for Python 3.10 through 3.14."
        ) from _GUROBI_IMPORT_ERROR
    universe_size = len(securities)
    observation_count = len(dates)
    model = gp.Model(model_name)
    model.Params.OutputFlag = 1 if solver_log else 0
    if time_limit_seconds is not None:
        model.Params.TimeLimit = time_limit_seconds
    if mip_gap is not None:
        model.Params.MIPGap = mip_gap

    weight = model.addVars(universe_size, lb=0.0, ub=u, name="w")
    activated = model.addVars(universe_size, vtype=GRB.BINARY, name="z")
    error = model.addVars(observation_count, lb=-GRB.INFINITY, name="e")
    for date_index, date in enumerate(dates):
        predicted = gp.quicksum(
            float(returns.at[date, security]) * weight[name_index]
            for name_index, security in enumerate(securities)
        )
        model.addConstr(
            error[date_index] == predicted - float(benchmark.at[date]),
            name=f"error_{date_index}",
        )
    squared_error = gp.quicksum(
        error[date_index] * error[date_index] for date_index in range(observation_count)
    )
    model.setObjective(squared_error / observation_count, GRB.MINIMIZE)
    model.addConstr(gp.quicksum(weight[name_index] for name_index in range(universe_size)) == 1, name="budget")
    model.addConstr(gp.quicksum(activated[name_index] for name_index in range(universe_size)) <= cardinality_limit, name="cardinality")
    for name_index in range(universe_size):
        model.addConstr(weight[name_index] >= ell * activated[name_index], name=f"lower_{name_index}")
        model.addConstr(weight[name_index] <= u * activated[name_index], name=f"upper_{name_index}")
    if gap_stall_seconds is not None:
        model._gap_stall_seconds = float(gap_stall_seconds)
        model._gap_stall_min_drop = float(gap_stall_min_drop)
        model._gap_stall_ref_gap = None
        model._gap_stall_ref_time = None
        model._gap_stall_triggered = False
        model.optimize(callback=_gurobi_gap_stall_callback)
    else:
        model.optimize()

    status = "gap_stall" if getattr(model, "_gap_stall_triggered", False) else _status_name(model.Status)
    weights = pd.Series(0.0, index=securities, name="weight")
    activation = pd.Series(0.0, index=securities, name="activated")
    objective = None
    if model.SolCount:
        weights = pd.Series(
            [weight[name_index].X for name_index in range(universe_size)],
            index=securities,
            name="weight",
        )
        activation = pd.Series(
            [activated[name_index].X for name_index in range(universe_size)],
            index=securities,
            name="activated",
        )
        objective = float(model.ObjVal)
    return {
        "status": status,
        "objective": objective,
        "runtime_seconds": float(model.Runtime),
        "weights": weights,
        "activated": activation,
        "mip_gap": _gurobi_number(model, "MIPGap"),
        "best_bound": _gurobi_number(model, "ObjBound"),
        "runtime_seconds": float(model.Runtime),
        "solver_version": _gurobi_version(),
        "has_solution": bool(model.SolCount),
    }


def _solve_scip(
    *,
    returns: pd.DataFrame,
    benchmark: pd.Series,
    securities: list[str],
    dates: list,
    cardinality_limit: int,
    ell: float,
    u: float,
    time_limit_seconds: float | None,
    mip_gap: float | None,
    gap_stall_seconds: float | None,
    gap_stall_min_drop: float,
    solver_log: bool,
) -> dict[str, object]:
    try:
        from pyscipopt import Eventhdlr, Model, SCIP_EVENTTYPE, quicksum
        from pyscipopt.recipes.nonlinear import set_nonlinear_objective
    except ImportError as error:
        raise RuntimeError("pyscipopt is not installed.") from error

    universe_size = len(securities)
    model = Model("spot_scip")
    # Default SCIP output goes to the terminal. Only mute it in summary mode.
    if not solver_log:
        model.hideOutput()
    weight = [model.addVar(lb=0.0, ub=u, name=f"w{name_index}") for name_index in range(universe_size)]
    activated = [model.addVar(vtype="B", name=f"z{name_index}") for name_index in range(universe_size)]
    error = [model.addVar(lb=None, name=f"e{date_index}") for date_index in range(len(dates))]
    for date_index, date in enumerate(dates):
        predicted = quicksum(
            float(returns.at[date, security]) * weight[name_index]
            for name_index, security in enumerate(securities)
        )
        model.addCons(error[date_index] == predicted - float(benchmark.at[date]))
    model.addCons(quicksum(weight) == 1)
    model.addCons(quicksum(activated) <= cardinality_limit)
    for name_index in range(universe_size):
        model.addCons(weight[name_index] >= ell * activated[name_index])
        model.addCons(weight[name_index] <= u * activated[name_index])
    squared_error = quicksum(error[date_index] * error[date_index] for date_index in range(len(dates)))
    set_nonlinear_objective(model, squared_error / len(dates), "minimize")
    if time_limit_seconds is not None:
        model.setParam("limits/time", time_limit_seconds)
    if mip_gap is not None:
        model.setParam("limits/gap", mip_gap)
    stall_handler = None
    if gap_stall_seconds is not None:

        class GapStallHandler(Eventhdlr):
            def __init__(self) -> None:
                super().__init__()
                self.stall_seconds = float(gap_stall_seconds)
                self.min_drop = float(gap_stall_min_drop)
                self.ref_gap: float | None = None
                self.ref_time: float | None = None
                self.last_check_time = -1.0
                self.triggered = False

            def eventinit(self) -> None:
                self.model.catchEvent(SCIP_EVENTTYPE.GAPUPDATED, self)
                self.model.catchEvent(SCIP_EVENTTYPE.NODESOLVED, self)

            def eventexec(self, event) -> None:
                if self.triggered:
                    return
                try:
                    gap = float(self.model.getGap())
                    now = float(self.model.getSolvingTime())
                except (TypeError, ValueError, AttributeError):
                    return
                if not math.isfinite(gap) or gap < 0:
                    return
                if now - self.last_check_time < 0.5 and self.ref_gap is not None:
                    return
                self.last_check_time = now
                if self.ref_gap is None or self.ref_time is None:
                    self.ref_gap = gap
                    self.ref_time = now
                    return
                relative_drop = (self.ref_gap - gap) / self.ref_gap if self.ref_gap > 1e-12 else 0.0
                if relative_drop >= self.min_drop:
                    self.ref_gap = gap
                    self.ref_time = now
                    return
                if now - self.ref_time >= self.stall_seconds:
                    self.triggered = True
                    self.model.interruptSolve()

        stall_handler = GapStallHandler()
        model.includeEventhdlr(stall_handler, "gap_stall", "Stop when relative gap stalls.")
    model.optimize()

    weights = pd.Series(0.0, index=securities, name="weight")
    activation = pd.Series(0.0, index=securities, name="activated")
    objective = None
    has_solution = model.getNSols() > 0
    if has_solution:
        weights = pd.Series(
            [model.getVal(weight[name_index]) for name_index in range(universe_size)],
            index=securities,
            name="weight",
        )
        activation = pd.Series(
            [model.getVal(activated[name_index]) for name_index in range(universe_size)],
            index=securities,
            name="activated",
        )
        objective = float(model.getObjVal())
    status_names = {
        "optimal": "optimal",
        "timelimit": "time_limit",
        "infeasible": "infeasible",
        "unbounded": "unbounded",
        "gaplimit": "gap_limit",
        "userinterrupt": "interrupted",
    }
    status = status_names.get(model.getStatus(), model.getStatus())
    if stall_handler is not None and stall_handler.triggered:
        status = "gap_stall"
    return {
        "status": status,
        "objective": objective,
        "runtime_seconds": float(model.getSolvingTime()),
        "weights": weights,
        "activated": activation,
        "mip_gap": _finite_number(model.getGap()),
        "best_bound": _finite_number(model.getDualbound()),
        "runtime_seconds": float(model.getSolvingTime()),
        "solver_version": _scip_version(model),
        "has_solution": has_solution,
    }


def _gurobi_gap_stall_callback(model, where) -> None:
    if gp is None or GRB is None:
        return
    if where != GRB.Callback.MIP:
        return
    if getattr(model, "_gap_stall_triggered", False):
        return
    try:
        runtime = float(model.cbGet(GRB.Callback.RUNTIME))
        primal = float(model.cbGet(GRB.Callback.MIP_OBJBST))
        dual = float(model.cbGet(GRB.Callback.MIP_OBJBND))
    except (TypeError, ValueError, AttributeError):
        return
    if not math.isfinite(primal) or not math.isfinite(dual) or abs(primal) <= 1e-12:
        return
    gap = abs(primal - dual) / abs(primal)
    if not math.isfinite(gap):
        return
    ref_gap = getattr(model, "_gap_stall_ref_gap", None)
    ref_time = getattr(model, "_gap_stall_ref_time", None)
    if ref_gap is None or ref_time is None:
        model._gap_stall_ref_gap = gap
        model._gap_stall_ref_time = runtime
        return
    relative_drop = (ref_gap - gap) / ref_gap if ref_gap > 1e-12 else 0.0
    if relative_drop >= float(model._gap_stall_min_drop):
        model._gap_stall_ref_gap = gap
        model._gap_stall_ref_time = runtime
        return
    if runtime - ref_time >= float(model._gap_stall_seconds):
        model._gap_stall_triggered = True
        model.terminate()


def _solve_bonmin(
    *,
    returns: pd.DataFrame,
    benchmark: pd.Series,
    securities: list[str],
    dates: list,
    cardinality_limit: int,
    ell: float,
    u: float,
    time_limit_seconds: float | None,
    mip_gap: float | None,
    solver_log: bool,
) -> dict[str, object]:
    try:
        from pyomo.environ import Binary, ConcreteModel, Constraint, Objective, SolverFactory, Var, minimize, value
        from pyomo.opt import TerminationCondition
    except ImportError as error:
        raise RuntimeError("pyomo could not be imported.") from error

    executable = _bonmin_executable()
    model = ConcreteModel()
    model.w = Var(securities, bounds=(0.0, u))
    model.z = Var(securities, domain=Binary)
    model.e = Var(dates)
    model.error = Constraint(
        dates,
        rule=lambda block, date: block.e[date]
        == sum(float(returns.at[date, security]) * block.w[security] for security in securities)
        - float(benchmark.at[date]),
    )
    model.budget = Constraint(expr=sum(model.w[security] for security in securities) == 1)
    model.cardinality = Constraint(expr=sum(model.z[security] for security in securities) <= cardinality_limit)
    model.lower = Constraint(securities, rule=lambda block, security: block.w[security] >= ell * block.z[security])
    model.upper = Constraint(securities, rule=lambda block, security: block.w[security] <= u * block.z[security])
    model.objective = Objective(
        expr=sum(model.e[date] ** 2 for date in dates) / len(dates),
        sense=minimize,
    )

    solver = SolverFactory("bonmin", executable=str(executable))
    solver.options["max_iter"] = 1000000
    solver.options["bonmin.iteration_limit"] = 100000000
    if time_limit_seconds is not None:
        solver.options["bonmin.time_limit"] = time_limit_seconds
    if mip_gap is not None:
        solver.options["bonmin.allowable_fraction_gap"] = mip_gap
    started = time.perf_counter()
    pyomo_logger = logging.getLogger("pyomo.core")
    previous_level = pyomo_logger.level
    pyomo_logger.setLevel(logging.ERROR)
    try:
        results = solver.solve(model, tee=solver_log)
    finally:
        pyomo_logger.setLevel(previous_level)
    runtime_seconds = time.perf_counter() - started

    status_names = {
        TerminationCondition.optimal: "optimal",
        TerminationCondition.maxTimeLimit: "time_limit",
        TerminationCondition.infeasible: "infeasible",
        TerminationCondition.unbounded: "unbounded",
        TerminationCondition.feasible: "feasible",
        TerminationCondition.maxIterations: "iteration_limit",
    }
    status = status_names.get(results.solver.termination_condition, str(results.solver.termination_condition))
    weights = pd.Series(0.0, index=securities, name="weight")
    activation = pd.Series(0.0, index=securities, name="activated")
    objective = None
    has_solution = False
    try:
        weights = pd.Series([value(model.w[security]) for security in securities], index=securities, name="weight")
        activation = pd.Series([value(model.z[security]) for security in securities], index=securities, name="activated")
        objective = float(value(model.objective))
        has_solution = True
    except ValueError:
        has_solution = False
    upper = _finite_number(getattr(results.problem, "upper_bound", None))
    lower = _finite_number(getattr(results.problem, "lower_bound", None))
    gap = None
    if status == "optimal" and objective is not None:
        lower = objective
        gap = 0.0
    elif upper is not None and lower is not None and abs(upper) > 1e-12:
        gap = _finite_number((upper - lower) / abs(upper))
    version = solver.version()
    solver_version = ".".join(str(part) for part in version) if version else "unknown"
    return {
        "status": status,
        "objective": objective,
        "mip_gap": gap,
        "best_bound": lower,
        "runtime_seconds": runtime_seconds,
        "solver_version": solver_version,
        "weights": weights,
        "activated": activation,
        "has_solution": has_solution,
    }


def _bonmin_executable() -> Path:
    configured = os.environ.get("BONMIN_EXECUTABLE")
    if configured:
        path = Path(configured)
    else:
        path = Path(__file__).resolve().parents[1] / ".tools" / "bonmin" / "bin" / "bonmin"
        if not path.is_file():
            found = shutil.which("bonmin")
            path = Path(found) if found else path
    if not path.is_file():
        raise RuntimeError(
            "bonmin is not installed. Expected .tools/bonmin/bin/bonmin or BONMIN_EXECUTABLE."
        )
    return path


def _check_solution(
    *,
    returns: pd.DataFrame,
    benchmark: pd.Series,
    weights: pd.Series,
    activated: pd.Series,
    cardinality_limit: int,
    ell: float,
    u: float,
    objective: float | None,
    tolerance: float,
    has_solution: bool,
) -> pd.DataFrame:
    if not has_solution:
        return pd.DataFrame([{"check": "solution_found", "value": 0, "passed": False}])
    active = returns.to_numpy(dtype="float64") @ weights.to_numpy(dtype="float64") - benchmark.to_numpy(dtype="float64")
    recomputed = float((active ** 2).mean())
    turned_on = activated >= 0.5
    lower_gap = float((ell * activated - weights).max())
    upper_gap = float((weights - u * activated).max())
    rows = [
        ("budget_gap", abs(float(weights.sum()) - 1.0), abs(float(weights.sum()) - 1.0) <= tolerance),
        ("minimum_weight", float(weights.min()), float(weights.min()) >= -tolerance),
        ("maximum_weight", float(weights.max()), float(weights.max()) <= u + tolerance),
        ("activated_count", int(turned_on.sum()), int(turned_on.sum()) <= cardinality_limit),
        ("lower_bound_gap", lower_gap, lower_gap <= tolerance),
        ("upper_bound_gap", upper_gap, upper_gap <= tolerance),
        ("objective_gap", abs(recomputed - float(objective)), abs(recomputed - float(objective)) <= tolerance),
    ]
    return pd.DataFrame(rows, columns=["check", "value", "passed"])


def _finite_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(number) and abs(number) < 1e19:
        return number
    return None


def _gurobi_number(model: object, attribute: str) -> float | None:
    try:
        return _finite_number(getattr(model, attribute))
    except gp.GurobiError:
        return None


def _gurobi_version() -> str:
    major, minor, technical = gp.gurobi.version()
    return f"{major}.{minor}.{technical}"


def _scip_version(model: object) -> str:
    return f"{model.getMajorVersion()}.{model.getMinorVersion()}.{model.getTechVersion()}"


def _status_name(status: int) -> str:
    names = {
        GRB.OPTIMAL: "optimal",
        GRB.INFEASIBLE: "infeasible",
        GRB.INF_OR_UNBD: "infeasible_or_unbounded",
        GRB.UNBOUNDED: "unbounded",
        GRB.TIME_LIMIT: "time_limit",
        GRB.SUBOPTIMAL: "suboptimal",
        GRB.INTERRUPTED: "interrupted",
        GRB.NUMERIC: "numeric",
    }
    return names.get(status, str(status))
