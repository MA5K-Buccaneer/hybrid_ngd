from __future__ import annotations

from typing import Callable

import numpy as np

METRICS = ("fisher", "gauss_newton", "hessian")

HostSolver = Callable[[np.ndarray, np.ndarray], np.ndarray]


def check_args(lr, damping: float, metric: str) -> None:
    if not callable(lr) and lr <= 0:
        raise ValueError(f"lr must be positive, got {lr}")
    if damping < 0:
        raise ValueError(f"damping must be non-negative, got {damping}")
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")


def resolve_solver(solver) -> HostSolver | None:
    """Return ``None`` for the framework-native solve, else a host callable."""
    if isinstance(solver, str):
        if solver == "direct":
            return None
        if solver == "numpy":
            from .solvers import NumpySolver

            return NumpySolver()
        if solver in ("qiskit", "hhl"):
            from .solvers import QiskitHHLSolver

            return QiskitHHLSolver()
        raise ValueError(f"unknown solver {solver!r}; use 'direct', 'numpy', 'qiskit' or a callable")
    if callable(solver):
        return solver
    raise TypeError(f"solver must be a string or a callable (A, b) -> x, got {type(solver)}")
