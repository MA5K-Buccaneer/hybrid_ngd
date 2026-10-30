"""Linear solvers for the natural-gradient system ``G d = g``.

A solver is any callable ``solver(A, b) -> x`` acting on float64 NumPy arrays,
where ``A`` is the (damped, symmetric) metric and ``b`` the flat gradient. The
torch and JAX optimisers call it on the host; pass ``solver="direct"`` to them
to stay on-device with the framework's own ``linalg.solve`` instead.
"""

from .classical import NumpySolver
from .hhl import HHLCircuit, HHLInfo, QiskitHHLSolver

__all__ = ["HHLCircuit", "HHLInfo", "NumpySolver", "QiskitHHLSolver"]
