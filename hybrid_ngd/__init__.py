"""Natural gradient descent for torch and JAX with classical or quantum (HHL) linear solves.

``hybrid_ngd.torch_ngd.NGD`` and ``hybrid_ngd.jax_ngd.NGD`` are imported lazily,
so only the framework you use needs to be installed.
"""

from .solvers import HHLCircuit, HHLInfo, NumpySolver, QiskitHHLSolver

__all__ = [
    "HHLCircuit", "HHLInfo", "NumpySolver", "QiskitHHLSolver",
    "TorchNGD", "TorchKFAC", "TorchMFAC", "JaxNGD", "JaxMFAC",
]


def __getattr__(name):
    if name == "TorchNGD":
        from .torch_ngd import NGD

        return NGD
    if name == "TorchKFAC":
        from .torch_kfac import KFAC

        return KFAC
    if name == "TorchMFAC":
        from .torch_mfac import MFAC

        return MFAC
    if name == "JaxMFAC":
        from .jax_mfac import MFAC

        return MFAC
    if name == "JaxNGD":
        from .jax_ngd import NGD

        return NGD
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
