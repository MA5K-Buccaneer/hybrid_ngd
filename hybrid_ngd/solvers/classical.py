"""Classical linear solvers."""

from __future__ import annotations

import numpy as np


class NumpySolver:
    """Classical host-side reference solver."""

    def __call__(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, b, rcond=None)[0]
