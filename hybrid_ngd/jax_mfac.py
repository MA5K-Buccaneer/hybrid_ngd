"""M-FAC for JAX pytrees: matrix-free natural gradient over a sliding window of gradients."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from ._common import check_args, resolve_solver


class MFACState(NamedTuple):
    count: jax.Array  # gradients seen so far
    grads: jax.Array  # [window, d] ring buffer
    gram: jax.Array  # [window, window] inner products of the buffer


class MFAC:
    r"""Matrix-free approximate curvature (Frantar, Kurtic & Alistarh, NeurIPS 2021).

    The same algorithm as :class:`~hybrid_ngd.torch_mfac.MFAC`: precondition with
    ``F = damping I + (1/k) sum_j g_j g_j^T`` over the last ``window`` gradients,
    computing ``F^{-1} g_t = k G^T (k damping I + G G^T)^{-1} e_t`` from a
    ``k x k`` system.

    ``loss_fn(params, *args, **kwargs)`` returns the loss (per-sample losses are
    averaged). With ``gradients="microbatch"`` it must return per-sample losses, and
    the curvature is the bias-corrected micro-batch estimate of the current step,
    ``damping I + R^T R`` with ``R = [gbar; sqrt(n_j/m)(g_j - gbar)]``, as in the torch
    version. :meth:`update` is pure and can be wrapped in ``jax.jit``. Buffers
    have a fixed size, and rows not yet filled are masked out. Host solvers such as
    :class:`~hybrid_ngd.QiskitHHLSolver` run through ``jax.pure_callback`` on the
    filled ``k x k`` block only.

    Args:
        loss_fn: See above.
        lr: Step size, or a schedule ``count -> lr``.
        damping: ``lambda`` in ``F = lambda I + ...``.
        window: Number of gradients in the sliding window, or of micro-batches.
            Defaults to the largest that fits the solver's ``max_dimension``
            (``max_dimension - 1`` for ``"microbatch"``), otherwise 32.
        solver: ``"direct"`` (``jnp.linalg.solve``), ``"qiskit"``, or any callable
            ``(A, b) -> x`` on NumPy arrays.
        gradients: ``"history"`` (the paper's sliding window) or ``"microbatch"``.
    """

    def __init__(self, loss_fn, lr=1e-3, damping=1e-4, window=None, solver="direct", gradients="history"):
        check_args(lr, damping, "fisher")
        if damping <= 0:
            raise ValueError("M-FAC needs damping > 0")
        if gradients not in ("history", "microbatch"):
            raise ValueError(f"gradients must be 'history' or 'microbatch', got {gradients!r}")
        self.gradients = gradients
        self.solver = resolve_solver(solver)
        limit = getattr(self.solver, "max_dimension", None)
        extra = 1 if gradients == "microbatch" else 0
        if window is None:
            window = limit - extra if limit else 32
        if limit is not None and window + extra > limit:
            raise ValueError(f"window={window} needs a {window + extra}x{window + extra} system, more than the "
                             f"solver's max_dimension={limit}")
        self.loss_fn = loss_fn
        self.lr = lr
        self.damping = damping
        self.window = window

    def init(self, params) -> MFACState:
        flat, _ = ravel_pytree(params)
        return MFACState(
            count=jnp.zeros((), jnp.int32),
            grads=jnp.zeros((self.window, flat.size), flat.dtype),
            gram=jnp.zeros((self.window, self.window), flat.dtype),
        )

    def update(self, params, state: MFACState, *args, **kwargs):
        """Take one step. Returns ``(new_params, new_state, loss)``."""
        flat, unravel = ravel_pytree(params)
        lr = self.lr(state.count) if callable(self.lr) else self.lr
        if self.gradients == "microbatch":
            loss, direction = self._microbatch_direction(flat, unravel, args, kwargs)
            return unravel(flat - lr * direction), state._replace(count=state.count + 1), loss
        loss, g = jax.value_and_grad(lambda x: jnp.mean(self.loss_fn(unravel(x), *args, **kwargs)))(flat)

        m = self.window
        slot = state.count % m
        grads = state.grads.at[slot].set(g)
        dots = grads @ g
        gram = state.gram.at[slot, :].set(dots).at[:, slot].set(dots)
        count = state.count + 1
        k = jnp.minimum(count, m)

        filled = jnp.arange(m) < k
        both = filled[:, None] & filled[None, :]
        # Unfilled rows become identity rows, decoupled from the filled block, and
        # their right-hand side is zero.
        system = jnp.where(both, gram, 0.0) + jnp.diag(jnp.where(filled, k * self.damping, 1.0))
        unit = jax.nn.one_hot(slot, m, dtype=flat.dtype)
        direction = k * (grads.T @ self._solve(system, unit, k))
        return unravel(flat - lr * direction), MFACState(count, grads, gram), loss

    def _microbatch_direction(self, flat, unravel, args, kwargs):
        m = self.window
        n = jax.eval_shape(lambda x: self.loss_fn(unravel(x), *args, **kwargs), flat).size
        if n < m:
            raise ValueError(f"loss_fn returned {n} per-sample losses, fewer than window={m}")
        groups = np.arange(n) * m // n  # static, matching the torch version's split
        counts = np.bincount(groups, minlength=m)

        def means(x):
            values = self.loss_fn(unravel(x), *args, **kwargs).reshape(-1)
            return jax.ops.segment_sum(values, groups, num_segments=m) / counts, values

        G, values = jax.jacrev(means, has_aux=True)(flat)  # [m, d]
        gbar = jnp.asarray(counts / n, dtype=flat.dtype) @ G  # the full-batch gradient
        R = jnp.concatenate([gbar[None], jnp.sqrt(jnp.asarray(counts / m, dtype=flat.dtype))[:, None] * (G - gbar)])
        system = R @ R.T + self.damping * jnp.eye(m + 1, dtype=flat.dtype)
        unit = jax.nn.one_hot(0, m + 1, dtype=flat.dtype)
        return values.mean(), R.T @ self._solve(system, unit, m + 1)

    def _solve(self, A, b, k):
        if self.solver is None:
            return jnp.linalg.solve(A, b)
        solver, dtype = self.solver, b.dtype

        def host(A, b, k):
            k = int(k)
            x = np.zeros(b.shape, dtype=dtype)
            x[:k] = solver(np.asarray(A, np.float64)[:k, :k], np.asarray(b, np.float64)[:k])
            return x

        return jax.pure_callback(host, jax.ShapeDtypeStruct(b.shape, dtype), A, b, k)
