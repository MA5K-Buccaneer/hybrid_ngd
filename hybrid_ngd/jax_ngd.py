"""Natural gradient descent for JAX pytrees."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from ._common import check_args, resolve_solver


class NGDState(NamedTuple):
    count: jax.Array


class NGD:
    r"""Natural gradient descent: ``theta <- theta - lr * (G + damping * I)^{-1} grad L``.

    ``loss_fn(params, *args, **kwargs)`` returns, depending on ``metric``:

    * ``"fisher"``: per-sample losses ``l_i`` (shape ``[N]``). ``L = mean(l_i)``
      and ``G`` is the empirical Fisher ``mean(grad l_i grad l_i^T)``.
    * ``"gauss_newton"``: residuals ``r_i`` (shape ``[N]``). ``L = mean(r_i^2) / 2``
      and ``G = mean(grad r_i grad r_i^T)``, the Fisher of a Gaussian likelihood.
    * ``"hessian"``: a scalar loss (per-sample losses are averaged), with
      ``G = hess L``. This gives a damped Newton step.

    ``params`` may be any pytree. :meth:`update` is pure and can be wrapped in
    ``jax.jit``. Host solvers such as :class:`~hybrid_ngd.QiskitHHLSolver` run
    through ``jax.pure_callback``, so they also work under jit (but not under
    ``grad`` of the update).

    Example::

        opt = NGD(loss_fn, lr=0.5, metric="gauss_newton", solver=QiskitHHLSolver())
        state = opt.init(params)
        step = jax.jit(opt.update)
        params, state, loss = step(params, state, x, y)

    Args:
        loss_fn: See above.
        lr: Step size, or a schedule ``count -> lr``.
        damping: Tikhonov damping added to the diagonal of ``G``.
        metric: ``"fisher"``, ``"gauss_newton"`` or ``"hessian"``.
        solver: ``"direct"`` uses ``jnp.linalg.solve``. ``"qiskit"`` uses a
            default :class:`~hybrid_ngd.QiskitHHLSolver`. Any callable
            ``(A, b) -> x`` on NumPy arrays also works.
    """

    def __init__(self, loss_fn, lr=0.1, damping=1e-3, metric="fisher", solver="direct"):
        check_args(lr, damping, metric)
        self.loss_fn = loss_fn
        self.lr = lr
        self.damping = damping
        self.metric = metric
        self.solver = resolve_solver(solver)

    def init(self, params) -> NGDState:
        return NGDState(count=jnp.zeros((), jnp.int32))

    def update(self, params, state: NGDState, *args, **kwargs):
        """Take one step. Returns ``(new_params, new_state, loss)``."""
        flat, unravel = ravel_pytree(params)

        def f(x):
            return self.loss_fn(unravel(x), *args, **kwargs)

        if self.metric == "hessian":

            def objective(x):
                return jnp.mean(f(x))

            loss, grad = jax.value_and_grad(objective)(flat)
            G = jax.hessian(objective)(flat)
        else:

            def values_twice(x):
                v = f(x).reshape(-1)
                return v, v

            jac, values = jax.jacrev(values_twice, has_aux=True)(flat)  # [N, n], [N]
            if self.metric == "fisher":
                loss = values.mean()
                grad = jac.mean(0)
            else:
                loss = 0.5 * jnp.mean(values**2)
                grad = jac.T @ values / values.size
            G = jac.T @ jac / values.size

        G = G + self.damping * jnp.eye(flat.size, dtype=flat.dtype)
        direction = self._solve(G, grad)
        lr = self.lr(state.count) if callable(self.lr) else self.lr
        return unravel(flat - lr * direction), NGDState(state.count + 1), loss

    def _solve(self, A, b):
        if self.solver is None:
            return jnp.linalg.solve(A, b)
        solver, dtype = self.solver, b.dtype

        def host(A, b):
            x = solver(np.asarray(A, np.float64), np.asarray(b, np.float64))
            return np.asarray(x, dtype=dtype)

        return jax.pure_callback(host, jax.ShapeDtypeStruct(b.shape, dtype), A, b)
