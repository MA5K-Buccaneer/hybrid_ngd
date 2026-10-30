"""M-FAC: matrix-free natural gradient over a sliding window of gradients, as a ``torch.optim.Optimizer``."""

from __future__ import annotations

import numpy as np
import torch

from ._common import check_args, resolve_solver
from .torch_ngd import _flatten, _jacobian


class MFAC(torch.optim.Optimizer):
    r"""Matrix-free approximate curvature (Frantar, Kurtic & Alistarh, NeurIPS 2021).

    Preconditions each gradient with the empirical Fisher of the last ``window``
    gradients,

        ``F = damping I + (1/k) sum_j g_j g_j^T``,  ``theta <- theta - lr F^{-1} g_t``,

    where the ``g_j`` are minibatch gradients from the most recent steps, the
    current one included, and ``k = min(steps taken, window)``. (The paper scales by
    ``1/window`` once the window is full; before that, only the gradients seen so
    far are used.)

    No ``d x d`` matrix is ever formed. ``F^{-1} g_t`` reduces to one ``k x k``
    linear system in the gradients' inner products. Because ``g_t`` is itself a row
    of the window ``G``, the push-through identity gives

        ``F^{-1} g_t = k G^T (k damping I + G G^T)^{-1} e_t``,

    which costs ``O(d k + k^3)`` per step, like the paper's dynamic algorithm, and
    avoids the ``1 / damping`` cancellation of the general Woodbury form. That
    ``k x k`` system is what ``solver`` solves, so an HHL circuit needs only
    ``ceil(log2 window)`` system qubits, whatever the model size.

    Use it like any torch optimiser: call ``loss.backward()`` then ``step()``, or
    pass a closure that returns the loss (per-sample losses are averaged) and lets
    the optimiser differentiate it. There is no momentum, as in the paper's
    experiments.

    ``gradients="microbatch"`` builds the curvature from the *current* step instead.
    The closure must return per-sample losses. They are split into ``window``
    micro-batches of sizes ``n_j``, with mean gradients ``g_j`` at the current
    parameters, and the per-sample empirical Fisher is estimated as

        ``F = damping I + gbar gbar^T + (1/m) sum_j n_j (g_j - gbar)(g_j - gbar)^T``,

    where ``gbar`` is the full-batch gradient. The ``n_j`` factor undoes the
    ``1/n_j`` shrinkage of a micro-batch mean's spread, so the estimate does not
    lose curvature as micro-batches grow. With rows ``R = [gbar; sqrt(n_j/m)(g_j - gbar)]``,
    ``F = damping I + R^T R`` and, since ``gbar`` is row 0,

        ``F^{-1} gbar = R^T (damping I + R R^T)^{-1} e_0``,

    an ``(m+1) x (m+1)`` solve with a basis-vector right-hand side. This sees the
    per-sample spread of gradients that a window of past mean gradients misses.
    With one sample per micro-batch it is exactly the damped empirical Fisher of
    :class:`~hybrid_ngd.torch_ngd.NGD`. It needs ``m`` gradient rows per step (or
    ``d`` Jacobian-vector products, whichever is fewer), not one gradient.

    Args:
        params: Parameters or parameter groups. They are preconditioned jointly;
            each group's ``lr`` is applied to its own parameters.
        lr: Step size.
        damping: ``lambda`` in ``F = lambda I + ...``.
        window: Number of gradients ``m`` in the sliding window, or of micro-batches.
            Defaults to the largest that fits the solver's ``max_dimension`` (for
            example ``QiskitHHLSolver(max_qubits=...)``): ``max_dimension``, or
            ``max_dimension - 1`` for ``"microbatch"``. Otherwise 32.
        solver: ``"direct"`` uses ``torch.linalg.solve`` on the ``k x k`` system. A
            host solver such as :class:`~hybrid_ngd.QiskitHHLSolver`, or any
            callable ``(A, b) -> x``, also works.
        gradients: ``"history"`` (the paper's sliding window of past gradients) or
            ``"microbatch"`` (micro-batch gradients of the current step; see above).
    """

    def __init__(self, params, lr=1e-3, damping=1e-4, window=None, solver="direct", gradients="history"):
        check_args(lr, damping, "fisher")
        if damping <= 0:
            raise ValueError("M-FAC needs damping > 0")
        if gradients not in ("history", "microbatch"):
            raise ValueError(f"gradients must be 'history' or 'microbatch', got {gradients!r}")
        self.solver = resolve_solver(solver)
        limit = getattr(self.solver, "max_dimension", None)
        extra = 1 if gradients == "microbatch" else 0  # the microbatch system has one row more than the window
        if window is None:
            window = limit - extra if limit else 32
        if window < 1:
            raise ValueError(f"window must be positive, got {window}")
        if limit is not None and window + extra > limit:
            raise ValueError(f"window={window} needs a {window + extra}x{window + extra} system, more than the "
                             f"solver's max_dimension={limit}")
        super().__init__(params, dict(lr=lr))
        self.damping = damping
        self.window = window
        self.gradients = gradients
        self.count = 0  # gradients seen so far
        self._micro: tuple[torch.Tensor, torch.Tensor] | None = None  # (R, R R^T) of the last microbatch step
        self._grads: torch.Tensor | None = None  # [window, d] ring buffer
        self._gram: torch.Tensor | None = None  # [window, window] inner products of the buffer

    def _params(self):
        return [p for group in self.param_groups for p in group["params"] if p.requires_grad]

    @torch.no_grad()
    def step(self, closure=None):
        params = self._params()
        loss = None
        if self.gradients == "microbatch":
            if closure is None:
                raise ValueError("gradients='microbatch' needs a closure that returns per-sample losses")
            loss, direction = self._microbatch_direction(closure, params)
            self._apply(direction)
            return loss
        if closure is not None:
            with torch.enable_grad():
                loss = closure().mean()
                grads = torch.autograd.grad(loss, params, allow_unused=True)
        else:
            grads = [p.grad for p in params]
        g = _flatten(grads, params)
        slot = self._push(g)
        self._apply(self._precondition_current(slot))
        return None if loss is None else loss.detach()

    def _apply(self, direction):
        offset = 0
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                p.add_(direction[offset : offset + p.numel()].view_as(p), alpha=-group["lr"])
                offset += p.numel()

    def _microbatch_direction(self, closure, params):
        m = self.window
        with torch.enable_grad():
            values = closure().reshape(-1)
            n = values.numel()
            if n < m:
                raise ValueError(f"the closure returned {n} per-sample losses, fewer than window={m}")
            groups = torch.arange(n, device=values.device) * m // n  # contiguous, sizes differ by at most 1
            counts = torch.bincount(groups, minlength=m).to(values.dtype)
            means = torch.zeros(m, dtype=values.dtype, device=values.device).index_add(0, groups, values) / counts
            G = _jacobian(means, params).detach()  # [m, d] micro-batch mean gradients
        gbar = (counts / n) @ G  # the full-batch gradient
        R = torch.cat([gbar[None], (counts / m).sqrt()[:, None] * (G - gbar)])
        gram = R @ R.T
        self._micro = (R, gram)  # so that ihvp() works in this mode too
        system = gram + self.damping * torch.eye(m + 1, dtype=gram.dtype, device=gram.device)
        unit = torch.zeros(m + 1, dtype=gram.dtype, device=gram.device)
        unit[0] = 1.0
        return values.mean().detach(), R.T @ self._solve(system, unit)

    def _push(self, g):
        """Store ``g`` in the ring buffer, update the inner products, return its row."""
        if self._grads is None:
            self._grads = g.new_zeros(self.window, g.numel())
            self._gram = g.new_zeros(self.window, self.window)
        slot = self.count % self.window
        self._grads[slot] = g
        dots = self._grads @ g
        self._gram[slot, :] = dots
        self._gram[:, slot] = dots
        self.count += 1
        return slot

    def _stored(self):
        k = min(self.count, self.window)
        return k, self._grads[:k], self._gram[:k, :k]

    def _precondition_current(self, slot):
        k, G, gram = self._stored()
        system = gram + k * self.damping * torch.eye(k, dtype=gram.dtype, device=gram.device)
        unit = torch.zeros(k, dtype=gram.dtype, device=gram.device)
        unit[slot] = 1.0
        return k * (G.T @ self._solve(system, unit))

    def ihvp(self, x):
        """``F^{-1} x`` for any flat vector ``x``, by the Woodbury identity
        ``F^{-1} x = (x - G^T (k damping I + G G^T)^{-1} G x) / damping``
        (with ``R`` in place of ``G`` and ``damping I`` in place of ``k damping I``
        after a microbatch step)."""
        if self.gradients == "microbatch" and self._micro is not None:
            R, gram = self._micro
            system = gram + self.damping * torch.eye(len(gram), dtype=gram.dtype, device=gram.device)
            return (x - R.T @ self._solve(system, R @ x)) / self.damping
        k, G, gram = self._stored()
        if k == 0:
            return x / self.damping
        system = gram + k * self.damping * torch.eye(k, dtype=gram.dtype, device=gram.device)
        return (x - G.T @ self._solve(system, G @ x)) / self.damping

    def _solve(self, A, b):
        if self.solver is None:
            return torch.linalg.solve(A, b)
        x = self.solver(A.cpu().double().numpy(), b.cpu().double().numpy())
        return torch.as_tensor(np.asarray(x), dtype=b.dtype, device=b.device)
