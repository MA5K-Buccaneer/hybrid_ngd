"""Natural gradient descent as a ``torch.optim.Optimizer``."""

from __future__ import annotations

import numpy as np
import torch

from ._common import check_args, resolve_solver


class NGD(torch.optim.Optimizer):
    r"""Natural gradient descent: ``theta <- theta - lr * (G + damping * I)^{-1} grad L``.

    ``step`` requires a closure. What it returns depends on ``metric``:

    * ``"fisher"``: per-sample losses ``l_i`` (shape ``[N]``). ``L = mean(l_i)``
      and ``G`` is the empirical Fisher ``mean(grad l_i grad l_i^T)``.
    * ``"gauss_newton"``: residuals ``r_i`` (shape ``[N]``). ``L = mean(r_i^2) / 2``
      and ``G = mean(grad r_i grad r_i^T)``, the Fisher of a Gaussian likelihood.
    * ``"hessian"``: a scalar loss (per-sample losses are averaged), with
      ``G = hess L``. This gives a damped Newton step.

    The closure should only run the forward pass, without calling ``backward``.
    The metric couples every parameter, so all groups are solved as one vector;
    each group's ``lr`` is still applied to its own parameters. ``G`` is dense
    ``n x n``, which suits models with up to a few thousand parameters, and far
    fewer when solving with :class:`~hybrid_ngd.QiskitHHLSolver`.

    Args:
        params: Parameters or parameter groups.
        lr: Step size.
        damping: Tikhonov damping added to the diagonal of ``G``.
        metric: ``"fisher"``, ``"gauss_newton"`` or ``"hessian"``.
        solver: ``"direct"`` uses ``torch.linalg.solve`` on the parameters'
            device. ``"qiskit"`` uses a default :class:`~hybrid_ngd.QiskitHHLSolver`.
            Any callable ``(A, b) -> x`` on NumPy arrays also works.
    """

    def __init__(self, params, lr=0.1, damping=1e-3, metric="fisher", solver="direct"):
        check_args(lr, damping, metric)
        super().__init__(params, dict(lr=lr))
        self.damping = damping
        self.metric = metric
        self.solver = resolve_solver(solver)

    @torch.no_grad()
    def step(self, closure):
        params = [p for group in self.param_groups for p in group["params"] if p.requires_grad]
        with torch.enable_grad():
            loss, grad, G = self._grad_and_metric(closure(), params)
        G.diagonal().add_(self.damping)
        direction = self._solve(G, grad)

        offset = 0
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                p.add_(direction[offset : offset + p.numel()].view_as(p), alpha=-group["lr"])
                offset += p.numel()
        return loss.detach()

    def _grad_and_metric(self, out, params):
        if self.metric == "hessian":
            loss = out.mean()
            grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
            grad = _flatten(grads, params)
            if grad.requires_grad:
                hess = _jacobian(grad, params)
            else:  # loss is linear in params
                hess = grad.new_zeros(grad.numel(), grad.numel())
            return loss, grad.detach(), hess

        values = out.reshape(-1)
        jac = _jacobian(values, params)  # [N, n]
        if self.metric == "fisher":
            loss = values.mean()
            grad = jac.mean(0)
        else:
            loss = 0.5 * values.pow(2).mean()
            grad = jac.T @ values.detach() / values.numel()
        return loss, grad, jac.T @ jac / values.numel()

    def _solve(self, A, b):
        if self.solver is None:
            return torch.linalg.solve(A, b)
        x = self.solver(A.cpu().double().numpy(), b.cpu().double().numpy())
        return torch.as_tensor(np.asarray(x), dtype=b.dtype, device=b.device)


def _flatten(tensors, params):
    return torch.cat(
        [(t if t is not None else torch.zeros_like(p)).reshape(-1) for t, p in zip(tensors, params)]
    )


def _jacobian(outputs, params):
    """Rows ``d outputs[i] / d params``, flattened to shape ``[len(outputs), n]``."""
    m = outputs.numel()
    if sum(p.numel() for p in params) < m:
        try:
            return _jacobian_by_columns(outputs, params)
        except RuntimeError:  # an op that is not twice differentiable; use the row method
            pass
    try:
        eye = torch.eye(m, dtype=outputs.dtype, device=outputs.device)
        rows = torch.autograd.grad(
            outputs, params, grad_outputs=eye, retain_graph=True, allow_unused=True, is_grads_batched=True
        )
        return torch.cat(
            [(r if r is not None else p.new_zeros(m, *p.shape)).reshape(m, -1) for r, p in zip(rows, params)],
            dim=1,
        )
    except RuntimeError:  # an op without a vmap batching rule; fall back to one backward per row
        pass
    return torch.stack(
        [
            _flatten(torch.autograd.grad(outputs[i], params, retain_graph=True, allow_unused=True), params)
            for i in range(m)
        ]
    )


def _jacobian_by_columns(outputs, params):
    """``J`` from one Jacobian-vector product per parameter, using the
    double-backward trick ``J v = d/du (J^T u) . v``. Cheaper than one backward
    per output when there are fewer parameters than outputs."""
    u = torch.zeros_like(outputs, requires_grad=True)
    vjp = _flatten(torch.autograd.grad(outputs, params, grad_outputs=u, create_graph=True, allow_unused=True), params)
    if not vjp.requires_grad:  # outputs do not depend on params
        return vjp.new_zeros(outputs.numel(), vjp.numel())
    eye = torch.eye(vjp.numel(), dtype=vjp.dtype, device=vjp.device)
    (jac_t,) = torch.autograd.grad(vjp, u, grad_outputs=eye, retain_graph=True, is_grads_batched=True)
    return jac_t.T
