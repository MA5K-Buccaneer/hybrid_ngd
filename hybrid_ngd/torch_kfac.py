"""K-FAC: natural gradient descent with a Kronecker-factored metric, as a ``torch.optim.Optimizer``."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ._common import check_args, resolve_solver
from .torch_ngd import _jacobian


@dataclass
class Block:
    """One independently solved part of the metric.

    ``"dense"`` blocks use the exact metric restricted to ``params``.
    ``"kronecker"`` blocks use ``A ⊗ G`` for one ``nn.Linear`` layer, with each
    factor split into diagonal chunks of at most ``max_dimension``.
    """

    kind: str
    label: str
    params: list
    layer: nn.Linear | None = None
    systems: tuple[int, ...] = ()  # sizes of the linear systems this block solves


class KFAC(torch.optim.Optimizer):
    r"""Kronecker-factored approximate curvature (Martens & Grosse, 2015), sized to the solver.

    The metric of each ``nn.Linear`` layer is approximated by a Kronecker product
    ``A ⊗ G``, where ``A = E[a a^T]`` is the covariance of the layer's inputs (with
    a column of ones for the bias) and ``G`` is the covariance of the gradients
    at its outputs. Blocks for different layers are taken as independent. The
    update for a layer's ``[W, b]`` is then

        ``(G + gamma_G I)^{-1} grad_[W, b] (A + gamma_A I)^{-1}``,

    so only matrices of size ``in_features + 1`` and ``out_features`` are inverted.

    **Sizing to the solver.** If the solver has a ``max_dimension`` (such as
    :class:`~hybrid_ngd.QiskitHHLSolver` with ``max_qubits``) or one is passed
    here, no linear system larger than that is ever solved. With
    ``structure="auto"``, the optimiser picks the most exact structure that fits:

    1. the dense metric over all parameters, exactly as :class:`~hybrid_ngd.torch_ngd.NGD`;
    2. otherwise, per layer, the dense block of that layer's parameters;
    3. otherwise, a Kronecker ``A ⊗ G`` block for that layer;
    4. a factor still too large is split into diagonal chunks that fit.

    ``structure="kfac"`` always uses step 3, with the chunking of step 4 when
    needed. :attr:`plan` lists the chosen blocks, and :meth:`describe` prints them.

    ``step`` takes the same closure as ``NGD``. It runs only the forward pass and
    returns:

    * ``"fisher"``: per-sample losses ``l_i``. ``L = mean(l_i)``, and the metric is
      the empirical Fisher.
    * ``"gauss_newton"``: residuals ``r_i``. ``L = mean(r_i^2) / 2``. Kronecker
      ``G`` factors are estimated by backpropagating ``v^T r`` with random signs
      ``v``, so that ``E[g g^T]`` equals the Gauss-Newton factor; dense blocks are
      exact.

    Damping: dense blocks solve ``(F + damping I) d = g``. Kronecker blocks use
    factored Tikhonov damping, ``gamma_A = pi sqrt(damping)`` and
    ``gamma_G = sqrt(damping) / pi`` with ``pi = sqrt((tr A / dim A) / (tr G / dim G))``.
    All curvature (dense blocks and factors) is an exponential moving average
    over steps with decay ``stat_decay``; ``stat_decay=0`` uses only the current batch.

    Limitations: only ``nn.Linear`` layers get per-layer blocks, each may be
    applied once per forward pass, and inputs are expected as
    ``(batch, features)``. Extra leading dimensions are flattened into the batch,
    which is only approximate for layers with shared weights. There is no
    momentum or adaptive damping.

    Args:
        model: The ``nn.Module`` to optimise.
        lr: Step size.
        damping: Tikhonov damping ``lambda``.
        metric: ``"fisher"`` or ``"gauss_newton"``.
        solver: ``"direct"`` uses ``torch.linalg.solve``. A host solver such as
            :class:`~hybrid_ngd.QiskitHHLSolver`, or any callable ``(A, b) -> x``, is
            called once per right-hand side.
        stat_decay: Decay of the curvature moving averages.
        other_params: Trainable parameters outside ``nn.Linear`` layers raise an
            error by default. ``"sgd"`` gives them plain gradient steps, unless
            ``structure="auto"`` covers them with the dense full metric.
        structure: ``"kfac"`` or ``"auto"`` (see above).
        max_dimension: Largest linear system to solve. Defaults to the solver's
            ``max_dimension`` attribute; ``None`` means unlimited.
    """

    def __init__(self, model: nn.Module, lr=0.1, damping=1e-3, metric="fisher", solver="direct", stat_decay=0.95,
                 other_params="error", structure="kfac", max_dimension=None):
        check_args(lr, damping, metric)
        if metric == "hessian":
            raise ValueError("K-FAC approximates the Fisher or Gauss-Newton matrix; metric='hessian' is not supported")
        if not 0 <= stat_decay < 1:
            raise ValueError(f"stat_decay must be in [0, 1), got {stat_decay}")
        if other_params not in ("error", "sgd"):
            raise ValueError(f"other_params must be 'error' or 'sgd', got {other_params!r}")
        if structure not in ("kfac", "auto"):
            raise ValueError(f"structure must be 'kfac' or 'auto', got {structure!r}")

        self.layers = [m for m in model.modules() if isinstance(m, nn.Linear) and m.weight.requires_grad]
        covered = {id(p) for layer in self.layers for p in layer.parameters()}
        others = [p for p in model.parameters() if p.requires_grad and id(p) not in covered]
        if others and other_params == "error":
            raise ValueError(
                f"K-FAC preconditions nn.Linear layers only, but the model has {len(others)} other trainable "
                "parameter tensors. Pass other_params='sgd' to update them with plain gradient steps."
            )
        layer_params = [p for layer in self.layers for p in layer.parameters() if p.requires_grad]
        super().__init__(layer_params + others, dict(lr=lr))
        self.damping = damping
        self.metric = metric
        self.solver = resolve_solver(solver)
        self.stat_decay = stat_decay
        self.structure = structure
        self.max_dimension = max_dimension if max_dimension is not None else getattr(self.solver, "max_dimension", None)
        if self.max_dimension is not None and self.max_dimension < 1:
            raise ValueError(f"max_dimension must be positive, got {self.max_dimension}")

        self.plan = self._make_plan(model, layer_params + others)
        planned = {id(p) for block in self.plan for p in block.params}
        self.other_params = [p for p in others if id(p) not in planned]  # plain gradient steps
        self.factors: dict[nn.Linear, tuple[torch.Tensor, torch.Tensor]] = {}  # Kronecker blocks
        self.dense_metrics: dict[str, torch.Tensor] = {}  # dense blocks, by label
        self._capturing = False
        self._inputs: dict[nn.Linear, torch.Tensor] = {}
        self._outputs: dict[nn.Linear, torch.Tensor] = {}
        hooked = [block.layer for block in self.plan if block.kind == "kronecker"]
        self._hooks = [layer.register_forward_hook(self._capture) for layer in hooked]

    # ---- planning -----------------------------------------------------------------

    def _fits(self, size: int) -> bool:
        return self.max_dimension is None or size <= self.max_dimension

    def _chunks(self, size: int) -> list[int]:
        if self._fits(size):
            return [size]
        d = self.max_dimension
        return [d] * (size // d) + ([size % d] if size % d else [])

    def _make_plan(self, model, all_params) -> list[Block]:
        n_total = sum(p.numel() for p in all_params)
        if self.structure == "auto" and self._fits(n_total):
            return [Block("dense", "all parameters", list(all_params), systems=(n_total,))]
        names = {m: name or type(m).__name__ for name, m in model.named_modules()}
        plan = []
        for layer in self.layers:
            params = [p for p in layer.parameters() if p.requires_grad]
            n_layer = sum(p.numel() for p in params)
            label = f"{names[layer]} ({layer.in_features}->{layer.out_features})"
            if self.structure == "auto" and self._fits(n_layer):
                plan.append(Block("dense", label, params, layer, systems=(n_layer,)))
            else:
                size_a = layer.in_features + (layer.bias is not None)
                plan.append(Block("kronecker", label, params, layer,
                                  systems=tuple(self._chunks(size_a)) + tuple(self._chunks(layer.out_features))))
        return plan

    def describe(self) -> str:
        """One line per block: its structure, the systems it solves and, for HHL, their qubit counts."""
        qubits = getattr(self.solver, "num_qubits_for", None)
        limit = "unlimited" if self.max_dimension is None else str(self.max_dimension)
        lines = [f"max system size: {limit}"]
        for block in self.plan:
            kind = {"dense": "dense block", "kronecker": "Kronecker A⊗G"}[block.kind]
            if block.kind == "kronecker" and len(block.systems) > 2:
                kind += " (chunked)"
            sizes = ", ".join(str(s) for s in sorted(set(block.systems), reverse=True))
            line = f"{block.label:<28}{kind:<26}systems of size {sizes}"
            if qubits is not None:
                line += f"  -> up to {max(qubits(s) for s in block.systems)} qubits"
            lines.append(line)
        if self.other_params:
            lines.append(f"{len(self.other_params)} other parameter tensors: plain gradient steps")
        return "\n".join(lines)

    # ---- hooks ----------------------------------------------------------------------

    def _capture(self, layer, inputs, output):
        if not self._capturing:
            return
        if layer in self._outputs:
            raise RuntimeError("K-FAC does not support an nn.Linear layer applied more than once per forward pass")
        self._inputs[layer] = inputs[0].detach()
        self._outputs[layer] = output

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    # ---- step -----------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure):
        self._inputs, self._outputs = {}, {}
        self._capturing = True
        try:
            with torch.enable_grad():
                values = closure().reshape(-1)
        finally:
            self._capturing = False
        params = [p for group in self.param_groups for p in group["params"]]
        kron_blocks = [b for b in self.plan if b.kind == "kronecker" and b.layer in self._outputs]
        dense_blocks = [b for b in self.plan if b.kind == "dense"]

        with torch.enable_grad():
            if self.metric == "fisher":
                loss = values.mean()
                stats_target = values.sum()  # row i of each output gradient is d l_i / d s_i
            else:
                loss = 0.5 * values.pow(2).mean()
                signs = torch.randint(0, 2, values.shape, device=values.device).to(values.dtype) * 2 - 1
                stats_target = (signs * values).sum()  # E[g g^T] over the signs is the Gauss-Newton factor
            output_grads = []
            if kron_blocks:
                output_grads = torch.autograd.grad(stats_target, [self._outputs[b.layer] for b in kron_blocks],
                                                   retain_graph=True)
            jacobians = [_jacobian(values, b.params) for b in dense_blocks]
            grads = dict(zip(params, torch.autograd.grad(loss, params, allow_unused=True)))

        def grad_of(p):
            return grads[p] if grads[p] is not None else torch.zeros_like(p)

        lr_of = {p: group["lr"] for group in self.param_groups for p in group["params"]}

        for block, jac in zip(dense_blocks, jacobians):
            metric = self._average(self.dense_metrics, block.label, jac.T @ jac / values.numel())
            g = torch.cat([grad_of(p).reshape(-1) for p in block.params])
            damped = metric + self.damping * torch.eye(len(g), dtype=g.dtype, device=g.device)
            direction = self._solve(damped, g[:, None])[:, 0]
            offset = 0
            for p in block.params:
                p.add_(direction[offset : offset + p.numel()].view_as(p), alpha=-lr_of[p])
                offset += p.numel()

        for block, g_out in zip(kron_blocks, output_grads):
            layer = block.layer
            A, G = self._kronecker_factors(layer, self._inputs[layer], g_out, values.numel())
            gw = grad_of(layer.weight)
            if layer.bias is not None:
                gw = torch.cat([gw, grad_of(layer.bias)[:, None]], dim=1)
            direction = self._precondition(A, G, gw)
            layer.weight.add_(direction[:, : layer.in_features], alpha=-lr_of[layer.weight])
            if layer.bias is not None:
                layer.bias.add_(direction[:, layer.in_features], alpha=-lr_of[layer.bias])

        for p in self.other_params:
            p.add_(grad_of(p), alpha=-lr_of[p])

        self._inputs, self._outputs = {}, {}
        return loss.detach()

    # ---- curvature ------------------------------------------------------------------

    def _average(self, store, key, new):
        if key in store and self.stat_decay:
            new = self.stat_decay * store[key] + (1 - self.stat_decay) * new
        store[key] = new
        return new

    def _kronecker_factors(self, layer, a, g, num_values):
        a = a.reshape(-1, layer.in_features)
        if layer.bias is not None:
            a = torch.cat([a, a.new_ones(len(a), 1)], dim=1)
        g = g.reshape(-1, layer.out_features)
        A, G = a.T @ a / len(a), g.T @ g / num_values
        if layer in self.factors and self.stat_decay:
            A0, G0 = self.factors[layer]
            A = self.stat_decay * A0 + (1 - self.stat_decay) * A
            G = self.stat_decay * G0 + (1 - self.stat_decay) * G
        self.factors[layer] = (A, G)
        return A, G

    def _precondition(self, A, G, V):
        """``(G + gamma_G I)^{-1} V (A + gamma_A I)^{-1}`` with factored Tikhonov damping."""
        mean_a = torch.trace(A) / A.shape[0]
        mean_g = torch.trace(G) / G.shape[0]
        pi = torch.sqrt(mean_a / mean_g) if mean_a > 0 and mean_g > 0 else torch.ones((), dtype=A.dtype)
        root = math.sqrt(self.damping)
        A_damped = A + float(pi * root) * torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
        G_damped = G + float(root / pi) * torch.eye(G.shape[0], dtype=G.dtype, device=G.device)
        left = self._solve(G_damped, V)
        return self._solve(A_damped, left.T).T  # A is symmetric

    def _solve(self, M, B):
        """Solve ``M X = B`` (``B`` has one right-hand side per column). Matrices
        larger than ``max_dimension`` are replaced by their block diagonal, with
        blocks of at most that size."""
        size = M.shape[0]
        bounds = np.cumsum([0, *self._chunks(size)])
        X = torch.empty_like(B)
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            block, rhs = M[lo:hi, lo:hi], B[lo:hi]
            if self.solver is None:
                X[lo:hi] = torch.linalg.solve(block, rhs)
            else:
                Mn, Bn = block.cpu().double().numpy(), rhs.cpu().double().numpy()
                cols = [np.asarray(self.solver(Mn, Bn[:, j])) for j in range(Bn.shape[1])]
                X[lo:hi] = torch.as_tensor(np.stack(cols, axis=1), dtype=B.dtype, device=B.device)
        return X
