import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from hybrid_ngd import QiskitHHLSolver  # noqa: E402
from hybrid_ngd.jax_ngd import NGD  # noqa: E402


def regression_data(seed=0, n=64, d=4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    return jnp.asarray(X), jnp.asarray(X @ np.arange(1.0, d + 1) + 0.1 * rng.normal(size=n))


def residuals(params, X, y):
    return X @ params["w"] + params["b"] - y


def test_hessian_step_solves_quadratic_on_pytree():
    rng = np.random.default_rng(0)
    M = rng.normal(size=(5, 5))
    A = jnp.asarray(M @ M.T + 5 * np.eye(5))
    c = jnp.asarray(rng.normal(size=5))

    def loss(p):
        x = jnp.concatenate([p["a"], p["b"]])
        return 0.5 * x @ A @ x - c @ x

    opt = NGD(loss, lr=1.0, damping=0.0, metric="hessian")
    params = {"a": jnp.zeros(2), "b": jnp.zeros(3)}
    params, state, _ = opt.update(params, opt.init(params))
    np.testing.assert_allclose(jnp.concatenate([params["a"], params["b"]]), jnp.linalg.solve(A, c))
    assert state.count == 1


def test_gauss_newton_with_qiskit_under_jit():
    X, y = regression_data()
    opt = NGD(residuals, lr=1.0, damping=1e-3, metric="gauss_newton", solver=QiskitHHLSolver(num_clock_qubits=8))
    params = {"w": jnp.zeros(4), "b": jnp.zeros(())}
    state = opt.init(params)
    step = jax.jit(opt.update)
    for _ in range(5):
        params, state, loss = step(params, state, X, y)

    design = jnp.column_stack([X, jnp.ones(len(y))])
    target = jnp.linalg.lstsq(design, y)[0]
    got = jnp.append(params["w"], params["b"])
    assert jnp.linalg.norm(got - target) / jnp.linalg.norm(target) < 1e-2


def test_fisher_with_lr_schedule():
    X, y = regression_data()
    labels = (y > 0).astype(float)

    def per_sample_bce(w, X, labels):
        logits = X @ w
        return jnp.logaddexp(0.0, logits) - labels * logits

    opt = NGD(per_sample_bce, lr=lambda count: 0.5 / (1 + count), damping=1e-2, metric="fisher")
    w, state = jnp.zeros(4), opt.init(jnp.zeros(4))
    losses = []
    for _ in range(10):
        w, state, loss = opt.update(w, state, X, labels)
        losses.append(float(loss))
    assert losses[-1] < 0.6 * losses[0]


def test_matches_torch_implementation():
    torch = pytest.importorskip("torch")
    from hybrid_ngd.torch_ngd import NGD as TorchNGD

    X, y = regression_data()
    labels = (y > 0).astype(float)

    def per_sample_bce(w, X, labels):
        logits = X @ w
        return jnp.logaddexp(0.0, logits) - labels * logits

    opt = NGD(per_sample_bce, lr=0.5, damping=1e-2, metric="fisher")
    w, state = jnp.zeros(4), opt.init(jnp.zeros(4))
    Xt, lt = torch.tensor(np.asarray(X)), torch.tensor(np.asarray(labels))
    wt = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    topt = TorchNGD([wt], lr=0.5, damping=1e-2, metric="fisher")
    for _ in range(3):
        w, state, _ = opt.update(w, state, X, labels)
        topt.step(lambda: torch.nn.functional.binary_cross_entropy_with_logits(Xt @ wt, lt, reduction="none"))
    np.testing.assert_allclose(np.asarray(w), wt.detach().numpy(), rtol=1e-10)
