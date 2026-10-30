import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hybrid_ngd import QiskitHHLSolver  # noqa: E402
from hybrid_ngd.torch_mfac import MFAC  # noqa: E402

torch.set_default_dtype(torch.float64)


def paper_ihvp(grads, x, damping):
    """Frantar et al. (2021), eqs. (8)-(13): the dynamic algorithm's F^{-1} x with
    F = damping I + (1/m) sum_j g_j g_j^T, written as plain loops for reference."""
    G = np.asarray(grads)
    m = len(G)
    gram = G @ G.T
    # D[i, j] = g_i^T F_{i-1}^{-1} g_j for i <= j, via the recursion over the partial inverses
    D = np.zeros((m, m))
    cur = gram / damping  # g_j^T F_0^{-1} g_k
    for i in range(m):
        D[i, i:] = cur[i, i:]
        cur = cur - np.outer(cur[:, i], cur[i, :]) / (m + cur[i, i])
    B = np.zeros((m, m))
    for i in range(m):
        B[i, i] = 1 / damping
        for j in range(i):
            B[i, j] = -sum(D[k, i] / (m + D[k, k]) * B[k, j] for k in range(j, i))
    # IHVP: q_i = (g_i^T F_{i-1}^{-1} x) / (m + D_ii), eqs. (11)-(12)
    p = G @ x / damping  # g_j^T F_0^{-1} x
    q = np.zeros(m)
    for i in range(m):
        q[i] = p[i] / (m + D[i, i])
        # eq. (12), rows j > i: (g_j^T F_i^{-1} x) = (g_j^T F_{i-1}^{-1} x) - D[i, j] q_i
        p = p - (np.arange(m) > i) * D[i, :] * q[i]
    coeffs = np.array([sum(q[k] * B[k, j] for k in range(j, m)) for j in range(m)])
    return x / damping - G.T @ coeffs


def dense_ihvp(grads, x, damping):
    G = np.asarray(grads)
    F = damping * np.eye(G.shape[1]) + G.T @ G / len(G)
    return np.linalg.solve(F, x)


def feed(opt, grads):
    """Push gradients through the optimiser's buffer without moving any parameters."""
    for g in grads:
        slot = opt._push(torch.tensor(g))
    return slot


def test_paper_recursion_reference_matches_dense_inverse():
    rng = np.random.default_rng(0)
    grads, x = rng.normal(size=(5, 9)), rng.normal(size=9)
    np.testing.assert_allclose(paper_ihvp(grads, x, 0.1), dense_ihvp(grads, x, 0.1), rtol=1e-8)


@pytest.mark.parametrize("damping", [1e-1, 1e-4])
def test_current_gradient_step_matches_paper_and_dense(damping):
    rng = np.random.default_rng(1)
    grads = rng.normal(size=(6, 11))
    opt = MFAC([torch.zeros(11, requires_grad=True)], damping=damping, window=6)
    slot = feed(opt, grads)
    step = opt._precondition_current(slot).numpy()
    np.testing.assert_allclose(step, dense_ihvp(grads, grads[slot], damping), rtol=1e-6)
    np.testing.assert_allclose(step, paper_ihvp(grads, grads[slot], damping), rtol=1e-6)


def test_sliding_window_and_general_ihvp():
    rng = np.random.default_rng(2)
    grads = rng.normal(size=(10, 7))
    opt = MFAC([torch.zeros(7, requires_grad=True)], damping=0.05, window=4)
    slot = feed(opt, grads)
    window = grads[-4:]  # only the last 4 gradients remain
    np.testing.assert_allclose(opt._precondition_current(slot).numpy(), dense_ihvp(window, grads[-1], 0.05), rtol=1e-8)
    x = rng.normal(size=7)
    np.testing.assert_allclose(opt.ihvp(torch.tensor(x)).numpy(), dense_ihvp(window, x, 0.05), rtol=1e-8)
    torch.testing.assert_close(opt._gram, opt._grads @ opt._grads.T)  # incrementally maintained


def test_partial_window_uses_seen_gradients():
    rng = np.random.default_rng(3)
    grads = rng.normal(size=(3, 5))
    opt = MFAC([torch.zeros(5, requires_grad=True)], damping=0.01, window=8)
    slot = feed(opt, grads)
    np.testing.assert_allclose(opt._precondition_current(slot).numpy(), dense_ihvp(grads, grads[-1], 0.01), rtol=1e-8)


def two_moons(n=200, seed=0):
    rng = np.random.default_rng(seed)
    t, labels = rng.uniform(0, np.pi, n), rng.integers(0, 2, n)
    x = np.where(labels[:, None] == 0, np.c_[np.cos(t), np.sin(t)], np.c_[1 - np.cos(t), 0.5 - np.sin(t)])
    return torch.tensor(x + 0.1 * rng.normal(size=x.shape)), torch.tensor(labels)


def mlp(hidden=8, seed=0):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(2, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, 2))


def test_closure_and_backward_styles_agree_and_train():
    X, y = two_moons()
    models = [mlp(), mlp()]
    opts = [MFAC(m.parameters(), lr=0.05, damping=1e-3, window=16) for m in models]
    losses = []
    for _ in range(40):
        losses.append(opts[0].step(lambda: torch.nn.functional.cross_entropy(models[0](X), y)).item())
        opts[1].zero_grad()
        torch.nn.functional.cross_entropy(models[1](X), y).backward()
        opts[1].step()
    for p, q in zip(models[0].parameters(), models[1].parameters()):
        torch.testing.assert_close(p, q)
    assert losses[-1] < 0.5 * losses[0]


def test_host_solver_matches_direct():
    X, y = two_moons(80)
    models = [mlp(3), mlp(3)]
    opts = [MFAC(models[0].parameters(), lr=0.05, damping=1e-3, window=8),
            MFAC(models[1].parameters(), lr=0.05, damping=1e-3, window=8, solver="numpy")]
    for _ in range(12):
        for model, opt in zip(models, opts):
            opt.step(lambda: torch.nn.functional.cross_entropy(model(X), y))
    for p, q in zip(models[0].parameters(), models[1].parameters()):
        torch.testing.assert_close(p, q)


def test_hhl_budget_sets_window_and_trains():
    # Window Gram systems are ill-conditioned (condition number ~400 here), so HHL needs
    # ~10 clock qubits; the system register only needs log2(window) = 3.
    X, y = two_moons()
    model = mlp(16)  # 82 parameters: the HHL system is window x window, never 82 x 82
    solver = QiskitHHLSolver(num_clock_qubits=10, max_qubits=14)
    errors = []

    def checked(A, b):
        x = solver(A, b)
        errors.append(np.linalg.norm(x - np.linalg.solve(A, b)) / np.linalg.norm(np.linalg.solve(A, b)))
        return x

    checked.max_dimension = solver.max_dimension
    opt = MFAC(model.parameters(), lr=0.05, damping=1e-3, solver=checked)
    assert opt.window == 8
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        losses = [opt.step(lambda: torch.nn.functional.cross_entropy(model(X), y)).item() for _ in range(30)]
    assert losses[-1] < 0.6 * losses[0]
    assert np.median(errors) < 0.2
    assert solver.last_info.num_qubits == 3 + 10 + 1
    with pytest.raises(ValueError, match="max_dimension"):
        MFAC(model.parameters(), window=16, solver=solver)


def test_jax_matches_torch():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from hybrid_ngd.jax_mfac import MFAC as JaxMFAC

    X, y = two_moons(100)
    model = mlp(4)
    params = {"W1": jnp.asarray(model[0].weight.detach().numpy()), "b1": jnp.asarray(model[0].bias.detach().numpy()),
              "W2": jnp.asarray(model[2].weight.detach().numpy()), "b2": jnp.asarray(model[2].bias.detach().numpy())}

    def loss_fn(p, X, y):
        logits = jnp.tanh(X @ p["W1"].T + p["b1"]) @ p["W2"].T + p["b2"]
        return jax.nn.logsumexp(logits, axis=1) - jnp.take_along_axis(logits, y[:, None], axis=1)[:, 0]

    Xj, yj = jnp.asarray(X.numpy()), jnp.asarray(y.numpy())
    for solver in ("direct", "numpy"):
        torch_model = mlp(4)
        torch_opt = MFAC(torch_model.parameters(), lr=0.05, damping=1e-3, window=5, solver=solver)
        opt = JaxMFAC(loss_fn, lr=0.05, damping=1e-3, window=5, solver=solver)
        state, p = opt.init(params), params
        update = jax.jit(opt.update)
        for _ in range(8):  # past the window size, so the ring buffer wraps
            p, state, _ = update(p, state, Xj, yj)
            torch_opt.step(lambda: torch.nn.functional.cross_entropy(torch_model(X), y))
        np.testing.assert_allclose(np.asarray(p["W1"]), torch_model[0].weight.detach().numpy(), rtol=1e-8)
        np.testing.assert_allclose(np.asarray(p["b2"]), torch_model[2].bias.detach().numpy(), rtol=1e-8)


def test_microbatch_with_one_sample_per_batch_is_ngd():
    from hybrid_ngd.torch_ngd import NGD

    X, y = two_moons(60)
    models = [mlp(3), mlp(3)]
    mfac = MFAC(models[0].parameters(), lr=0.5, damping=1e-2, window=60, gradients="microbatch")
    ngd = NGD(models[1].parameters(), lr=0.5, damping=1e-2, metric="fisher")
    for _ in range(5):
        mfac.step(lambda: torch.nn.functional.cross_entropy(models[0](X), y, reduction="none"))
        ngd.step(lambda: torch.nn.functional.cross_entropy(models[1](X), y, reduction="none"))
    for p, q in zip(models[0].parameters(), models[1].parameters()):
        torch.testing.assert_close(p, q)


def test_microbatch_direction_matches_dense_formula():
    X, y = two_moons(50)  # 50 samples in 8 micro-batches: sizes 6 and 7
    model = mlp(4)
    opt = MFAC(model.parameters(), lr=1.0, damping=0.03, window=8, gradients="microbatch")
    params = list(model.parameters())
    before = torch.cat([p.detach().reshape(-1).clone() for p in params])
    losses = torch.nn.functional.cross_entropy(model(X), y, reduction="none")
    groups = torch.arange(50) * 8 // 50
    G = torch.stack([torch.cat([g.reshape(-1) for g in torch.autograd.grad(losses[groups == j].mean(), params,
                                                                           retain_graph=True)]) for j in range(8)])
    full = torch.cat([g.reshape(-1) for g in torch.autograd.grad(losses.mean(), params)])
    sizes = torch.bincount(groups).double()
    deviations = G - full
    F = 0.03 * torch.eye(G.shape[1]) + torch.outer(full, full) + (deviations.T * sizes) @ deviations / 8
    opt.step(lambda: torch.nn.functional.cross_entropy(model(X), y, reduction="none"))
    after = torch.cat([p.detach().reshape(-1) for p in params])
    torch.testing.assert_close(before - after, torch.linalg.solve(F, full))
    np.testing.assert_allclose(opt.ihvp(full).numpy(), torch.linalg.solve(F, full).numpy(), rtol=1e-8)


def test_microbatch_estimate_does_not_shrink_with_batch_size():
    # With the n_j correction, the estimate of the per-sample spread stays about the same
    # as micro-batches grow; without it, the curvature would shrink roughly as 1 / n_j.
    X, y = two_moons(400)
    model = mlp(4)
    traces = []
    for m in (400, 40, 10):
        opt = MFAC(model.parameters(), lr=1e-9, damping=1e-2, window=m, gradients="microbatch")
        opt.step(lambda: torch.nn.functional.cross_entropy(model(X), y, reduction="none"))
        traces.append(torch.trace(opt._micro[1]).item())  # tr(R R^T) = tr(R^T R)
    assert max(traces) / min(traces) < 2


def test_microbatch_errors_and_jax_match():
    X, y = two_moons(40)
    model = mlp(3)
    opt = MFAC(model.parameters(), window=8, gradients="microbatch")
    with pytest.raises(ValueError, match="closure"):
        opt.step()
    with pytest.raises(ValueError, match="fewer than window"):
        opt.step(lambda: torch.nn.functional.cross_entropy(model(X[:5]), y[:5], reduction="none"))

    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from hybrid_ngd.jax_mfac import MFAC as JaxMFAC

    torch_model = mlp(3)
    params = {"W1": jnp.asarray(torch_model[0].weight.detach().numpy()), "b1": jnp.asarray(torch_model[0].bias.detach().numpy()),
              "W2": jnp.asarray(torch_model[2].weight.detach().numpy()), "b2": jnp.asarray(torch_model[2].bias.detach().numpy())}

    def loss_fn(p, X, y):
        logits = jnp.tanh(X @ p["W1"].T + p["b1"]) @ p["W2"].T + p["b2"]
        return jax.nn.logsumexp(logits, axis=1) - jnp.take_along_axis(logits, y[:, None], axis=1)[:, 0]

    torch_opt = MFAC(torch_model.parameters(), lr=0.3, damping=1e-2, window=6, gradients="microbatch")
    jax_opt = JaxMFAC(loss_fn, lr=0.3, damping=1e-2, window=6, gradients="microbatch")
    state, p = jax_opt.init(params), params
    update = jax.jit(jax_opt.update)
    Xj, yj = jnp.asarray(X.numpy()), jnp.asarray(y.numpy())
    for _ in range(5):
        p, state, _ = update(p, state, Xj, yj)
        torch_opt.step(lambda: torch.nn.functional.cross_entropy(torch_model(X), y, reduction="none"))
    np.testing.assert_allclose(np.asarray(p["W1"]), torch_model[0].weight.detach().numpy(), rtol=1e-8)
