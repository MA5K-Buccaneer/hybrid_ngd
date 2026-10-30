import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hybrid_ngd import QiskitHHLSolver  # noqa: E402
from hybrid_ngd.torch_kfac import KFAC  # noqa: E402
from hybrid_ngd.torch_ngd import NGD  # noqa: E402

torch.set_default_dtype(torch.float64)


def regression_data(seed=0, n=64, d=4):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    y = X @ torch.arange(1.0, d + 1) + 0.5 + 0.1 * torch.randn(n, generator=g)
    return X, y


def two_moons(n=200, seed=0):
    rng = np.random.default_rng(seed)
    t, labels = rng.uniform(0, np.pi, n), rng.integers(0, 2, n)
    x = np.where(labels[:, None] == 0, np.c_[np.cos(t), np.sin(t)], np.c_[1 - np.cos(t), 0.5 - np.sin(t)])
    return torch.tensor(x + 0.1 * rng.normal(size=x.shape)), torch.tensor(labels)


def mlp(*sizes, seed=0):
    torch.manual_seed(seed)
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(torch.nn.Linear(a, b))
        if i < len(sizes) - 2:
            layers.append(torch.nn.Tanh())
    return torch.nn.Sequential(*layers)


def test_matches_full_gauss_newton_for_a_single_linear_layer():
    # One output and residual = s - y: G is exactly 1, so A ⊗ G is the exact Gauss-Newton matrix.
    X, y = regression_data()
    kfac_model, ngd_model = mlp(4, 1), mlp(4, 1)
    kfac = KFAC(kfac_model, lr=1.0, damping=0.0, metric="gauss_newton", stat_decay=0.0)
    ngd = NGD(ngd_model.parameters(), lr=1.0, damping=0.0, metric="gauss_newton")
    kfac.step(lambda: kfac_model(X).squeeze(-1) - y)
    ngd.step(lambda: ngd_model(X).squeeze(-1) - y)
    for p, q in zip(kfac_model.parameters(), ngd_model.parameters()):
        torch.testing.assert_close(p, q)
    design = torch.column_stack([X, torch.ones(len(y))])
    solution = torch.linalg.lstsq(design, y).solution
    torch.testing.assert_close(torch.cat([kfac_model[0].weight.reshape(-1), kfac_model[0].bias]), solution)


def test_factors_match_definition():
    X, labels = two_moons(50)
    model = mlp(2, 3, 2)
    opt = KFAC(model, lr=0.1, metric="fisher", stat_decay=0.0)
    opt.step(lambda: torch.nn.functional.cross_entropy(model(X), labels, reduction="none"))
    # recompute layer 0's factors by hand (weights have moved, so rebuild from the pre-step copy)
    model0 = mlp(2, 3, 2)
    h = model0[0](X)
    losses = torch.nn.functional.cross_entropy(model0[2](torch.tanh(h)), labels, reduction="none")
    (g,) = torch.autograd.grad(losses.sum(), h)
    a = torch.column_stack([X, torch.ones(len(X))])
    A, G = opt.factors[model[0]]
    torch.testing.assert_close(A, a.T @ a / len(X))
    torch.testing.assert_close(G, g.T @ g / len(X))


@pytest.mark.parametrize("metric", ["fisher", "gauss_newton"])
def test_trains_mlp(metric):
    X, labels = two_moons()
    model = mlp(2, 8, 2)
    opt = KFAC(model, lr=0.3, damping=1e-2, metric=metric)

    def closure():
        logits = model(X)
        if metric == "fisher":
            return torch.nn.functional.cross_entropy(logits, labels, reduction="none")
        return (logits - torch.nn.functional.one_hot(labels, 2)).reshape(-1)

    losses = [opt.step(closure).item() for _ in range(40)]
    assert losses[-1] < 0.5 * losses[0]
    accuracy = (model(X).argmax(1) == labels).double().mean()
    assert accuracy > 0.85


def test_host_solver_matches_direct():
    X, labels = two_moons(80)
    models = [mlp(2, 3, 2), mlp(2, 3, 2)]
    opts = [KFAC(models[0], lr=0.5, damping=1e-2), KFAC(models[1], lr=0.5, damping=1e-2, solver="numpy")]
    for _ in range(3):
        for model, opt in zip(models, opts):
            opt.step(lambda: torch.nn.functional.cross_entropy(model(X), labels, reduction="none"))
    for p, q in zip(models[0].parameters(), models[1].parameters()):
        torch.testing.assert_close(p, q)


def test_qiskit_solver_on_kronecker_factors():
    X, y = regression_data()
    model = mlp(4, 2, 1)
    solver = QiskitHHLSolver(num_clock_qubits=8, match_damping_to_clock=True)
    opt = KFAC(model, lr=0.5, damping=1e-2, metric="gauss_newton", solver=solver)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        losses = [opt.step(lambda: model(X).squeeze(-1) - y).item() for _ in range(15)]
    assert losses[-1] < 0.2 * losses[0]
    assert solver.last_info.num_qubits <= 3 + 8 + 1  # factors of size 5 and 2, never the full 13 x 13 metric


def test_unsupported_parameters():
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.LayerNorm(3), torch.nn.Linear(3, 1))
    with pytest.raises(ValueError, match="other_params='sgd'"):
        KFAC(model)
    X, y = regression_data(d=2)
    opt = KFAC(model, lr=0.1, metric="gauss_newton", other_params="sgd")
    before = model[1].weight.clone()
    opt.step(lambda: model(X).squeeze(-1) - y)
    assert not torch.equal(before, model[1].weight)
    with pytest.raises(ValueError, match="hessian"):
        KFAC(mlp(2, 1), metric="hessian")


def test_layer_applied_twice_raises():
    layer = torch.nn.Linear(2, 2)
    model = torch.nn.Sequential(layer)
    opt = KFAC(model, metric="gauss_newton")
    X = torch.randn(8, 2)
    with pytest.raises(RuntimeError, match="more than once"):
        opt.step(lambda: layer(layer(X)).reshape(-1))


def moons_closure(model, X, labels):
    return lambda: torch.nn.functional.cross_entropy(model(X), labels, reduction="none")


def test_auto_with_unlimited_budget_is_full_ngd():
    X, labels = two_moons(80)
    kfac_model, ngd_model = mlp(2, 3, 2), mlp(2, 3, 2)
    kfac = KFAC(kfac_model, lr=0.5, damping=1e-2, structure="auto", stat_decay=0.0)
    ngd = NGD(ngd_model.parameters(), lr=0.5, damping=1e-2)
    assert [b.kind for b in kfac.plan] == ["dense"]
    for _ in range(3):
        kfac.step(moons_closure(kfac_model, X, labels))
        ngd.step(moons_closure(ngd_model, X, labels))
    for p, q in zip(kfac_model.parameters(), ngd_model.parameters()):
        torch.testing.assert_close(p, q)


@pytest.mark.parametrize(
    "max_dimension, kinds, systems",
    [
        (17, ["dense"], [(17,)]),                                   # whole model fits
        (10, ["dense", "dense"], [(9,), (8,)]),                     # each layer's block fits
        (8, ["kronecker", "dense"], [(3, 3), (8,)]),                # layer 0 (9 params) needs factors
        (2, ["kronecker", "kronecker"], [(2, 1, 2, 1), (2, 2, 2)]),  # factors chunked
    ],
)
def test_auto_plan_follows_budget(max_dimension, kinds, systems):
    opt = KFAC(mlp(2, 3, 2), structure="auto", max_dimension=max_dimension)
    assert [b.kind for b in opt.plan] == kinds
    assert [b.systems for b in opt.plan] == systems


@pytest.mark.parametrize("structure", ["auto", "kfac"])
def test_never_solves_more_than_the_budget(structure):
    sizes = []

    def recording_solver(A, b):
        sizes.append(A.shape[0])
        return np.linalg.solve(A, b)

    X, labels = two_moons(80)
    model = mlp(2, 6, 2)
    opt = KFAC(model, lr=0.3, damping=1e-2, solver=recording_solver, structure=structure, max_dimension=4)
    losses = [opt.step(moons_closure(model, X, labels)).item() for _ in range(15)]
    assert max(sizes) <= 4
    assert losses[-1] < losses[0]


def test_hhl_qubit_budget_sets_the_factorisation():
    X, y = regression_data()
    model = mlp(4, 2, 1)  # 13 parameters: needs 4 system qubits for the full metric
    solver = QiskitHHLSolver(num_clock_qubits=4, max_qubits=7, match_damping_to_clock=True)  # 2 system qubits
    opt = KFAC(model, lr=0.5, damping=1e-2, metric="gauss_newton", solver=solver, structure="auto")
    assert opt.max_dimension == 4
    assert [b.kind for b in opt.plan] == ["kronecker", "dense"]  # 4->2 layer (10 params) factorised, 2->1 dense (3)
    assert "up to 7 qubits" in opt.describe()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        losses = [opt.step(lambda: model(X).squeeze(-1) - y).item() for _ in range(15)]
    assert losses[-1] < 0.3 * losses[0]


def test_chunked_solve_is_block_diagonal():
    rng = np.random.default_rng(0)
    M = torch.tensor(rng.normal(size=(5, 5)))
    M = M @ M.T + 5 * torch.eye(5)
    B = torch.tensor(rng.normal(size=(5, 3)))
    opt = KFAC(mlp(2, 1), max_dimension=2)
    expected = torch.cat([torch.linalg.solve(M[a:b, a:b], B[a:b]) for a, b in [(0, 2), (2, 4), (4, 5)]])
    torch.testing.assert_close(opt._solve(M, B), expected)
