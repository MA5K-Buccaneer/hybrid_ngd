import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hybrid_ngd import QiskitHHLSolver  # noqa: E402
from hybrid_ngd.torch_ngd import NGD  # noqa: E402

torch.set_default_dtype(torch.float64)


def regression_data(seed=0, n=64, d=4):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    y = X @ torch.arange(1.0, d + 1) + 0.1 * torch.randn(n, generator=g)
    return X, y


def test_hessian_step_solves_quadratic_across_groups():
    rng = np.random.default_rng(0)
    M = rng.normal(size=(5, 5))
    A = torch.tensor(M @ M.T + 5 * np.eye(5))
    c = torch.tensor(rng.normal(size=5))
    a, b = torch.zeros(2, requires_grad=True), torch.zeros(3, requires_grad=True)
    opt = NGD([{"params": [a]}, {"params": [b]}], lr=1.0, damping=0.0, metric="hessian")

    def closure():
        x = torch.cat([a, b])
        return 0.5 * x @ A @ x - c @ x

    opt.step(closure)
    torch.testing.assert_close(torch.cat([a, b]).detach(), torch.linalg.solve(A, c))


def test_gauss_newton_step_is_least_squares():
    X, y = regression_data()
    w = torch.zeros(4, requires_grad=True)
    opt = NGD([w], lr=1.0, damping=0.0, metric="gauss_newton")
    opt.step(lambda: X @ w - y)
    torch.testing.assert_close(w.detach(), torch.linalg.lstsq(X, y).solution)


def test_fisher_decreases_logistic_loss():
    X, y = regression_data()
    labels = (y > 0).double()
    w = torch.zeros(4, requires_grad=True)
    opt = NGD([w], lr=0.5, damping=1e-2, metric="fisher")

    def closure():
        return torch.nn.functional.binary_cross_entropy_with_logits(X @ w, labels, reduction="none")

    losses = [opt.step(closure).item() for _ in range(10)]
    assert losses[-1] < 0.5 * losses[0]


def test_frozen_group_does_not_move():
    X, y = regression_data()
    w1, w2 = torch.zeros(2, requires_grad=True), torch.zeros(2, requires_grad=True)
    opt = NGD([{"params": [w1]}, {"params": [w2], "lr": 1e-12}], lr=1.0, metric="gauss_newton")
    opt.step(lambda: X @ torch.cat([w1, w2]) - y)
    assert w1.abs().sum() > 0.1 and w2.abs().max() < 1e-9


def test_qiskit_solver_matches_direct():
    X, y = regression_data()
    target = torch.linalg.lstsq(X, y).solution
    w = torch.zeros(4, requires_grad=True)
    opt = NGD([w], lr=1.0, damping=1e-3, metric="gauss_newton", solver=QiskitHHLSolver(num_clock_qubits=8))
    for _ in range(5):
        opt.step(lambda: X @ w - y)
    assert torch.linalg.norm(w.detach() - target) / torch.linalg.norm(target) < 1e-2
    assert opt.solver.last_info.success_probability > 0


def test_sampled_qiskit_backend_trains():
    X, y = regression_data()
    target = torch.linalg.lstsq(X, y).solution
    w = torch.zeros(4, requires_grad=True)
    solver = QiskitHHLSolver(num_clock_qubits=8, backend="statevector_sampler", shots=50_000, seed=0)
    opt = NGD([w], lr=1.0, damping=1e-3, metric="gauss_newton", solver=solver)
    for _ in range(5):
        opt.step(lambda: X @ w - y)
    assert torch.linalg.norm(w.detach() - target) / torch.linalg.norm(target) < 0.05


def test_jacobian_methods_agree():
    from hybrid_ngd.torch_ngd import _jacobian, _jacobian_by_columns

    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2))
    X, labels = torch.randn(50, 3), torch.randint(0, 2, (50,))
    params = list(model.parameters())
    losses = torch.nn.functional.cross_entropy(model(X), labels, reduction="none")
    rows = torch.stack([torch.cat([g.reshape(-1) for g in torch.autograd.grad(losses[i], params, retain_graph=True)]) for i in range(50)])
    torch.testing.assert_close(_jacobian_by_columns(losses, params), rows)
    torch.testing.assert_close(_jacobian(losses, params), rows)
