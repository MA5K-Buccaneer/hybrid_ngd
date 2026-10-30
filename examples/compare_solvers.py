"""Train a small tanh network with natural gradient descent, solving each step
classically and with the simulated HHL circuit, in both torch and JAX.

    python examples/compare_solvers.py
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import torch

from hybrid_ngd import QiskitHHLSolver
from hybrid_ngd.jax_ngd import NGD as JaxNGD
from hybrid_ngd.torch_ngd import NGD as TorchNGD

jax.config.update("jax_enable_x64", True)
warnings.simplefilter("ignore", UserWarning)  # ill-conditioning warnings from small clocks

rng = np.random.default_rng(0)
X = rng.uniform(-2, 2, size=(128, 1))
y = np.sin(X[:, 0])
HIDDEN = 2  # 1 -> 2 -> 1 network: 7 parameters, so 3 system qubits
STEPS = 30


def init_params():
    r = np.random.default_rng(1)
    return {
        "W1": r.normal(size=(1, HIDDEN)),
        "b1": np.zeros(HIDDEN),
        "W2": r.normal(size=(HIDDEN,)) / HIDDEN,
        "b2": np.zeros(()),
    }


def run_torch(solver):
    p = {k: torch.tensor(v, requires_grad=True) for k, v in init_params().items()}
    Xt, yt = torch.tensor(X), torch.tensor(y)
    opt = TorchNGD(p.values(), lr=0.5, damping=1e-3, metric="gauss_newton", solver=solver)

    def residuals():
        return torch.tanh(Xt @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"] - yt

    return [opt.step(residuals).item() for _ in range(STEPS)]


def run_jax(solver):
    def residuals(p, X, y):
        return jnp.tanh(X @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"] - y

    opt = JaxNGD(residuals, lr=0.5, damping=1e-3, metric="gauss_newton", solver=solver)
    params = jax.tree.map(jnp.asarray, init_params())
    state, step = opt.init(params), jax.jit(opt.update)
    losses = []
    for _ in range(STEPS):
        params, state, loss = step(params, state, X, y)
        losses.append(float(loss))
    return losses


if __name__ == "__main__":
    runs = {
        "torch / direct": run_torch("direct"),
        "torch / HHL (6 clock qubits)": run_torch(QiskitHHLSolver(num_clock_qubits=6)),
        "jax   / direct": run_jax("direct"),
        "jax   / HHL (10 clock qubits)": run_jax(QiskitHHLSolver(num_clock_qubits=10)),
    }
    width = max(map(len, runs))
    print(f"{'':{width}}  " + "  ".join(f"step {s:>2}" for s in (0, 5, 10, 20, STEPS - 1)))
    for name, losses in runs.items():
        print(f"{name:{width}}  " + "  ".join(f"{losses[s]:.1e}" for s in (0, 5, 10, 20, STEPS - 1)))
