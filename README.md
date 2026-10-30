# hybrid-ngd

Natural gradient descent for **PyTorch** and **JAX**, where the linear solve
`(G + damping·I) d = ∇L` can run classically or through a simulated **HHL
quantum linear solver built in Qiskit**.

```bash
uv pip install -e ".[torch,jax,qiskit]"   # pick the extras you need
pytest
python examples/compare_solvers.py
```

[`examples/ngd_cross_entropy_torch.ipynb`](examples/ngd_cross_entropy_torch.ipynb) walks through training a small
torch classifier on cross-entropy with Adam, NGD with a direct solve, and NGD with HHL. Install its extras with
`pip install -e ".[notebook]"`.

[`examples/ngd_diffusion_torch.ipynb`](examples/ngd_diffusion_torch.ipynb) trains a 50-parameter DDPM denoiser with
`metric="gauss_newton"`. It shows why HHL's clock resolution, not the system size, limits the quantum solve, and
how `match_damping_to_clock=True` recovers it.

[`examples/ngd_score_sde_torch.ipynb`](examples/ngd_score_sde_torch.ipynb) does the same in the score-based SDE
formulation (VP SDE, denoising score matching). The toy data has an analytic score, so it measures the exact score
error and compares three samplers: the reverse SDE, the probability-flow ODE and predictor-corrector.

[`examples/benchmark_optimisers.ipynb`](examples/benchmark_optimisers.ipynb) benchmarks SGD, Adam, NGD (torch and
JAX), K-FAC, M-FAC (torch and JAX) and the HHL variants. It covers three tasks with tuned settings and 3 seeds,
the cost per step as the model grows, K-FAC (auto) across qubit budgets, and M-FAC (HHL) across clock sizes, with a
noise-injection control. The first run takes about 10 minutes. Results are saved to
`examples/results/benchmark_optimisers/`, so later runs load them in seconds. A section recomputes only when its
settings or the `hybrid_ngd` source change, or when you set `RECOMPUTE = True`.

## Metrics

| `metric`         | loss function / closure returns | `G`                                |
|------------------|---------------------------------|------------------------------------|
| `"fisher"`       | per-sample losses `[N]`         | empirical Fisher `mean ∇ℓᵢ∇ℓᵢᵀ`    |
| `"gauss_newton"` | residuals `[N]`                 | `mean ∇rᵢ∇rᵢᵀ` (Gaussian Fisher)   |
| `"hessian"`      | scalar loss                     | `∇²L` (damped Newton)              |

## Solvers

- `"direct"` (default): `torch.linalg.solve` / `jnp.linalg.solve`, on-device.
- `QiskitHHLSolver(num_clock_qubits=6)` (or `"qiskit"`): builds the HHL circuit
  (state prep → QPE of `exp(iAt)` → `arcsin(C/λ)` rotation → uncompute),
  postselects and rescales by `‖b‖/C`. Diagnostics such as success probability,
  qubit count and transpiled depth are in `solver.last_info`, and
  `solver.build_circuit(A, b)` returns the bare circuit.
- Any callable `(A, b) -> x` on NumPy arrays.

### Circuit backends for `QiskitHHLSolver`

| option                                         | runs on                                                   |
|------------------------------------------------|-----------------------------------------------------------|
| `backend=None` / `"statevector"` (default)     | exact `Statevector` amplitudes, no shots                  |
| `backend="statevector_sampler"`                | ideal circuit, `shots` measurements (shot noise only)     |
| `backend="aer"`                                | `AerSimulator` (`pip install -e ".[aer]"`)                |
| `backend=<BackendV2>`                          | transpiled for it, run with `BackendSamplerV2`            |
| `backend=<BackendV2>, sampler=<SamplerV2>`     | transpiled for the backend, run with your sampler         |
| `sampler=<SamplerV2>`                          | transpiled to `cx`/`u`, run with your sampler             |

```python
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

QiskitHHLSolver(8, backend="statevector_sampler", shots=50_000, seed=0)       # shot noise
QiskitHHLSolver(8, backend=AerSimulator.from_backend(FakeSherbrooke()))       # device noise model

# IBM hardware
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
backend = QiskitRuntimeService().least_busy(operational=True)
QiskitHHLSolver(4, backend=backend, sampler=SamplerV2(mode=backend), optimization_level=3)
```

`QiskitHHLSolver(..., match_damping_to_clock=True)` solves `(A + μI) x = b` with the smallest μ that lifts
λ_min onto the first clock step, so no eigenvalue is dropped. It is implemented as one phase gate per clock
qubit, since `exp(i(A+μI)t) = exp(iμt)·exp(iAt)`, and `last_info.damping_shift` reports μ. It only helps
positive semi-definite `A`, such as the Fisher and Gauss-Newton metrics.

Shot-based backends see only `|xᵢ|²`. To recover signs, each index is
interfered with the largest amplitude (CNOTs plus a Hadamard per mask). The
overall sign is the one that minimises `‖Ax − b‖`. Each solve runs
`2^⌈log₂ n⌉` circuits of `shots` shots in a single sampler job.

## Usage

```python
# torch
from hybrid_ngd import QiskitHHLSolver
from hybrid_ngd.torch_ngd import NGD

opt = NGD(model.parameters(), lr=0.5, metric="gauss_newton", solver=QiskitHHLSolver(8))
loss = opt.step(lambda: model(X).squeeze(-1) - y)   # closure: forward only, no backward()

# JAX (any pytree, jit-able; host solvers go through jax.pure_callback)
from hybrid_ngd.jax_ngd import NGD

opt = NGD(lambda p, X, y: f(p, X) - y, lr=0.5, metric="gauss_newton", solver=QiskitHHLSolver(8))
state = opt.init(params)
params, state, loss = jax.jit(opt.update)(params, state, X, y)
```

## K-FAC (torch)

`hybrid_ngd.torch_kfac.KFAC` replaces the dense metric with K-FAC ([Martens & Grosse, 2015](https://arxiv.org/abs/1503.05671)).
Each `nn.Linear` layer gets a block `A ⊗ G`, where `A` is the covariance of the layer's inputs (with a bias
column) and `G` is the covariance of the gradients at its outputs. The update `G⁻¹ ∇[W, b] A⁻¹` only inverts
matrices of size `in_features + 1` and `out_features`.

```python
from hybrid_ngd.torch_kfac import KFAC

opt = KFAC(model, lr=0.5, damping=1e-2, metric="fisher")       # takes the model, to hook its Linear layers
loss = opt.step(lambda: loss_fn(model(X), y))                  # same closure as NGD

KFAC(model, metric="gauss_newton", solver=QiskitHHLSolver(6, match_damping_to_clock=True))  # HHL on the factors
```

- `metric` is `"fisher"` (per-sample losses) or `"gauss_newton"` (residuals; `G` is estimated with random-sign
  backprop). `"hessian"` is not supported.
- Damping is factored Tikhonov damping using Martens & Grosse's π rule. `A` and `G` are moving averages over
  steps (`stat_decay=0.95`).
- `solver` works on the factors. Host solvers such as HHL run once per column, so the circuits stay small: each
  layer needs about `⌈log₂(in+1)⌉` or `⌈log₂ out⌉` system qubits, instead of `⌈log₂ n⌉` for the whole model.
- Only `nn.Linear` layers are preconditioned. Other trainable parameters raise an error unless you pass
  `other_params="sgd"`. A layer may be applied only once per forward pass. There is no JAX version yet.

### Factorising to fit the available qubits

`QiskitHHLSolver(max_qubits=...)` caps the circuit size. It defaults to `backend.num_qubits` for a device or
device model, so the largest system HHL can take is `solver.max_dimension = 2^(max_qubits − clock − 1)`.
`KFAC(..., structure="auto")` reads that limit and uses the most exact structure that fits:

1. the dense metric over all parameters (identical to `NGD`);
2. otherwise a dense block per layer;
3. otherwise a Kronecker block `A ⊗ G` per layer;
4. a factor that is still too large is replaced by its block diagonal, in chunks that fit.

`structure="kfac"` always uses step 3, and step 4 when needed. `max_dimension=` sets the limit directly, for
example with classical solvers. `opt.describe()` prints the plan. For a 2 → 8 → 2 MLP (42 parameters) with
6 clock qubits:

```
max_qubits=13  all parameters  dense block              systems of size 42      -> up to 13 qubits
max_qubits=12  0 (2->8)        dense block              systems of size 24      -> up to 12 qubits
               2 (8->2)        dense block              systems of size 18      -> up to 12 qubits
max_qubits=10  0 (2->8)        Kronecker A⊗G            systems of size 8, 3    -> up to 10 qubits
               2 (8->2)        Kronecker A⊗G (chunked)  systems of size 8, 2, 1 -> up to 10 qubits
max_qubits=9   both layers     Kronecker A⊗G (chunked)  systems of size 4, 3 / 4, 2, 1
```
- On the 2-moons classifier with 322 parameters, 60 steps took 0.5 s with K-FAC and 15 s with full NGD, for a
  final loss of 0.0049 against 0.0029.

## M-FAC (torch and JAX)

`hybrid_ngd.torch_mfac.MFAC` and `hybrid_ngd.jax_mfac.MFAC` implement M-FAC
([Frantar, Kurtic & Alistarh, NeurIPS 2021](https://arxiv.org/abs/2107.03356)). They precondition each gradient
with the empirical Fisher of the last `window` gradients, `F = λI + (1/m) Σ gⱼgⱼᵀ`, and never form a `d × d`
matrix.

```python
from hybrid_ngd.torch_mfac import MFAC

opt = MFAC(model.parameters(), lr=0.1, damping=1e-2, window=32)   # a drop-in torch optimiser
loss.backward(); opt.step()                                        # or opt.step(closure)

MFAC(model.parameters(), solver=QiskitHHLSolver(10, max_qubits=14))  # window defaults to max_dimension = 8
```

- The current gradient is a row of the window `G`, so `F⁻¹g = m·Gᵀ(mλI + GGᵀ)⁻¹eₜ`. That is one `m × m` solve
  per step, `O(dm + m³)` like the paper's dynamic algorithm, and it avoids the `1/λ` cancellation of the general
  Woodbury form. The tests check it against the paper's recursion (eqs. 8–13) and against a dense inverse.
- `solver` solves that `m × m` system, so HHL needs only `⌈log₂ m⌉` system qubits, whatever the model size. The
  window defaults to the solver's `max_dimension`.
- The catch for HHL: consecutive gradients are nearly parallel, so `mλI + GGᵀ` can be badly conditioned, and the
  damping λ largely decides how badly. In the tests, λ = 1e-3 gave condition numbers of about 300–450, which need
  about 10 clock qubits. On the benchmark's task A, λ = 1e-2 gave at most about 25, and 6 clock qubits were
  accurate to about 2% per solve.
- There is no momentum, as in the paper's experiments. Before the window fills, the `k < m` gradients seen so far
  are used with `1/k` scaling.

### Micro-batch curvature: `gradients="microbatch"`

The paper's window holds past *batch-mean* gradients, which miss most of the per-sample spread a Fisher is built
from. On the benchmark's full-batch task A the mean gradient carries only about 2–4% of the per-sample gradient
energy. `MFAC(gradients="microbatch", window=m)` builds the curvature from the current step instead. The closure
returns per-sample losses, which are split into `m` micro-batches with mean gradients `gⱼ` and sizes `nⱼ`:

```
F = λI + ḡḡᵀ + (1/m) Σⱼ nⱼ (gⱼ − ḡ)(gⱼ − ḡ)ᵀ,   F⁻¹ḡ = Rᵀ(λI + RRᵀ)⁻¹e₀,   R = [ḡ; √(nⱼ/m)(gⱼ − ḡ)]
```

```python
opt = MFAC(model.parameters(), lr=0.5, damping=1e-2, window=8, gradients="microbatch")
opt.step(lambda: loss_fn(model(X), y))            # per-sample losses, e.g. reduction="none"
```

- With one sample per micro-batch it is exactly NGD's damped empirical Fisher, which the tests check. The `nⱼ`
  factor undoes the `1/nⱼ` shrinkage of a micro-batch mean's spread. Without it, the curvature is underestimated
  about `nⱼ`-fold and steps diverge.
- The HHL system is `(m+1) × (m+1)` with a basis-vector right-hand side, so the window defaults to
  `max_dimension − 1`. On the benchmark's task A, `m = 7` (an 8×8 system, 10 qubits with 6 clock qubits) trained
  as well as dense NGD.
- It needs `m` gradient rows per step instead of one. A low-rank estimate (`m ≪ d`) leaves the other directions
  with only the damping, so it may need a smaller learning rate than NGD.

## Caveats

- `G` is dense `n × n` for `NGD`. HHL needs `⌈log₂ n⌉ + num_clock_qubits + 1` qubits and
  statevector cost grows as `2^qubits`, so keep models to roughly a hundred
  parameters or fewer on the HHL path, or use `KFAC`.
- This is a faithful simulation, not a speed-up. The `exp(iAt)` oracle and the
  default eigenvalue bounds come from a classical diagonalisation of `A`.
- The clock is read as a signed integer with resolution `max|λ| / (2^(nc−1) − 1)`.
  Eigenvalues below about half that are cut off, and off-grid eigenvalues are
  inverted approximately. Raise `damping` or `num_clock_qubits` if you see the
  conditioning warning.
- Shot-based readout assumes real amplitudes (true for real `A`, `b`) and costs
  `n` circuits per optimiser step. Postselection succeeds with probability
  `last_info.success_probability`, so the effective shot count is lower than `shots`.
- On some machines OpenBLAS multithreading makes small dense solves in NumPy and JAX 100–1,000× slower. A
  322×322 solve took 723 ms with 12 threads and 0.9 ms with 1. If JAX `NGD` is unexpectedly slow, set
  `OPENBLAS_NUM_THREADS=1` before importing NumPy. `QiskitHHLSolver` already limits BLAS threads during its
  simulation.
- Device noise hurts quickly. On a generic noisy 9-qubit model, transpiled
  depths of 500–2000 gave 25–60% solution error, against 0.3–5% noiseless.
