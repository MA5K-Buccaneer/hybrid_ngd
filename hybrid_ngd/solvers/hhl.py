"""HHL quantum linear solver, built in Qiskit and run on a choice of backend."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np


@dataclass
class HHLCircuit:
    """An HHL circuit plus what is needed to read the solution back out.

    Qubit layout (little-endian, as in Qiskit): ``sys`` then ``clock`` then
    ``anc``. Postselect ``anc = 1`` and ``clock = 0``; the remaining ``sys``
    amplitudes ``a`` give ``x = norm_b / resolution * a[:n]``.
    """

    circuit: "QuantumCircuit"  # noqa: F821 - qiskit is an optional import
    num_system_qubits: int
    num_clock_qubits: int
    resolution: float  # eigenvalue of one clock step; also the rotation constant C
    eigenvalue_bounds: tuple[float, float]
    norm_b: float
    n: int
    damping_shift: float = 0.0  # mu added to every eigenvalue by match_damping_to_clock


@dataclass
class HHLInfo:
    """Diagnostics from the most recent :class:`QiskitHHLSolver` call."""

    success_probability: float  # probability of the postselected outcome (estimated when sampling)
    num_qubits: int
    resolution: float
    eigenvalue_bounds: tuple[float, float]
    imag_norm: float | None  # discarded imaginary part (exact mode only); ~0 when eigenvalues sit on the clock grid
    num_circuits: int = 1
    shots: int | None = None  # per circuit; None for the exact statevector
    depth: int | None = None  # depth of the (transpiled) base circuit when sampling
    damping_shift: float = 0.0  # mu when match_damping_to_clock is on; the solve is then (A + mu I) x = b


class QiskitHHLSolver:
    """Solve ``A x = b`` with the HHL algorithm in Qiskit.

    The circuit is: amplitude-encode ``b`` -> quantum phase estimation of
    ``exp(iAt)`` on ``num_clock_qubits`` -> rotate an ancilla by
    ``arcsin(C / lambda)`` -> uncompute the phase estimation. Postselecting the
    ancilla gives ``A^{-1} b`` up to the known factor ``||b|| / C``, so the
    returned vector has the correct norm as well as direction.

    The clock register is read as a signed (two's-complement) integer, so
    eigenvalues are resolved in steps of ``resolution = max|lambda| / k_max`` with
    ``k_max = 2**(num_clock_qubits - 1) - 1``. That handles indefinite ``A`` (for
    example a Hessian), and for positive-definite ``A`` it stops phase-estimation
    leakage from the top of the spectrum wrapping onto the smallest, most
    heavily weighted eigenvalues. Eigenvalues that are not multiples of
    ``resolution`` are inverted only approximately, and those below
    ``resolution / 2`` are dropped, which acts as a spectral cut-off. Damping the
    metric keeps the condition number, and hence the clock size, under control.

    ``match_damping_to_clock=True`` makes that damping automatic. It solves
    ``(A + mu I) x = b`` with the smallest ``mu`` that lifts ``lambda_min`` onto the
    first clock step, ``mu = max(0, (lambda_max - k_max lambda_min) / (k_max - 1))``.
    Then no eigenvalue is dropped: directions HHL could not resolve get a
    gradient-like step ``b_j / mu`` instead of none (Levenberg-Marquardt damping
    rather than a hard spectral cut-off). The shift needs no change to the ``A``
    oracle, because ``exp(i(A + mu I)t) = exp(i mu t) exp(iAt)``: inside each
    controlled ``U^(2^j)`` the global phase becomes a phase gate
    ``P(mu t 2^j)`` on clock qubit ``j``. That only helps positive semi-definite
    ``A``, so for indefinite ``A`` it warns and does nothing.

    Backends:

    * ``backend=None`` or ``"statevector"`` (default): read the postselected
      amplitudes exactly from a :class:`~qiskit.quantum_info.Statevector`.
    * ``"statevector_sampler"``: ideal circuit, but read out from ``shots``
      measurements with :class:`~qiskit.primitives.StatevectorSampler`.
    * ``"aer"``: :class:`qiskit_aer.AerSimulator` (needs ``qiskit-aer``).
    * Any ``BackendV2``, e.g. ``AerSimulator.from_backend(fake)`` for a noisy
      device model or a real IBM backend. Circuits are transpiled for it and run
      with :class:`~qiskit.primitives.BackendSamplerV2`, unless ``sampler`` is
      given. For IBM Runtime pass
      ``backend=service.backend(...)`` and ``sampler=SamplerV2(mode=backend)``.
    * ``sampler`` alone: any ``BaseSamplerV2``. Circuits are transpiled to a
      generic ``cx``/``u`` basis first.

    Measurements only give ``|x_i|^2``. For the signs, each other index ``j`` is
    interfered with the largest amplitude ``r``: for the mask ``M = r ^ j``,
    CNOTs from one bit of ``M`` onto the rest followed by a Hadamard turn the
    measurement probabilities into the product ``x_r x_j``. Linking every index
    to ``r`` directly keeps the sign of each significant entry tied to a large
    product rather than to a chain through near-zero amplitudes. The overall
    sign is unobservable, so the one that minimises ``||A x - b||`` is chosen.
    Each solve therefore runs ``2**num_system_qubits`` circuits of ``shots``
    shots (the base circuit plus one per mask, all in one sampler job), and
    assumes the solution amplitudes are real, which holds for real ``A`` and ``b``.

    There is no speed-up here. Building the ``exp(iAt)`` oracle diagonalises
    ``A`` classically, and the default ``eigenvalue_bounds`` come from the same
    diagonalisation. Use it to study how a quantum linear solver's errors
    (clock precision, shot noise, device noise) affect training.

    Args:
        num_clock_qubits: Size of the phase-estimation register.
        eigenvalue_bounds: ``(lambda_min, lambda_max)`` of ``A``. They must
            contain the whole spectrum or the phases wrap around. Defaults to the
            exact extremes.
        backend: See above.
        sampler: A ``BaseSamplerV2`` to run the measurement circuits.
        shots: Shots per circuit when sampling.
        optimization_level: Transpiler optimisation level for ``BackendV2`` targets.
        seed: Seed for the simulators this class constructs itself.
        match_damping_to_clock: Shift the spectrum so that no eigenvalue falls
            below the clock resolution (see above). Needs ``num_clock_qubits >= 3``.
        max_qubits: Qubits available for one HHL circuit. Defaults to
            ``backend.num_qubits`` for a ``BackendV2``, else unlimited. Systems
            larger than :attr:`max_dimension` raise ``ValueError``; optimisers such
            as :class:`~hybrid_ngd.torch_kfac.KFAC` read :attr:`max_dimension` to
            factorise the metric so that every solve fits.
    """

    def __init__(
        self,
        num_clock_qubits: int = 6,
        eigenvalue_bounds: tuple[float, float] | None = None,
        backend=None,
        sampler=None,
        shots: int = 8192,
        optimization_level: int = 1,
        seed: int | None = None,
        match_damping_to_clock: bool = False,
        max_qubits: int | None = None,
    ):
        if num_clock_qubits < 2:
            raise ValueError("num_clock_qubits must be at least 2")
        if match_damping_to_clock and num_clock_qubits < 3:
            raise ValueError("match_damping_to_clock needs num_clock_qubits >= 3")
        if shots < 1:
            raise ValueError("shots must be positive")
        if isinstance(backend, str) and backend not in ("statevector", "statevector_sampler", "aer"):
            raise ValueError(f"unknown backend {backend!r}; use 'statevector', 'statevector_sampler', 'aer' or a BackendV2")
        if sampler is not None and backend in ("statevector", "statevector_sampler"):
            raise ValueError(f"backend={backend!r} cannot be combined with a sampler")
        self.num_clock_qubits = num_clock_qubits
        self.eigenvalue_bounds = eigenvalue_bounds
        self.backend = backend
        self.sampler = sampler
        self.shots = shots
        self.optimization_level = optimization_level
        self.seed = seed
        self.match_damping_to_clock = match_damping_to_clock
        if max_qubits is None and backend is not None and not isinstance(backend, str):
            max_qubits = getattr(backend, "num_qubits", None)
        if max_qubits is not None and max_qubits < num_clock_qubits + 2:
            raise ValueError(
                f"max_qubits={max_qubits} leaves no system qubit: {num_clock_qubits} clock qubits and 1 ancilla "
                "are needed on top"
            )
        self.max_qubits = max_qubits
        self.last_info: HHLInfo | None = None
        self._runner = None  # (sampler, pass_manager), built on the first sampled solve

    @property
    def exact(self) -> bool:
        return self.sampler is None and self.backend in (None, "statevector")

    def num_qubits_for(self, n: int) -> int:
        """Qubits in the HHL circuit for an ``n x n`` system: system, clock and ancilla."""
        return max(1, math.ceil(math.log2(n))) + self.num_clock_qubits + 1

    @property
    def max_dimension(self) -> int | None:
        """Largest system size that fits in ``max_qubits`` (``None`` if unlimited)."""
        if self.max_qubits is None:
            return None
        return 2 ** (self.max_qubits - self.num_clock_qubits - 1)

    def clock_matched_shift(self, lam_min: float, lam_max: float) -> float:
        """The ``mu`` that ``match_damping_to_clock`` adds for a spectrum in
        ``[lam_min, lam_max]``: the smallest shift with
        ``lam_min + mu >= (lam_max + mu) / k_max``. Zero if that already holds."""
        k_max = 2 ** (self.num_clock_qubits - 1) - 1
        return max(0.0, (lam_max - k_max * lam_min) / (k_max - 1))

    def build_circuit(self, A: np.ndarray, b: np.ndarray) -> HHLCircuit:
        from qiskit import QuantumCircuit, QuantumRegister
        from qiskit.circuit.library import StatePreparation, UCRYGate, UnitaryGate

        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)
        n = b.shape[0]
        if A.shape != (n, n):
            raise ValueError(f"A has shape {A.shape}, expected {(n, n)}")
        if self.max_qubits is not None and self.num_qubits_for(n) > self.max_qubits:
            raise ValueError(
                f"a {n}x{n} system needs {self.num_qubits_for(n)} qubits but max_qubits={self.max_qubits}; "
                f"systems up to {self.max_dimension}x{self.max_dimension} fit"
            )
        norm_b = float(np.linalg.norm(b))
        if norm_b == 0:
            raise ValueError("b must be non-zero")
        A = 0.5 * (A + A.T)

        evals, evecs = np.linalg.eigh(A)
        lam_min, lam_max = self.eigenvalue_bounds or (float(evals[0]), float(evals[-1]))
        mu = 0.0
        if self.match_damping_to_clock:
            if lam_min < 0:
                warnings.warn(
                    "match_damping_to_clock only helps positive semi-definite A; A has a negative "
                    "eigenvalue, so no shift is applied.",
                    stacklevel=3,
                )
            else:
                mu = self.clock_matched_shift(lam_min, lam_max)
                lam_min, lam_max = lam_min + mu, lam_max + mu
        scale = max(abs(lam_min), abs(lam_max))
        if scale == 0:
            raise ValueError("A must be non-zero")

        # Pad to a power of two. b is zero on the padding, so the padded
        # eigenvalue only has to be representable (after the shift by mu).
        num_sys = max(1, math.ceil(math.log2(n)))
        dim = 2**num_sys
        w = np.concatenate([evals, np.full(dim - n, scale - mu)])
        V = np.eye(dim)
        V[:n, :n] = evecs

        nc = self.num_clock_qubits
        k_max = 2 ** (nc - 1) - 1
        t = 2 * math.pi * k_max / (2**nc * scale)  # maps |lambda| = scale to clock value k_max
        resolution = scale / k_max
        if np.min(np.abs(evals + mu)) < resolution * (1 - 1e-9):
            warnings.warn(
                f"Condition number exceeds the clock's {k_max}:1 range; eigenvalues below "
                f"{resolution:.3g} are inverted poorly or dropped. Increase num_clock_qubits "
                "or the damping.",
                stacklevel=3,
            )

        sys = QuantumRegister(num_sys, "sys")
        clock = QuantumRegister(nc, "clock")
        anc = QuantumRegister(1, "anc")

        qpe = QuantumCircuit(sys, clock, name="QPE")
        qpe.h(clock)
        p0, p1 = np.diag([1.0, 0.0]), np.diag([0.0, 1.0])
        for j in range(nc):
            U = (V * np.exp(1j * w * t * 2**j)) @ V.T
            # Qiskit orders kron factors last-qubit-first: control is qubit 0.
            cU = np.kron(U, p1) + np.kron(np.eye(dim), p0)
            # Unitary by construction (V is orthogonal), so skip Qiskit's O(dim^3) check.
            qpe.append(UnitaryGate(cU, label=f"c-U^{2**j}", check_input=False), [clock[j], *sys])
            if mu:
                # exp(i(A + mu I)t) = exp(i mu t) exp(iAt). Controlled on clock[j], that
                # global phase is a relative phase on the control: a phase gate.
                qpe.p(mu * t * 2**j, clock[j])
        qpe.append(_inverse_qft(nc), clock)

        def signed_k(k: int) -> int:
            return k - 2**nc if k >= 2 ** (nc - 1) else k

        # sin(theta/2) = C / lambda_k with C = resolution and lambda_k = k * resolution.
        angles = [0.0] + [2 * math.asin(1 / signed_k(k)) for k in range(1, 2**nc)]

        b_pad = np.zeros(dim)
        b_pad[:n] = b / norm_b
        qc = QuantumCircuit(sys, clock, anc, name="HHL")
        qc.append(StatePreparation(b_pad), sys)
        qc.append(qpe.to_gate(), [*sys, *clock])
        qc.append(UCRYGate(angles), [anc[0], *clock])
        qc.append(qpe.to_gate().inverse(), [*sys, *clock])
        return HHLCircuit(qc, num_sys, nc, resolution, (lam_min, lam_max), norm_b, n, damping_shift=mu)

    def measurement_circuits(self, hhl: HHLCircuit) -> list:
        """Circuit ``M`` measures the HHL output after interfering every pair of
        system indices ``(i, i ^ M)``; ``M = 0`` is the plain computational basis."""
        from qiskit import ClassicalRegister

        circuits = []
        for mask in range(2**hhl.num_system_qubits):
            qc = hhl.circuit.copy(name=f"HHL_mask{mask}")
            if mask:
                pivot = _lowest_bit(mask)
                for q in range(hhl.num_system_qubits):
                    if q != pivot and mask >> q & 1:
                        qc.cx(pivot, q)
                qc.h(pivot)
            qc.add_register(ClassicalRegister(qc.num_qubits, "meas"))
            qc.measure(range(qc.num_qubits), range(qc.num_qubits))
            circuits.append(qc)
        return circuits

    def __call__(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        b = np.asarray(b, dtype=float)
        if not np.any(b):
            return np.zeros_like(b)
        with _single_threaded_blas():
            return self._solve(A, b)

    def _solve(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        hhl = self.build_circuit(A, b)
        if self.exact:
            post = self._postselect_exact(hhl)
        else:
            post = self._postselect_sampled(hhl, np.asarray(A, dtype=float), b)
        x = hhl.norm_b / hhl.resolution * post[: hhl.n]
        if self.exact:
            self.last_info.imag_norm = float(np.linalg.norm(x.imag))
        return x.real

    def _postselect_exact(self, hhl: HHLCircuit) -> np.ndarray:
        from qiskit.quantum_info import Statevector

        # Start from the prepared |b> rather than simulating StatePreparation's
        # gate decomposition; the rest of the circuit is simulated as built.
        prep, *rest = hhl.circuit.data
        assert prep.operation.name == "state_preparation"
        dim = 2**hhl.num_system_qubits
        initial = np.zeros(2**hhl.circuit.num_qubits, dtype=complex)
        initial[:dim] = prep.operation.params  # sys is the low-order register; clock and anc start in |0>
        body = hhl.circuit.copy_empty_like()
        for instruction in rest:
            body.append(instruction)
        psi = Statevector(initial).evolve(body).data.reshape(2, 2**hhl.num_clock_qubits, dim)
        post = psi[1, 0]  # anc = 1, clock = 0
        self.last_info = HHLInfo(
            success_probability=float(np.vdot(post, post).real),
            num_qubits=hhl.circuit.num_qubits,
            resolution=hhl.resolution,
            eigenvalue_bounds=hhl.eigenvalue_bounds,
            imag_norm=None,
            damping_shift=hhl.damping_shift,
        )
        return post

    def _postselect_sampled(self, hhl: HHLCircuit, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        sampler, pass_manager = self._get_runner()
        circuits = self.measurement_circuits(hhl)
        if pass_manager is not None:
            circuits = pass_manager.run(circuits)
        result = sampler.run(circuits, shots=self.shots).result()
        probs = [_postselected_probs(result[k].data.meas.get_counts(), hhl) for k in range(len(circuits))]

        base = probs[0]
        success = float(base.sum())
        self.last_info = HHLInfo(
            success_probability=success,
            num_qubits=hhl.circuit.num_qubits,
            resolution=hhl.resolution,
            eigenvalue_bounds=hhl.eigenvalue_bounds,
            imag_norm=None,
            num_circuits=len(circuits),
            shots=self.shots,
            depth=circuits[0].depth(),
            damping_shift=hhl.damping_shift,
        )
        if success == 0:
            warnings.warn(
                f"No shot passed postselection out of {self.shots}; returning zeros. "
                "Increase shots or damping, or reduce num_clock_qubits.",
                stacklevel=3,
            )
            return np.zeros(2**hhl.num_system_qubits)

        post = _recover_signs(np.sqrt(base), probs)
        x = post[: hhl.n]
        Ax = (A + A.T) / 2 @ x + hhl.damping_shift * x  # the system actually solved is (A + mu I) x = b
        if b @ Ax < 0:  # global sign: the one with the smaller residual
            post = -post
        return post

    def _get_runner(self):
        if self._runner is None:
            from qiskit.primitives import BackendSamplerV2, StatevectorSampler
            from qiskit.transpiler import generate_preset_pass_manager

            backend = self.backend
            if backend == "statevector_sampler":
                self._runner = (StatevectorSampler(seed=self.seed), None)
            elif backend is None:
                pass_manager = generate_preset_pass_manager(self.optimization_level, basis_gates=["cx", "u"])
                self._runner = (self.sampler, pass_manager)
            else:
                if backend == "aer":
                    from qiskit_aer import AerSimulator

                    backend = AerSimulator(seed_simulator=self.seed)
                pass_manager = generate_preset_pass_manager(self.optimization_level, backend=backend)
                sampler = self.sampler
                if sampler is None:
                    options = {} if self.seed is None else {"seed_simulator": self.seed}
                    sampler = BackendSamplerV2(backend=backend, options=options)
                self._runner = (sampler, pass_manager)
        return self._runner


def _postselected_probs(counts: dict[str, int], hhl: HHLCircuit) -> np.ndarray:
    """Probabilities of ``(anc = 1, clock = 0, sys = i)``, unnormalised by the postselection."""
    m, nc = hhl.num_system_qubits, hhl.num_clock_qubits
    probs = np.zeros(2**m)
    total = 0
    for bits, count in counts.items():
        idx = int(bits, 2)  # clbit i measures qubit i, and bitstrings are little-endian
        total += count
        if idx >> (m + nc) == 1 and (idx >> m) % 2**nc == 0:
            probs[idx % 2**m] += count
    return probs / total


def _recover_signs(magnitudes: np.ndarray, mask_probs: list[np.ndarray]) -> np.ndarray:
    """Signed real amplitudes from ``|a_i|`` and the postselected probabilities of
    each mask circuit. With pivot ``q`` the lowest bit of ``M`` and ``i`` the
    member of the pair with bit ``q`` clear, ``a_i a_{i^M} = (p_M[i] - p_M[i ^ 2**q]) / 2``."""
    root = int(np.argmax(magnitudes))
    signs = np.ones(len(magnitudes))
    for j in range(len(magnitudes)):
        if j == root:
            continue
        mask = root ^ j
        bit = 1 << _lowest_bit(mask)
        i = root if not root & bit else j
        product = (mask_probs[mask][i] - mask_probs[mask][i ^ bit]) / 2
        signs[j] = 1.0 if product >= 0 else -1.0
    return signs * magnitudes


def _lowest_bit(mask: int) -> int:
    return (mask & -mask).bit_length() - 1


def _single_threaded_blas():
    """The simulation is many small dense products, where BLAS threading costs
    far more than it saves (3-8x slower solves with 12 threads). Limit it to
    one thread when threadpoolctl is available."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        import contextlib

        return contextlib.nullcontext()
    return threadpool_limits(limits=1, user_api="blas")


def _inverse_qft(num_qubits: int):
    try:
        from qiskit.circuit.library import QFTGate
    except ImportError:  # qiskit < 1.2
        from qiskit.circuit.library import QFT

        return QFT(num_qubits, inverse=True).to_gate()
    return QFTGate(num_qubits).inverse()
