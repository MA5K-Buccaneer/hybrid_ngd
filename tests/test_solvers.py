import warnings

import numpy as np
import pytest

from hybrid_ngd import QiskitHHLSolver


def random_symmetric(rng, evals):
    Q, _ = np.linalg.qr(rng.normal(size=(len(evals), len(evals))))
    return Q @ np.diag(evals) @ Q.T


def test_exact_when_eigenvalues_on_clock_grid():
    # 4 clock qubits -> k_max = 7, so multiples of scale / 7 are represented exactly.
    rng = np.random.default_rng(0)
    A = random_symmetric(rng, np.array([1, 3, 4, 7]) * 0.5)
    b = rng.normal(size=4)
    solver = QiskitHHLSolver(num_clock_qubits=4)
    np.testing.assert_allclose(solver(A, b), np.linalg.solve(A, b), atol=1e-10)
    assert solver.last_info.imag_norm < 1e-10
    assert 0 < solver.last_info.success_probability <= 1


def test_indefinite_and_padded():
    rng = np.random.default_rng(1)
    A = random_symmetric(rng, np.array([-7.0, 2.0, 5.0]))  # n = 3 is padded to 4
    b = rng.normal(size=3)
    x = QiskitHHLSolver(num_clock_qubits=4)(A, b)
    np.testing.assert_allclose(x, np.linalg.solve(A, b), atol=1e-10)


def test_off_grid_error_shrinks_with_clock_size():
    rng = np.random.default_rng(2)
    cases = [(random_symmetric(rng, rng.uniform(1, 10, 6)), rng.normal(size=6)) for _ in range(10)]

    def median_error(nc):
        errs = []
        for A, b in cases:
            exact = np.linalg.solve(A, b)
            errs.append(np.linalg.norm(QiskitHHLSolver(nc)(A, b) - exact) / np.linalg.norm(exact))
        return np.median(errs)

    coarse, fine = median_error(5), median_error(9)
    assert fine < 0.01
    assert fine < coarse / 4


def test_warns_when_ill_conditioned():
    A = np.diag([1e-3, 1.0])
    with pytest.warns(UserWarning, match="Condition number"):
        QiskitHHLSolver(num_clock_qubits=4)(A, np.ones(2))


def test_zero_rhs():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        np.testing.assert_array_equal(QiskitHHLSolver()(np.eye(3), np.zeros(3)), np.zeros(3))


def test_build_circuit_layout():
    hhl = QiskitHHLSolver(num_clock_qubits=5).build_circuit(np.eye(5) * 2, np.ones(5))
    assert (hhl.num_system_qubits, hhl.num_clock_qubits) == (3, 5)
    assert hhl.circuit.num_qubits == 3 + 5 + 1


def sampled_cases():
    rng = np.random.default_rng(3)
    pd = random_symmetric(rng, np.array([1, 3, 4, 7]) * 0.5)
    # Opposite-signed large entries at indices 01 and 10, which differ in both
    # bits, plus a near-zero one: sign recovery must not route through the
    # small amplitudes.
    indefinite = random_symmetric(rng, np.array([-7.0, 2.0, 5.0]))
    return [(pd, rng.normal(size=4)), (indefinite, indefinite @ np.array([0.01, -1.0, 0.8]))]


def relative_error(solver, A, b):
    exact = np.linalg.solve(A, b)
    return np.linalg.norm(solver(A, b) - exact) / np.linalg.norm(exact)


@pytest.mark.parametrize("case", range(2))
def test_statevector_sampler_backend(case):
    A, b = sampled_cases()[case]
    solver = QiskitHHLSolver(num_clock_qubits=4, backend="statevector_sampler", shots=100_000, seed=0)
    assert relative_error(solver, A, b) < 0.03
    info = solver.last_info
    assert (info.num_circuits, info.shots, info.imag_norm) == (4, 100_000, None)


def test_sign_recovery_with_non_adjacent_large_entries():
    A, b = sampled_cases()[1]
    x = np.linalg.solve(A, b)
    assert abs(x[0]) < 0.05 * np.abs(x).max()  # the setup the test relies on
    solver = QiskitHHLSolver(num_clock_qubits=4, backend="statevector_sampler", shots=20_000, seed=1)
    np.testing.assert_array_equal(np.sign(solver(A, b)[1:]), np.sign(x[1:]))


def test_transpiled_backend():
    from qiskit.providers.fake_provider import GenericBackendV2

    A, b = sampled_cases()[0]
    solver = QiskitHHLSolver(4, backend=GenericBackendV2(7, noise_info=False, seed=0), shots=100_000, seed=0)
    assert relative_error(solver, A, b) < 0.03
    assert solver.last_info.depth > 5  # decomposed to the backend's basis


def test_aer_backend_and_bare_sampler():
    pytest.importorskip("qiskit_aer")
    from qiskit_aer.primitives import SamplerV2

    A, b = sampled_cases()[1]
    for solver in (
        QiskitHHLSolver(4, backend="aer", shots=100_000, seed=0),
        QiskitHHLSolver(4, sampler=SamplerV2(seed=0), shots=100_000),
    ):
        assert relative_error(solver, A, b) < 0.05


def test_invalid_backend_options():
    with pytest.raises(ValueError, match="unknown backend"):
        QiskitHHLSolver(backend="gpu")
    with pytest.raises(ValueError, match="cannot be combined"):
        QiskitHHLSolver(backend="statevector", sampler=object())


def test_ibm_runtime_sampler():
    pytest.importorskip("qiskit_ibm_runtime")
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime import SamplerV2

    A, b = sampled_cases()[0]
    backend = AerSimulator(seed_simulator=0)
    solver = QiskitHHLSolver(4, backend=backend, sampler=SamplerV2(mode=backend), shots=100_000)
    assert relative_error(solver, A, b) < 0.03


def ill_conditioned_case():
    rng = np.random.default_rng(4)
    return random_symmetric(rng, np.array([1e-3, 0.01, 0.1, 0.5, 2.0, 9.0])), rng.normal(size=6)


def test_match_damping_to_clock_is_a_phase_gate_shift():
    A, b = ill_conditioned_case()
    solver = QiskitHHLSolver(6, match_damping_to_clock=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the shift must lift every eigenvalue onto the clock grid
        x = solver(A, b)
    mu = solver.last_info.damping_shift
    assert mu == pytest.approx(solver.clock_matched_shift(1e-3, 9.0))
    shifted = A + mu * np.eye(6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        via_oracle = QiskitHHLSolver(6)(shifted, b)  # same maths, with the shift built into exp(iAt)
    np.testing.assert_allclose(x, via_oracle, atol=1e-10)

    exact = np.linalg.solve(shifted, b)
    assert np.linalg.norm(x - exact) / np.linalg.norm(exact) < 0.05
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = QiskitHHLSolver(6)(A, b)
    assert np.linalg.norm(plain - exact) / np.linalg.norm(exact) > 0.5  # drops most of the spectrum


def test_match_damping_to_clock_sampled():
    A, b = ill_conditioned_case()
    solver = QiskitHHLSolver(6, match_damping_to_clock=True, backend="statevector_sampler", shots=100_000, seed=0)
    x = solver(A, b)
    exact = np.linalg.solve(A + solver.last_info.damping_shift * np.eye(6), b)
    assert np.linalg.norm(x - exact) / np.linalg.norm(exact) < 0.06


def test_match_damping_to_clock_skips_indefinite():
    A = np.diag([-1.0, 0.01, 2.0])
    solver = QiskitHHLSolver(6, match_damping_to_clock=True)
    with pytest.warns(UserWarning, match="positive semi-definite"):
        solver(A, np.ones(3))
    assert solver.last_info.damping_shift == 0.0
    with pytest.raises(ValueError, match="num_clock_qubits >= 3"):
        QiskitHHLSolver(2, match_damping_to_clock=True)
