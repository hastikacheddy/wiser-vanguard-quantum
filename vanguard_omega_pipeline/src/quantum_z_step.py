"""
ADMM z-update: penalty-free, topology-matched QAOA+ (with ZNE + fallback).
==========================================================================

The z-update is the combinatorial heart of the pipeline.  For cluster c it
solves, over binary supports of exact cardinality :math:`K_c`,

.. math::

    z^{k+1} = \\arg\\min_{|z| = K_c} \\Big\\{
        \\underbrace{w_{mv}\\big(q\\, z^{\\top}\\Sigma z - \\mu^{\\top}z\\big)}_{
            \\text{discrete mean-variance}}
        \\;+\\; \\underbrace{\\tfrac{\\rho}{2}\\,\\|x^{k+1} + u^{k} -
            \\alpha z\\|_2^2}_{\\text{ADMM consensus anchor}} \\Big\\} .

**Key algebraic fact** (why the ADMM anchor costs *zero* circuit structure):
because :math:`z_i^2 = z_i`, the quadratic anchor is **linear** in z,

.. math::

    \\tfrac{\\rho}{2}\\|v - \\alpha z\\|^2
    \\;=\\; \\tfrac{\\rho}{2}\\sum_i \\big(\\alpha^2 - 2\\alpha v_i\\big) z_i
    \\;+\\; const, \\qquad v \\equiv x^{k+1} + u^{k},

so it folds into the single-qubit Z coefficients of the phase Hamiltonian.
The circuit's two-qubit structure comes ONLY from the covariance — the ADMM
coordination is free, and there is still **no cardinality penalty anywhere**:

* **Initial state**: Dicke :math:`|D^n_{K}\\rangle` (deterministic
  Bärtschi–Eidenbenz SCS cascade — exact uniform superposition over the
  feasible shell; every cluster has n ≤ 20, so the exact preparation always
  fits the budget; no heuristic approximation needed at this scale).
* **Mixer**: the HRP-compiled **BFS spanning-tree XY mixer** — the edge list
  comes verbatim from :class:`src.hrp_clustering.MixerTopology` (degree ≤ 3
  → embeds on heavy-hex without SWAPs).  Every
  :math:`e^{-i\\beta(X_iX_j+Y_iY_j)/2}` block conserves total excitation
  number, so Hamming weight :math:`K_c` is an exact invariant of the whole
  circuit for ALL parameter values, Trotterization included.
* **Trotterized shallow (γ, β) initialization**: parameters start on a
  discretized-annealing ramp (γ ramps up, β ramps down across the p layers
  — the Trotterization of an adiabatic sweep), then warm-start from the
  previous ADMM iteration's optimum: consecutive z-updates differ only by
  a bounded shift of the linear Ising field, so parameter transfer is the
  cheap, principled initialization.

**Zero-Noise Extrapolation** (:func:`zne_expectation`): global unitary
folding :math:`U \\mapsto U (U^{\\dagger} U)^m` scales the physical noise by
:math:`c = 2m + 1` while leaving the ideal unitary invariant; energies
measured at :math:`c \\in \\{1, 3, 5\\}` under a depolarizing model are
Richardson-extrapolated to :math:`c \\to 0`.  ZNE wraps the *audit* path
(NISQ-readiness evidence), not the inner ADMM loop, where it would multiply
iteration latency by the number of scale factors for no coordination gain.

**Production fail-safe** (:func:`classical_z_exact`): exhaustive enumeration
of the :math:`\\binom{n}{K}` shell (n ≤ 20 ⇒ at most 184,756 supports —
milliseconds in vectorized form).  The coordinator swaps it in whenever the
quantum path raises or exceeds its latency budget, logging a warning; the
pipeline never stalls.

Qiskit Primitives V2 only (``StatevectorSampler``, ``AerSimulator`` for the
noisy ZNE runs).  No ``qiskit.aqua``, no ``execute``.
"""

from __future__ import annotations

import itertools
import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RYGate, XXPlusYYGate
from qiskit.primitives import StatevectorSampler

from .hrp_clustering import MixerTopology

logger = logging.getLogger("omega.quantum_z")

__all__ = [
    "qubo_to_ising",
    "dicke_state_circuit",
    "build_tree_qaoa_ansatz",
    "ZStepResult",
    "solve_z_update",
    "classical_z_exact",
    "zne_expectation",
    "fold_global",
]


# ===========================================================================
# Binary-quadratic → Ising (generic, penalty-free by construction).
# ===========================================================================
def qubo_to_ising(Q: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Map  f(z) = z'Qz + a'z  (z ∈ {0,1}^n, Q symmetric) to Ising form
    :math:`f = c + \\sum_i h_i Z_i + \\sum_{i<j} J_{ij} Z_i Z_j` via
    :math:`z_i = (1 - Z_i)/2`.

    Diagonal terms fold into the linear part (:math:`z_i^2 = z_i`):

    .. math::

        J_{ij} = \\tfrac{Q_{ij}}{2}\\;(i<j), \\quad
        h_i = -\\tfrac{a_i + Q_{ii}}{2} - \\tfrac{1}{2}\\sum_{j \\ne i} Q_{ij}, \\quad
        c = \\tfrac{\\mathbf{1}^{\\top}a}{2} + \\tfrac{\\mathrm{tr}Q}{2}
            + \\tfrac{1}{4}\\sum_{i \\ne j} Q_{ij} .

    Verified in the QA suite against brute-force enumeration of f(z).
    """
    Q = np.asarray(Q, dtype=float)
    a = np.asarray(a, dtype=float).ravel()
    n = len(a)
    if Q.shape != (n, n):
        raise ValueError(f"Q {Q.shape} incompatible with a ({n},)")
    Q = 0.5 * (Q + Q.T)
    a_eff = a + np.diag(Q)                      # fold diagonal (z² = z)
    Q_off = Q.copy()
    np.fill_diagonal(Q_off, 0.0)
    J = 0.5 * Q_off                             # used strictly i<j
    h = -0.5 * a_eff - 0.5 * Q_off.sum(axis=1)
    offset = 0.5 * a_eff.sum() + 0.25 * Q_off.sum()
    return h, J, float(offset)


def binary_objective(z: np.ndarray, Q: np.ndarray, a: np.ndarray) -> float:
    """f(z) = z'Qz + a'z — the classical scorer shared by all solvers."""
    z = np.asarray(z, dtype=float)
    return float(z @ Q @ z + a @ z)


# ===========================================================================
# Feasible-subspace state preparation (exact — clusters are ≤ 20 qubits).
# ===========================================================================
def dicke_state_circuit(n: int, k: int) -> QuantumCircuit:
    """Deterministic Dicke state |D^n_k⟩ via the Bärtschi–Eidenbenz split-&-
    cycle-shift cascade (arXiv:1904.07358): O(nk) gates, O(n) depth.
    Numerically verified uniform over the weight-k shell in the QA suite."""
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got n={n}, k={k}")
    qc = QuantumCircuit(n, name=f"Dicke|{n},{k}>")
    if k == 0:
        return qc
    for q in range(n - k, n):
        qc.x(q)
    if k == n:
        return qc
    for m in range(n, k, -1):
        _scs_block(qc, m, k)
    for m in range(k, 1, -1):
        _scs_block(qc, m, m - 1)
    return qc


def _scs_block(qc: QuantumCircuit, m: int, k: int) -> None:
    """SCS_{m,k} unitary on qubits [0, m): one CRY split + (k−1) CCRY cascades."""
    qc.cx(m - 2, m - 1)
    qc.cry(2 * np.arccos(np.sqrt(1.0 / m)), m - 1, m - 2)
    qc.cx(m - 2, m - 1)
    for ell in range(2, k + 1):
        qc.cx(m - ell - 1, m - 1)
        qc.append(RYGate(2 * np.arccos(np.sqrt(ell / m))).control(2),
                  [m - 1, m - ell, m - ell - 1])
        qc.cx(m - ell - 1, m - 1)


# ===========================================================================
# Topology-matched QAOA+ ansatz.
# ===========================================================================
def build_tree_qaoa_ansatz(
    h: np.ndarray,
    J: np.ndarray,
    k: int,
    topology: MixerTopology,
    reps: int = 1,
    measure: bool = True,
) -> QuantumCircuit:
    """Assemble the penalty-free ansatz on the HRP-compiled tree.

    .. math::

        |\\gamma,\\beta\\rangle = \\prod_{l=1}^{p}
            e^{-i\\beta_l H_{mix}^{tree}}\\, e^{-i\\gamma_l H_C}\\,
            |D^n_K\\rangle,
        \\qquad
        H_{mix}^{tree} = \\tfrac12 \\sum_{(i,j) \\in T_{BFS}} (X_iX_j + Y_iY_j).

    The phase layer is exact (diagonal factors commute): RZ(2γh_i) and
    RZZ(2γJ_ij).  The mixer applies ``XXPlusYYGate(2β)`` edge-by-edge along
    the topology's parallel layers (Qiskit's XXPlusYY(θ) = exp[−iθ(XX+YY)/4],
    so θ = 2β realizes exp[−iβ(XX+YY)/2]).  Every gate conserves Hamming
    weight ⇒ the ansatz never leaves the |z| = K shell.
    """
    n = len(h)
    if topology.n != n:
        raise ValueError(f"topology n={topology.n} != Hamiltonian n={n}")
    gammas = ParameterVector("γ", reps)
    betas = ParameterVector("β", reps)
    qc = QuantumCircuit(n, name=f"QAOA-tree(p={reps})")
    qc.compose(dicke_state_circuit(n, k), inplace=True)
    for layer in range(reps):
        for i in range(n):
            if abs(h[i]) > 1e-14:
                qc.rz(2.0 * h[i] * gammas[layer], i)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(J[i, j]) > 1e-14:
                    qc.rzz(2.0 * J[i, j] * gammas[layer], i, j)
        for parallel_group in topology.layers:
            for (a_q, b_q) in parallel_group:
                qc.append(XXPlusYYGate(2.0 * betas[layer]), [a_q, b_q])
    if measure:
        qc.measure_all()
    return qc


# ===========================================================================
# The z-update solver (variational loop) + classical exact fallback.
# ===========================================================================
@dataclass
class ZStepResult:
    """Outcome of one z-update (quantum or fallback), with provenance."""
    z: np.ndarray
    objective: float
    method: str                        # 'qaoa_tree' | 'classical_exact'
    params: np.ndarray | None
    history: list[float]
    feasible_fraction: float
    elapsed_s: float
    fallback_used: bool = False
    fallback_reason: str = ""
    circuit: QuantumCircuit | None = field(default=None, repr=False)


def classical_z_exact(
    Q: np.ndarray, a: np.ndarray, k: int, max_supports: int = 400_000,
) -> tuple[np.ndarray, float]:
    """Exact minimizer of z'Qz + a'z over the weight-k shell by enumeration.

    Clusters are capped at 20 assets, so :math:`\\binom{20}{10} = 184{,}756`
    bounds the shell — exhaustive search IS the branch-and-bound endgame at
    this size, with zero optimality gap.  Used as the production fail-safe
    and for early ADMM iterations under the latency policy.
    """
    n = len(a)
    if math.comb(n, k) > max_supports:
        raise ValueError(f"C({n},{k}) exceeds enumeration budget {max_supports:,}")
    best_val, best_support = np.inf, None
    for support in itertools.combinations(range(n), k):
        idx = list(support)
        val = Q[np.ix_(idx, idx)].sum() + a[idx].sum()
        if val < best_val:
            best_val, best_support = val, idx
    z = np.zeros(n, dtype=int)
    z[best_support] = 1
    return z, float(best_val)


def solve_z_update(
    Q: np.ndarray,
    a: np.ndarray,
    k: int,
    topology: MixerTopology,
    reps: int = 1,
    shots: int = 1024,
    maxiter: int = 20,
    seed: int = 7,
    warm_params: np.ndarray | None = None,
    cvar_alpha: float = 0.25,
    use_quantum: bool = True,
    quantum_time_budget_s: float = 120.0,
) -> ZStepResult:
    """Solve one z-update; quantum QAOA+ path with automatic classical fail-safe.

    The variational loop minimizes the CVaR_α of sampled objective values
    (Barkoutsos et al., Quantum 4, 256 (2020)) via COBYLA; the reported z is
    the best *observed* feasible sample re-scored classically.  Any exception
    on the quantum path — or exceeding ``quantum_time_budget_s`` between
    iterations — triggers :func:`classical_z_exact` with a logged warning:
    the ADMM loop keeps moving no matter what the simulator does.
    """
    t0 = time.perf_counter()
    n = len(a)
    if not use_quantum:
        z, val = classical_z_exact(Q, a, k)
        return ZStepResult(z=z, objective=val, method="classical_exact",
                           params=None, history=[val], feasible_fraction=1.0,
                           elapsed_s=time.perf_counter() - t0)
    try:
        h, J, _ = qubo_to_ising(Q, a)
        ansatz = build_tree_qaoa_ansatz(h, J, k, topology, reps=reps)
        sampler = StatevectorSampler(seed=seed)
        rng = np.random.default_rng(seed)
        p = reps
        param_order = sorted(ansatz.parameters, key=lambda prm: prm.name)

        def bind(theta: np.ndarray) -> QuantumCircuit:
            mapping = {}
            for prm in param_order:
                vec, idx = prm.name.split("[")
                idx = int(idx.rstrip("]"))
                mapping[prm] = theta[idx] if vec == "γ" else theta[p + idx]
            return ansatz.assign_parameters(mapping)

        history: list[float] = []
        best_f, best_z = np.inf, None
        infeasible_shots = total_shots = 0

        def evaluate(theta: np.ndarray) -> float:
            nonlocal best_f, best_z, infeasible_shots, total_shots
            if time.perf_counter() - t0 > quantum_time_budget_s:
                raise TimeoutError(f"quantum z-update exceeded "
                                   f"{quantum_time_budget_s}s budget")
            counts = sampler.run([(bind(theta),)], shots=shots).result()[0] \
                .data.meas.get_counts()
            energies, weights = [], []
            for bitstr, cnt in counts.items():
                total_shots += cnt
                zv = np.fromiter((int(c) for c in bitstr[::-1]), dtype=int, count=n)
                if zv.sum() != k:              # symmetry guard (never trips)
                    infeasible_shots += cnt
                    continue
                fv = binary_objective(zv, Q, a)
                energies.append(fv)
                weights.append(cnt)
                if fv < best_f:
                    best_f, best_z = fv, zv
            e = np.asarray(energies)
            w = np.asarray(weights, dtype=float)
            order = np.argsort(e)
            e, w = e[order], w[order]
            cutoff = cvar_alpha * w.sum()
            cum = np.cumsum(w)
            tail = np.minimum(w, np.maximum(cutoff - (cum - w), 0.0))
            val = float(e @ tail / max(tail.sum(), 1e-12))
            history.append(val)
            return val

        if warm_params is not None and len(warm_params) == 2 * p:
            theta0 = np.asarray(warm_params, dtype=float)
        else:
            ell = (np.arange(p) + 1) / (p + 1)   # Trotterized-annealing ramp
            theta0 = np.concatenate([0.75 * ell, 0.75 * (1 - ell)])
            theta0 = theta0 + 0.05 * rng.standard_normal(2 * p)

        theta_opt = _minimize_cobyla(evaluate, theta0, maxiter)
        evaluate(theta_opt)                      # final sampling at optimum
        if best_z is None:
            raise RuntimeError("no feasible sample (symmetry violated?)")
        feas = 1.0 - infeasible_shots / max(total_shots, 1)
        return ZStepResult(z=best_z, objective=best_f, method="qaoa_tree",
                           params=np.asarray(theta_opt), history=history,
                           feasible_fraction=feas,
                           elapsed_s=time.perf_counter() - t0, circuit=ansatz)
    except Exception as exc:  # noqa: BLE001 — production fail-safe, logged
        logger.warning("quantum z-update failed (%s: %s) — falling back to "
                       "exact classical enumeration", type(exc).__name__, exc)
        z, val = classical_z_exact(Q, a, k)
        return ZStepResult(z=z, objective=val, method="classical_exact",
                           params=None, history=[val], feasible_fraction=1.0,
                           elapsed_s=time.perf_counter() - t0,
                           fallback_used=True,
                           fallback_reason=f"{type(exc).__name__}: {exc}")


def _minimize_cobyla(fun, theta0: np.ndarray, maxiter: int) -> np.ndarray:
    """COBYLA via qiskit-algorithms when importable, SciPy otherwise."""
    try:
        from qiskit_algorithms.optimizers import COBYLA
        return np.asarray(COBYLA(maxiter=maxiter, rhobeg=0.3)
                          .minimize(fun, theta0).x)
    except ImportError:
        from scipy.optimize import minimize
        return np.asarray(minimize(fun, theta0, method="COBYLA",
                                   options={"maxiter": maxiter, "rhobeg": 0.3}).x)


# ===========================================================================
# Zero-Noise Extrapolation (global unitary folding) — NISQ-readiness audit.
# ===========================================================================
def fold_global(circuit: QuantumCircuit, scale: float) -> QuantumCircuit:
    """Global unitary folding  U → U (U†U)^m  with noise scale c = 2m + 1.

    ``scale`` must be an odd integer ≥ 1.  The measurement layer is stripped,
    the unitary body folded, and measurements re-appended — the ideal channel
    is the identity-composed U, while physical error rates multiply by ~c.
    """
    if scale < 1 or int(scale) != scale or int(scale) % 2 == 0:
        raise ValueError(f"scale must be an odd integer >= 1, got {scale}")
    m = (int(scale) - 1) // 2
    body = circuit.remove_final_measurements(inplace=False)
    folded = body.copy()
    for _ in range(m):
        folded.compose(body.inverse(), inplace=True)
        folded.compose(body, inplace=True)
    folded.measure_all()
    return folded


def zne_expectation(
    bound_circuit: QuantumCircuit,
    Q: np.ndarray,
    a: np.ndarray,
    k: int,
    scales: tuple[int, ...] = (1, 3, 5),
    shots: int = 4096,
    two_qubit_error: float = 0.01,
    one_qubit_error: float = 0.001,
    seed: int = 7,
) -> dict:
    """Richardson ZNE of the feasible-shell energy under depolarizing noise.

    Runs the (already parameter-bound) circuit on ``AerSimulator`` with a
    depolarizing noise model at each folding scale, computes the mean
    post-selected objective, then extrapolates linearly to zero noise:

    .. math::

        E(c) \\approx E_0 + s\\,c \\;\\Rightarrow\\;
        \\hat E_{ZNE} = E(0) \\text{ from the least-squares fit.}

    Returns ``{"scales", "energies", "extrapolated", "feasible_fractions"}``.
    Under symmetry-breaking noise the post-selection rate ALSO degrades with
    c — reported so the audit shows both the mitigation and its cost.
    """
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(one_qubit_error, 1),
                                      ["rz", "sx", "x", "ry"])
    noise.add_all_qubit_quantum_error(depolarizing_error(two_qubit_error, 2),
                                      ["cx", "rzz", "xx_plus_yy", "cry"])
    backend = AerSimulator(noise_model=noise, seed_simulator=seed)
    n = len(a)

    energies, feas_fracs = [], []
    for c in scales:
        folded = fold_global(bound_circuit, c)
        tqc = transpile(folded, backend, optimization_level=0, seed_transpiler=7)
        counts = backend.run(tqc, shots=shots).result().get_counts()
        num = den = tot = 0.0
        for bitstr, cnt in counts.items():
            tot += cnt
            zv = np.fromiter((int(ch) for ch in bitstr.replace(" ", "")[::-1]),
                             dtype=int, count=n)
            if zv.sum() != k:
                continue
            num += cnt * binary_objective(zv, Q, a)
            den += cnt
        energies.append(num / den if den else np.nan)
        feas_fracs.append(den / tot if tot else 0.0)

    xs = np.asarray(scales, dtype=float)
    ys = np.asarray(energies, dtype=float)
    ok = ~np.isnan(ys)
    slope, intercept = (np.polyfit(xs[ok], ys[ok], 1)
                        if ok.sum() >= 2 else (0.0, ys[ok][0] if ok.any() else np.nan))
    return {
        "scales": list(scales),
        "energies": [float(e) for e in energies],
        "extrapolated": float(intercept),
        "slope": float(slope),
        "feasible_fractions": [float(f) for f in feas_fracs],
    }
