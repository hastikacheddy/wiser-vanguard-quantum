"""
Competitor baseline — faithful STEELMAN of the heuristic stack.
===============================================================

Adversarial-benchmark integrity rules enforced in this module:

1. **The competitor is implemented correctly, at full strength.**  A
   benchmark against a buggy or weakened baseline is void.  The penalty-
   QUBO Ising mapping below is exact (verified against brute-force
   enumeration in the QA suite); DBSCAN is the real scikit-learn
   implementation; the QAOA runs on Qiskit Primitives V2 with the same
   optimizer budget our own z-step gets.
2. **Failure modes are MEASURED, never scripted.**  Constraint leakage is
   read off the sampled shot distribution; DBSCAN's structural mismatch is
   quantified against known sector labels; hardware costs come from the
   real transpiler (see :func:`src.metrics.heavy_hex_transpile_audit`).
   If the penalty stack behaves well at some λ, the numbers will say so —
   that is what makes the numbers credible when it doesn't.

The competitor pipeline replicated here:

* **Clustering**: DBSCAN on the correlation distance
  :math:`D_{ij} = \\sqrt{2(1 - \\rho_{ij})}` with ``metric='precomputed'``.
  Structural critique (measured by :func:`dbscan_quality_report`): DBSCAN
  assumes clusters are *density-connected regions of uniform density*; an
  equity correlation structure is hierarchical with graded density, so
  DBSCAN typically returns one merged super-cluster (ε too large), shatters
  the universe into noise points (ε too small), or both across regimes —
  and it offers no dendrogram to cut at a qubit budget.
* **Formulation**: the textbook soft-penalty QUBO

  .. math::

      f_{\\lambda}(z) = q\\, z^{\\top}\\Sigma z - \\mu^{\\top} z
        + \\lambda\\Big(\\textstyle\\sum_i z_i - K\\Big)^2 ,

  whose exact binary-quadratic form is
  :math:`Q = q\\Sigma + \\lambda \\mathbf{1}\\mathbf{1}^{\\top}` (the λ block
  is rank-1 but **dense** — every asset pair acquires the uniform coupling
  λ/2 after the Ising map) and
  :math:`a = -\\mu - 2\\lambda K \\mathbf{1}`, constant :math:`\\lambda K^2`.
  Note the global reach of the penalty: because cardinality couples ALL
  assets, this formulation cannot be decomposed into sub-problems — the
  competitor must execute one monolithic N-qubit circuit.
* **Circuit**: standard QAOA — uniform :math:`|+\\rangle^{\\otimes N}`
  initialization and transverse-field X mixer
  :math:`H_{mix} = \\sum_i X_i`, which does NOT commute with the excitation
  number: the walk explores all :math:`2^N` states and feasibility is only
  *encouraged* by λ.  Mean-energy objective (the textbook choice).
* **Allocation**: equal weight over the selected support, as specified.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorSampler

from .quantum_z_step import (
    binary_objective, classical_z_exact, qubo_to_ising, _minimize_cobyla,
)

__all__ = [
    "CompetitorResult",
    "CompetitorHeuristicPipeline",
    "dbscan_quality_report",
    "penalty_lambda_sweep",
]


@dataclass
class CompetitorResult:
    """Measured outcome of one competitor-pipeline run."""
    z_argmax: np.ndarray               # their selection rule: most probable
    cardinality_argmax: int
    cardinality_breach: int            # |cardinality − K| (measured)
    feasible_shot_fraction: float      # P(|z| = K) in the final sampling
    z_best_feasible: np.ndarray | None
    objective_mv_argmax: float         # pure mean-variance score of argmax
    objective_mv_best_feasible: float | None
    weights: np.ndarray                # equal-weight allocation on argmax
    penalty_lambda: float
    elapsed_s: float
    history: list[float] = field(default_factory=list)
    circuit: QuantumCircuit | None = field(default=None, repr=False)


class CompetitorHeuristicPipeline:
    """Correct, full-strength implementation of the DBSCAN + penalty-QUBO
    + X-mixer stack (see module docstring for the integrity rules)."""

    def __init__(self, num_assets: int, target_k: int,
                 penalty_lambda: float = 10.0, risk_aversion: float = 5.0,
                 seed: int = 7) -> None:
        self.N = int(num_assets)
        self.K = int(target_k)
        self.lam = float(penalty_lambda)
        self.q = float(risk_aversion)
        self.seed = int(seed)

    # ------------------------------------------------------------- DBSCAN
    def run_dbscan_clustering(self, covariance_matrix: np.ndarray,
                              eps: float = 0.5, min_samples: int = 2) -> np.ndarray:
        """DBSCAN on :math:`D_{ij} = \\sqrt{2(1-\\rho_{ij})}` (their metric,
        note the missing 1/2 vs. the López de Prado normalization — kept
        verbatim to replicate their stack).  Returns sklearn labels with
        −1 = noise.  API note: the correct call is ``fit_predict`` —
        ``fit_labels`` does not exist in scikit-learn."""
        from sklearn.cluster import DBSCAN

        std = np.sqrt(np.diag(covariance_matrix))
        corr = covariance_matrix / (np.outer(std, std) + 1e-12)
        corr = np.clip(corr, -1.0, 1.0)
        dist = np.sqrt(np.maximum(2.0 * (1.0 - corr), 0.0))
        np.fill_diagonal(dist, 0.0)
        return DBSCAN(eps=eps, min_samples=min_samples,
                      metric="precomputed").fit_predict(dist)

    # ------------------------------------------------- penalty QUBO (exact)
    def build_penalty_qubo(self, mu: np.ndarray, sigma: np.ndarray,
                           ) -> tuple[np.ndarray, np.ndarray, float]:
        """Exact binary-quadratic form of the soft-penalty objective.

        .. math::

            f_{\\lambda}(z) = z^{\\top}\\!\\big(q\\Sigma +
                \\lambda\\mathbf{1}\\mathbf{1}^{\\top}\\big) z
                + \\big(-\\mu - 2\\lambda K\\mathbf{1}\\big)^{\\top} z
                + \\lambda K^2

        using :math:`(\\sum_i z_i)^2 = z^{\\top}\\mathbf{1}\\mathbf{1}^{\\top}z`
        (the diagonal of the ones-matrix supplies the linear
        :math:`\\lambda z_i` terms since :math:`z_i^2 = z_i`).  Verified
        against brute force in the QA suite.  Returns ``(Q, a, const)``.
        """
        n = self.N
        Q = self.q * np.asarray(sigma, dtype=float) + self.lam * np.ones((n, n))
        a = -np.asarray(mu, dtype=float) - 2.0 * self.lam * self.K * np.ones(n)
        return Q, a, self.lam * self.K**2

    # ---------------------------------------------------- X-mixer QAOA (V2)
    def build_x_mixer_ansatz(self, h: np.ndarray, J: np.ndarray,
                             reps: int = 1) -> QuantumCircuit:
        """Textbook QAOA: |+⟩^N init, RZ/RZZ phase separator, RX(2β) mixer.

        The mixer :math:`H_{mix} = \\sum_i X_i` does not commute with
        :math:`\\hat n`, so the state has support on ALL Hamming weights —
        the structural source of the leakage measured downstream."""
        n = len(h)
        gammas = ParameterVector("γ", reps)
        betas = ParameterVector("β", reps)
        qc = QuantumCircuit(n, name=f"QAOA-X(p={reps},λ={self.lam:g})")
        qc.h(range(n))
        for layer in range(reps):
            for i in range(n):
                if abs(h[i]) > 1e-14:
                    qc.rz(2.0 * h[i] * gammas[layer], i)
            for i in range(n):
                for j in range(i + 1, n):
                    if abs(J[i, j]) > 1e-14:
                        qc.rzz(2.0 * J[i, j] * gammas[layer], i, j)
            for i in range(n):
                qc.rx(2.0 * betas[layer], i)
        qc.measure_all()
        return qc

    def execute_naive_qaoa(self, mu: np.ndarray, sigma: np.ndarray,
                           reps: int = 1, shots: int = 2048,
                           maxiter: int = 60) -> CompetitorResult:
        """Run the penalty-QAOA end to end and MEASURE its behavior.

        Selection rule replicated from the competitor spec: the most
        probable bitstring of the final sampling (their
        ``argmax``-of-distribution rule, ported to the Primitives V2 API —
        the draft's ``result.eigenstate.argmax()`` is not a valid call on
        the modern stack).  We additionally record the best *feasible*
        sample so the benchmark can grade the stack at its most charitable.
        """
        t0 = time.perf_counter()
        n = self.N
        Q, a, _ = self.build_penalty_qubo(mu, sigma)
        Q_mv = self.q * np.asarray(sigma, dtype=float)   # pure MV scorer
        a_mv = -np.asarray(mu, dtype=float)
        h, J, _ = qubo_to_ising(Q, a)
        ansatz = self.build_x_mixer_ansatz(h, J, reps=reps)
        sampler = StatevectorSampler(seed=self.seed)
        param_order = sorted(ansatz.parameters, key=lambda p: p.name)
        p = reps
        history: list[float] = []

        def bind(theta: np.ndarray) -> QuantumCircuit:
            mapping = {}
            for prm in param_order:
                vec, idx = prm.name.split("[")
                mapping[prm] = (theta[int(idx.rstrip(']'))] if vec == "γ"
                                else theta[p + int(idx.rstrip(']'))])
            return ansatz.assign_parameters(mapping)

        def counts_at(theta: np.ndarray) -> dict:
            return sampler.run([(bind(theta),)], shots=shots) \
                .result()[0].data.meas.get_counts()

        def evaluate(theta: np.ndarray) -> float:
            counts = counts_at(theta)
            tot = sum(counts.values())
            val = sum(cnt * binary_objective(_decode(b, n), Q, a)
                      for b, cnt in counts.items()) / tot
            history.append(val)                       # mean energy (textbook)
            return val

        rng = np.random.default_rng(self.seed)
        theta0 = 0.1 * rng.standard_normal(2 * p)
        theta_opt = _minimize_cobyla(evaluate, theta0, maxiter)

        counts = counts_at(theta_opt)
        tot = sum(counts.values())
        feasible = sum(cnt for b, cnt in counts.items()
                       if _decode(b, n).sum() == self.K)
        z_arg = _decode(max(counts, key=counts.get), n)
        best_feas, best_feas_val = None, np.inf
        for b, _cnt in counts.items():
            zv = _decode(b, n)
            if zv.sum() == self.K:
                fv = binary_objective(zv, Q_mv, a_mv)
                if fv < best_feas_val:
                    best_feas, best_feas_val = zv, fv

        weights = self.allocate_weights_heuristically(z_arg)
        return CompetitorResult(
            z_argmax=z_arg,
            cardinality_argmax=int(z_arg.sum()),
            cardinality_breach=abs(int(z_arg.sum()) - self.K),
            feasible_shot_fraction=feasible / tot,
            z_best_feasible=best_feas,
            objective_mv_argmax=binary_objective(z_arg, Q_mv, a_mv),
            objective_mv_best_feasible=(None if best_feas is None
                                        else float(best_feas_val)),
            weights=weights, penalty_lambda=self.lam,
            elapsed_s=time.perf_counter() - t0,
            history=history, circuit=ansatz,
        )

    # -------------------------------------------------------- allocation
    def allocate_weights_heuristically(self, z: np.ndarray) -> np.ndarray:
        """Equal weight over the selection (their rule, verbatim); uniform
        1/N book when the selection is empty."""
        z = np.asarray(z, dtype=float)
        return z / z.sum() if z.sum() > 0 else np.ones(self.N) / self.N


def _decode(bitstr: str, n: int) -> np.ndarray:
    """Little-endian Qiskit bitstring → z vector (qubit i = z_i)."""
    return np.fromiter((int(c) for c in bitstr.replace(" ", "")[::-1]),
                       dtype=int, count=n)


# ===========================================================================
# Measured critiques.
# ===========================================================================
def dbscan_quality_report(labels: np.ndarray, sectors: np.ndarray,
                          max_size: int = 20) -> dict:
    """Quantify DBSCAN's structural mismatch against the known hierarchy.

    Reports: number of clusters found, noise-point count (label −1 assets
    that DBSCAN simply refuses to place — unusable for a full-investment
    mandate), size of the largest cluster vs the qubit budget, and sector
    purity (share of each cluster's members belonging to its dominant
    sector).  These are measured numbers; the comparison table pairs them
    with the HRP plan's equivalents from
    :func:`src.hrp_clustering.cluster_report`."""
    labels = np.asarray(labels)
    sectors = np.asarray(sectors)
    ids = [c for c in np.unique(labels) if c != -1]
    sizes = [int((labels == c).sum()) for c in ids]
    purities = []
    for c in ids:
        secs = sectors[labels == c]
        _, counts = np.unique(secs, return_counts=True)
        purities.append(counts.max() / counts.sum())
    return {
        "n_clusters": len(ids),
        "n_noise_assets": int((labels == -1).sum()),
        "largest_cluster": max(sizes) if sizes else 0,
        "fits_qubit_budget": bool(sizes and max(sizes) <= max_size),
        "mean_sector_purity": float(np.mean(purities)) if purities else np.nan,
        "cluster_sizes": sizes,
    }


def penalty_lambda_sweep(
    mu: np.ndarray, sigma: np.ndarray, k: int,
    lambdas: tuple[float, ...] = (0.05, 0.5, 5.0, 50.0),
    risk_aversion: float = 5.0, reps: int = 1, shots: int = 2048,
    maxiter: int = 60, seed: int = 7,
) -> pd.DataFrame:
    """THE leakage experiment: sweep λ and measure the double bind.

    For each λ: feasible-shot fraction, argmax cardinality breach, and the
    approximation ratio of the best feasible sample against the exact
    weight-K optimum (from :func:`classical_z_exact` on the PURE
    mean-variance objective).  The structural prediction — small λ leaks
    the constraint, large λ drowns the covariance signal and degrades
    solution quality — is thereby demonstrated with measured numbers, and
    any λ that happens to work at this instance size is reported honestly.
    """
    n = len(mu)
    Q_mv = risk_aversion * np.asarray(sigma, dtype=float)
    a_mv = -np.asarray(mu, dtype=float)
    z_star, f_star = classical_z_exact(Q_mv, a_mv, k)
    # Worst feasible value for a normalized [0, 1] ratio.
    f_worst = -np.inf
    rng = np.random.default_rng(0)
    for _ in range(2000):
        z_r = np.zeros(n, dtype=int)
        z_r[rng.choice(n, k, replace=False)] = 1
        f_worst = max(f_worst, binary_objective(z_r, Q_mv, a_mv))

    rows = []
    for lam in lambdas:
        pipe = CompetitorHeuristicPipeline(n, k, penalty_lambda=lam,
                                           risk_aversion=risk_aversion, seed=seed)
        res = pipe.execute_naive_qaoa(mu, sigma, reps=reps, shots=shots,
                                      maxiter=maxiter)
        if res.objective_mv_best_feasible is not None and f_worst > f_star:
            ratio = (f_worst - res.objective_mv_best_feasible) / (f_worst - f_star)
        else:
            ratio = np.nan
        rows.append({
            "lambda": lam,
            "feasible_shot_fraction": res.feasible_shot_fraction,
            "argmax_cardinality": res.cardinality_argmax,
            "argmax_breach": res.cardinality_breach,
            "best_feasible_mv_objective": res.objective_mv_best_feasible,
            "approx_ratio_vs_exact": ratio,
            "time_s": res.elapsed_s,
        })
    df = pd.DataFrame(rows)
    df.attrs["exact_optimum"] = f_star
    df.attrs["random_worst"] = f_worst
    return df
