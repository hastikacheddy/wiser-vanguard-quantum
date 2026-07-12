"""
Classical baselines for the asset-selection layer.
==================================================

Two reference solvers against which the penalty-free QAOA-XY layer is judged:

1. **Exact MIQP** — the global optimum of the joint selection + allocation
   problem, via ``cvxpy`` with boolean variables (big-M linking):

   .. math::

       \\min_{w, z} \\quad & w^{\\top} \\Sigma w - \\lambda_{ret}\\, \\mu^{\\top} w \\\\
       \\text{s.t.} \\quad & \\mathbf{1}^{\\top} w = 1, \\quad 0 \\le w_i \\le z_i W_{max}, \\\\
                          & \\mathbf{1}^{\\top} z = K, \\quad z \\in \\{0, 1\\}^N .

   For the *selection-only* comparison (apples-to-apples with the quantum
   layer, which optimizes over binary supports), we also provide an exact
   exhaustive enumerator over the :math:`\\binom{N}{K}` feasible supports —
   tractable up to a configurable budget, and the ground truth used to
   compute QAOA approximation ratios.

2. **Simulated annealing (SA)** — the classical heuristic benchmark.
   Crucially, SA here uses the *same feasibility-preserving philosophy* as
   the quantum layer: its proposal move is a **swap** (deactivate one held
   asset, activate one unheld asset), so every visited state satisfies
   :math:`\\sum_i z_i = K` exactly.  This is the classical analogue of the
   XY-mixer's Hamming-weight conservation and makes the benchmark honest:
   neither method wastes time in the infeasible region.

Shared discrete objective (identical to the quantum cost Hamiltonian's
classical function, see :mod:`src.qaoa_xy_mixer`):

.. math::

    f(z) \\;=\\; q \\, z^{\\top} \\Sigma z \\;-\\; \\mu^{\\top} z ,
    \\qquad z \\in \\{0,1\\}^N,\\; \\textstyle\\sum_i z_i = K .
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "selection_objective",
    "SelectionBaselineResult",
    "solve_selection_exhaustive",
    "solve_selection_sa",
    "solve_joint_miqp",
    "exact_selection_baseline",
]


def selection_objective(z: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                        risk_aversion: float) -> float:
    """Discrete mean-variance score  f(z) = q z'Σz − μ'z  (lower is better).

    This is *exactly* the classical function whose Ising encoding is the
    QAOA phase Hamiltonian — so quantum and classical solvers are compared
    on the same landscape.
    """
    z = np.asarray(z, dtype=float)
    return float(risk_aversion * z @ sigma @ z - mu @ z)


@dataclass
class SelectionBaselineResult:
    """Outcome of a classical selection baseline run."""
    method: str
    z: np.ndarray
    objective: float
    elapsed_s: float
    is_global_optimum: bool
    status: str = "ok"
    history: list[float] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 1a. Exact selection by exhaustive enumeration over the feasible subspace.
# ---------------------------------------------------------------------------
def solve_selection_exhaustive(
    mu: np.ndarray, sigma: np.ndarray, k: int, risk_aversion: float,
    max_supports: int = 500_000,
) -> SelectionBaselineResult:
    """Enumerate all C(N, K) supports and return the global argmin of f(z).

    Complexity is exactly the size of the feasible subspace — the same
    subspace the XY-mixer walks — which makes this the canonical ground
    truth for approximation ratios.  Refuses to run past ``max_supports``
    (e.g. C(40, 8) ≈ 7.7e7 is enumerable offline but not interactively).
    """
    n = len(mu)
    n_supports = math.comb(n, k)
    if n_supports > max_supports:
        raise ValueError(
            f"C({n},{k}) = {n_supports:,} exceeds the enumeration budget "
            f"({max_supports:,}); use solve_joint_miqp or SA instead."
        )
    t0 = time.perf_counter()
    best_val, best_support = np.inf, None
    q = risk_aversion
    for support in itertools.combinations(range(n), k):
        idx = list(support)
        val = q * sigma[np.ix_(idx, idx)].sum() - mu[idx].sum()
        if val < best_val:
            best_val, best_support = val, idx
    z = np.zeros(n, dtype=int)
    z[best_support] = 1
    return SelectionBaselineResult(
        method="Exhaustive (exact)", z=z, objective=float(best_val),
        elapsed_s=time.perf_counter() - t0, is_global_optimum=True,
        extra={"supports_enumerated": n_supports},
    )


# ---------------------------------------------------------------------------
# 1b. Exact joint MIQP (selection + weights) via cvxpy boolean variables.
# ---------------------------------------------------------------------------
def solve_joint_miqp(
    mu: np.ndarray, sigma: np.ndarray, k: int,
    w_max: float = 0.25, return_weight: float = 1.0,
    solver_order: tuple[str, ...] = ("SCIP", "ECOS_BB"),
) -> SelectionBaselineResult:
    """Global optimum of the joint problem via branch-and-bound MIQP.

    .. math::

        \\min_{w,z}\\; w^{\\top}\\Sigma w - \\lambda\\,\\mu^{\\top} w
        \\;\\;\\text{s.t.}\\;\\;
        \\mathbf{1}^{\\top}w = 1,\\;
        0 \\le w \\le W_{max} z,\\;
        \\mathbf{1}^{\\top}z = K,\\; z\\in\\{0,1\\}^N .

    Tries solvers in ``solver_order`` (SCIP if installed, then the
    open-source ECOS_BB branch-and-bound).  Returns ``status='failed'``
    rather than raising if no MIP-capable solver is available, so the
    dashboard can degrade gracefully.
    """
    import cvxpy as cp

    n = len(mu)
    if k * w_max < 1.0 - 1e-9:
        raise ValueError(f"Infeasible: K*W_max = {k * w_max:.3f} < 1 — no valid budget allocation.")
    w = cp.Variable(n, nonneg=True)
    z = cp.Variable(n, boolean=True)
    constraints = [cp.sum(w) == 1, w <= w_max * z, cp.sum(z) == k]
    objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(sigma)) - return_weight * mu @ w)
    prob = cp.Problem(objective, constraints)

    t0 = time.perf_counter()
    last_err = "no solver attempted"
    for solver_name in solver_order:
        solver = getattr(cp, solver_name, None)
        if solver is None or solver_name not in cp.installed_solvers():
            continue
        try:
            prob.solve(solver=solver)
            if prob.status in ("optimal", "optimal_inaccurate"):
                z_val = np.round(np.asarray(z.value).ravel()).astype(int)
                return SelectionBaselineResult(
                    method=f"MIQP ({solver_name})", z=z_val,
                    objective=float(prob.value),
                    elapsed_s=time.perf_counter() - t0,
                    is_global_optimum=True, status=prob.status,
                    extra={"w": np.asarray(w.value).ravel()},
                )
            last_err = f"{solver_name}: status={prob.status}"
        except (cp.error.SolverError, Exception) as exc:  # noqa: BLE001
            last_err = f"{solver_name}: {exc}"
    return SelectionBaselineResult(
        method="MIQP", z=np.zeros(n, dtype=int), objective=np.inf,
        elapsed_s=time.perf_counter() - t0, is_global_optimum=False,
        status=f"failed ({last_err})",
    )


# ---------------------------------------------------------------------------
# 2. Simulated annealing with cardinality-preserving swap moves.
# ---------------------------------------------------------------------------
def solve_selection_sa(
    mu: np.ndarray, sigma: np.ndarray, k: int, risk_aversion: float,
    n_sweeps: int = 400, t_initial: float | None = None,
    t_final: float = 1e-4, seed: int = 7,
) -> SelectionBaselineResult:
    """Simulated annealing restricted to the Hamming-weight-K shell.

    Proposal move: swap one selected asset with one unselected asset —
    the classical counterpart of one XY-mixer excitation hop.  Uses an
    O(K) incremental Δf update instead of recomputing z'Σz from scratch:

    .. math::

        \\Delta f = q\\big[\\Sigma_{jj} - \\Sigma_{ii}
                    + 2\\textstyle\\sum_{l \\in S \\setminus \\{i\\}}
                      (\\Sigma_{lj} - \\Sigma_{li})\\big] - (\\mu_j - \\mu_i)

    for deactivating *i* and activating *j* given current support *S*.
    Geometric cooling from ``t_initial`` (auto-calibrated to the typical
    |Δf| if not given) to ``t_final``.
    """
    rng = np.random.default_rng(seed)
    n = len(mu)
    q = risk_aversion
    t0 = time.perf_counter()

    support = list(rng.choice(n, size=k, replace=False))
    in_support = np.zeros(n, dtype=bool)
    in_support[support] = True
    z = in_support.astype(float)
    current = selection_objective(z, mu, sigma, q)
    best_val, best_support = current, list(support)

    # Auto-calibrate the initial temperature from a burst of random moves.
    if t_initial is None:
        deltas = []
        for _ in range(64):
            i = support[rng.integers(k)]
            j = int(rng.choice(np.flatnonzero(~in_support)))
            row = sigma[:, j] - sigma[:, i]
            d = q * (sigma[j, j] - sigma[i, i] + 2 * (row[support].sum() - row[i])) - (mu[j] - mu[i])
            deltas.append(abs(d))
        t_initial = max(np.median(deltas), 1e-6) * 2.0

    n_iters = n_sweeps * n
    cooling = (t_final / t_initial) ** (1.0 / max(n_iters - 1, 1))
    temp = t_initial
    history = [current]

    out_pool = np.flatnonzero(~in_support)
    for _ in range(n_iters):
        pos = rng.integers(k)
        i = support[pos]
        j = int(out_pool[rng.integers(len(out_pool))])
        row = sigma[:, j] - sigma[:, i]
        delta = q * (sigma[j, j] - sigma[i, i] + 2 * (row[support].sum() - row[i])) - (mu[j] - mu[i])
        if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-12)):
            support[pos] = j
            in_support[i], in_support[j] = False, True
            out_pool[out_pool == j] = i
            current += delta
            if current < best_val - 1e-12:
                best_val, best_support = current, list(support)
        temp *= cooling
        history.append(current)

    z_best = np.zeros(n, dtype=int)
    z_best[best_support] = 1
    # Re-evaluate exactly to eliminate accumulated floating-point drift.
    best_val = selection_objective(z_best, mu, sigma, q)
    return SelectionBaselineResult(
        method="Simulated Annealing (swap moves)", z=z_best, objective=best_val,
        elapsed_s=time.perf_counter() - t0, is_global_optimum=False,
        history=history[:: max(1, n_iters // 500)],
        extra={"n_iters": n_iters, "t_initial": t_initial},
    )


def exact_selection_baseline(
    mu: np.ndarray, sigma: np.ndarray, k: int, risk_aversion: float,
    max_supports: int = 500_000,
) -> SelectionBaselineResult:
    """Best available *exact* selection baseline.

    Prefers exhaustive enumeration (guaranteed exact, no solver caveats);
    past the enumeration budget it falls back to long-run SA and flags the
    result as non-certified so the dashboard reports it honestly.
    """
    n = len(mu)
    if math.comb(n, k) <= max_supports:
        return solve_selection_exhaustive(mu, sigma, k, risk_aversion, max_supports)
    result = solve_selection_sa(mu, sigma, k, risk_aversion, n_sweeps=1200, seed=1)
    result.method = "SA (long run — exact baseline unavailable at this size)"
    return result
