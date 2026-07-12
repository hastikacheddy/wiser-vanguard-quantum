"""
Classical continuous allocation layer (CVXPY).
==============================================

Phase 2 of the hybrid decomposition.  The quantum combinatorial layer emits
a binary support vector :math:`z \\in \\{0,1\\}^N` with :math:`|z| = K`;
this module solves the *convex* restriction of the portfolio problem on
that support:

.. math::

    \\min_{w} \\quad & w^{\\top} \\Sigma w \\\\
    \\text{s.t.} \\quad
      & \\mathbf{1}^{\\top} w = 1                       && \\text{(budget)} \\\\
      & w_i \\ge 0                                      && \\text{(long-only)} \\\\
      & w_i \\le z_i \\, W_{max}                        && \\text{(support gating + box)} \\\\
      & \\mu^{\\top} w \\ge R_{target}                  && \\text{(return floor)} \\\\
      & \\textstyle\\sum_{i \\in g} w_i \\le C_g \\;\\forall g && \\text{(sector caps, optional)} \\\\
      & \\|w - w_{prev}\\|_1 \\le \\tau                 && \\text{(turnover budget, optional)}

The gating constraint :math:`w_i \\le z_i W_{max}` is the hand-off contract
between the layers: assets the quantum layer did not select are *hard-zeroed*
(when :math:`z_i = 0` the box collapses to :math:`w_i = 0`), while selected
assets inherit the concentration cap :math:`W_{max}`.

Because z is fixed, this is a plain convex QP — solved to global optimality
in milliseconds by any conic solver.  This is precisely the payoff of the
decomposition: the NP-hard cardinality combinatorics live in the quantum
layer; everything continuous stays convex and certifiable.

Infeasibility policy (production behavior, not an afterthought):
  * ``K · W_max < 1``  → raised immediately with an actionable message.
  * Return floor unattainable on the given support → automatic *graceful
    relaxation*: re-solve maximizing :math:`\\mu^{\\top} w` subject to the
    remaining constraints, report the achieved return, and flag
    ``relaxed_return_target=True`` so the UI can surface the warning.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np

__all__ = ["AllocationResult", "allocate_weights", "efficient_frontier"]

_SOLVER_CHAIN = ("CLARABEL", "OSQP", "ECOS", "SCS")


@dataclass
class AllocationResult:
    """Solved continuous allocation with full diagnostics."""
    w: np.ndarray
    variance: float
    volatility: float
    expected_return: float
    status: str
    solver: str
    elapsed_s: float
    relaxed_return_target: bool = False
    messages: list[str] = field(default_factory=list)


def _solve_with_fallback(problem: cp.Problem) -> str:
    """Try the open-source conic solver chain; return the solver that worked."""
    last_exc: Exception | None = None
    for name in _SOLVER_CHAIN:
        if name not in cp.installed_solvers():
            continue
        try:
            problem.solve(solver=getattr(cp, name))
            if problem.status in ("optimal", "optimal_inaccurate"):
                return name
        except (cp.error.SolverError, Exception) as exc:  # noqa: BLE001
            last_exc = exc
    if problem.status in ("infeasible", "infeasible_inaccurate"):
        return "infeasible"
    raise RuntimeError(f"All conic solvers failed: {last_exc}")


def allocate_weights(
    z: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    w_max: float = 0.25,
    target_return: float | None = None,
    sectors: np.ndarray | None = None,
    sector_cap: float | None = None,
    w_prev: np.ndarray | None = None,
    turnover_budget: float | None = None,
) -> AllocationResult:
    """Solve the convex allocation QP on the quantum-selected support.

    Parameters
    ----------
    z : (N,) binary support from the quantum layer (exactly K ones).
    mu, sigma : annualized moments (same objects fed to the quantum layer —
        the two phases must price the same market).
    w_max : per-asset concentration cap :math:`W_{max}`.
    target_return : annualized floor :math:`R_{target}`; ``None`` disables it.
    sectors, sector_cap : integer sector labels and a uniform per-sector
        weight cap :math:`C_g`; both ``None`` disables the constraint.
    w_prev, turnover_budget : previous book and L1 turnover budget τ
        (:math:`\\|w - w_{prev}\\|_1 \\le \\tau`); both ``None`` disables it.

    Returns
    -------
    AllocationResult — globally optimal w on the support (convexity ⇒ the
    reported solution is certifiably optimal for the *given* z).
    """
    z = np.asarray(z, dtype=float).ravel()
    n = len(mu)
    k = int(round(z.sum()))
    messages: list[str] = []

    if k == 0:
        raise ValueError("Support vector z selects no assets.")
    if k * w_max < 1.0 - 1e-9:
        raise ValueError(
            f"Infeasible hand-off: K·W_max = {k}×{w_max} = {k * w_max:.3f} < 1. "
            f"Raise W_max above {1.0 / k:.3f} or increase K."
        )

    w = cp.Variable(n, nonneg=True)
    base_constraints: list[cp.Constraint] = [
        cp.sum(w) == 1,
        w <= z * w_max,               # gating: unselected ⇒ w_i ≤ 0 ⇒ w_i = 0
    ]
    if sectors is not None and sector_cap is not None:
        for g in np.unique(np.asarray(sectors)):
            base_constraints.append(cp.sum(w[np.asarray(sectors) == g]) <= sector_cap)
    if w_prev is not None and turnover_budget is not None:
        base_constraints.append(cp.norm1(w - np.asarray(w_prev)) <= turnover_budget)

    risk = cp.quad_form(w, cp.psd_wrap(sigma))
    t0 = time.perf_counter()
    relaxed = False

    constraints = list(base_constraints)
    if target_return is not None:
        constraints.append(mu @ w >= target_return)
    problem = cp.Problem(cp.Minimize(risk), constraints)
    solver = _solve_with_fallback(problem)

    if solver == "infeasible" and target_return is not None:
        # Graceful relaxation: the support cannot reach R_target under the
        # remaining constraints → report the max attainable return instead.
        messages.append(
            f"Return floor {target_return:.2%} unattainable on this support; "
            f"relaxed to maximum-achievable-return portfolio."
        )
        relaxed = True
        problem = cp.Problem(cp.Maximize(mu @ w), base_constraints)
        solver = _solve_with_fallback(problem)
    if solver == "infeasible":
        raise RuntimeError(
            "Allocation infeasible even without the return floor — check the "
            "sector caps / turnover budget against K·W_max."
        )

    w_val = np.asarray(w.value).ravel()
    w_val[w_val < 1e-10] = 0.0
    w_val = w_val / w_val.sum()       # renormalize away solver epsilon
    variance = float(w_val @ sigma @ w_val)
    return AllocationResult(
        w=w_val,
        variance=variance,
        volatility=float(np.sqrt(max(variance, 0.0))),
        expected_return=float(mu @ w_val),
        status=problem.status,
        solver=solver,
        elapsed_s=time.perf_counter() - t0,
        relaxed_return_target=relaxed,
        messages=messages,
    )


def efficient_frontier(
    z: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    w_max: float = 0.25,
    n_points: int = 25,
    **constraint_kwargs,
) -> list[AllocationResult]:
    """Trace the efficient frontier *restricted to the quantum support*.

    Sweeps the return floor from the minimum-variance portfolio's return up
    to the maximum attainable return on the support, re-solving the QP at
    each level.  Points that become infeasible near the max-return corner
    are skipped (the frontier is naturally open at that end under W_max).
    """
    lo = allocate_weights(z, mu, sigma, w_max, target_return=None, **constraint_kwargs)
    # A finite-but-unattainable floor (μ'w ≤ max μ on the simplex) triggers
    # the graceful relaxation, which returns the max-return corner portfolio.
    hi = allocate_weights(z, mu, sigma, w_max,
                          target_return=float(np.max(mu)) + 1.0, **constraint_kwargs)
    r_lo, r_hi = lo.expected_return, hi.expected_return
    results = []
    for r in np.linspace(r_lo, r_hi, n_points):
        try:
            res = allocate_weights(z, mu, sigma, w_max, target_return=float(r),
                                   **constraint_kwargs)
            if not res.relaxed_return_target:
                results.append(res)
        except RuntimeError:
            continue
    return results or [lo]
