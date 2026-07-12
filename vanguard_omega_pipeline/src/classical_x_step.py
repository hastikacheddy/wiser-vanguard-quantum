"""
ADMM x-update: the continuous convex sub-problem (CVXPY).
=========================================================

Within one HRP cluster c (n assets, capital budget :math:`b_c`), the scaled-
form ADMM x-update solves

.. math::

    x^{k+1} \\;=\\; \\arg\\min_{x \\in \\mathcal{X}}
        \\Big\\{\\; x^{\\top} \\Sigma x \\;-\\; \\lambda_{ret}\\, \\mu^{\\top} x
        \\;+\\; c_{tc}^{\\top} |x - w_{prev}|
        \\;+\\; \\tfrac{\\rho}{2}\\, \\| x - \\alpha z^{k} + u^{k} \\|_2^2 \\;\\Big\\}

over the convex feasible set

.. math::

    \\mathcal{X} = \\Big\\{ x :\\;
        \\mathbf{1}^{\\top} x = b_c,\\;
        0 \\le x \\le x^{max},\\;
        S x \\le b_c \\cdot \\text{cap}_{sec} \\Big\\},

where

* :math:`\\alpha = b_c / K_c` is the **consensus scale**: the binary z lives
  on {0,1} while x carries capital, so the consensus point is "budget spread
  over the selected support" (:math:`x \\approx \\alpha z`).  The final
  portfolio is *not* forced to equal weights — after ADMM terminates, a
  polish QP re-optimizes weights freely on the converged support (see
  :func:`src.admm_coordinator.final_polish`).
* :math:`c_{tc}^{\\top}|x - w_{prev}|` is the L1 transaction-cost drag
  (per-asset linear costs, Vanguard-style), convex and handled exactly by
  the conic solver — no smoothing.
* sector caps, when supplied, are budget-proportional
  (:math:`S x \\le b_c \\cdot cap`) — a *sufficient* condition for the global
  cap after aggregation since budgets sum to 1.  **Coordinator policy**: the
  caps are applied only at the global polish (budget = 1), NOT inside ADMM
  iterations, because HRP clusters are sector-coherent by construction and
  a sub-unit cap on a one-sector cluster is structurally infeasible.

Everything here is a plain convex QP: strong duality holds, the solution is
certifiable, and solver failures degrade along an explicit fallback chain
(CLARABEL → OSQP → ECOS → SCS) rather than crashing the ADMM loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np

__all__ = ["XStepResult", "solve_x_update"]

_SOLVER_CHAIN = ("CLARABEL", "OSQP", "ECOS", "SCS")


@dataclass
class XStepResult:
    """Solved x-update with diagnostics for the coordinator's audit trail."""
    x: np.ndarray
    objective: float
    status: str
    solver: str
    elapsed_s: float
    messages: list[str] = field(default_factory=list)


def _solve_chain(problem: cp.Problem) -> tuple[str, str]:
    """Try the open-source conic chain; return (solver_used, status)."""
    last_exc: Exception | None = None
    for name in _SOLVER_CHAIN:
        if name not in cp.installed_solvers():
            continue
        try:
            problem.solve(solver=getattr(cp, name))
            if problem.status in ("optimal", "optimal_inaccurate"):
                return name, problem.status
        except (cp.error.SolverError, Exception) as exc:  # noqa: BLE001
            last_exc = exc
    if problem.status in ("infeasible", "infeasible_inaccurate"):
        return "none", problem.status
    raise RuntimeError(f"x-update: all conic solvers failed ({last_exc})")


def solve_x_update(
    z: np.ndarray,
    u: np.ndarray,
    rho: float,
    mu: np.ndarray,
    sigma: np.ndarray,
    budget: float,
    alpha: float,
    x_max: float,
    return_weight: float = 1.0,
    tc_linear: np.ndarray | None = None,
    w_prev: np.ndarray | None = None,
    sectors: np.ndarray | None = None,
    sector_cap: float | None = None,
    gate_to_support: bool = False,
    x_min: float = 0.0,
) -> XStepResult:
    """One scaled-form ADMM x-update on a cluster (see module docstring).

    Parameters
    ----------
    z, u : (n,) current binary iterate and scaled dual variable.
    rho : ADMM penalty weight ρ (0 disables the ADMM term — used by the
        final polish QP, which reuses this exact function).
    mu, sigma : cluster-restricted moments (μ_c, Σ_cc).
    budget : cluster capital b_c (Σx = b_c).
    alpha : consensus scale α = b_c / K_c.
    x_max : per-asset box upper bound (must satisfy K_c·x_max ≥ b_c).
    return_weight : λ_ret multiplying the expected-return term.
    tc_linear, w_prev : L1 transaction-cost coefficients and previous book
        (both None disables the t-cost term).
    sectors, sector_cap : cluster-local sector labels + budget-proportional
        cap (both None disables).
    gate_to_support : if True the box becomes ``x ≤ x_max · z`` (hard support
        gating) — used by the final polish, NOT during ADMM iterations
        (gating during iterations would make the x-update blind to assets
        the z-update might still switch on).
    x_min : minimum position size on the support (``x ≥ x_min · z``, active
        only with ``gate_to_support``).  An "exactly K names" mandate needs
        this floor: without it the convex optimum may legally zero a
        selected asset, silently shrinking the held cardinality.

    Returns
    -------
    XStepResult; raises ValueError on structurally infeasible inputs.
    """
    z = np.asarray(z, dtype=float).ravel()
    u = np.asarray(u, dtype=float).ravel()
    n = len(mu)
    if not (z.shape == u.shape == (n,)) or sigma.shape != (n, n):
        raise ValueError(f"dimension mismatch: z{z.shape} u{u.shape} "
                         f"mu({n},) sigma{sigma.shape}")
    k_eff = max(int(round(z.sum())), 1)
    if gate_to_support and k_eff * x_max < budget - 1e-9:
        raise ValueError(f"gated x-update infeasible: K·x_max = {k_eff * x_max:.4f} "
                         f"< budget {budget:.4f}")
    if gate_to_support and x_min > 0 and k_eff * x_min > budget + 1e-9:
        raise ValueError(f"support floor infeasible: K·x_min = "
                         f"{k_eff * x_min:.4f} > budget {budget:.4f}")
    if x_max * n < budget - 1e-9:
        raise ValueError(f"box infeasible: n·x_max = {n * x_max:.4f} < budget {budget:.4f}")

    messages: list[str] = []
    x = cp.Variable(n, nonneg=True)
    constraints: list[cp.Constraint] = [cp.sum(x) == budget]
    if gate_to_support:
        constraints.append(x <= x_max * z)
        if x_min > 0:
            constraints.append(x >= x_min * z)
    else:
        constraints.append(x <= x_max)
    if sectors is not None and sector_cap is not None:
        sectors = np.asarray(sectors)
        for g in np.unique(sectors):
            constraints.append(cp.sum(x[sectors == g]) <= budget * sector_cap)

    objective = cp.quad_form(x, cp.psd_wrap(sigma)) - return_weight * (mu @ x)
    if tc_linear is not None and w_prev is not None:
        objective = objective + tc_linear @ cp.abs(x - np.asarray(w_prev))
    if rho > 0:
        objective = objective + (rho / 2.0) * cp.sum_squares(x - alpha * z + u)

    t0 = time.perf_counter()
    problem = cp.Problem(cp.Minimize(objective), constraints)
    solver, status = _solve_chain(problem)
    if solver == "none":
        # Budget-proportional sector caps can bind against an adversarial
        # support; retry once without the ADMM anchor to localize the cause.
        raise RuntimeError(
            f"x-update infeasible (status={status}) — check sector caps "
            f"({sector_cap}) and box ({x_max}) against budget {budget:.4f}."
        )
    x_val = np.asarray(x.value).ravel()
    x_val[x_val < 1e-12] = 0.0
    return XStepResult(
        x=x_val, objective=float(problem.value), status=status,
        solver=solver, elapsed_s=time.perf_counter() - t0, messages=messages,
    )
