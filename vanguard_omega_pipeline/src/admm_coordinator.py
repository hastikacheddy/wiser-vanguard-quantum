"""
ADMM master coordinator — the classical/quantum handshake.
==========================================================

Per HRP cluster c, the mixed-binary program

.. math::

    \\min_{x, z}\\;\\; \\phi(x) + g(z)
    \\quad \\text{s.t.} \\quad x = \\alpha z,\\;\\;
    x \\in \\mathcal{X},\\;\\; z \\in \\{0,1\\}^n,\\; |z| = K_c

(:math:`\\phi` = continuous Markowitz + t-costs on the convex set
:math:`\\mathcal{X}`; :math:`g` = discrete mean-variance;
:math:`\\alpha = b_c / K_c` the consensus scale) is attacked with
scaled-form ADMM on the augmented Lagrangian

.. math::

    \\mathcal{L}_{\\rho}(x, z, u) = \\phi(x) + g(z)
        + \\tfrac{\\rho}{2}\\,\\|x - \\alpha z + u\\|_2^2
        - \\tfrac{\\rho}{2}\\,\\|u\\|_2^2 ,

iterating (Boyd et al., *Distributed Optimization and Statistical Learning
via ADMM*, Found. Trends ML 2011 — §3.1 scaled form, §9 for the mixed-
integer heuristic caveat):

.. math::

    x^{k+1} &= \\arg\\min_{x \\in \\mathcal{X}}
        \\phi(x) + \\tfrac{\\rho}{2}\\|x - \\alpha z^{k} + u^{k}\\|^2
        && \\text{(convex QP — CVXPY)} \\\\
    z^{k+1} &= \\arg\\min_{|z| = K_c}
        g(z) + \\tfrac{\\rho}{2}\\|x^{k+1} - \\alpha z + u^{k}\\|^2
        && \\text{(QAOA+ on the tree mixer)} \\\\
    u^{k+1} &= u^{k} + x^{k+1} - \\alpha z^{k+1}
        && \\text{(scaled dual ascent)}

with residuals

.. math::

    r^{k+1} = \\|x^{k+1} - \\alpha z^{k+1}\\|_2
    \\qquad
    s^{k+1} = \\rho\\,\\alpha\\,\\|z^{k+1} - z^{k}\\|_2 .

**Adaptive ρ policy** (residual balancing, Boyd §3.4.1 — the OMEGA
correction): with :math:`\\mu = 10, \\tau = 2`,

.. math::

    \\rho_{k+1} = \\begin{cases}
        \\tau\\,\\rho_k & \\text{if } r_k > \\mu\\, s_k
            \\;\\;(\\text{primal lagging — tighten consensus})\\\\
        \\rho_k / \\tau & \\text{if } s_k > \\mu\\, r_k
            \\;\\;(\\text{dual lagging — relax consensus})\\\\
        \\rho_k & \\text{otherwise,}
    \\end{cases}

and the scaled dual is rescaled (:math:`u \\leftarrow u\\,\\rho_k /
\\rho_{k+1}`) so the underlying multiplier :math:`y = \\rho u` is preserved.
This is what keeps the loop stable when the quantum sampler returns a
suboptimal z under noise: an off-support z inflates r, ρ tightens, the next
x-update pulls capital back toward consensus rather than oscillating.

**Honesty note carried into the docs**: with a nonconvex z-set, ADMM is a
*principled heuristic* (Boyd §9.1–9.2) — monotone residual decay is not
guaranteed a priori; that is exactly why the dashboard plots the *measured*
residuals rather than asserting convergence.

**Latency policy**: the z-update runs the exact classical enumerator for
early iterations and switches to the quantum QAOA+ path for the final
``quantum_last_n`` iterations (both solve the SAME argmin — the policy
trades simulator latency, not mathematics).  Set ``quantum_last_n >=
max_iter`` to run quantum on every iteration.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from .classical_x_step import solve_x_update
from .hrp_clustering import ClusterPlan
from .quantum_z_step import ZStepResult, solve_z_update

logger = logging.getLogger("omega.admm")

__all__ = ["ADMMConfig", "ClusterSolution", "OmegaResult",
           "solve_cluster", "allocate_cardinality", "solve_universe",
           "forecast_robustness_stress"]


@dataclass
class ADMMConfig:
    """All coordinator knobs in one auditable object."""
    max_iter: int = 8
    rho0: float = 1.0
    tau: float = 2.0                    # adaptive-ρ multiplier
    mu_ratio: float = 10.0              # residual-balance threshold
    rho_min: float = 1e-3
    rho_max: float = 1e3
    eps_abs: float = 1e-4
    eps_rel: float = 1e-3
    risk_aversion: float = 5.0          # q in the discrete MV term
    mv_weight: float = 1.0              # w_mv scaling of the discrete term
    return_weight: float = 1.0          # GROWTH goal: λ_ret weight
    box_multiple: float = 2.0           # x_max = box_multiple · b_c / K_c
    sector_cap: float | None = 0.60     # asset-class cap (global polish)
    use_tcosts: bool = True
    # --- Investor goals (challenge: growth / income / drawdown / cost). ----
    # Growth = return_weight above; the three below act on the continuous
    # capital layer (the polish QP), where they are convex — the discrete
    # selection layer stays a pure risk-return backbone.
    income_floor: float | None = None   # INCOME: y'w ≥ floor (graceful relax)
    cvar_weight: float = 0.0            # DRAWDOWN CONTROL: CVaR penalty wt
    cvar_alpha: float = 0.15            # tail fraction for CVaR
    cost_multiplier: float = 1.0        # COST SENSITIVITY: scales t-costs
    # --- quantum z-update knobs -------------------------------------------
    reps: int = 1
    shots: int = 1024
    cobyla_maxiter: int = 20
    quantum_last_n: int = 2             # latency policy (see module docstring)
    quantum_time_budget_s: float = 120.0
    seed: int = 7


@dataclass
class ClusterSolution:
    """Full ADMM trajectory + outcome for one cluster."""
    cluster_id: int
    asset_idx: np.ndarray               # global indices
    z: np.ndarray                       # final binary support (local)
    x: np.ndarray                       # final continuous iterate (local)
    k: int
    budget: float
    alpha: float
    primal_residuals: list[float]
    dual_residuals: list[float]
    rho_trace: list[float]
    z_objectives: list[float]
    z_methods: list[str]
    fallbacks: int
    converged: bool
    iterations: int
    elapsed_s: float
    last_z_result: ZStepResult | None = field(default=None, repr=False)


def allocate_cardinality(budgets: np.ndarray, sizes: list[int],
                         k_total: int) -> list[int]:
    """Split the global cardinality K across clusters ∝ capital budgets.

    Largest-remainder rounding with per-cluster clamps
    :math:`1 \\le K_c \\le |c|`; the post-clamp deficit/surplus is corrected
    greedily on the largest remainders that stay within clamp bounds.
    Asserts :math:`\\sum_c K_c = K` on return.
    """
    n_clusters = len(sizes)
    if not (n_clusters <= k_total <= sum(sizes)):
        raise ValueError(f"K={k_total} incompatible with {n_clusters} clusters "
                         f"of total size {sum(sizes)}")
    raw = budgets * k_total
    k_alloc = np.maximum(np.floor(raw).astype(int), 1)
    k_alloc = np.minimum(k_alloc, sizes)
    remainders = raw - k_alloc
    while k_alloc.sum() < k_total:
        cand = [i for i in range(n_clusters) if k_alloc[i] < sizes[i]]
        i = max(cand, key=lambda j: remainders[j])
        k_alloc[i] += 1
        remainders[i] -= 1.0
    while k_alloc.sum() > k_total:
        cand = [i for i in range(n_clusters) if k_alloc[i] > 1]
        i = min(cand, key=lambda j: remainders[j])
        k_alloc[i] -= 1
        remainders[i] += 1.0
    assert k_alloc.sum() == k_total
    return k_alloc.tolist()


def solve_cluster(
    cluster_id: int,
    asset_idx: np.ndarray,
    mu_c: np.ndarray,
    sigma_c: np.ndarray,
    k_c: int,
    budget: float,
    topology,
    config: ADMMConfig,
    tc_linear: np.ndarray | None = None,
    w_prev: np.ndarray | None = None,
    sectors_c: np.ndarray | None = None,
    z_warm: np.ndarray | None = None,
) -> ClusterSolution:
    """Run the full ADMM loop (x → z → u, adaptive ρ) on one cluster.

    ``z_warm`` seeds the binary iterate (the coordinator passes the top-K_c
    HRP weights); the quantum z-update warm-starts its (γ, β) from the
    previous iteration's optimum (parameter transfer between consecutive
    z-updates, which differ only by a bounded linear-field shift).
    """
    t0 = time.perf_counter()
    n = len(mu_c)
    alpha = budget / k_c
    x_max = config.box_multiple * alpha
    if k_c * x_max < budget:
        raise ValueError("box_multiple must be >= 1 for budget feasibility")

    rho = config.rho0
    u = np.zeros(n)
    if z_warm is not None and int(np.asarray(z_warm).sum()) == k_c:
        z = np.asarray(z_warm, dtype=int).copy()
    else:
        z = np.zeros(n, dtype=int)
        z[np.argsort(-mu_c)[:k_c]] = 1       # naive high-return warm start

    Q_mv = config.mv_weight * config.risk_aversion * sigma_c
    a_mv = -config.mv_weight * mu_c

    primal_res: list[float] = []
    dual_res: list[float] = []
    rho_trace: list[float] = []
    z_objs: list[float] = []
    z_methods: list[str] = []
    fallbacks = 0
    warm_params = None
    converged = False
    last_zres: ZStepResult | None = None
    it = 0

    for it in range(1, config.max_iter + 1):
        # ---- Step 1: x-update (convex QP). --------------------------------
        xres = solve_x_update(
            z=z, u=u, rho=rho, mu=mu_c, sigma=sigma_c, budget=budget,
            alpha=alpha, x_max=x_max, return_weight=config.return_weight,
            tc_linear=tc_linear if config.use_tcosts else None,
            w_prev=w_prev if config.use_tcosts else None,
            sectors=sectors_c, sector_cap=config.sector_cap,
        )
        x = xres.x

        # ---- Step 2: z-update (quantum QAOA+ / exact classical). ----------
        #   (ρ/2)||v − αz||², v = x + u  →  linear coeff (ρα/2)(α − 2 v_i).
        v = x + u
        a_admm = (rho * alpha / 2.0) * (alpha - 2.0 * v)
        use_quantum = it > config.max_iter - config.quantum_last_n
        zres = solve_z_update(
            Q=Q_mv, a=a_mv + a_admm, k=k_c, topology=topology,
            reps=config.reps, shots=config.shots,
            maxiter=config.cobyla_maxiter, seed=config.seed + it,
            warm_params=warm_params, use_quantum=use_quantum,
            quantum_time_budget_s=config.quantum_time_budget_s,
        )
        z_new = zres.z
        if zres.params is not None:
            warm_params = zres.params
        if zres.fallback_used:
            fallbacks += 1
        last_zres = zres
        z_objs.append(zres.objective)
        z_methods.append(zres.method + ("*" if zres.fallback_used else ""))

        # ---- Step 3: dual update + residuals. ------------------------------
        r = float(np.linalg.norm(x - alpha * z_new))
        s = float(rho * alpha * np.linalg.norm(z_new - z))
        u = u + x - alpha * z_new
        primal_res.append(r)
        dual_res.append(s)
        rho_trace.append(rho)
        z = z_new

        # ---- Stopping: Boyd §3.3.1 (absolute + relative tolerances). -------
        eps_pri = np.sqrt(n) * config.eps_abs + config.eps_rel * max(
            np.linalg.norm(x), alpha * np.linalg.norm(z))
        eps_dual = np.sqrt(n) * config.eps_abs + config.eps_rel * rho * np.linalg.norm(u)
        if r < eps_pri and s < eps_dual:
            converged = True
            # Quantum confirmation: if convergence arrived before the
            # latency policy's quantum window, run ONE QAOA+ z-update on the
            # converged sub-problem so every cluster exercises the hardware
            # track.  The exact enumerator is optimal at ≤20 assets, so the
            # quantum result can only tie it — it is recorded in the method
            # trail ("(confirm)") and the converged support is never
            # degraded (adopted only on tie/improvement).
            if (config.quantum_last_n > 0
                    and not any(m.startswith("qaoa") for m in z_methods)):
                zq = solve_z_update(
                    Q=Q_mv, a=a_mv + a_admm, k=k_c, topology=topology,
                    reps=config.reps, shots=config.shots,
                    maxiter=config.cobyla_maxiter, seed=config.seed + 1000,
                    warm_params=warm_params, use_quantum=True,
                    quantum_time_budget_s=config.quantum_time_budget_s,
                )
                if zq.fallback_used:
                    fallbacks += 1
                z_objs.append(zq.objective)
                z_methods.append(zq.method + "(confirm)")
                last_zres = zq
            break

        # ---- Adaptive ρ (residual balancing) + scaled-dual rescale. --------
        if r > config.mu_ratio * s:
            rho_new = min(rho * config.tau, config.rho_max)
        elif s > config.mu_ratio * r:
            rho_new = max(rho / config.tau, config.rho_min)
        else:
            rho_new = rho
        if rho_new != rho:
            u = u * (rho / rho_new)          # preserve y = ρu
            rho = rho_new

    return ClusterSolution(
        cluster_id=cluster_id, asset_idx=np.asarray(asset_idx), z=z, x=x,
        k=k_c, budget=budget, alpha=alpha,
        primal_residuals=primal_res, dual_residuals=dual_res,
        rho_trace=rho_trace, z_objectives=z_objs, z_methods=z_methods,
        fallbacks=fallbacks, converged=converged, iterations=it,
        elapsed_s=time.perf_counter() - t0, last_z_result=last_zres,
    )


@dataclass
class OmegaResult:
    """Universe-level outcome: final book + per-cluster ADMM provenance."""
    weights: np.ndarray                  # (N,) final polished weights
    support: np.ndarray                  # (N,) binary union support
    k_total: int
    cluster_solutions: list[ClusterSolution]
    polish_status: str
    elapsed_s: float
    income_floor_relaxed: bool = False
    polish_messages: list[str] = field(default_factory=list)

    @property
    def total_fallbacks(self) -> int:
        return sum(cs.fallbacks for cs in self.cluster_solutions)


def solve_universe(
    md,
    plan: ClusterPlan,
    k_total: int = 20,
    w_max_global: float = 0.10,
    w_min_global: float = 0.01,
    config: ADMMConfig | None = None,
    scenario_matrix: np.ndarray | None = None,
) -> OmegaResult:
    """Full pipeline: cardinality split → per-cluster ADMM → global polish.

    The polish step re-solves the FREE convex allocation on the union of the
    converged supports against the full covariance (cross-cluster risk
    restored), with global sector caps and t-costs — reusing the x-update QP
    with ρ = 0 and hard support gating.  This removes the equal-weight bias
    of the consensus scale α from the final book.  ``w_min_global`` floors
    every held name (``w ≥ w_min·z``): an "exactly K positions" mandate is
    only enforceable with a minimum position size, otherwise the convex
    optimum may legally zero a selected asset (caught by QA, kept fixed).
    """
    t0 = time.perf_counter()
    config = config or ADMMConfig()
    n = md.n_assets
    sizes = [len(c) for c in plan.clusters]
    k_alloc = allocate_cardinality(plan.budgets, sizes, k_total)

    solutions: list[ClusterSolution] = []
    for c_id, (cl, k_c, b_c, topo) in enumerate(
            zip(plan.clusters, k_alloc, plan.budgets, plan.topologies)):
        w_hrp_c = plan.hrp_benchmark_weights[cl]
        z_warm = np.zeros(len(cl), dtype=int)
        z_warm[np.argsort(-w_hrp_c)[:k_c]] = 1
        # Sector caps are deliberately NOT imposed inside clusters: HRP
        # clusters are sector-coherent by construction (~96% purity in QA),
        # so an intra-cluster cap below 100% of the cluster budget is
        # structurally infeasible for a one-sector cluster.  The global cap
        # is enforced where it is meaningful — at the final polish, whose
        # support spans clusters (and hence sectors).
        sol = solve_cluster(
            cluster_id=c_id, asset_idx=cl, mu_c=md.mu[cl],
            sigma_c=md.sigma[np.ix_(cl, cl)], k_c=k_c, budget=float(b_c),
            topology=topo, config=config,
            tc_linear=md.tc_linear[cl] * config.cost_multiplier,
            w_prev=md.w_prev[cl],
            sectors_c=None, z_warm=z_warm,
        )
        logger.info("cluster %d: %d iters, converged=%s, fallbacks=%d",
                    c_id, sol.iterations, sol.converged, sol.fallbacks)
        solutions.append(sol)

    support = np.zeros(n, dtype=int)
    for sol in solutions:
        support[sol.asset_idx[sol.z == 1]] = 1
    assert int(support.sum()) == k_total, \
        f"support {int(support.sum())} != K {k_total}"

    # ---- Global polish: free convex weights on the union support. ---------
    polish = solve_x_update(
        z=support, u=np.zeros(n), rho=0.0, mu=md.mu, sigma=md.sigma,
        budget=1.0, alpha=1.0 / k_total, x_max=w_max_global,
        return_weight=config.return_weight,
        tc_linear=(md.tc_linear * config.cost_multiplier
                   if config.use_tcosts else None),
        w_prev=md.w_prev if config.use_tcosts else None,
        sectors=md.sectors,
        sector_cap=config.sector_cap,
        gate_to_support=True,
        x_min=w_min_global,
        yields=getattr(md, "asset_yields", None),
        income_floor=config.income_floor,
        scenario_matrix=scenario_matrix,
        cvar_weight=config.cvar_weight,
        cvar_alpha=config.cvar_alpha,
    )
    weights = polish.x / polish.x.sum()
    return OmegaResult(
        weights=weights, support=support, k_total=k_total,
        cluster_solutions=solutions, polish_status=polish.status,
        elapsed_s=time.perf_counter() - t0,
        income_floor_relaxed=polish.relaxed_income_floor,
        polish_messages=list(polish.messages),
    )


def forecast_robustness_stress(
    md,
    plan: ClusterPlan,
    k_total: int = 20,
    w_max_global: float = 0.10,
    config: ADMMConfig | None = None,
    scenario_matrix: np.ndarray | None = None,
    n_trials: int = 5,
    noise_scale: float = 0.20,
    seed: int = 123,
):
    """Guardrail robustness under forecast error: μ is an INPUT assumption
    (per the challenge statement), so the defensible question is not "is μ
    right?" but "what breaks when μ is wrong?".

    Each trial perturbs expected returns multiplicatively,
    :math:`\\mu' = \\mu \\odot (1 + \\eta\\,\\varepsilon)`,
    :math:`\\varepsilon \\sim \\mathcal{N}(0, I)` (default η = 20%), re-runs
    the full pipeline on the perturbed beliefs, then grades the resulting
    book against the *unperturbed* assumptions.  Reported per trial:

    * hard-guardrail breach count — the structural claim is that this stays
      **0 for every trial**, because feasibility is enforced by symmetry
      and convex constraints, never by the forecast;
    * Sharpe under base assumptions — quantifies graceful quality decay;
    * support overlap with the base book — selection stability.

    Runs with ``quantum_last_n = 0`` (exact classical z-updates): the same
    argmin at these cluster sizes, and robustness of the *pipeline
    contract* is what is being measured, not sampler variance.
    Returns a pandas DataFrame (trial 0 = unperturbed base).
    """
    import dataclasses

    import pandas as pd

    from .metrics import compute_metrics, constraint_breach_audit, total_breaches

    config = config or ADMMConfig()
    fast_cfg = dataclasses.replace(config, quantum_last_n=0)
    rng = np.random.default_rng(seed)

    def _grade(res, label):
        m = compute_metrics(res.weights, md.mu, md.sigma, md.risk_free_rate,
                            md.tc_linear, md.w_prev)
        audit = constraint_breach_audit(
            res.weights, k_target=k_total, w_max=w_max_global,
            sectors=md.sectors, sector_cap=fast_cfg.sector_cap)
        return {"trial": label,
                "sharpe_under_base_mu": m["sharpe_net"],
                "volatility": m["volatility"],
                "hard_breaches": total_breaches(audit),
                "positions": m["num_positions"]}

    base = solve_universe(md, plan, k_total=k_total,
                          w_max_global=w_max_global, config=fast_cfg,
                          scenario_matrix=scenario_matrix)
    rows = [dict(_grade(base, "base (unperturbed)"), support_overlap=1.0)]
    for t in range(1, n_trials + 1):
        mu_p = md.mu * (1.0 + noise_scale * rng.standard_normal(md.n_assets))
        md_p = dataclasses.replace(md, mu=mu_p)
        res = solve_universe(md_p, plan, k_total=k_total,
                             w_max_global=w_max_global, config=fast_cfg,
                             scenario_matrix=scenario_matrix)
        overlap = float((res.support & base.support).sum()) / k_total
        rows.append(dict(_grade(res, f"μ ± {noise_scale:.0%} #{t}"),
                         support_overlap=overlap))
    return pd.DataFrame(rows)
