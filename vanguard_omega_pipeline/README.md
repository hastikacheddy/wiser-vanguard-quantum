# Vanguard OMEGA — ADMM-Coordinated HRP + Penalty-Free QAOA+

> Scales penalty-free hybrid portfolio construction to N = 100+ via
> hierarchical decomposition, ADMM coordination, and hardware-matched
> quantum circuits.

Part of the [WISER Vanguard Challenge submission](../README.md).

## Four integrated pillars

1. **HRP decomposition** — [`src/hrp_clustering.py`](src/hrp_clustering.py)
   Ward linkage on the correlation distance carves N = 100+ into ≤ 20-asset
   sub-problems (measured: 9 clusters, 96% asset-class purity), assigns
   inverse-risk capital budgets, and compiles a **degree-≤3 BFS
   maximum-correlation spanning tree per cluster** — the XY-mixer graph,
   heavy-hex-embeddable by construction (3 parallel two-qubit layers per
   mixer application).

2. **ADMM coordinator** — [`src/admm_coordinator.py`](src/admm_coordinator.py)
   Scaled-form x → z → u loop per cluster with **adaptive ρ** (residual
   balancing, μ = 10, τ = 2, dual rescaled to preserve y = ρu), Boyd-style
   stopping, cardinality split across clusters, and a global convex polish
   (with support floor `w ≥ w_min·z` — an "exactly K names" mandate is
   unenforceable without it).

3. **Classical x-step** — [`src/classical_x_step.py`](src/classical_x_step.py)
   CVXPY QP: Markowitz − return + L1 transaction costs + `(ρ/2)‖x−αz+u‖²`,
   with a CLARABEL → OSQP → ECOS → SCS solver fallback chain and the four
   investor goals (growth, income floor, drawdown-control CVaR penalty, cost
   sensitivity).

4. **Quantum z-step** — [`src/quantum_z_step.py`](src/quantum_z_step.py)
   The ADMM anchor is *linear* in binary z (`z² = z`), so it folds into the
   Ising field: **zero penalty terms, zero extra circuit structure**. Dicke
   `|D^n_K⟩` init + tree-mixer QAOA+ (Qiskit Primitives V2), CVaR objective,
   warm-started (γ, β), **zero-noise extrapolation** via global unitary
   folding (measured 6× error reduction under depolarizing noise), and an
   exact-enumeration classical fail-safe that keeps the pipeline moving on any
   quantum-path failure (logged).

**Adversarial benchmark** — [`src/competitor_heuristic.py`](src/competitor_heuristic.py)
A *steelman* of the DBSCAN + soft-penalty-QUBO + X-mixer stack: encoding
verified exact against brute force, failure modes **measured, never
scripted**. [`src/metrics.py`](src/metrics.py) transpiles real circuits onto
real heavy-hex coupling maps — no analytic gate estimates anywhere.

## Measured results

Seed 7, N = 100, K = 20, all reproducible from code.

| Metric | Result |
|---|---|
| End-to-end pipeline (classical z + quantum final iterations) | 1.3 s |
| Hard-guardrail breaches | **0** |
| Positions / Sharpe (net) | 20 · **0.47** |
| ADMM residuals (primal, all clusters) | converged, ~10⁻⁵ (9/9) |
| Ansatz leakage off the weight-K shell (random params) | ≈ 10⁻³³ |
| Competitor, λ = 0.05 | 2.3% feasible shots; most probable = **0 assets** |
| Competitor, λ = 5 | signal-to-penalty ratio 4.9 × 10⁻² |
| DBSCAN at defaults | **all 100 assets labeled noise** |
| Largest circuit — routed CX (heavy-hex, SABRE) | **1,735** |
| Competitor monolithic n = 100 — routed CX | **24,530** (SWAP tax 14,630) |

## Run

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/vanguard_dashboard.py
```

Tabs: **Allocation** (weights by cluster, breach audit, goal metrics,
trade-offs vs. the HRP baseline, forecast-robustness stress) · **ADMM
Convergence** (recorded residual / ρ trajectories) · **Hardware Audit**
(real transpilation + ZNE demo). A data-source toggle switches between the
synthetic universe and a committed real-ETF snapshot.

## Benchmark-integrity rules

The competitor is implemented correctly and at full strength; every number
comes from an actual execution; and results unfavorable to us (matched-size
gate counts, any λ that happens to work) are reported alongside the rest.
