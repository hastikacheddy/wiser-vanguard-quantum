# Vanguard OMEGA — ADMM-Coordinated HRP + Penalty-Free QAOA+

WISER Global Quantum+AI — Vanguard Challenge. Four integrated pillars:

1. **HRP decomposition** ([hrp_clustering.py](src/hrp_clustering.py)) — Ward
   linkage on the correlation distance carves N=100+ into ≤20-asset
   sub-problems (measured: 9 clusters, 96% sector purity), assigns
   inverse-risk capital budgets, and compiles a **degree-≤3 BFS
   maximum-correlation spanning tree per cluster** — the XY-mixer graph,
   heavy-hex-embeddable by construction (3 parallel two-qubit layers per
   mixer application).
2. **ADMM coordinator** ([admm_coordinator.py](src/admm_coordinator.py)) —
   scaled-form x→z→u loop per cluster with **adaptive ρ** (residual
   balancing, μ=10, τ=2, dual rescaled to preserve y=ρu), Boyd-style
   stopping, cardinality split across clusters, and a global convex polish
   (with support floor w ≥ w_min·z — an "exactly K names" mandate is
   unenforceable without it).
3. **Classical x-step** ([classical_x_step.py](src/classical_x_step.py)) —
   CVXPY QP: Markowitz − return + L1 transaction costs + (ρ/2)‖x−αz+u‖²,
   solver fallback chain CLARABEL→OSQP→ECOS→SCS.
4. **Quantum z-step** ([quantum_z_step.py](src/quantum_z_step.py)) — the
   ADMM anchor is *linear* in binary z (z²=z), so it folds into the Ising
   field: **zero penalty terms, zero extra circuit structure**. Dicke |D^n_K⟩
   init + tree-mixer QAOA+ (Primitives V2), CVaR objective, warm-started
   (γ,β), **ZNE via global unitary folding** (measured 6× error reduction
   under depolarizing noise), and an exact-enumeration classical fail-safe
   that keeps the pipeline moving on any quantum-path failure (logged).

**Adversarial benchmark** ([competitor_heuristic.py](src/competitor_heuristic.py)):
a *steelman* of the DBSCAN + soft-penalty-QUBO + X-mixer stack — encoding
verified exact against brute force; failure modes **measured, never
scripted**. [metrics.py](src/metrics.py) transpiles real circuits onto real
heavy-hex coupling maps (no analytic gate estimates anywhere).

## Measured results (QA suite, seed 7, N=100, K=20)

- Full pipeline: 1.3 s end-to-end (classical z-updates + quantum final
  iterations), **zero hard-guardrail breaches**, 20 positions, Sharpe(net) 0.47.
- ADMM: residuals decayed in 9/9 clusters (final primal ~1e-5), all
  converged, adaptive ρ activated in 9/9.
- Ansatz feasibility: probability leak off the weight-K shell ≈ 1e-33 at
  random parameters (symmetry, not penalty).
- Competitor at λ=0.05: **2.3% feasible shots; most probable bitstring
  selects 0 assets**. At λ=5: feasible but signal/penalty ratio 4.9e-2 —
  covariance rides on <5% of each rotation. DBSCAN at its defaults:
  **all 100 assets labeled noise**.
- Heavy-hex (SABRE, measured): OMEGA's largest circuit 1,735 routed CX;
  competitor matched-size 530 (honest context — their per-circuit cost is
  lower); competitor **monolithic n=100 (required, penalty is global):
  24,530 routed CX, SWAP tax 14,630**.

## Run

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/vanguard_dashboard.py
```

Tabs: **Allocation** (weights by cluster, breach audit, measured
head-to-head + λ-sweep) · **ADMM Convergence** (recorded residual/ρ
trajectories) · **Hardware Audit** (real transpilation + ZNE demo).

Benchmark-integrity rules: competitor implemented correctly at full
strength; every number from an actual execution; unfavorable-to-us results
(matched-size gate counts, any λ that works) reported alongside.
