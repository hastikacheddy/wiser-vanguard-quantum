# WISER Global Quantum+AI 2026 — Vanguard Challenge

Two hybrid quantum-classical portfolio construction pipelines built for the
Vanguard Challenge, both strictly penalty-free (no QUBO cardinality
penalties — feasibility is enforced by circuit symmetry).

## Projects

### [vanguard_quantum](vanguard_quantum/) — 2-Phase Hybrid Pipeline
Quantum asset selection (Dicke-state init + XY ring-mixer QAOA, Hamming
weight conserved by construction) followed by classical convex weight
allocation (CVXPY). Includes exact MIQP + simulated-annealing baselines and
a 3-tab Streamlit copilot dashboard. Verified up to N=40 via
matrix-product-state (tensor network) simulation.

### [vanguard_omega_pipeline](vanguard_omega_pipeline/) — ADMM-HRP-QAOA+ (OMEGA)
Scales to N=100+: Hierarchical Risk Parity decomposes the universe into
≤20-asset clusters; an ADMM coordinator (adaptive ρ) alternates between a
CVXPY x-update and a quantum z-update running penalty-free QAOA+ on
heavy-hex-embeddable (degree ≤ 3) correlation spanning-tree mixers, with
zero-noise extrapolation and an exact classical fail-safe. Ships with a
measured (never mocked) adversarial benchmark against a DBSCAN +
soft-penalty-QUBO baseline, including real heavy-hex transpilation audits.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate                      # Windows
pip install -r vanguard_omega_pipeline/requirements.txt

# OMEGA dashboard (N=100 ADMM pipeline + benchmarks)
streamlit run vanguard_omega_pipeline/app/vanguard_dashboard.py

# 2-phase pipeline dashboard
streamlit run vanguard_quantum/app/copilot_ui.py
```

Each project's README documents its architecture, the mathematics, and the
numerically verified results (Dicke-state exactness, feasible-subspace
confinement to ~1e-33, ADMM residual convergence, measured competitor
failure modes, heavy-hex gate counts).
