# Vanguard Quantum — Two-Phase Hybrid Pipeline

> Penalty-free multi-asset portfolio construction: constraint-preserving
> QAOA asset selection followed by classical convex allocation.

Part of the [WISER Vanguard Challenge submission](../README.md). This is the
compact, single-circuit demonstrator; the decomposed N = 100+ engine lives in
[`../vanguard_omega_pipeline`](../vanguard_omega_pipeline).

## Design

A two-phase hybrid decomposition that **never uses a QUBO penalty term**:

1. **Quantum combinatorial layer** — [`src/qaoa_xy_mixer.py`](src/qaoa_xy_mixer.py)
   Selects exactly K of N assets with a constraint-preserving QAOA:
   Dicke-state `|D^N_K⟩` initialization (Bärtschi–Eidenbenz, with a documented
   shallow block-W fallback past 24 qubits) and an XY **ring** mixer. Every
   gate commutes with the total-excitation operator, confining the walk to the
   `C(N,K)`-dimensional feasible shell *by symmetry* — verified to leak
   < 10⁻³⁰ probability at random parameters.

2. **Classical allocation layer** — [`src/weight_allocator.py`](src/weight_allocator.py)
   A CVXPY convex QP on the selected support: budget, long-only, `W_max` box
   with support gating, return floor, sector caps, and an L1 turnover budget.

**Baselines** ([`src/classical_miqp.py`](src/classical_miqp.py)): exhaustive
exact selection, joint MIQP (SCIP / ECOS_BB), and cardinality-preserving
simulated annealing. Shared analytics in [`src/metrics.py`](src/metrics.py);
synthetic factor-model market in [`src/data_gen.py`](src/data_gen.py).

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
streamlit run app/copilot_ui.py
```

Dashboard tabs:

- **Allocation Copilot** — weights, restricted efficient frontier, constraint audit
- **Classical vs. Quantum Benchmark** — objective / Sharpe / wall time vs. SA and the exact optimum
- **Architecture Audit** — transpiled depth and CX counts vs. N, and the quantitative penalty-free case

## Simulation backends

| Universe | Backend | Notes |
|---|---|---|
| N ≤ 20 | `StatevectorSampler` (exact) | seconds per run |
| N > 20 | `AerSimulator(method="matrix_product_state")` | tensor-network fallback; N = 40, p = 1 in ~5 min on a laptop |

Weight-K post-selection is applied to counts as a safety net against MPS
bond-truncation leakage (measured survival: 100%).

## Verified properties

| Property | Result |
|---|---|
| Dicke state uniform over the weight-k shell | to 10⁻¹⁶ (n ≤ 6, exhaustive) |
| Ising encoding reproduces `q·zᵀΣz − μᵀz` | exact, all basis states |
| Ansatz leakage outside the feasible shell (random γ, β) | ≈ 10⁻³³ |
| N = 10, K = 4 — QAOA vs. certified global optimum | reproduced exactly |
| N = 40, K = 8 — MPS run | exactly 8 assets, 100% feasible shots |
