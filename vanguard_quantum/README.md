# Vanguard Quantum — Hybrid Penalty-Free Portfolio Construction

WISER Global Quantum+AI Program 2026 (Vanguard Challenge).

A 2-phase hybrid decomposition for multi-asset portfolio construction that
**never uses a QUBO penalty term**:

1. **Quantum combinatorial layer** ([src/qaoa_xy_mixer.py](src/qaoa_xy_mixer.py)) —
   selects exactly K of N assets with a constraint-preserving QAOA:
   Dicke-state |D^N_K⟩ initialization (Bärtschi–Eidenbenz, with a documented
   shallow block-W fallback past 24 qubits) + XY **ring** mixer. Every gate
   commutes with total excitation number, so the walk is confined to the
   C(N,K)-dimensional feasible shell *by symmetry* — verified numerically to
   leak < 1e-30 probability at random parameters.
2. **Classical allocation layer** ([src/weight_allocator.py](src/weight_allocator.py)) —
   CVXPY convex QP on the selected support: budget, long-only, W_max box +
   support gating, return floor, sector caps, L1 turnover budget.

Baselines ([src/classical_miqp.py](src/classical_miqp.py)): exhaustive exact
selection, joint MIQP (SCIP/ECOS_BB), and cardinality-preserving simulated
annealing. Shared analytics in [src/metrics.py](src/metrics.py); synthetic
factor-model market in [src/data_gen.py](src/data_gen.py).

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
streamlit run app/copilot_ui.py
```

Dashboard tabs: **Allocation Copilot** (weights, restricted efficient
frontier, constraint audit) · **Classical vs. Quantum Benchmark** (objective /
Sharpe / wall time vs SA and the exact optimum) · **Architecture Audit**
(transpiled depth & CX counts vs N, and the quantitative penalty-free case).

## Simulation backends

| Universe | Backend | Notes |
|---|---|---|
| N ≤ 20 | `StatevectorSampler` (exact) | seconds per run |
| N > 20 | `AerSimulator(method="matrix_product_state")` | tensor-network fallback; N=40, p=1 runs in ~5 min on a laptop |

Weight-K post-selection is applied to counts as a safety net against MPS
bond-truncation leakage (measured survival: 100%).

## Verified properties (numerical QA)

- Dicke |D^n_k⟩ uniform over the weight-k shell to 1e-16 (n ≤ 6 exhaustive).
- Ising encoding satisfies ⟨z|H_C|z⟩ + c = q·z'Σz − μ'z for **all** basis states.
- Full ansatz at random (γ, β): probability outside the feasible shell ≈ 1e-33.
- N=10, K=4: QAOA (p=2, 40 COBYLA iters) reproduces the certified global optimum.
- N=40, K=8: MPS run selects exactly 8 assets, 100% feasible shots.
