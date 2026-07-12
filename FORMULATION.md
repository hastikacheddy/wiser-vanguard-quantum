# Mathematical Formulation — Multi-Asset Portfolio Construction

Consolidated problem statement for the WISER Vanguard Challenge submission.
Every equation below is implemented verbatim in the referenced module and
verified numerically in the QA suites.

## 1. Decision variables

| Symbol | Type | Meaning |
|---|---|---|
| `z ∈ {0,1}^N` | **binary** | asset selection (`z_i = 1` ⇔ asset *i* held) |
| `w ∈ R^N` | continuous | capital weights |

## 2. Master problem (quadratic objective, linear constraints)

```
min_{w,z}   w'Σw  −  λ_ret μ'w  +  c_tc'|w − w_prev|  +  w_cvar · CVaR_α(−Rw)
subject to  1'w = 1                      (budget — linear)
            0 ≤ w ≤ W_max · z            (long-only, box, support gating — linear)
            w ≥ w_min · z                (min position: "exactly K names" — linear)
            1'z = K                      (cardinality — linear, the hard part)
            S_class · w ≤ c_cap          (asset-class caps — linear)
            y'w ≥ y_floor                (income goal — linear)
            ‖w − w_prev‖₁ ≤ τ            (turnover budget — linear via lift)
```

- Quadratic terms: risk `w'Σw`; after selection restriction, the discrete
  score `q z'Σz − μ'z` (implemented: `classical_x_step.py`, `quantum_z_step.py`).
- The four **investor goals** map to: growth = `λ_ret`; income = `y_floor`;
  drawdown control = `w_cvar` on the Rockafellar–Uryasev CVaR of monthly
  stress-scenario losses `−Rw` (convex LP lift with auxiliary `ζ`, slack `u ≥ 0`:
  `CVaR_α = ζ + (αS)⁻¹ Σ_s max(−(Rw)_s − ζ, 0)`); cost sensitivity = a
  multiplier on `c_tc`.
- Mixed-binary + quadratic + cardinality ⇒ NP-hard; classical exact solvers
  scale exponentially in the branch-and-bound tree.

## 3. Decomposition (HRP) and coordination (ADMM)

Ward-linkage hierarchical clustering on `d_ij = √((1−ρ_ij)/2)` partitions the
universe into clusters of ≤ 20 assets with capital budgets `b_c` ∝ inverse
cluster variance; the global `K` splits as `K_c ∝ b_c` (`hrp_clustering.py`,
`admm_coordinator.py`). Per cluster, scaled-form ADMM on consensus `x = αz`,
`α = b_c/K_c`:

```
x^{k+1} = argmin_{x∈X}  φ(x) + (ρ/2)‖x − αz^k + u^k‖²      (convex QP)
z^{k+1} = argmin_{|z|=K_c}  g(z) + (ρ/2)‖x^{k+1} − αz + u^k‖²  (quantum)
u^{k+1} = u^k + x^{k+1} − αz^{k+1}
```

with residual-balancing adaptive ρ (μ=10, τ=2; dual rescaled to preserve
y = ρu) and Boyd §3.3 stopping. A final convex polish re-optimizes free
weights on the union support against the full Σ with all goals active.

## 4. Quantum-compatible derivation (penalty-free)

Substituting `z_i = (1 − Z_i)/2` into any binary-quadratic `f(z) = z'Qz + a'z`:

```
H_C = Σ_i h_i Z_i + Σ_{i<j} J_ij Z_i Z_j + const
J_ij = Q_ij/2,   h_i = −(a_i + Q_ii)/2 − ½ Σ_{j≠i} Q_ij
```

For the z-update, `Q = w_mv q Σ_cc` and `a = −w_mv μ_c + a_ADMM` where the
ADMM anchor is **exactly linear** in z (`z_i² = z_i`):
`(ρ/2)‖v − αz‖² = Σ_i (ρα/2)(α − 2v_i) z_i + const`, `v = x + u`.

**No cardinality penalty appears anywhere.** The constraint `1'z = K` is
enforced by symmetry: the circuit starts in the Dicke state `|D^n_K⟩`
(uniform superposition of the weight-K shell, deterministic
Bärtschi–Eidenbenz preparation) and every gate — RZ/RZZ phase separators and
each XY-mixer block `exp[−iβ(X_iX_j + Y_iY_j)/2]` on the HRP-compiled,
degree-≤3 BFS spanning tree — commutes with the total-excitation operator
`n̂ = Σ_i (I − Z_i)/2`. Hence the state stays in the `C(n,K)`-dimensional
feasible subspace for **all** parameter values, Trotterization included
(measured leakage at random parameters: ~10⁻³³).

Contrast (implemented as a full-strength benchmark in
`competitor_heuristic.py`): the textbook soft-penalty QUBO
`f_λ(z) = q z'Σz − μ'z + λ(1'z − K)²` ⇒ `Q = qΣ + λ11'` adds the uniform
coupling λ/2 to every qubit pair (graph completion), requires
`λ ≳ range(f)` (signal-to-penalty ≈ 5×10⁻² measured at λ=5), cannot be
decomposed (the penalty couples all N assets), and leaks feasibility under
the X mixer (measured: 2.3% feasible shots at λ=0.05).

## 5. Validation

Classical validation routines (`classical_miqp.py`, `quantum_z_step.classical_z_exact`):
exhaustive enumeration of the C(N,K) shell (certified optimum), joint big-M
MIQP (branch-and-bound), swap-move simulated annealing, and the full-universe
HRP allocator. Measured at the tested sizes: QAOA approximation ratio 1.0000
against the certified optimum; final books carry zero hard-constraint
breaches (structured audit in `metrics.py`).
