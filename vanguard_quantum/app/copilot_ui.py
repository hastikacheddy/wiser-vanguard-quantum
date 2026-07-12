"""
Vanguard Challenge — Allocation Copilot dashboard.
==================================================

Three-tab Streamlit front-end over the hybrid penalty-free pipeline:

  Tab 1 · Allocation Copilot        — final weights, restricted efficient
                                      frontier, risk metrics, constraint audit.
  Tab 2 · Classical vs. Quantum     — QAOA-XY vs simulated annealing vs exact
                                      baseline: objective, Sharpe, wall time.
  Tab 3 · Architecture Audit        — circuit depth / gate counts vs N, and
                                      the quantitative case for penalty-free
                                      (feasible-subspace vs Hilbert-space
                                      dimension, energy-scale inflation).

Run with:
    streamlit run app/copilot_ui.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classical_miqp import (  # noqa: E402
    exact_selection_baseline, selection_objective, solve_selection_sa,
)
from src.data_gen import generate_market_data  # noqa: E402
from src.metrics import (  # noqa: E402
    approximation_ratio, benchmark_table, compute_metrics, constraint_report,
)
from src.qaoa_xy_mixer import (  # noqa: E402
    QAOAXYSelector, build_qaoa_ansatz, circuit_resource_report, penalty_free_audit,
)
from src.weight_allocator import allocate_weights, efficient_frontier  # noqa: E402

st.set_page_config(page_title="Vanguard Quantum Allocation Copilot",
                   page_icon="⚛️", layout="wide")

ACCENT = "#7A3FF2"


# ---------------------------------------------------------------- sidebar
st.sidebar.title("⚛️ Pipeline Controls")
st.sidebar.caption("Hybrid penalty-free decomposition — WISER 2026")

seed = st.sidebar.number_input("Random seed", 0, 9999, 7)
n_assets = st.sidebar.slider("Universe size N", 6, 40, 16,
                             help="N ≤ 20 samples exactly (statevector); larger N "
                                  "switches to the tensor-network (MPS) fallback.")
k_assets = st.sidebar.slider("Cardinality K", 2, min(12, n_assets - 1),
                             min(5, n_assets - 1))
risk_aversion = st.sidebar.slider("Risk aversion q", 0.5, 20.0, 5.0, 0.5)
w_max = st.sidebar.slider("Per-asset cap W_max", 0.05, 1.0, 0.35, 0.05)
target_return = st.sidebar.slider("Return floor R* (annual)", 0.0, 0.20, 0.08, 0.01)
sector_cap = st.sidebar.slider("Sector cap", 0.10, 1.0, 0.60, 0.05)
turnover_budget = st.sidebar.slider("Turnover budget τ (L1)", 0.1, 2.0, 2.0, 0.1,
                                    help="2.0 = unconstrained (full rebalance).")

st.sidebar.divider()
st.sidebar.subheader("Quantum layer (QAOA-XY)")
reps = st.sidebar.slider("Depth p (layers)", 1, 5, 2)
shots = st.sidebar.select_slider("Shots / evaluation", [512, 1024, 2048, 4096], 1024)
maxiter = st.sidebar.slider("COBYLA iterations", 10, 150, 40)
init_mode = st.sidebar.selectbox("Feasible initial state", ["auto", "dicke", "block_w"],
                                 help="auto: exact Dicke ≤ 24 qubits, block-W beyond "
                                      "(documented depth trade-off).")

if k_assets * w_max < 1.0:
    st.sidebar.error(f"K·W_max = {k_assets * w_max:.2f} < 1 — allocation infeasible. "
                     f"Raise W_max or K.")

run_clicked = st.sidebar.button("▶ Run hybrid pipeline", type="primary",
                                use_container_width=True,
                                disabled=k_assets * w_max < 1.0)


@st.cache_data(show_spinner=False)
def _market(n: int, s: int):
    return generate_market_data(n_assets=n, seed=s)


md = _market(n_assets, int(seed))


def _run_pipeline():
    selector = QAOAXYSelector(reps=reps, shots=int(shots), maxiter=maxiter,
                              init_mode=init_mode, seed=int(seed))
    with st.spinner("Phase 1 — QAOA-XY combinatorial selection "
                    "(feasible-subspace walk)…"):
        sel = selector.select(md.mu, md.sigma, k_assets, risk_aversion)
    with st.spinner("Phase 2 — convex allocation on the quantum support…"):
        alloc = allocate_weights(
            sel.z, md.mu, md.sigma, w_max=w_max, target_return=target_return,
            sectors=md.sectors, sector_cap=sector_cap,
            w_prev=md.w_prev, turnover_budget=turnover_budget,
        )
        frontier = efficient_frontier(sel.z, md.mu, md.sigma, w_max=w_max,
                                      sectors=md.sectors, sector_cap=sector_cap)
    st.session_state["sel"] = sel
    st.session_state["alloc"] = alloc
    st.session_state["frontier"] = frontier
    st.session_state["run_config"] = dict(n=n_assets, k=k_assets, q=risk_aversion,
                                          seed=int(seed))


if run_clicked:
    _run_pipeline()

st.title("Multi-Asset Portfolio Construction — Quantum Allocation Copilot")
st.caption(
    "Phase 1: penalty-free QAOA (Dicke init + XY ring mixer) walks *only* the "
    f"C({n_assets},{k_assets}) = {math.comb(n_assets, k_assets):,} feasible supports. "
    "Phase 2: CVXPY solves the convex weighting on the selected support."
)

tab1, tab2, tab3 = st.tabs(
    ["🧭 Allocation Copilot", "⚔️ Classical vs. Quantum Benchmark", "🔬 Architecture Audit"]
)

# ===========================================================================
# TAB 1 — Allocation Copilot
# ===========================================================================
with tab1:
    if "alloc" not in st.session_state:
        st.info("Configure the pipeline in the sidebar and press **Run hybrid pipeline**.")
    else:
        sel = st.session_state["sel"]
        alloc = st.session_state["alloc"]
        frontier = st.session_state["frontier"]
        pm = compute_metrics(alloc.w, md.mu, md.sigma, md.risk_free_rate)

        if alloc.relaxed_return_target:
            st.warning(" ".join(alloc.messages))

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Sharpe", f"{pm.sharpe:.2f}")
        c2.metric("Exp. return", f"{pm.expected_return:.2%}")
        c3.metric("Volatility", f"{pm.volatility:.2%}")
        c4.metric("Effective # bets", f"{pm.effective_num_bets:.1f}")
        c5.metric("Quantum backend", sel.backend,
                  help=f"init={sel.init_mode}, p={sel.reps}, "
                       f"feasible shots={sel.feasible_fraction:.1%}")

        left, right = st.columns([3, 2])
        with left:
            idx = np.argsort(-alloc.w)
            active = [i for i in idx if alloc.w[i] > 1e-6]
            fig = go.Figure(go.Bar(
                x=[md.asset_names[i] for i in active],
                y=[alloc.w[i] for i in active],
                marker_color=ACCENT,
                text=[f"{alloc.w[i]:.1%}" for i in active], textposition="outside",
            ))
            fig.add_hline(y=w_max, line_dash="dot", line_color="crimson",
                          annotation_text="W_max")
            fig.update_layout(title="Final capital weights (quantum support ∩ convex optimum)",
                              yaxis_tickformat=".0%", height=420, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with right:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[r.volatility for r in frontier],
                y=[r.expected_return for r in frontier],
                mode="lines+markers", name="Frontier (on support)",
                line=dict(color=ACCENT, width=3),
            ))
            fig.add_trace(go.Scatter(
                x=[pm.volatility], y=[pm.expected_return], mode="markers",
                name="Chosen portfolio",
                marker=dict(size=14, color="crimson", symbol="star"),
            ))
            fig.update_layout(title="Efficient frontier restricted to quantum support",
                              xaxis_title="Volatility", yaxis_title="Expected return",
                              xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                              height=420, legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Constraint compliance audit")
        breaches = constraint_report(
            alloc.w, sel.z, md.mu, k=k_assets, w_max=w_max,
            target_return=None if alloc.relaxed_return_target else target_return,
            sectors=md.sectors, sector_cap=sector_cap,
            w_prev=md.w_prev, turnover_budget=turnover_budget,
        )
        audit_df = pd.DataFrame([{
            "Constraint": b.name, "Measured": f"{b.value:.4f}",
            "Bound": f"{b.bound:.4f}", "Status": "✅ PASS" if b.ok else "❌ BREACH",
            "Note": b.detail,
        } for b in breaches])
        st.dataframe(audit_df, use_container_width=True, hide_index=True)

# ===========================================================================
# TAB 2 — Classical vs Quantum benchmark
# ===========================================================================
with tab2:
    st.markdown(
        "Head-to-head on the **identical discrete landscape** "
        r"$f(z) = q\,z^\top\Sigma z - \mu^\top z$ over $\{z : |z| = K\}$. "
        "The exact baseline certifies the global optimum; approximation "
        "ratios are normalized so 1.0 = optimal."
    )
    if st.button("⚔️ Run benchmark suite", type="primary"):
        rows: dict[str, dict] = {}

        with st.spinner("Exact baseline (exhaustive / MIQP)…"):
            exact = exact_selection_baseline(md.mu, md.sigma, k_assets, risk_aversion)
        with st.spinner("Simulated annealing (cardinality-preserving swaps)…"):
            sa = solve_selection_sa(md.mu, md.sigma, k_assets, risk_aversion,
                                    seed=int(seed))
        with st.spinner("QAOA-XY (penalty-free feasible-subspace walk)…"):
            selector = QAOAXYSelector(reps=reps, shots=int(shots), maxiter=maxiter,
                                      init_mode=init_mode, seed=int(seed))
            qsel = selector.select(md.mu, md.sigma, k_assets, risk_aversion,
                                   compute_resources=False)

        f_opt = exact.objective
        for name, z, f_val, t_s, is_exact in [
            (exact.method, exact.z, exact.objective, exact.elapsed_s, exact.is_global_optimum),
            ("Simulated Annealing", sa.z, sa.objective, sa.elapsed_s, False),
            (f"QAOA-XY (p={reps}, {qsel.backend})", qsel.z, qsel.objective,
             qsel.elapsed_s, False),
        ]:
            alloc_b = allocate_weights(z, md.mu, md.sigma, w_max=w_max)
            pm_b = compute_metrics(alloc_b.w, md.mu, md.sigma, md.risk_free_rate)
            rows[name] = dict(
                objective=f_val,
                approx_ratio=approximation_ratio(f_val, f_opt) if exact.is_global_optimum else np.nan,
                sharpe=pm_b.sharpe, expected_return=pm_b.expected_return,
                volatility=pm_b.volatility, time_s=t_s, exact=is_exact,
            )
        st.session_state["bench"] = benchmark_table(rows)
        st.session_state["bench_traces"] = {"sa": sa.history, "qaoa": qsel.history}

    if "bench" in st.session_state:
        df = st.session_state["bench"]
        st.dataframe(
            df.style.format({
                "objective": "{:.5f}", "approx_ratio": "{:.4f}", "sharpe": "{:.3f}",
                "expected_return": "{:.2%}", "volatility": "{:.2%}", "time_s": "{:.3f}s",
            }),
            use_container_width=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Bar(
                x=df.index, y=df["sharpe"], marker_color=[ACCENT, "#888", "#2ca02c"],
                text=[f"{v:.2f}" for v in df["sharpe"]], textposition="outside",
            ))
            fig.update_layout(title="Post-allocation Sharpe by selection method", height=380)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            traces = st.session_state["bench_traces"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=traces["qaoa"], mode="lines",
                                     name="QAOA-XY (CVaR cost / iteration)",
                                     line=dict(color=ACCENT)))
            fig.add_trace(go.Scatter(y=traces["sa"], mode="lines",
                                     name="SA (energy / sampled step)",
                                     line=dict(color="#888")))
            fig.update_layout(title="Optimizer convergence traces",
                              xaxis_title="Iteration (method-native scale)",
                              yaxis_title="Objective f(z)", height=380,
                              legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Wall-times are simulator times — they benchmark *solution quality "
            "per oracle model*, not hardware speed. The QAOA column inherits "
            "exponential-simulation overhead that vanishes on real QPUs."
        )

# ===========================================================================
# TAB 3 — Architecture audit
# ===========================================================================
with tab3:
    st.subheader("Why penalty-free: the quantitative case")
    audit = penalty_free_audit(md.mu, md.sigma, k_assets, risk_aversion)

    c1, c2, c3 = st.columns(3)
    c1.metric("Feasible subspace C(N,K)", f"{audit['feasible_dim']:,}")
    c2.metric("Full Hilbert space 2^N", f"{audit['hilbert_dim']:,}")
    c3.metric("Feasible fraction", f"{audit['feasible_fraction']:.2e}",
              help="Share of a penalty-QAOA walk that is not provably wasted.")

    st.latex(r"""
        H_{QUBO} = \underbrace{q\,z^\top \Sigma z - \mu^\top z}_{\text{signal}}
        \;+\; \underbrace{\lambda\Big(\textstyle\sum_i z_i - K\Big)^2}_{\text{penalty
        — absent in our formulation}}
    """)
    st.markdown(
        f"""
A correct penalty needs $\\lambda$ above the objective's spread — here
$\\lambda \\gtrsim {audit['lambda_required']:.2f}$ — and after the Ising map the
penalty injects a **uniform coupling $\\lambda/2$ on every one of the
{audit['penalty_edge_count']} qubit pairs**. The largest genuine covariance
coupling is only $|J|_{{max}} = {audit['max_cost_coupling']:.4f}$, i.e. a
**signal-to-penalty ratio of {audit['signal_to_penalty']:.2e}**: the market
structure is drowned by constraint bookkeeping ("graph collapse"), and the
physical rotation angles encoding the actual objective shrink below realistic
NISQ pulse resolution.

Our architecture deletes that term entirely. Feasibility is enforced by
**symmetry**: the Dicke initial state has Hamming weight exactly $K$, and
every gate — RZ/RZZ phase separators and each XY-mixer block
$e^{{-i\\beta(X_iX_j + Y_iY_j)/2}}$ — commutes with the total-excitation
operator $\\hat n = \\sum_i (I - Z_i)/2$. The walk is confined to the
$\\binom{{N}}{{K}}$-dimensional shell **by construction, even under
Trotterization and for any parameter values** — there is no penalty weight
to mistune and zero amplitude ever leaks into infeasible states.
        """
    )
    st.latex(r"""
        [\,e^{-i\beta(X_iX_j+Y_iY_j)/2},\ \hat n\,] = 0
        \qquad\Longrightarrow\qquad
        |\psi(\gamma,\beta)\rangle \in
        \mathrm{span}\{\,|z\rangle : |z| = K\,\}\ \ \forall\,\gamma,\beta .
    """)

    st.divider()
    st.subheader("NISQ resource accounting (transpiled to {RZ, SX, X, CX})")
    col_a, col_b = st.columns([1, 2])
    with col_a:
        if "sel" in st.session_state and st.session_state["sel"].resources:
            res = st.session_state["sel"].resources
            st.metric("Qubits", res["num_qubits"])
            st.metric("Transpiled depth", res["transpiled_depth"])
            st.metric("Two-qubit (CX) gates", res["two_qubit_gates"])
            st.metric("Two-qubit depth", res["two_qubit_depth"])
            st.json(res["gate_counts"])
        else:
            st.info("Run the pipeline (sidebar) to profile the exact circuit used.")
    with col_b:
        if st.button("📐 Profile depth scaling across N"):
            sizes = [n for n in (8, 12, 16, 24, 32, 40) if n >= k_assets + 1]
            prog = st.progress(0.0, "Transpiling…")
            rows = []
            for i, n in enumerate(sizes):
                md_n = generate_market_data(n_assets=n, seed=int(seed))
                qc, mode_used = build_qaoa_ansatz(
                    md_n.mu, md_n.sigma, min(k_assets, n - 1), risk_aversion,
                    reps=reps, init_mode=init_mode)
                r = circuit_resource_report(qc)
                r["init"] = mode_used
                rows.append(r)
                prog.progress((i + 1) / len(sizes),
                              f"N={n}: depth {r['transpiled_depth']}, "
                              f"CX {r['two_qubit_gates']}")
            st.session_state["scaling"] = pd.DataFrame(rows)
            prog.empty()
        if "scaling" in st.session_state:
            sc = st.session_state["scaling"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sc["num_qubits"], y=sc["transpiled_depth"],
                                     mode="lines+markers", name="Transpiled depth",
                                     line=dict(color=ACCENT, width=3)))
            fig.add_trace(go.Scatter(x=sc["num_qubits"], y=sc["two_qubit_gates"],
                                     mode="lines+markers", name="CX count",
                                     line=dict(color="crimson", width=3)))
            fig.update_layout(title=f"Circuit cost vs universe size (p={reps})",
                              xaxis_title="Qubits N", yaxis_title="Count",
                              height=400, legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(sc[["num_qubits", "init", "logical_depth",
                             "transpiled_depth", "two_qubit_depth",
                             "two_qubit_gates", "total_gates"]],
                         use_container_width=True, hide_index=True)
            st.caption(
                "The dominant CX cost is the dense RZZ phase layer "
                "(≈ N(N−1)/2 pairs per layer — a property of the dense "
                "covariance, not of the constraint). The penalty-free design "
                "keeps the *mixer* at N shallow XX+YY blocks per layer and "
                "the initial state polynomial (Dicke O(N·K), or block-W "
                "O(N/K)-depth beyond the exact-Dicke budget)."
            )
