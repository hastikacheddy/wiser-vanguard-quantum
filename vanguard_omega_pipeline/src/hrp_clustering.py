"""
Hierarchical Risk Parity decomposition + topology-matched mixer synthesis.
==========================================================================

This module is the *classical decomposition pillar*: it converts an
N = 100+ universe into (a) hardware-executable sub-problems of size ≤ 20,
(b) capital budgets per sub-problem, and (c) — the compiler-level payload —
a **BFS spanning tree per cluster** that the quantum z-step uses verbatim as
its XY-mixer interaction graph, eliminating SWAP routing.

Pipeline (López de Prado, *Building Diversified Portfolios that Outperform
Out-of-Sample*, J. Portfolio Management 2016 — steps 1–3; step 4–5 are our
quantum-specific extensions):

1. **Correlation distance.**

   .. math::

       d_{ij} \\;=\\; \\sqrt{\\tfrac{1}{2}\\,(1 - \\rho_{ij})} \\;\\in\\; [0, 1]

   a proper metric on assets (0 = perfectly correlated substitutes).

2. **Agglomerative linkage** (``scipy.cluster.hierarchy.linkage``).
   *Deviation from canon, documented:* textbook HRP uses **single** linkage,
   which chains and yields wildly unbalanced clusters — useless for carving
   qubit-budgeted sub-problems.  We default to **Ward**, which produces
   compact, balanced clusters; the ``method`` argument restores canon when
   the full-universe HRP weights are used as a classical benchmark.

3. **Quasi-diagonalization + recursive bisection** →
   :func:`hrp_weights`, the full classical HRP allocator (kept as an
   ML-free benchmark and as the intra-cluster seed allocation).

4. **Dendrogram chunking** → :func:`chunk_clusters`: recursive descent of
   the linkage tree emits clusters of size ≤ ``max_size`` (default 20 —
   the statevector-exact qubit budget), then greedily re-merges runt
   clusters along dendrogram order.  The result is a *partition* — proved
   disjoint + exhaustive in the QA suite.

5. **Mixer topology synthesis** → :func:`mixer_spanning_tree`: per cluster,
   the minimum spanning tree of the distance matrix (equivalently the
   *maximum-correlation* spanning tree — XY excitation hops travel between
   the closest economic substitutes first), BFS-ordered from a hub root,
   plus a greedy edge-coloring into parallel layers.  Because every
   :math:`XX+YY` edge term conserves total excitation number and a spanning
   tree is connected, Hamming-weight-K feasibility AND full reachability of
   the :math:`\\binom{n}{K}` shell are both preserved — while the transpiler
   routing cost on a matched device topology drops to **zero SWAPs**.
   The honest trade-off: a tree's diameter exceeds a ring's, so mixing is
   slower per layer; the hardware-audit tab quantifies both sides.

No sklearn, no DBSCAN, no heuristic ML — SciPy hierarchy + csgraph only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage, to_tree
from scipy.spatial.distance import squareform

__all__ = [
    "correlation_from_returns",
    "correlation_distance",
    "hrp_linkage",
    "quasi_diagonal_order",
    "hrp_weights",
    "chunk_clusters",
    "cluster_budgets",
    "MixerTopology",
    "mixer_spanning_tree",
    "ring_topology",
    "ClusterPlan",
    "build_cluster_plan",
    "cluster_report",
]


# ===========================================================================
# 1–2. Correlation → metric distance → linkage.
# ===========================================================================
def correlation_from_returns(returns: np.ndarray) -> np.ndarray:
    """Sample correlation from a (T, N) return panel, clipped into [−1, 1]
    with an exact unit diagonal (guards the metric property of d_ij)."""
    returns = np.asarray(returns, dtype=float)
    if returns.ndim != 2 or returns.shape[0] < 3:
        raise ValueError(f"returns must be (T>=3, N), got {returns.shape}")
    corr = np.corrcoef(returns, rowvar=False)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def correlation_distance(corr: np.ndarray) -> np.ndarray:
    """López de Prado metric  d_ij = sqrt((1 − ρ_ij)/2)  ∈ [0, 1]."""
    corr = np.clip(np.asarray(corr, dtype=float), -1.0, 1.0)
    d = np.sqrt(np.maximum(0.5 * (1.0 - corr), 0.0))
    np.fill_diagonal(d, 0.0)
    return d


def hrp_linkage(
    returns: np.ndarray | None = None,
    corr: np.ndarray | None = None,
    method: str = "ward",
) -> tuple[np.ndarray, np.ndarray]:
    """Agglomerative linkage on the correlation-distance metric.

    Exactly one of ``returns`` / ``corr`` may be supplied (``corr`` wins if
    both).  Returns ``(linkage_matrix, corr)``.

    ``method='ward'`` (default) → balanced clusters for qubit budgeting;
    ``method='single'`` → canonical HRP chaining (benchmark use).
    """
    if corr is None:
        if returns is None:
            raise ValueError("provide returns or corr")
        corr = correlation_from_returns(returns)
    dist = correlation_distance(corr)
    condensed = squareform(dist, checks=False)
    return linkage(condensed, method=method), corr


# ===========================================================================
# 3. Classical HRP allocation (benchmark + intra-cluster seed weights).
# ===========================================================================
def quasi_diagonal_order(link: np.ndarray) -> np.ndarray:
    """Seriation: dendrogram leaf order that quasi-diagonalizes Σ (similar
    assets adjacent).  This ordering drives the recursive bisection."""
    return np.asarray(leaves_list(link), dtype=int)


def _inverse_variance_weights(cov_sub: np.ndarray) -> np.ndarray:
    """IVP weights  w_i ∝ 1/Σ_ii  (the HRP intra-split allocator)."""
    iv = 1.0 / np.clip(np.diag(cov_sub), 1e-12, None)
    return iv / iv.sum()


def _cluster_variance(cov: np.ndarray, idx: np.ndarray) -> float:
    """Variance of the IVP portfolio of the sub-cluster ``idx``:
    :math:`\\sigma^2_c = w^{\\top} \\Sigma_{cc} w`, w = IVP weights."""
    sub = cov[np.ix_(idx, idx)]
    w = _inverse_variance_weights(sub)
    return float(w @ sub @ w)


def hrp_weights(cov: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Full recursive-bisection HRP allocation over the whole universe.

    Walking down the seriated list, each cluster [c] is split into halves
    (c₁, c₂) and capital is divided inversely to cluster risk:

    .. math::

        \\alpha = 1 - \\frac{\\sigma^2_{c_1}}{\\sigma^2_{c_1} + \\sigma^2_{c_2}},
        \\qquad w_{c_1} \\leftarrow \\alpha\\, w_{c_1},
        \\quad  w_{c_2} \\leftarrow (1-\\alpha)\\, w_{c_2} .

    Long-only and fully invested by construction (Σw = 1, w > 0) — verified
    in QA.  Serves as the ML-free classical benchmark for the dashboard.
    """
    n = cov.shape[0]
    if cov.shape != (n, n) or len(order) != n:
        raise ValueError(f"dimension mismatch: cov {cov.shape}, order {len(order)}")
    w = np.ones(n)
    stack: list[np.ndarray] = [np.asarray(order, dtype=int)]
    while stack:
        cluster = stack.pop()
        if len(cluster) < 2:
            continue
        mid = len(cluster) // 2
        left, right = cluster[:mid], cluster[mid:]
        v_l, v_r = _cluster_variance(cov, left), _cluster_variance(cov, right)
        alpha = 1.0 - v_l / (v_l + v_r)
        w[left] *= alpha
        w[right] *= 1.0 - alpha
        stack.extend((left, right))
    return w / w.sum()


# ===========================================================================
# 4. Dendrogram chunking into qubit-budgeted sub-problems.
# ===========================================================================
def chunk_clusters(
    link: np.ndarray,
    n_assets: int,
    max_size: int = 20,
    min_size: int = 4,
) -> list[np.ndarray]:
    """Partition the universe into dendrogram-respecting clusters of size
    ≤ ``max_size`` (the hardware/simulator qubit budget).

    Algorithm: recursive descent from the linkage root — any subtree whose
    leaf count fits the budget is emitted whole (preserving maximal
    hierarchical coherence); larger subtrees recurse into their children.
    A post-pass then greedily merges runt clusters (< ``min_size``) with
    their smaller dendrogram-adjacent neighbor whenever the merge respects
    ``max_size`` — runts waste a quantum-circuit invocation on a trivially
    enumerable problem.

    Returns clusters in dendrogram (seriation) order; the QA suite asserts
    they are pairwise disjoint and exhaustive.
    """
    if max_size < 2:
        raise ValueError("max_size must be >= 2")
    root = to_tree(link)
    if root.count != n_assets:
        raise ValueError(f"linkage has {root.count} leaves, expected {n_assets}")

    clusters: list[np.ndarray] = []

    def descend(node) -> None:
        if node.count <= max_size:
            clusters.append(np.array(node.pre_order(lambda leaf: leaf.id), dtype=int))
        else:
            descend(node.get_left())
            descend(node.get_right())

    descend(root)

    # Greedy runt-merging along dendrogram order (deterministic).
    merged = True
    while merged:
        merged = False
        for i, cl in enumerate(clusters):
            if len(cl) >= min_size:
                continue
            neighbors = [j for j in (i - 1, i + 1) if 0 <= j < len(clusters)]
            neighbors = [j for j in neighbors if len(clusters[j]) + len(cl) <= max_size]
            if not neighbors:
                continue                      # runt stays (max_size dominates)
            j = min(neighbors, key=lambda j: len(clusters[j]))
            keep = min(i, j)
            clusters[keep] = np.concatenate([clusters[min(i, j)], clusters[max(i, j)]])
            del clusters[max(i, j)]
            merged = True
            break
    return clusters


def cluster_budgets(cov: np.ndarray, clusters: list[np.ndarray]) -> np.ndarray:
    """Capital budget per cluster ∝ inverse IVP-portfolio variance —
    the same inverse-risk logic HRP applies at every bisection, lifted to
    the cluster level.  The ADMM coordinator scales each sub-problem's
    budget constraint by these (they sum to 1)."""
    inv_var = np.array([1.0 / max(_cluster_variance(cov, cl), 1e-12) for cl in clusters])
    return inv_var / inv_var.sum()


# ===========================================================================
# 5. Topology-matched tree-mixer synthesis.
# ===========================================================================
@dataclass
class MixerTopology:
    """XY-mixer interaction graph for one cluster (local qubit indices).

    ``edges`` are in BFS discovery order from ``root``; ``layers`` groups
    them into sets acting on disjoint qubits, so one mixer application has
    circuit depth = ``len(layers)`` two-qubit slabs.  ``kind`` records
    whether this is the compiled tree or a comparison ring.
    """
    n: int
    kind: str                                # 'bfs_tree' | 'ring'
    root: int
    edges: list[tuple[int, int]]
    layers: list[list[tuple[int, int]]]
    bfs_depth: int
    max_degree: int

    def __post_init__(self) -> None:
        expect = self.n - 1 if self.kind == "bfs_tree" else (self.n if self.n > 2 else self.n - 1)
        assert len(self.edges) == max(expect, 0), \
            f"{self.kind}: {len(self.edges)} edges for n={self.n}"
        flat = [q for layer in self.layers for e in layer for q in e]
        assert len(flat) == 2 * len(self.edges), "layers must cover every edge once"
        for layer in self.layers:
            qubits = [q for e in layer for q in e]
            assert len(qubits) == len(set(qubits)), "layer reuses a qubit (not parallel)"


def _greedy_edge_layers(edges: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """Greedy edge-coloring: place each edge in the first layer where both
    endpoints are free.  For a tree (chromatic index = Δ) greedy uses at
    most 2Δ − 1 layers; the audit tab reports the achieved number."""
    layers: list[list[tuple[int, int]]] = []
    used: list[set[int]] = []
    for (a, b) in edges:
        for layer, busy in zip(layers, used):
            if a not in busy and b not in busy:
                layer.append((a, b))
                busy.update((a, b))
                break
        else:
            layers.append([(a, b)])
            used.append({a, b})
    return layers


def mixer_spanning_tree(corr_sub: np.ndarray,
                        max_degree: int | None = 3) -> MixerTopology:
    """Compile the cluster's XY-mixer graph: BFS-ordered, degree-capped
    maximum-correlation spanning tree.

    * Kruskal on :math:`d_{ij} = \\sqrt{(1-\\rho_{ij})/2}` ≡ **maximum**-
      correlation spanning tree (d is monotone-decreasing in ρ): excitation
      swaps travel between the closest economic substitutes first, so early
      mixer layers explore the portfolio moves with the least risk impact.
    * ``max_degree`` (default 3 = the heavy-hex lattice degree) constrains
      every vertex during Kruskal so the tree can embed on the device graph
      without SWAP insertion.  Degree-constrained MST is NP-hard in general;
      the greedy cap is the standard compiler heuristic, and a final
      uncapped pass reconnects any leftover components (logged via the
      resulting degree, never silently disconnected).  ``None`` = uncapped.
    * Root = maximum-degree node (hub) → minimal BFS depth in practice.
    * Deterministic: edges sorted by (distance, i, j).

    Physics guarantee carried by ANY connected topology: each
    :math:`e^{-i\\beta(X_iX_j+Y_iY_j)/2}` commutes with
    :math:`\\hat n = \\sum_i (I - Z_i)/2`, and connectivity makes the
    excitation-transport walk irreducible on the weight-K shell — so the
    tree keeps exact feasibility AND reachability while matching the chip.
    """
    n = corr_sub.shape[0]
    if corr_sub.shape != (n, n):
        raise ValueError(f"corr_sub must be square, got {corr_sub.shape}")
    if n == 1:
        return MixerTopology(n=1, kind="bfs_tree", root=0, edges=[], layers=[],
                             bfs_depth=0, max_degree=0)

    dist = correlation_distance(corr_sub)
    cap = max_degree if max_degree is not None else n
    all_edges = sorted(
        ((dist[i, j], i, j) for i in range(n) for j in range(i + 1, n)),
        key=lambda t: (t[0], t[1], t[2]),
    )

    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    degree = np.zeros(n, dtype=int)
    adj: list[list[int]] = [[] for _ in range(n)]
    accepted = 0
    for pass_capped in (True, False):          # 2nd pass: reconnect uncapped
        for (_, i, j) in all_edges:
            if accepted == n - 1:
                break
            if pass_capped and (degree[i] >= cap or degree[j] >= cap):
                continue
            ri, rj = find(i), find(j)
            if ri == rj:
                continue
            parent[ri] = rj
            degree[i] += 1
            degree[j] += 1
            adj[i].append(j)
            adj[j].append(i)
            accepted += 1
    assert accepted == n - 1, f"spanning tree must have n-1 edges, got {accepted}"

    root = int(np.argmax(degree))
    # Manual BFS from the hub root → edges in discovery order + depth levels.
    edges: list[tuple[int, int]] = []
    depth = np.full(n, -1, dtype=int)
    depth[root] = 0
    frontier = [root]
    while frontier:
        nxt: list[int] = []
        for p in frontier:
            for v in sorted(adj[p]):
                if depth[v] < 0:
                    depth[v] = depth[p] + 1
                    edges.append((p, v))
                    nxt.append(v)
        frontier = nxt
    assert (depth >= 0).all(), "BFS failed to reach every vertex (tree disconnected?)"

    return MixerTopology(
        n=n, kind="bfs_tree", root=root, edges=edges,
        layers=_greedy_edge_layers(edges),
        bfs_depth=int(depth.max()), max_degree=int(degree.max()),
    )


def ring_topology(n: int) -> MixerTopology:
    """Reference ring mixer (even / odd / wrap layering) for the hardware
    audit's tree-vs-ring comparison.  Same conservation physics; requires
    SWAP insertion on non-ring device graphs."""
    if n == 1:
        return MixerTopology(n=1, kind="ring", root=0, edges=[], layers=[],
                             bfs_depth=0, max_degree=0)
    even = [(i, i + 1) for i in range(0, n - 1, 2)]
    odd = [(i, i + 1) for i in range(1, n - 1, 2)]
    wrap = [(n - 1, 0)] if n > 2 else []
    edges = even + odd + wrap
    layers = [lst for lst in (even, odd, wrap) if lst]
    return MixerTopology(n=n, kind="ring", root=0, edges=edges, layers=layers,
                         bfs_depth=n // 2, max_degree=2)


# ===========================================================================
# Orchestration: one call from the ADMM coordinator.
# ===========================================================================
@dataclass
class ClusterPlan:
    """Complete decomposition artifact consumed by the ADMM coordinator."""
    clusters: list[np.ndarray]               # global asset indices per cluster
    budgets: np.ndarray                      # (n_clusters,) capital budgets, Σ=1
    topologies: list[MixerTopology]          # tree mixer per cluster
    linkage_matrix: np.ndarray
    seriation: np.ndarray                    # quasi-diagonal leaf order
    hrp_benchmark_weights: np.ndarray        # (N,) classical HRP allocation
    corr: np.ndarray
    method: str
    max_size: int

    n_assets: int = field(init=False)

    def __post_init__(self) -> None:
        all_idx = np.concatenate(self.clusters) if self.clusters else np.array([], int)
        self.n_assets = len(all_idx)
        # Partition proof: disjoint + exhaustive + size-budgeted.
        assert len(np.unique(all_idx)) == self.n_assets, "clusters overlap"
        assert set(all_idx.tolist()) == set(range(self.n_assets)), "clusters not exhaustive"
        assert all(len(c) <= self.max_size for c in self.clusters), "cluster exceeds qubit budget"
        assert abs(self.budgets.sum() - 1.0) < 1e-9
        assert len(self.topologies) == len(self.clusters)


def build_cluster_plan(
    sigma: np.ndarray,
    returns: np.ndarray | None = None,
    corr: np.ndarray | None = None,
    max_size: int = 20,
    min_size: int = 4,
    method: str = "ward",
    mixer_max_degree: int | None = 3,
) -> ClusterPlan:
    """One-shot decomposition: linkage → chunks → budgets → tree mixers.

    ``sigma`` is the risk model used for budgets / HRP weights; the
    correlation used for clustering may come from the same risk model
    (``corr``) or be estimated from a return panel (``returns``).
    ``mixer_max_degree=3`` matches the heavy-hex lattice degree so every
    tree embeds SWAP-free on the device graph.
    """
    link, corr_used = hrp_linkage(returns=returns, corr=corr, method=method)
    n = corr_used.shape[0]
    if sigma.shape != (n, n):
        raise ValueError(f"sigma {sigma.shape} vs corr {corr_used.shape}")
    order = quasi_diagonal_order(link)
    clusters = chunk_clusters(link, n, max_size=max_size, min_size=min_size)
    return ClusterPlan(
        clusters=clusters,
        budgets=cluster_budgets(sigma, clusters),
        topologies=[mixer_spanning_tree(corr_used[np.ix_(cl, cl)], mixer_max_degree)
                    for cl in clusters],
        linkage_matrix=link,
        seriation=order,
        hrp_benchmark_weights=hrp_weights(sigma, order),
        corr=corr_used,
        method=method,
        max_size=max_size,
    )


def cluster_report(plan: ClusterPlan, sectors: np.ndarray | None = None,
                   sector_names: list[str] | None = None) -> pd.DataFrame:
    """Per-cluster summary table for the dashboard: size, budget, internal
    correlation, mixer-tree geometry (depth / max degree / parallel layers)."""
    rows = []
    for c_id, (cl, topo) in enumerate(zip(plan.clusters, plan.topologies)):
        sub = plan.corr[np.ix_(cl, cl)]
        off = sub[np.triu_indices(len(cl), 1)]
        row = {
            "cluster": c_id,
            "size": len(cl),
            "budget": plan.budgets[c_id],
            "intra_corr": float(off.mean()) if off.size else 1.0,
            "tree_edges": len(topo.edges),
            "tree_depth": topo.bfs_depth,
            "tree_max_degree": topo.max_degree,
            "mixer_layers": len(topo.layers),
        }
        if sectors is not None:
            labels, counts = np.unique(np.asarray(sectors)[cl], return_counts=True)
            top = int(labels[np.argmax(counts)])
            row["dominant_sector"] = (sector_names[top] if sector_names else top)
            row["sector_purity"] = float(counts.max() / counts.sum())
        rows.append(row)
    return pd.DataFrame(rows)
