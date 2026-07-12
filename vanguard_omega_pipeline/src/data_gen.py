"""
Synthetic large-universe market generator (N = 100+) with a genuinely
HIERARCHICAL covariance structure.
=====================================================================

The HRP decomposition layer is only meaningful if the covariance actually
contains a hierarchy to discover.  We therefore generate returns from a
three-level nested factor model — market → sector → industry — so that the
dendrogram produced by :mod:`src.hrp_clustering` recovers real structure
(validated in the QA suite, not assumed):

.. math::

    r_{i,t} \\;=\\; \\tfrac{\\mu_i}{T_{yr}}
        \\;+\\; \\beta^{mkt}_i \\, m_t
        \\;+\\; \\beta^{sec}_i \\, s_{g(i),t}
        \\;+\\; \\beta^{ind}_i \\, f_{h(i),t}
        \\;+\\; \\varepsilon_{i,t}

with mutually independent zero-mean Gaussian factors (market volatility >
sector > industry) and idiosyncratic noise :math:`\\varepsilon`.  Stacking
loadings into :math:`B \\in \\mathbb{R}^{N \\times F}` with diagonal factor
covariance :math:`\\Sigma_F` gives the *exact* population covariance

.. math::

    \\Sigma \\;=\\; B \\Sigma_F B^{\\top} + D \\;\\succeq\\; 0
    \\qquad (D = \\mathrm{diag}(\\sigma^2_{\\varepsilon,i}) \\succ 0),

which is what the optimizers consume — the simulated panel exists for the
HRP correlation estimate and dashboard realism, mirroring production where
the risk model and the return history are distinct artifacts.

Expected returns embed factor risk premia (market premium dominant) plus
bounded cross-sectional alpha noise; per-asset **linear transaction cost
coefficients** (in weight-per-unit terms, i.e. cost = Σ_i c_i |x_i − x^{prev}_i|)
and a sparse previous book ``w_prev`` are generated for the ADMM x-update's
turnover / t-cost terms.

Everything is seeded: identical universes feed the quantum pipeline and
every classical baseline, which is a fairness requirement of the judged
benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["OmegaMarketData", "generate_market_data", "TRADING_DAYS_PER_YEAR"]

TRADING_DAYS_PER_YEAR = 252

# Multi-asset universe: labels are ASSET CLASSES (equities, fixed income,
# commodities, alternatives, FX), per the challenge's multi-asset mandate.
# The variable name `sectors` is retained across the API for stability —
# read it as "asset-class label".
_SECTOR_POOL = [
    "US Equity", "Intl Equity", "EM Equity", "Govt Bonds",
    "Corp Bonds", "High Yield", "Commodities", "REITs",
    "Infrastructure", "Private Alts", "FX Overlay", "Cash Equiv",
]

# Base annual income yield per asset class (dividends / coupons / carry),
# aligned index-wise with _SECTOR_POOL. Bonds, high yield and REITs carry
# income; commodities and FX overlays carry essentially none.
_CLASS_BASE_YIELD = [
    0.018, 0.024, 0.028, 0.033, 0.042, 0.065, 0.004, 0.046,
    0.038, 0.012, 0.006, 0.035,
]


@dataclass
class OmegaMarketData:
    """One synthetic universe with hierarchical risk structure.

    Attributes
    ----------
    mu : (N,) annualized expected returns.
    sigma : (N, N) exact population covariance (annualized, PSD).
    returns : (T, N) simulated daily return panel (HRP input).
    sectors : (N,) sector labels in ``[0, n_sectors)``.
    industries : (N,) global industry labels (nested inside sectors) —
        the second hierarchy level HRP should resolve.
    sector_names : display names per sector label.
    asset_names : tickers, e.g. ``"AST042 (Tech/3)"``.
    w_prev : (N,) previous portfolio (sums to 1; sparse) for turnover terms.
    tc_linear : (N,) linear transaction-cost coefficients c_i (annualized
        drag per unit of |Δweight|; ~5–40 bps).
    asset_yields : (N,) annual income yield y_i (dividends/coupons/carry) —
        drives the INCOME goal (:math:`y^{\\top}w \\ge` income floor).
    risk_free_rate : annualized r_f.
    seed : RNG seed used (provenance).
    """

    mu: np.ndarray
    sigma: np.ndarray
    returns: np.ndarray
    sectors: np.ndarray
    industries: np.ndarray
    sector_names: list[str]
    asset_names: list[str]
    w_prev: np.ndarray
    tc_linear: np.ndarray
    asset_yields: np.ndarray
    risk_free_rate: float
    seed: int

    n_assets: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_assets = int(self.mu.shape[0])
        n = self.n_assets
        # --- Tensor-dimension validation (explicit, per execution rules) ---
        assert self.sigma.shape == (n, n), f"sigma {self.sigma.shape} != ({n},{n})"
        assert self.returns.ndim == 2 and self.returns.shape[1] == n, \
            f"returns panel {self.returns.shape} incompatible with N={n}"
        for name in ("sectors", "industries", "w_prev", "tc_linear", "asset_yields"):
            arr = getattr(self, name)
            assert arr.shape == (n,), f"{name} {arr.shape} != ({n},)"
        assert np.allclose(self.sigma, self.sigma.T, atol=1e-12), "sigma not symmetric"
        min_eig = float(np.linalg.eigvalsh(self.sigma).min())
        assert min_eig > 0.0, f"sigma must be PD (min eig {min_eig:.3e})"
        assert abs(self.w_prev.sum() - 1.0) < 1e-9, "w_prev must sum to 1"

    def sector_matrix(self) -> np.ndarray:
        """One-hot map ``S[g, i] = 1`` iff asset *i* is in sector *g*, so the
        x-update writes sector caps as the single vectorized constraint
        ``S @ x <= cap``."""
        s = np.zeros((len(self.sector_names), self.n_assets))
        s[self.sectors, np.arange(self.n_assets)] = 1.0
        return s

    def correlation(self) -> np.ndarray:
        """Population correlation implied by ``sigma`` (preferred HRP input —
        noiseless; the sample-estimate path is exercised via ``returns``)."""
        vol = np.sqrt(np.diag(self.sigma))
        corr = self.sigma / np.outer(vol, vol)
        np.fill_diagonal(corr, 1.0)
        return np.clip(corr, -1.0, 1.0)

    def scenario_matrix(self, horizon_days: int = 21) -> np.ndarray:
        """Stress-scenario return matrix R ∈ R^{S×N} for scenario penalties.

        Chops the simulated daily panel into non-overlapping ``horizon_days``
        windows (≈ monthly) and compounds each into one scenario:
        :math:`R_{s,i} = \\prod_{t \\in s}(1 + r_{t,i}) - 1`.  The allocation
        layer's CVaR penalty (drawdown-control goal) operates on portfolio
        losses :math:`-R w` over these scenarios — the standard convex
        instrument for tail/drawdown control (Rockafellar & Uryasev, 2000).
        """
        n_windows = self.returns.shape[0] // horizon_days
        if n_windows < 5:
            raise ValueError("panel too short for scenario construction")
        r = self.returns[: n_windows * horizon_days]
        r = r.reshape(n_windows, horizon_days, self.n_assets)
        return np.prod(1.0 + r, axis=1) - 1.0


def generate_market_data(
    n_assets: int = 100,
    n_sectors: int = 10,
    industries_per_sector: int = 3,
    n_days: int = 3 * TRADING_DAYS_PER_YEAR,
    risk_free_rate: float = 0.03,
    seed: int = 7,
) -> OmegaMarketData:
    """Generate the N=100+ hierarchical universe.

    Hierarchy: ``n_sectors`` sector factors, each containing
    ``industries_per_sector`` industry sub-factors; every asset loads on
    (market, its sector, its industry).  Calibration targets: asset vols
    ≈ 15–40% annualized, intra-industry correlation > intra-sector >
    cross-sector — the signature HRP is designed to exploit.

    Returns a fully validated :class:`OmegaMarketData`.
    """
    if n_sectors > len(_SECTOR_POOL):
        raise ValueError(f"n_sectors must be <= {len(_SECTOR_POOL)}")
    rng = np.random.default_rng(seed)
    sector_names = _SECTOR_POOL[:n_sectors]

    # --- Assign hierarchy labels (every sector/industry populated). --------
    sectors = rng.integers(0, n_sectors, size=n_assets)
    sectors[:n_sectors] = np.arange(n_sectors)          # no empty sector
    industries = np.empty(n_assets, dtype=int)
    for i in range(n_assets):
        industries[i] = sectors[i] * industries_per_sector + rng.integers(industries_per_sector)
    n_industries = n_sectors * industries_per_sector

    asset_names = [
        f"AST{i:03d} ({sector_names[sectors[i]][:4]}/{industries[i] % industries_per_sector})"
        for i in range(n_assets)
    ]

    # --- Factor loadings B: [market | sectors | industries]. ----------------
    n_factors = 1 + n_sectors + n_industries
    B = np.zeros((n_assets, n_factors))
    B[:, 0] = rng.uniform(0.75, 1.25, n_assets)                       # market
    B[np.arange(n_assets), 1 + sectors] = rng.uniform(0.55, 1.05, n_assets)
    B[np.arange(n_assets), 1 + n_sectors + industries] = rng.uniform(0.45, 0.95, n_assets)

    # --- Diagonal factor covariance (annualized vols; market > sector > ind).
    factor_vols = np.concatenate([
        [0.16],
        rng.uniform(0.075, 0.11, n_sectors),
        rng.uniform(0.05, 0.08, n_industries),
    ])
    sigma_f = np.diag(factor_vols**2)

    idio_vol = rng.uniform(0.08, 0.20, n_assets)
    D = np.diag(idio_vol**2)

    sigma = B @ sigma_f @ B.T + D
    sigma = 0.5 * (sigma + sigma.T)

    # --- Expected returns: premia proportional to systematic risk taken. ----
    premia = np.concatenate([
        [0.050],
        rng.uniform(0.004, 0.020, n_sectors),
        rng.uniform(0.002, 0.012, n_industries),
    ])
    mu = risk_free_rate + B @ premia + 0.012 * rng.standard_normal(n_assets)
    mu = np.clip(mu, 0.005, 0.28)

    # --- Daily panel (independent Gaussian factors → cheap exact simulation).
    f_t = rng.standard_normal((n_days, n_factors)) * (factor_vols / np.sqrt(TRADING_DAYS_PER_YEAR))
    eps = rng.standard_normal((n_days, n_assets)) * (idio_vol / np.sqrt(TRADING_DAYS_PER_YEAR))
    returns = mu / TRADING_DAYS_PER_YEAR + f_t @ B.T + eps

    # --- Previous book (sparse) + linear t-cost coefficients (5–40 bps). ----
    k_prev = max(5, n_assets // 6)
    prev_idx = rng.choice(n_assets, size=k_prev, replace=False)
    w_prev = np.zeros(n_assets)
    w_prev[prev_idx] = rng.dirichlet(np.ones(k_prev) * 2.0)
    tc_linear = rng.uniform(0.0005, 0.0040, n_assets)

    # --- Income yields (drawn LAST so every previously verified field is
    # bit-identical to earlier releases for the same seed). ----------------
    base_y = np.asarray(_CLASS_BASE_YIELD[:n_sectors])
    asset_yields = np.clip(
        base_y[sectors] * rng.uniform(0.7, 1.3, n_assets), 0.0, 0.12)

    return OmegaMarketData(
        mu=mu, sigma=sigma, returns=returns, sectors=sectors,
        industries=industries, sector_names=sector_names,
        asset_names=asset_names, w_prev=w_prev, tc_linear=tc_linear,
        asset_yields=asset_yields, risk_free_rate=risk_free_rate, seed=seed,
    )


if __name__ == "__main__":  # smoke check
    md = generate_market_data(n_assets=100)
    corr = md.correlation()
    same_ind = (md.industries[:, None] == md.industries[None, :]) & ~np.eye(md.n_assets, dtype=bool)
    same_sec = (md.sectors[:, None] == md.sectors[None, :]) & ~same_ind & ~np.eye(md.n_assets, dtype=bool)
    cross = (md.sectors[:, None] != md.sectors[None, :])
    print(f"corr(industry)={corr[same_ind].mean():.3f} > corr(sector)={corr[same_sec].mean():.3f} "
          f"> corr(cross)={corr[cross].mean():.3f}")
