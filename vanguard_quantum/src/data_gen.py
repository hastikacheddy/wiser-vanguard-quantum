"""
Synthetic multi-asset market data generator.
============================================

Produces a realistic, reproducible investment universe for the hybrid
quantum/classical portfolio construction pipeline:

* Expected (annualized) returns  :math:`\\mu \\in \\mathbb{R}^N`
* Annualized covariance matrix   :math:`\\Sigma \\in \\mathbb{R}^{N \\times N}`
* Sector membership map (for sector-cap constraints in the classical layer)
* A previous portfolio ``w_prev`` (for turnover-budget constraints)
* A simulated daily return panel (for realism / plotting)

Mathematical model
------------------
Returns follow a strict factor model, which guarantees a positive
semi-definite covariance by construction:

.. math::

    r_t = \\alpha + B f_t + \\varepsilon_t,
    \\qquad
    f_t \\sim \\mathcal{N}(0, \\Sigma_F),
    \\qquad
    \\varepsilon_t \\sim \\mathcal{N}(0, D)

so that

.. math::

    \\Sigma \\;=\\; B \\Sigma_F B^{\\top} + D  \\;\\succeq\\; 0 ,

with :math:`B \\in \\mathbb{R}^{N \\times F}` sector-tilted factor loadings,
:math:`\\Sigma_F \\succ 0` the factor covariance, and :math:`D` a diagonal
idiosyncratic variance matrix.  Expected returns embed a risk premium so that
the mean-variance objective has non-trivial structure:

.. math::

    \\mu_i = r_f + B_i \\lambda_F + \\eta_i ,

where :math:`\\lambda_F` is the vector of factor risk premia.

Everything is seeded for exact reproducibility across the quantum and
classical benchmark runs (a hard requirement for a fair judged comparison).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["MarketData", "generate_market_data"]

TRADING_DAYS_PER_YEAR = 252

# Multi-asset universe: labels are ASSET CLASSES per the challenge mandate
# (the variable name `sectors` is kept across the API — read as class label).
_SECTOR_POOL = [
    "US Equity", "Intl Equity", "EM Equity", "Govt Bonds",
    "Corp Bonds", "High Yield", "Commodities", "REITs",
    "Infrastructure", "Cash Equiv",
]


@dataclass
class MarketData:
    """Container for one synthetic market universe.

    Attributes
    ----------
    mu : (N,) annualized expected returns.
    sigma : (N, N) annualized covariance matrix (symmetric PSD).
    sectors : (N,) integer sector labels in ``[0, n_sectors)``.
    sector_names : names for each sector label.
    asset_names : human-readable tickers, e.g. ``"AST07 (Energy)"``.
    w_prev : (N,) previous portfolio weights (sums to 1) — used by the
        turnover-budget constraint in the classical allocation layer.
    returns : (T, N) simulated daily returns panel.
    risk_free_rate : annualized risk-free rate used to build the premia.
    seed : RNG seed used, for provenance.
    """

    mu: np.ndarray
    sigma: np.ndarray
    sectors: np.ndarray
    sector_names: list[str]
    asset_names: list[str]
    w_prev: np.ndarray
    returns: np.ndarray
    risk_free_rate: float
    seed: int

    n_assets: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_assets = int(self.mu.shape[0])
        # Defensive invariants — the optimizer layers rely on these.
        assert self.sigma.shape == (self.n_assets, self.n_assets)
        assert np.allclose(self.sigma, self.sigma.T, atol=1e-10), "Sigma must be symmetric"
        min_eig = float(np.linalg.eigvalsh(self.sigma).min())
        assert min_eig > -1e-10, f"Sigma must be PSD (min eig = {min_eig:.3e})"

    def sector_matrix(self) -> np.ndarray:
        """One-hot sector membership matrix ``S`` with ``S[g, i] = 1`` iff
        asset *i* belongs to sector *g*.  Used to vectorize sector caps:
        the constraint ``S @ w <= cap`` bounds every sector's total weight."""
        n_sectors = len(self.sector_names)
        s = np.zeros((n_sectors, self.n_assets))
        s[self.sectors, np.arange(self.n_assets)] = 1.0
        return s


def generate_market_data(
    n_assets: int = 40,
    n_sectors: int = 8,
    n_factors: int = 5,
    n_days: int = 2 * TRADING_DAYS_PER_YEAR,
    risk_free_rate: float = 0.03,
    seed: int = 7,
) -> MarketData:
    """Generate a seeded synthetic universe with factor-model covariance.

    Parameters
    ----------
    n_assets : universe size N (challenge spec: 40).
    n_sectors : number of sectors for the sector-cap constraints.
    n_factors : number of latent risk factors F (F << N gives realistic
        low-rank-plus-diagonal covariance spectra).
    n_days : length of the simulated daily return panel.
    risk_free_rate : annualized r_f.
    seed : RNG seed (NumPy ``default_rng``).

    Returns
    -------
    MarketData
        Fully populated, internally consistent universe.

    Notes
    -----
    Calibration targets (annualized): single-asset volatilities in roughly
    15%–45%, expected returns in roughly 2%–18%, average pairwise
    correlation ~0.3 with visible sector clustering — i.e. numbers a
    practitioner would recognize as an equity universe.
    """
    if n_sectors > len(_SECTOR_POOL):
        raise ValueError(f"n_sectors must be <= {len(_SECTOR_POOL)}")
    rng = np.random.default_rng(seed)

    sector_names = _SECTOR_POOL[:n_sectors]
    sectors = rng.integers(0, n_sectors, size=n_assets)
    # Guarantee every sector is populated (so sector caps are meaningful).
    for g in range(min(n_sectors, n_assets)):
        sectors[g] = g
    asset_names = [f"AST{i:02d} ({sector_names[sectors[i]][:4]})" for i in range(n_assets)]

    # ---- Factor loadings B: market beta + sector tilt + idiosyncratic style.
    B = np.zeros((n_assets, n_factors))
    B[:, 0] = rng.uniform(0.7, 1.3, size=n_assets)          # market factor
    for i in range(n_assets):
        style = 1 + (sectors[i] % (n_factors - 1)) if n_factors > 1 else 0
        B[i, style] = rng.uniform(0.4, 1.0)                  # sector/style tilt
    B += 0.10 * rng.standard_normal((n_assets, n_factors))

    # ---- Factor covariance Sigma_F (annualized, PD by construction).
    factor_vols = np.concatenate([[0.16], rng.uniform(0.05, 0.12, n_factors - 1)])
    A = rng.standard_normal((n_factors, n_factors))
    corr_f = A @ A.T
    d_inv = 1.0 / np.sqrt(np.diag(corr_f))
    corr_f = 0.3 * (d_inv[:, None] * corr_f * d_inv[None, :]) + 0.7 * np.eye(n_factors)
    sigma_f = np.outer(factor_vols, factor_vols) * corr_f

    # ---- Idiosyncratic variances D (annualized).
    idio_vol = rng.uniform(0.10, 0.25, size=n_assets)
    D = np.diag(idio_vol**2)

    sigma = B @ sigma_f @ B.T + D
    sigma = 0.5 * (sigma + sigma.T)                          # exact symmetry

    # ---- Expected returns: factor risk premia + cross-sectional alpha noise.
    factor_premia = np.concatenate([[0.055], rng.uniform(0.005, 0.03, n_factors - 1)])
    mu = risk_free_rate + B @ factor_premia + 0.015 * rng.standard_normal(n_assets)
    mu = np.clip(mu, 0.01, 0.25)

    # ---- Simulated daily panel (for plots / realism, not for estimation).
    L_f = np.linalg.cholesky(sigma_f / TRADING_DAYS_PER_YEAR)
    f_t = rng.standard_normal((n_days, n_factors)) @ L_f.T
    eps = rng.standard_normal((n_days, n_assets)) * (idio_vol / np.sqrt(TRADING_DAYS_PER_YEAR))
    returns = mu / TRADING_DAYS_PER_YEAR + f_t @ B.T + eps

    # ---- Previous book: a random sparse feasible portfolio (for turnover).
    k_prev = max(3, n_assets // 5)
    prev_idx = rng.choice(n_assets, size=k_prev, replace=False)
    w_prev = np.zeros(n_assets)
    w_prev[prev_idx] = rng.dirichlet(np.ones(k_prev))

    return MarketData(
        mu=mu, sigma=sigma, sectors=sectors, sector_names=sector_names,
        asset_names=asset_names, w_prev=w_prev, returns=returns,
        risk_free_rate=risk_free_rate, seed=seed,
    )


if __name__ == "__main__":  # quick self-check
    md = generate_market_data()
    vols = np.sqrt(np.diag(md.sigma))
    print(f"N={md.n_assets}  vol range=[{vols.min():.1%}, {vols.max():.1%}]  "
          f"mu range=[{md.mu.min():.1%}, {md.mu.max():.1%}]")
    corr = md.sigma / np.outer(vols, vols)
    print(f"mean pairwise corr = {corr[np.triu_indices_from(corr, 1)].mean():.2f}")
