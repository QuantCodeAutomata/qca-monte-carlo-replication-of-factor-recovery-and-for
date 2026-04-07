"""
Data Generating Process (DGP) for Monte Carlo simulations.

Implements the DGP described in Stock & Watson (2002) Table 1 scenarios,
including dynamic factors, time-varying loadings, spatial dependence,
and GARCH-type heteroskedasticity.

Custom — Context7 found no library equivalent for this specific DGP (paper Eq. 13-18).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ScenarioParams:
    """
    Parameters for a single Monte Carlo scenario (Table 1 row).

    Attributes
    ----------
    T : int
        Number of time periods.
    N : int
        Number of cross-sectional units (predictors).
    r_tilde : int
        Number of dynamic factors.
    k : int
        Maximum candidate factor dimension for PC estimation.
    q : int
        Number of lags in the dynamic factor structure.
        True static factor dimension = r_tilde * (1 + q).
    alpha : float
        AR(1) coefficient for factor dynamics. |alpha| < 1 required.
    a : float
        AR(1) coefficient for idiosyncratic disturbances.
    b : float
        Spatial MA coefficient for cross-sectional dependence.
    c : float
        Drift coefficient for time-varying loadings (c=0 => time-invariant).
    pi : float
        Probability that a predictor has zero factor loading (irrelevant).
    delta_0 : float
        GARCH intercept (unconditional variance target when delta_1=delta_2=0).
    delta_1 : float
        GARCH persistence coefficient. delta_1 + delta_2 < 1 required.
    delta_2 : float
        GARCH ARCH coefficient.
    label : str
        Scenario label (e.g., 'A1', 'B1', etc.).
    """

    T: int
    N: int
    r_tilde: int
    k: int
    q: int
    alpha: float
    a: float
    b: float
    c: float
    pi: float
    delta_0: float
    delta_1: float
    delta_2: float
    label: str = ""

    def __post_init__(self) -> None:
        assert abs(self.alpha) < 1.0, f"alpha={self.alpha} must satisfy |alpha| < 1"
        assert self.delta_1 + self.delta_2 < 1.0, (
            f"GARCH stationarity violated: delta_1+delta_2="
            f"{self.delta_1+self.delta_2} >= 1"
        )
        assert 0.0 <= self.pi <= 1.0, f"pi={self.pi} must be in [0,1]"
        assert self.k >= self.r_tilde * (1 + self.q), (
            f"k={self.k} must be >= r_true={self.r_tilde*(1+self.q)}"
        )

    @property
    def r_true(self) -> int:
        """True static factor dimension."""
        return self.r_tilde * (1 + self.q)


def lambda_star(R2_i: float, r_true: int) -> float:
    """
    Calibrated loading scale mapping lambda*(R_i^2).

    Derived so that the expected R^2 of predictor i (fraction of variance
    explained by common factors) equals R_i^2 in the baseline scenario
    (alpha=0, a=b=0, homoskedastic errors with Var(e_it)=1).

    For x_it = lambda_i' F_t + e_it with F_t ~ N(0, I_{r_true}),
    Var(lambda_i' F_t) = r_true * lambda*(R_i^2)^2 (in expectation over
    the N(0,1) loading draws), and Var(e_it) = 1, so:

        R_i^2 = r_true * lambda*(R_i^2)^2 / (r_true * lambda*(R_i^2)^2 + 1)
        => lambda*(R_i^2) = sqrt(R_i^2 / (r_true * (1 - R_i^2)))

    Custom — Context7 found no library equivalent (paper calibration dependency).

    Parameters
    ----------
    R2_i : float
        Target R^2 for predictor i. Must be in (0, 1).
    r_true : int
        True static factor dimension (r_tilde * (1 + q)).

    Returns
    -------
    float
        Scaling factor lambda*(R_i^2).
    """
    if R2_i <= 0.0:
        return 0.0
    if R2_i >= 1.0:
        return 1e6  # effectively infinite loading
    return np.sqrt(R2_i / (r_true * (1.0 - R2_i)))


def simulate_dgp(
    params: ScenarioParams,
    rng: np.random.Generator,
) -> dict:
    """
    Simulate one replication of the DGP for a given scenario.

    Implements Steps 2-7 of the paper's Monte Carlo methodology:
      - Dynamic factors f_t = alpha * f_{t-1} + u_t
      - Time-varying loadings lambda_{ijt} = lambda_{ij,t-1} + (c/T) * zeta_{ijt}
      - GARCH idiosyncratic volatility
      - Spatial MA cross-sectional dependence (Eq. 13)
      - Predictors x_it = sum_{j=0}^q lambda_{ijt}' f_{t-j} + e_it
      - Target y_{t+1} = sum_{j=0}^q iota' f_{t-j} + epsilon_{t+1}

    Custom — Context7 found no library equivalent for this DGP (paper Eq. 13-18).

    Parameters
    ----------
    params : ScenarioParams
        Scenario-specific parameters.
    rng : np.random.Generator
        NumPy random generator for reproducibility.

    Returns
    -------
    dict with keys:
        'X'       : np.ndarray, shape (T, N) — predictor panel
        'y'       : np.ndarray, shape (T+1,) — target series (y[t] = y_t)
        'F_true'  : np.ndarray, shape (T, r_true) — true static factors
        'params'  : ScenarioParams
    """
    T = params.T
    N = params.N
    r_tilde = params.r_tilde
    q = params.q
    r_true = params.r_true
    alpha = params.alpha
    a = params.a
    b = params.b
    c = params.c
    pi = params.pi
    delta_0 = params.delta_0
    delta_1 = params.delta_1
    delta_2 = params.delta_2

    # ------------------------------------------------------------------ #
    # Step 2: Initialize DGP
    # ------------------------------------------------------------------ #

    # Draw R_i^2 for each predictor
    R2 = np.where(
        rng.uniform(size=N) < pi,
        0.0,
        rng.uniform(low=0.1, high=0.8, size=N),
    )

    # Draw provisional loading shocks lambda_tilde_{ij0} ~ N(0,1)
    # Shape: (N, r_tilde, q+1) — one set per lag j=0,...,q
    lambda_tilde_init = rng.standard_normal(size=(N, r_tilde, q + 1))

    # Compute initial loadings lambda_{ij0} = lambda*(R_i^2) * lambda_tilde_{ij0}
    # lambda_star_i is a scalar per predictor
    lambda_star_i = np.array([lambda_star(R2[i], r_true) for i in range(N)])

    # lambda[i, j, t] = loading of predictor i on factor j at lag-index l
    # Initialize at t=0
    # Shape: (N, r_tilde, q+1, T+1) — last dim is time (0=init, 1..T=simulation)
    # To save memory, store only current and previous loading
    lambda_curr = lambda_star_i[:, None, None] * lambda_tilde_init  # (N, r_tilde, q+1)

    # Pre-sample factors: simulate burn-in to approximate stationary distribution
    # Need q lags before t=1, so simulate burn_in + q steps
    burn_in = max(50, q + 1)
    f_presample = np.zeros((burn_in + q, r_tilde))
    u_presample = rng.standard_normal(size=(burn_in + q, r_tilde))
    for t in range(1, burn_in + q):
        f_presample[t] = alpha * f_presample[t - 1] + u_presample[t]
    # f_presample[-q:] are the q pre-sample factors (f_{0}, f_{-1}, ..., f_{-(q-1)})
    f_lag_buffer = f_presample[-q:] if q > 0 else np.zeros((0, r_tilde))

    # Initialize GARCH states: sigma_{i0}^2 = unconditional variance
    # Unconditional variance = delta_0 / (1 - delta_1 - delta_2)
    if delta_1 + delta_2 > 0:
        sigma2_uncond = delta_0 / (1.0 - delta_1 - delta_2)
    else:
        sigma2_uncond = delta_0
    sigma2_curr = np.full(N, sigma2_uncond)  # sigma_{i,0}^2
    v_curr = np.zeros(N)  # v_{i,0} = 0

    # Initialize idiosyncratic AR state e_{i,0} = 0
    e_curr = np.zeros(N)

    # ------------------------------------------------------------------ #
    # Steps 3-7: Simulate T periods
    # ------------------------------------------------------------------ #

    # Storage
    F_dynamic = np.zeros((T, r_tilde))  # dynamic factors f_t
    X = np.zeros((T, N))
    y = np.zeros(T + 1)  # y[0] unused; y[t] = y_t for t=1,...,T

    # Innovations
    u_all = rng.standard_normal(size=(T, r_tilde))          # factor innovations
    zeta_all = rng.standard_normal(size=(T, N, r_tilde, q + 1))  # loading drift
    eta_all = rng.standard_normal(size=(T, N))              # idiosyncratic
    epsilon_all = rng.standard_normal(size=T + 1)           # target noise

    # Factor at t=0 (last pre-sample value)
    f_prev = f_presample[-1] if burn_in + q > 0 else np.zeros(r_tilde)

    # Separate lag buffer for y: y_{t+1} = sum_{j=0}^q iota' f_{t-j} + epsilon_{t+1}
    # At the start of iteration t (0-indexed), f_prev = f_t (1-indexed).
    # y_lag_buffer_y[j] = f_{t-j} for j=1,...,q (1-indexed).
    # Initialize: f_lag_buffer_y = [f_{-1}, ..., f_{-(q)}] = f_presample[-(q+1):-1]
    if q > 0:
        # f_presample has shape (burn_in+q, r_tilde); last q+1 elements are f_{-(q)},...,f_0
        # We need [f_{-1}, ..., f_{-(q)}] = f_presample[-(q+1):-1]
        f_lag_buffer_y = f_presample[-(q + 1):-1]  # shape (q, r_tilde)
    else:
        f_lag_buffer_y = np.zeros((0, r_tilde))

    iota = np.ones(r_tilde)

    for t in range(T):
        # Step 3: Dynamic factors
        f_t = alpha * f_prev + u_all[t]
        F_dynamic[t] = f_t

        # Build lag buffer for predictors: [f_t, f_{t-1}, ..., f_{t-q}]
        # f_t is the factor just computed (= f_{t+1} in 1-indexed).
        # f_lag_buffer contains [f_{t-1}, ..., f_{t-q}] (1-indexed) before update.
        if q > 0:
            f_lags = np.vstack([f_t[None, :], f_lag_buffer])  # (q+1, r_tilde)
        else:
            f_lags = f_t[None, :]  # (1, r_tilde)

        # Step 4: Update loadings (time-varying)
        if c != 0.0:
            lambda_curr = lambda_curr + (c / T) * zeta_all[t]  # (N, r_tilde, q+1)

        # Step 5: Idiosyncratic disturbances
        # GARCH update: sigma_{it}^2 = delta_0 + delta_1*sigma_{i,t-1}^2 + delta_2*v_{i,t-1}^2
        sigma2_new = delta_0 + delta_1 * sigma2_curr + delta_2 * v_curr ** 2
        sigma2_new = np.maximum(sigma2_new, 1e-10)  # numerical floor
        eta_t = eta_all[t]
        v_new = np.sqrt(sigma2_new) * eta_t  # v_{it} = sigma_{it} * eta_{it}

        # Spatial MA: (1 - a*L) e_it = (1+b^2)*v_it + b*v_{i+1,t} + b*v_{i-1,t}
        # Zero-padding boundary: v_{0,t} = v_{N+1,t} = 0
        # Custom — Context7 found no library equivalent (paper Eq. 13)
        v_padded = np.concatenate([[0.0], v_new, [0.0]])  # length N+2
        spatial_innov = (1.0 + b ** 2) * v_new + b * v_padded[2:] + b * v_padded[:-2]
        # AR(1) recursion: e_it = a * e_{i,t-1} + spatial_innov_it
        e_new = a * e_curr + spatial_innov

        # Step 6: Predictors x_it = sum_{j=0}^q lambda_{ijt}' f_{t-j} + e_it
        # lambda_curr shape: (N, r_tilde, q+1)
        # f_lags shape: (q+1, r_tilde)
        # signal[i] = sum_{j=0}^q lambda_curr[i, :, j] . f_lags[j, :]
        signal = np.einsum("irj,jr->i", lambda_curr, f_lags)
        X[t] = signal + e_new

        # Step 7: Target y_{t+1} = sum_{j=0}^q iota' f_{t-j} + epsilon_{t+1}
        # Paper model: y_{t+1} depends on f_t (same time as x_{it}).
        # In the loop, f_prev = f_t (1-indexed), f_lag_buffer_y = [f_{t-1},...,f_{t-q}].
        # Custom — Context7 found no library equivalent (paper Eq. 7).
        if q > 0:
            f_y_lags = np.vstack([f_prev[None, :], f_lag_buffer_y])  # (q+1, r_tilde)
        else:
            f_y_lags = f_prev[None, :]  # (1, r_tilde)
        y_signal = sum(iota @ f_y_lags[j] for j in range(q + 1))
        y[t + 1] = y_signal + epsilon_all[t + 1]

        # Update lag buffers
        if q > 0:
            f_lag_buffer = np.vstack([f_t[None, :], f_lag_buffer[:-1]])
            f_lag_buffer_y = np.vstack([f_prev[None, :], f_lag_buffer_y[:-1]])

        # Update states
        f_prev = f_t
        sigma2_curr = sigma2_new
        v_curr = v_new
        e_curr = e_new

    # Build true static factor matrix F_true: shape (T, r_true)
    # F_t = (f_t', f_{t-1}', ..., f_{t-q}')
    # We need to reconstruct the lag structure from F_dynamic
    F_true = np.zeros((T, r_true))
    for t in range(T):
        for j in range(q + 1):
            if t - j >= 0:
                F_true[t, j * r_tilde:(j + 1) * r_tilde] = F_dynamic[t - j]
            # else: pre-sample factors (already 0 by initialization — acceptable approximation)

    return {
        "X": X,
        "y": y,
        "F_true": F_true,
        "params": params,
    }
