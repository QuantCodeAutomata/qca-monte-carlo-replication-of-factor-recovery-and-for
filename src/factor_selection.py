"""
Factor number selection criteria.

Implements:
  - Bai & Ng (2002) information criteria ICp1, ICp2, ICp3
  - Forecasting-equation AIC and BIC

Custom — Context7 found no library equivalent for Bai-Ng panel IC criteria.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


def bai_ng_criteria(
    V_hat: np.ndarray,
    N: int,
    T: int,
    k_max: int,
) -> Tuple[int, int, int]:
    """
    Compute Bai & Ng (2002) information criteria ICp1, ICp2, ICp3.

    For each candidate j=0,...,k_max:
        IC_pell(j) = ln(V_hat_j) + j * g_ell(N, T)

    Penalty functions:
        g1(N,T) = ((N+T)/(NT)) * ln(NT/(N+T))
        g2(N,T) = ((N+T)/(NT)) * ln(C_NT^2)
        g3(N,T) = ln(C_NT^2) / C_NT^2
    where C_NT^2 = min(N, T).

    Custom — Context7 found no library equivalent (Bai & Ng 2002 Eq. 9-11).

    Parameters
    ----------
    V_hat : np.ndarray, shape (k_max+1,)
        Objective function values V_hat_j for j=0,...,k_max.
    N : int
        Number of cross-sectional units.
    T : int
        Number of time periods.
    k_max : int
        Maximum candidate factor dimension.

    Returns
    -------
    r_hat_ic1 : int
        Selected factor count by ICp1.
    r_hat_ic2 : int
        Selected factor count by ICp2.
    r_hat_ic3 : int
        Selected factor count by ICp3.
    """
    assert len(V_hat) == k_max + 1, (
        f"V_hat length {len(V_hat)} != k_max+1={k_max+1}"
    )

    C_NT2 = float(min(N, T))

    g1 = ((N + T) / (N * T)) * np.log(N * T / (N + T))
    g2 = ((N + T) / (N * T)) * np.log(C_NT2)
    g3 = np.log(C_NT2) / C_NT2

    # Compute IC values for j=0,...,k_max
    # For j=0, V_hat[0] is the total variance (no factors)
    j_range = np.arange(k_max + 1)

    # Avoid log(0) by flooring V_hat
    V_safe = np.maximum(V_hat, 1e-15)

    ic1 = np.log(V_safe) + j_range * g1
    ic2 = np.log(V_safe) + j_range * g2
    ic3 = np.log(V_safe) + j_range * g3

    r_hat_ic1 = int(np.argmin(ic1))
    r_hat_ic2 = int(np.argmin(ic2))
    r_hat_ic3 = int(np.argmin(ic3))

    return r_hat_ic1, r_hat_ic2, r_hat_ic3


def forecasting_aic_bic(
    y: np.ndarray,
    F_hat: np.ndarray,
    k_max: int,
) -> Tuple[int, int]:
    """
    Select factor count by minimizing AIC or BIC of the forecasting regression.

    For each candidate j=0,...,k_max, regress y on the first j estimated
    factors (plus intercept) and compute Gaussian-likelihood-based AIC/BIC.

    AIC = T_reg * ln(RSS/T_reg) + 2 * (j + 1)
    BIC = T_reg * ln(RSS/T_reg) + (j + 1) * ln(T_reg)

    where T_reg is the number of regression observations and (j+1) counts
    the j factor coefficients plus the intercept.

    Custom — Context7 found no library equivalent for this specific criterion.

    Parameters
    ----------
    y : np.ndarray, shape (T_reg,)
        Target variable for the regression (y_{t+h} for t=1,...,T_reg).
    F_hat : np.ndarray, shape (T_reg, k_max)
        Estimated factors aligned with y.
    k_max : int
        Maximum candidate factor dimension.

    Returns
    -------
    r_hat_aic : int
        Selected factor count by AIC.
    r_hat_bic : int
        Selected factor count by BIC.
    """
    T_reg = len(y)
    assert F_hat.shape[0] == T_reg, (
        f"F_hat rows {F_hat.shape[0]} != T_reg={T_reg}"
    )
    assert F_hat.shape[1] >= k_max, (
        f"F_hat columns {F_hat.shape[1]} < k_max={k_max}"
    )

    aic_vals = np.zeros(k_max + 1)
    bic_vals = np.zeros(k_max + 1)

    for j in range(k_max + 1):
        if j == 0:
            # Intercept-only model
            X_reg = np.ones((T_reg, 1))
        else:
            X_reg = np.column_stack([np.ones(T_reg), F_hat[:, :j]])

        n_params = X_reg.shape[1]  # j factors + 1 intercept

        # OLS
        try:
            beta, _, _, _ = np.linalg.lstsq(X_reg, y, rcond=None)
            residuals = y - X_reg @ beta
        except np.linalg.LinAlgError:
            aic_vals[j] = np.inf
            bic_vals[j] = np.inf
            continue

        rss = np.sum(residuals ** 2)
        rss = max(rss, 1e-15)  # numerical floor

        log_likelihood_term = T_reg * np.log(rss / T_reg)
        aic_vals[j] = log_likelihood_term + 2.0 * n_params
        bic_vals[j] = log_likelihood_term + n_params * np.log(T_reg)

    r_hat_aic = int(np.argmin(aic_vals))
    r_hat_bic = int(np.argmin(bic_vals))

    return r_hat_aic, r_hat_bic


def select_factors(
    V_hat: np.ndarray,
    F_hat: np.ndarray,
    y_reg: np.ndarray,
    N: int,
    T: int,
    k_max: int,
    r_true: int,
) -> dict:
    """
    Apply all factor selection rules and return selected counts.

    Parameters
    ----------
    V_hat : np.ndarray, shape (k_max+1,)
        PC objective values.
    F_hat : np.ndarray, shape (T, k_max)
        Estimated factors.
    y_reg : np.ndarray, shape (T_reg,)
        Target for forecasting regression (aligned with F_hat[:T_reg]).
    N : int
        Number of predictors.
    T : int
        Number of time periods.
    k_max : int
        Maximum candidate dimension.
    r_true : int
        True static factor dimension.

    Returns
    -------
    dict with keys: 'true', 'ic1', 'ic2', 'ic3', 'aic', 'bic'
        Each value is the selected factor count (int).
    """
    T_reg = len(y_reg)
    r_ic1, r_ic2, r_ic3 = bai_ng_criteria(V_hat, N, T, k_max)
    r_aic, r_bic = forecasting_aic_bic(y_reg, F_hat[:T_reg], k_max)

    return {
        "true": r_true,
        "ic1": r_ic1,
        "ic2": r_ic2,
        "ic3": r_ic3,
        "aic": r_aic,
        "bic": r_bic,
    }
