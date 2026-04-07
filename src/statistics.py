"""
Factor recovery and forecast closeness statistics.

Implements:
  - R^2_{Fhat,F} (Eq. 17): factor recovery statistic
  - S^2_{yhat,ytilde} (Eq. 18): forecast closeness statistic

Custom — Context7 found no library equivalent (paper Eq. 17-18).
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


def compute_r2_factor_recovery(
    F_hat: np.ndarray,
    F_true: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Compute the factor recovery statistic R^2_{Fhat,F} (paper Eq. 17).

    R^2_{Fhat,F} = tr(F_hat' P_F F_hat) / tr(F_hat' F_hat)

    where P_F = F_true (F_true' F_true)^{-1} F_true' is the projection
    onto the column space of F_true.

    Custom — Context7 found no library equivalent (paper Eq. 17).

    Parameters
    ----------
    F_hat : np.ndarray, shape (T, k)
        Estimated factor matrix.
    F_true : np.ndarray, shape (T, r_true)
        True static factor matrix.

    Returns
    -------
    r2 : float
        R^2_{Fhat,F} statistic in [0, 1].
    numerator : float
        tr(F_hat' P_F F_hat).
    denominator : float
        tr(F_hat' F_hat).
    """
    T, k = F_hat.shape
    T2, r_true = F_true.shape
    assert T == T2, f"T mismatch: F_hat has {T}, F_true has {T2}"

    # Projection matrix P_F = F_true (F_true' F_true)^{-1} F_true'
    # tr(F_hat' P_F F_hat) = ||P_F F_hat||^2_F = ||F_true^+ F_hat||^2_F * ||F_true||^2
    # More efficiently: use QR decomposition of F_true
    Q, _ = np.linalg.qr(F_true, mode="reduced")  # Q: (T, r_true), orthonormal columns
    # P_F F_hat = Q Q' F_hat
    QTF = Q.T @ F_hat  # (r_true, k)
    numerator = np.sum(QTF ** 2)  # tr(F_hat' Q Q' F_hat) = ||Q' F_hat||^2_F

    denominator = np.sum(F_hat ** 2)  # tr(F_hat' F_hat)

    if denominator < 1e-15:
        return 0.0, numerator, denominator

    r2 = numerator / denominator
    # Clip to [0, 1] for numerical safety
    r2 = float(np.clip(r2, 0.0, 1.0))
    return r2, float(numerator), float(denominator)


def compute_forecast_closeness(
    y_hat: float,
    y_tilde: float,
) -> Tuple[float, float]:
    """
    Compute per-replication components for S^2_{yhat,ytilde} (paper Eq. 18).

    Returns the squared difference and the squared benchmark for aggregation.

    Custom — Context7 found no library equivalent (paper Eq. 18).

    Parameters
    ----------
    y_hat : float
        Feasible forecast (based on estimated factors).
    y_tilde : float
        Infeasible benchmark forecast (based on true factors).

    Returns
    -------
    sq_diff : float
        (y_hat - y_tilde)^2
    sq_tilde : float
        y_tilde^2
    """
    sq_diff = (y_hat - y_tilde) ** 2
    sq_tilde = y_tilde ** 2
    return float(sq_diff), float(sq_tilde)


def aggregate_s2(
    sq_diffs: np.ndarray,
    sq_tildes: np.ndarray,
    version: str = "tilde",
) -> float:
    """
    Aggregate per-replication components into S^2_{yhat,ytilde}.

    Two candidate versions (paper Eq. 18 is ambiguous):
      version='tilde': S^2 = 1 - E[(y_hat - y_tilde)^2] / E[y_tilde^2]
      version='hat':   S^2 = 1 - E[(y_hat - y_tilde)^2] / E[y_hat^2]

    Custom — Context7 found no library equivalent (paper Eq. 18).

    Parameters
    ----------
    sq_diffs : np.ndarray, shape (n_reps,)
        Per-replication (y_hat - y_tilde)^2 values.
    sq_tildes : np.ndarray, shape (n_reps,)
        Per-replication y_tilde^2 values (used for version='tilde').
    version : str
        'tilde' or 'hat'.

    Returns
    -------
    s2 : float
        S^2 statistic.
    """
    mean_sq_diff = np.mean(sq_diffs)
    if version == "tilde":
        denom = np.mean(sq_tildes)
    else:
        raise ValueError(f"Unknown version: {version}. Use 'tilde'.")

    if denom < 1e-15:
        return 0.0

    s2 = 1.0 - mean_sq_diff / denom
    return float(s2)


def compute_infeasible_forecast(
    y: np.ndarray,
    F_true: np.ndarray,
    T: int,
) -> float:
    """
    Compute the infeasible one-step-ahead forecast using true factors.

    Regresses y_{t+1} on F_true_t for t=1,...,T-1 (using observations
    where both y_{t+1} and F_t are available), then forecasts y_{T+1|T}.

    Parameters
    ----------
    y : np.ndarray, shape (T+1,)
        Target series. y[t] = y_t for t=0,...,T. y[0] is unused.
    F_true : np.ndarray, shape (T, r_true)
        True static factor matrix. Row t corresponds to time t+1 (1-indexed).
    T : int
        Number of time periods.

    Returns
    -------
    y_tilde : float
        Infeasible forecast y_tilde_{T+1|T}.
    """
    # Regression: y[t+1] on F_true[t-1] for t=1,...,T-1
    # i.e., y[2],...,y[T] on F_true[0],...,F_true[T-2]
    # (y is 0-indexed: y[1]=y_1, ..., y[T]=y_T, y[T+1] is what we forecast)
    # Wait: y has shape (T+1,), y[0] unused, y[1],...,y[T] are the T observations
    # We want to forecast y[T+1] but y only goes to index T.
    # Actually y has T+1 elements: y[0] unused, y[1],...,y[T] are y_1,...,y_T
    # The target is y_{T+1} which is y[T] in 0-indexed if we shift...
    # Let me re-read: y[t+1] = signal + epsilon_{t+1} for t=0,...,T-1
    # So y[1],...,y[T] are the T observations of y_{1},...,y_{T}
    # The forecast target is y_{T+1} = y[T] in the array? No...
    # In simulate_dgp: y has shape (T+1,), y[0] unused, y[t+1] is set for t=0,...,T-1
    # So y[1],...,y[T] are y_1,...,y_T. The forecast target y_{T+1} is NOT in the array.
    # For the Monte Carlo, we need to generate y[T+1] separately or use y[T] as the target.
    # Actually in the DGP, y[t] = y_t for t=1,...,T, and we forecast y_{T+1}.
    # The array y has indices 0,...,T, so y[T] = y_T is the last observation.
    # We need y_{T+1} which would be y[T+1] but the array only goes to T.
    # Let me check: in simulate_dgp, y = np.zeros(T+1), and y[t+1] is set for t=0,...,T-1
    # So y[1],...,y[T] are set. y[0] is 0 (unused). y[T+1] is NOT generated.
    # For the forecast, we need to generate y_{T+1} separately.
    # But for the infeasible forecast, we just need the regression coefficients.
    # The forecast is y_tilde_{T+1|T} = beta_hat_0 + beta_hat_F' F_true[T-1]
    # (F_true[T-1] is the last row, corresponding to t=T)

    # Regression sample: t=1,...,T-1 (use y[2],...,y[T] as targets, F_true[0],...,F_true[T-2] as regressors)
    # y[t+1] for t=1,...,T-1 means y[2],...,y[T]
    # F_true[t-1] for t=1,...,T-1 means F_true[0],...,F_true[T-2]
    # Wait, let me be more careful:
    # We want: y_{t+1} = beta_0 + beta_F' F_t + error for t=1,...,T-1
    # y_{t+1} = y[t+1] for t=1,...,T-1 => y[2],...,y[T]
    # F_t = F_true[t-1] (0-indexed) for t=1,...,T-1 => F_true[0],...,F_true[T-2]

    y_target = y[2:T + 1]  # y_2,...,y_T (length T-1)
    F_reg = F_true[:T - 1]  # F_1,...,F_{T-1} (length T-1)

    X_reg = np.column_stack([np.ones(T - 1), F_reg])
    beta, _, _, _ = np.linalg.lstsq(X_reg, y_target, rcond=None)

    # Forecast: y_tilde_{T+1|T} = beta_0 + beta_F' F_T
    F_T = F_true[T - 1]  # F_T (last row)
    y_tilde = beta[0] + F_T @ beta[1:]
    return float(y_tilde)


def compute_feasible_forecast(
    y: np.ndarray,
    F_hat: np.ndarray,
    r_hat: int,
    T: int,
) -> float:
    """
    Compute the feasible one-step-ahead forecast using estimated factors.

    Parameters
    ----------
    y : np.ndarray, shape (T+1,) or longer
        Target series. y[t] = y_t for t=1,...,T.
    F_hat : np.ndarray, shape (T, k)
        Estimated factor matrix.
    r_hat : int
        Selected number of factors.
    T : int
        Number of time periods.

    Returns
    -------
    y_hat : float
        Feasible forecast y_hat_{T+1|T}.
    """
    if r_hat == 0:
        # Intercept-only model
        y_target = y[2:T + 1]
        y_hat = float(np.mean(y_target))
        return y_hat

    y_target = y[2:T + 1]  # y_2,...,y_T (length T-1)
    F_reg = F_hat[:T - 1, :r_hat]  # F_1,...,F_{T-1}

    X_reg = np.column_stack([np.ones(T - 1), F_reg])
    beta, _, _, _ = np.linalg.lstsq(X_reg, y_target, rcond=None)

    F_T = F_hat[T - 1, :r_hat]
    y_hat = beta[0] + F_T @ beta[1:]
    return float(y_hat)
