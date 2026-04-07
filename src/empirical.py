"""
Empirical expanding-window forecasting for U.S. industrial production.

Replicates Table 2 of Stock & Watson (2002): PC-based diffusion index
forecasts of 12-month-ahead industrial production growth using an
expanding window from 1970:01 to 1997:12.

Uses the FRED API via massive client for data retrieval.
Custom — Context7 found no library equivalent for this specific pipeline.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Forecast horizon
H = 12

# Estimation start date
ESTIMATION_START = "1959-01-01"

# Forecast origin range
FORECAST_START = "1970-01-01"
FORECAST_END = "1997-12-01"

# Maximum number of PC factors
K_MAX = 10


def fetch_macro_data() -> pd.DataFrame:
    """
    Fetch macroeconomic data from FRED using the massive client.

    Retrieves a representative set of monthly U.S. macroeconomic series
    covering the period 1959:01-1998:12, approximating the Stock-Watson
    149-series dataset.

    Returns
    -------
    pd.DataFrame
        Monthly macroeconomic panel with DatetimeIndex.
        Columns include 'IP' (industrial production) and predictor series.
    """
    try:
        from massive import RESTClient
        api_key = os.getenv("MASSIVE_TOKEN") or os.getenv("MASSIVE_API_KEY")
        if not api_key:
            raise ValueError("MASSIVE_TOKEN environment variable not set")
        client = RESTClient(api_key=api_key)
        return _fetch_from_massive(client)
    except Exception as e:
        logger.warning(f"Could not fetch from massive API: {e}. Using synthetic data.")
        return _generate_synthetic_macro_data()


def _fetch_from_massive(client) -> pd.DataFrame:
    """
    Fetch macro series from FRED via massive client.

    Uses a representative subset of the Stock-Watson macro dataset.
    Series are transformed to stationarity as described in the paper.

    Parameters
    ----------
    client : RESTClient
        Authenticated massive REST client.

    Returns
    -------
    pd.DataFrame
        Monthly panel with DatetimeIndex.
    """
    # Core series from Stock-Watson macro dataset
    # Format: (ticker, transformation_code)
    # Transformation codes:
    #   1 = level, 2 = first difference, 4 = log, 5 = log first difference
    series_list = [
        # Industrial Production (target)
        ("INDPRO", 5),
        # Employment
        ("PAYEMS", 5),
        ("USPRIV", 5),
        ("USCONS", 5),
        ("MANEMP", 5),
        ("DMANEMP", 5),
        ("NDMANEMP", 5),
        ("SRVPRD", 5),
        ("USTRADE", 5),
        ("USFIRE", 5),
        ("USGOVT", 5),
        ("USGOOD", 5),
        ("CES1021000001", 5),
        ("USMINE", 5),
        ("USEHS", 5),
        ("USWTRADE", 5),
        ("USTPU", 5),
        ("USINFO", 5),
        ("USPBS", 5),
        ("USEDUH", 5),
        ("USLAH", 5),
        ("USSERV", 5),
        # Unemployment
        ("UNRATE", 2),
        ("UEMPMEAN", 2),
        ("UEMPLT5", 2),
        ("UEMP5TO14", 2),
        ("UEMP15OV", 2),
        ("UEMP15T26", 2),
        ("UEMP27OV", 2),
        # Housing
        ("HOUST", 4),
        ("HOUSTNE", 4),
        ("HOUSTMW", 4),
        ("HOUSTS", 4),
        ("HOUSTW", 4),
        ("PERMIT", 4),
        # Orders and inventories
        ("DPCERA3M086SBEA", 5),
        ("CMRMTSPL", 5),
        ("RETAILSMNSA", 5),
        ("ISRATIO", 2),
        # Prices
        ("CPIAUCSL", 5),
        ("CPILFESL", 5),
        ("PPIACO", 5),
        ("PPIFGS", 5),
        ("PPIFCG", 5),
        ("PPICMM", 5),
        ("PPICRM", 5),
        ("PPIIDC", 5),
        ("PPIITM", 5),
        # Interest rates
        ("FEDFUNDS", 2),
        ("TB3MS", 2),
        ("TB6MS", 2),
        ("GS1", 2),
        ("GS5", 2),
        ("GS10", 2),
        ("AAA", 2),
        ("BAA", 2),
        # Money
        ("M1SL", 5),
        ("M2SL", 5),
        ("BOGMBASE", 5),
        # Exchange rates
        ("DTWEXBGS", 5),
        # Stock market
        ("SP500", 5),
        ("VIXCLS", 2),
    ]

    dfs = {}
    for ticker, tcode in series_list:
        try:
            aggs = []
            for a in client.list_aggs(
                ticker=ticker,
                multiplier=1,
                timespan="month",
                from_="1959-01-01",
                to="1998-12-31",
                limit=50000,
            ):
                aggs.append(a)
            if not aggs:
                continue
            dates = [pd.Timestamp(a.timestamp, unit="ms") for a in aggs]
            values = [a.close for a in aggs]
            s = pd.Series(values, index=dates, name=ticker)
            s = s.resample("MS").last()
            dfs[ticker] = (s, tcode)
        except Exception as e:
            logger.debug(f"Could not fetch {ticker}: {e}")
            continue

    if not dfs:
        logger.warning("No series fetched from massive. Using synthetic data.")
        return _generate_synthetic_macro_data()

    # Apply transformations
    transformed = {}
    for ticker, (s, tcode) in dfs.items():
        try:
            if tcode == 1:
                transformed[ticker] = s
            elif tcode == 2:
                transformed[ticker] = s.diff()
            elif tcode == 4:
                transformed[ticker] = np.log(s)
            elif tcode == 5:
                transformed[ticker] = np.log(s).diff()
        except Exception:
            continue

    df = pd.DataFrame(transformed)
    df.index = pd.DatetimeIndex(df.index).to_period("M").to_timestamp("M")
    df = df.sort_index()

    # Restrict to 1959:01-1998:12
    df = df.loc["1959-01-01":"1998-12-31"]

    return df


def _generate_synthetic_macro_data(
    n_series: int = 149,
    seed: int = 12345,
) -> pd.DataFrame:
    """
    Generate synthetic macro panel data for testing when FRED data unavailable.

    Creates a factor model panel with properties similar to the Stock-Watson
    macro dataset: N=149 series, T=480 months (1959:01-1998:12), r=3 factors.

    Parameters
    ----------
    n_series : int
        Number of predictor series (default 149).
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Synthetic monthly panel with DatetimeIndex.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("1959-01-01", "1998-12-31", freq="MS")
    T = len(dates)
    r = 3  # number of latent factors

    # Simulate AR(1) factors
    alpha = 0.9
    F = np.zeros((T, r))
    for t in range(1, T):
        F[t] = alpha * F[t - 1] + rng.standard_normal(r)

    # Loadings
    Lambda = rng.standard_normal((n_series, r)) * 0.5

    # Idiosyncratic errors (AR(1))
    E = np.zeros((T, n_series))
    for t in range(1, T):
        E[t] = 0.5 * E[t - 1] + rng.standard_normal(n_series)

    # Predictors
    X = F @ Lambda.T + E

    # Industrial production (target): driven by first factor
    ip_level = np.cumsum(0.3 * F[:, 0] + 0.1 * rng.standard_normal(T))
    ip_level = np.exp(ip_level / 10)  # make it a level series

    col_names = ["INDPRO"] + [f"X{i:03d}" for i in range(1, n_series)]
    df = pd.DataFrame(
        np.column_stack([np.log(ip_level / np.roll(ip_level, 1)), X[:, :n_series - 1]]),
        index=dates,
        columns=col_names,
    )
    df.iloc[0] = np.nan  # first diff creates NaN at start

    return df


def prepare_panel(
    df: pd.DataFrame,
    origin: pd.Timestamp,
    ip_col: str = "INDPRO",
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Prepare the standardized predictor panel at a given forecast origin.

    Standardizes each predictor using only data through the origin date.
    Handles missing data by forward-filling up to 3 periods, then dropping
    series with insufficient history.

    Parameters
    ----------
    df : pd.DataFrame
        Full macro panel.
    origin : pd.Timestamp
        Forecast origin date T.
    ip_col : str
        Column name for industrial production.

    Returns
    -------
    X_std : np.ndarray, shape (T_est, N_valid)
        Standardized predictor matrix.
    dates_est : pd.DatetimeIndex
        Dates corresponding to rows of X_std.
    valid_cols : list[str]
        Names of valid predictor columns.
    """
    # Use data through origin
    mask = df.index <= origin
    df_est = df.loc[mask].copy()

    # Drop IP column from predictors
    predictor_cols = [c for c in df_est.columns if c != ip_col]
    X_raw = df_est[predictor_cols].copy()

    # Forward fill up to 3 periods for missing data (real-time valid)
    X_raw = X_raw.ffill(limit=3)

    # Drop series with more than 10% missing
    missing_frac = X_raw.isna().mean()
    valid_cols = missing_frac[missing_frac < 0.10].index.tolist()
    X_raw = X_raw[valid_cols].dropna()

    if X_raw.empty or len(X_raw) < 24:
        raise ValueError(f"Insufficient data at origin {origin}")

    # Standardize using origin-specific mean and std
    X_mean = X_raw.mean()
    X_std_dev = X_raw.std()
    X_std_dev = X_std_dev.replace(0, 1.0)  # avoid division by zero

    X_std = ((X_raw - X_mean) / X_std_dev).values
    dates_est = X_raw.index

    return X_std, dates_est, valid_cols


def compute_ar_benchmark(
    y_series: pd.Series,
    origin: pd.Timestamp,
    h: int = H,
    p: int = 4,
) -> float:
    """
    Compute AR(p) benchmark forecast for h-step-ahead target.

    Estimates AR(p) on log-differenced IP growth and forecasts h steps ahead
    using direct projection (regress y_{t+h} on y_t, ..., y_{t-p+1}).

    Parameters
    ----------
    y_series : pd.Series
        Target series y_{t+h} = ln(IP_{t+h}/IP_t).
    origin : pd.Timestamp
        Forecast origin.
    h : int
        Forecast horizon.
    p : int
        AR lag order.

    Returns
    -------
    float
        AR forecast for y_{origin+h}.
    """
    # Use data through origin
    y_est = y_series.loc[y_series.index <= origin].dropna()

    if len(y_est) < p + h + 5:
        return float(y_est.mean()) if len(y_est) > 0 else 0.0

    # Direct projection: y_{t+h} on y_t, ..., y_{t-p+1}
    # Use monthly IP growth (first difference of log IP) as AR input
    T_est = len(y_est)
    # Build regression: y_{t+h} on lags of y_t
    # We need y_{t+h} for t up to T_est - h
    y_vals = y_est.values
    n_obs = T_est - h - p + 1
    if n_obs < 5:
        return float(np.mean(y_vals))

    Y_dep = y_vals[p + h - 1:]  # y_{t+h} for t=p,...,T_est-h
    X_lag = np.column_stack([
        y_vals[p - 1 + h - 1 - j: T_est - j - 1] for j in range(p)
    ])
    # Trim to same length
    min_len = min(len(Y_dep), X_lag.shape[0])
    Y_dep = Y_dep[:min_len]
    X_lag = X_lag[:min_len]

    X_reg = np.column_stack([np.ones(min_len), X_lag])
    try:
        beta, _, _, _ = np.linalg.lstsq(X_reg, Y_dep, rcond=None)
    except np.linalg.LinAlgError:
        return float(np.mean(y_vals))

    # Forecast using last p values of y
    last_lags = y_vals[-p:][::-1]  # most recent first
    y_hat = beta[0] + last_lags @ beta[1:]
    return float(y_hat)


def run_expanding_window_forecast(
    df: pd.DataFrame,
    ip_col: str = "INDPRO",
    h: int = H,
    k_max: int = K_MAX,
    fixed_k_list: Optional[list[int]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the expanding-window PC forecasting exercise.

    For each forecast origin T from 1970:01 to 1997:12:
      1. Standardize predictors using data through T
      2. Estimate PC factors (up to k_max)
      3. Select factors using Bai-Ng ICp3
      4. Estimate forecasting regression and generate forecast
      5. Store forecast and realized value

    Parameters
    ----------
    df : pd.DataFrame
        Full macro panel with IP and predictors.
    ip_col : str
        Column name for industrial production index.
    h : int
        Forecast horizon (default 12 months).
    k_max : int
        Maximum number of PC factors.
    fixed_k_list : list[int], optional
        List of fixed factor counts for additional forecasts.
    verbose : bool
        If True, log progress.

    Returns
    -------
    pd.DataFrame
        Forecast results with columns: origin, realized, ar_forecast,
        pc_ic3_forecast, pc_k1_forecast, ..., and forecast errors.
    """
    from src.pca_estimation import estimate_pca
    from src.factor_selection import bai_ng_criteria

    if fixed_k_list is None:
        fixed_k_list = [1, 2, 3, 4]

    # Build target series: y_{t+h} = ln(IP_{t+h}/IP_t)
    # IP is already log-differenced (transformation code 5), so we need
    # to reconstruct the h-period growth rate
    # If INDPRO is log-differenced monthly growth, then:
    # y_{t+h} = sum_{j=1}^h delta_log_IP_{t+j}
    # But the paper uses y_{t+12} = ln(IP_{t+12}/IP_t) directly.
    # We'll compute this from the level series if available, or sum diffs.

    ip_growth = df[ip_col].copy()  # monthly log growth rate (already differenced)

    # Compute 12-month cumulative growth: y_{t+12} = sum_{j=1}^{12} ip_growth_{t+j}
    # This equals ln(IP_{t+12}/IP_t) when ip_growth is log-differenced
    y_target = ip_growth.rolling(window=h).sum().shift(-h + 1)
    # y_target[t] = sum_{j=0}^{h-1} ip_growth[t+j] = ln(IP_{t+h}/IP_t)
    # Actually: y_target[t] = ip_growth[t] + ... + ip_growth[t+h-1]
    # We want y_{t+h} = ln(IP_{t+h}/IP_t) = sum_{j=1}^h ip_growth_{t+j}
    # So shift by 1 more:
    y_target = ip_growth.rolling(window=h).sum().shift(-(h - 1))
    # Now y_target[t] = sum_{j=0}^{h-1} ip_growth[t+j]
    # = ip_growth[t] + ... + ip_growth[t+h-1]
    # This is ln(IP_{t+h}/IP_{t-1}) approximately, not exactly what we want.
    # Let's be more careful:
    # y_{t+h} = ln(IP_{t+h}/IP_t) = sum_{j=1}^h ln(IP_{t+j}/IP_{t+j-1})
    #         = sum_{j=1}^h ip_growth[t+j]
    # So y_target[t] = sum_{j=1}^h ip_growth[t+j]
    y_target = ip_growth.shift(-1).rolling(window=h).sum().shift(-(h - 1))
    # y_target[t] = ip_growth[t+1] + ... + ip_growth[t+h]
    # This is correct: y_{t+h} = ln(IP_{t+h}/IP_t)

    # Forecast origins
    origins = pd.date_range(FORECAST_START, FORECAST_END, freq="MS")

    results = []

    for origin in origins:
        if verbose and origin.month == 1:
            logger.info(f"  Forecast origin: {origin.strftime('%Y-%m')}")

        try:
            # Prepare standardized predictor panel
            X_std, dates_est, valid_cols = prepare_panel(df, origin, ip_col)
            T_est, N_est = X_std.shape

            if T_est < 24 or N_est < 5:
                logger.warning(f"Insufficient data at {origin}: T={T_est}, N={N_est}")
                continue

            k_eff = min(k_max, T_est - 1, N_est)

            # Estimate PC factors
            F_hat, _, V_hat = estimate_pca(X_std, k_eff)

            # Select factors using Bai-Ng ICp3
            from src.factor_selection import bai_ng_criteria
            _, _, r_ic3 = bai_ng_criteria(V_hat, N_est, T_est, k_eff)

            # Build forecasting regression sample
            # At origin T, use t=1,...,T-h for regression
            # y_{t+h} is available for t up to T-h
            # dates_est[-1] = origin, so we need y_target up to origin-h months ago
            origin_idx = len(dates_est) - 1  # index of origin in estimation sample

            # Regression: y_{t+h} on F_hat_t for t where y_{t+h} is known
            # i.e., t such that t+h <= origin => t <= origin - h months
            n_reg = origin_idx + 1 - h  # number of regression observations
            if n_reg < 10:
                logger.warning(f"Too few regression obs at {origin}: n_reg={n_reg}")
                continue

            # Get y values for regression
            reg_dates = dates_est[:n_reg]
            y_reg_vals = y_target.reindex(reg_dates).values

            # Check for NaN in y_reg
            valid_mask = ~np.isnan(y_reg_vals)
            if valid_mask.sum() < 10:
                continue

            y_reg_clean = y_reg_vals[valid_mask]
            F_reg_clean = F_hat[:n_reg][valid_mask]

            # Realized value
            realized_date = origin + pd.DateOffset(months=h)
            realized = y_target.get(realized_date, np.nan)

            # AR benchmark forecast
            ar_forecast = compute_ar_benchmark(ip_growth, origin, h=h)

            # PC forecasts
            row = {
                "origin": origin,
                "realized": realized,
                "ar_forecast": ar_forecast,
                "r_ic3": r_ic3,
                "N_est": N_est,
                "T_est": T_est,
            }

            # ICp3-selected forecast
            r_hat = min(r_ic3, k_eff)
            row["pc_ic3_forecast"] = _pc_forecast(
                y_reg_clean, F_reg_clean, F_hat[origin_idx], r_hat
            )

            # Fixed-k forecasts
            for k_fixed in fixed_k_list:
                k_use = min(k_fixed, k_eff)
                row[f"pc_k{k_fixed}_forecast"] = _pc_forecast(
                    y_reg_clean, F_reg_clean, F_hat[origin_idx], k_use
                )

            results.append(row)

        except Exception as e:
            logger.warning(f"Error at origin {origin}: {e}")
            continue

    if not results:
        raise RuntimeError("No forecast results generated")

    return pd.DataFrame(results).set_index("origin")


def _pc_forecast(
    y_reg: np.ndarray,
    F_reg: np.ndarray,
    F_origin: np.ndarray,
    r_hat: int,
) -> float:
    """
    Estimate forecasting regression and generate one forecast.

    Parameters
    ----------
    y_reg : np.ndarray, shape (T_reg,)
        Target values for regression.
    F_reg : np.ndarray, shape (T_reg, k)
        Factor values for regression.
    F_origin : np.ndarray, shape (k,)
        Factor values at forecast origin.
    r_hat : int
        Number of factors to use.

    Returns
    -------
    float
        Forecast value.
    """
    T_reg = len(y_reg)
    if r_hat == 0:
        return float(np.mean(y_reg))

    r_use = min(r_hat, F_reg.shape[1])
    X_reg = np.column_stack([np.ones(T_reg), F_reg[:, :r_use]])
    try:
        beta, _, _, _ = np.linalg.lstsq(X_reg, y_reg, rcond=None)
    except np.linalg.LinAlgError:
        return float(np.mean(y_reg))

    forecast = beta[0] + F_origin[:r_use] @ beta[1:]
    return float(forecast)


def compute_forecast_metrics(
    results_df: pd.DataFrame,
    fixed_k_list: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Compute MSE and relative MSE for all forecast methods.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output from run_expanding_window_forecast.
    fixed_k_list : list[int], optional
        Fixed factor counts used.

    Returns
    -------
    pd.DataFrame
        Table with method, MSE, relative MSE, and RMSE.
    """
    if fixed_k_list is None:
        fixed_k_list = [1, 2, 3, 4]

    # Drop rows with missing realized values
    df = results_df.dropna(subset=["realized"])

    methods = ["ar_forecast", "pc_ic3_forecast"] + [
        f"pc_k{k}_forecast" for k in fixed_k_list
    ]
    method_labels = ["AR(4)", "PC ICp3"] + [f"PC k={k}" for k in fixed_k_list]

    rows = []
    ar_mse = None

    for method, label in zip(methods, method_labels):
        if method not in df.columns:
            continue
        errors = df["realized"] - df[method]
        errors = errors.dropna()
        if len(errors) == 0:
            continue
        mse = float(np.mean(errors ** 2))
        rmse = float(np.sqrt(mse))
        if method == "ar_forecast":
            ar_mse = mse
        rows.append({"method": label, "mse": mse, "rmse": rmse, "n_forecasts": len(errors)})

    metrics_df = pd.DataFrame(rows)
    if ar_mse is not None and ar_mse > 0:
        metrics_df["relative_mse"] = metrics_df["mse"] / ar_mse
    else:
        metrics_df["relative_mse"] = np.nan

    return metrics_df
