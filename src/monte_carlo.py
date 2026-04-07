"""
Monte Carlo orchestration for Table 1 replication.

Runs 2,000 replications per scenario, computes R^2_{Fhat,F} and
S^2_{yhat,ytilde} for each factor selection rule, and aggregates results.

Custom — Context7 found no library equivalent for this specific MC framework.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.dgp import ScenarioParams, simulate_dgp
from src.pca_estimation import estimate_pca
from src.factor_selection import select_factors
from src.statistics import (
    compute_r2_factor_recovery,
    compute_infeasible_forecast,
    compute_feasible_forecast,
    compute_forecast_closeness,
    aggregate_s2,
)

logger = logging.getLogger(__name__)

SELECTION_RULES = ["true", "ic1", "ic2", "ic3", "aic", "bic"]


def run_single_replication(
    params: ScenarioParams,
    rng: np.random.Generator,
) -> dict:
    """
    Execute one Monte Carlo replication for a given scenario.

    Implements Steps 2-12 of the paper's methodology.

    Parameters
    ----------
    params : ScenarioParams
        Scenario parameters.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    dict with keys:
        'r2_num'    : float — numerator of R^2 (tr(F_hat' P_F F_hat))
        'r2_den'    : float — denominator of R^2 (tr(F_hat' F_hat))
        'sq_diffs'  : dict[str, float] — (y_hat - y_tilde)^2 per rule
        'sq_tildes' : float — y_tilde^2
        'r_selected': dict[str, int] — selected factor counts per rule
        'y_tilde'   : float — infeasible forecast
        'y_hats'    : dict[str, float] — feasible forecasts per rule
    """
    T = params.T
    N = params.N
    k = params.k
    r_true = params.r_true

    # Steps 2-7: Simulate DGP
    sim = simulate_dgp(params, rng)
    X = sim["X"]          # (T, N)
    y = sim["y"]          # (T+1,) — y[1],...,y[T] are y_1,...,y_T
    F_true = sim["F_true"]  # (T, r_true)

    # Step 8: Estimate PC factors
    k_eff = min(k, T - 1, N)  # ensure k doesn't exceed matrix dimensions
    F_hat, Lambda_hat, V_hat = estimate_pca(X, k_eff)

    # Step 9: Factor recovery statistic (using k_eff factors)
    r2, r2_num, r2_den = compute_r2_factor_recovery(F_hat, F_true)

    # Forecasting regression sample: t=1,...,T-1
    # y_reg = y[2],...,y[T] (length T-1)
    y_reg = y[2:T + 1]  # shape (T-1,)

    # Step 10: Factor selection
    selections = select_factors(
        V_hat=V_hat,
        F_hat=F_hat,
        y_reg=y_reg,
        N=N,
        T=T,
        k_max=k_eff,
        r_true=r_true,
    )

    # Infeasible forecast (true factors)
    y_tilde = compute_infeasible_forecast(y, F_true, T)

    # Feasible forecasts for each selection rule
    y_hats = {}
    for rule in SELECTION_RULES:
        r_hat = min(selections[rule], k_eff)
        y_hats[rule] = compute_feasible_forecast(y, F_hat, r_hat, T)

    # Step 11: S^2 components
    sq_diffs = {}
    for rule in SELECTION_RULES:
        sq_diff, _ = compute_forecast_closeness(y_hats[rule], y_tilde)
        sq_diffs[rule] = sq_diff
    sq_tilde = y_tilde ** 2

    return {
        "r2_num": r2_num,
        "r2_den": r2_den,
        "sq_diffs": sq_diffs,
        "sq_tilde": sq_tilde,
        "r_selected": selections,
        "y_tilde": y_tilde,
        "y_hats": y_hats,
    }


def run_scenario(
    params: ScenarioParams,
    n_reps: int = 2000,
    base_seed: int = 42,
    verbose: bool = False,
) -> dict:
    """
    Run Monte Carlo simulation for a single scenario.

    Parameters
    ----------
    params : ScenarioParams
        Scenario parameters.
    n_reps : int
        Number of Monte Carlo replications.
    base_seed : int
        Base random seed for reproducibility.
    verbose : bool
        If True, log progress every 200 replications.

    Returns
    -------
    dict with aggregated results:
        'label'     : str
        'params'    : ScenarioParams
        'r2'        : float — R^2_{Fhat,F}
        's2'        : dict[str, float] — S^2 per selection rule
        'r_mean'    : dict[str, float] — mean selected factor count per rule
        'n_reps'    : int
    """
    rng = np.random.default_rng(base_seed)

    r2_nums = np.zeros(n_reps)
    r2_dens = np.zeros(n_reps)
    sq_diffs_all = {rule: np.zeros(n_reps) for rule in SELECTION_RULES}
    sq_tildes_all = np.zeros(n_reps)
    r_selected_all = {rule: np.zeros(n_reps, dtype=int) for rule in SELECTION_RULES}

    for rep in range(n_reps):
        if verbose and rep % 200 == 0:
            logger.info(f"  Scenario {params.label}: replication {rep}/{n_reps}")

        result = run_single_replication(params, rng)

        r2_nums[rep] = result["r2_num"]
        r2_dens[rep] = result["r2_den"]
        sq_tildes_all[rep] = result["sq_tilde"]
        for rule in SELECTION_RULES:
            sq_diffs_all[rule][rep] = result["sq_diffs"][rule]
            r_selected_all[rule][rep] = result["r_selected"][rule]

    # Step 12: Aggregate
    # R^2_{Fhat,F} = E[tr(F_hat' P_F F_hat)] / E[tr(F_hat' F_hat)]
    r2_agg = np.mean(r2_nums) / max(np.mean(r2_dens), 1e-15)
    r2_agg = float(np.clip(r2_agg, 0.0, 1.0))

    # S^2 per selection rule
    s2_agg = {}
    for rule in SELECTION_RULES:
        s2_agg[rule] = aggregate_s2(sq_diffs_all[rule], sq_tildes_all, version="tilde")

    # Mean selected factor counts
    r_mean = {rule: float(np.mean(r_selected_all[rule])) for rule in SELECTION_RULES}

    return {
        "label": params.label,
        "params": params,
        "r2": r2_agg,
        "s2": s2_agg,
        "r_mean": r_mean,
        "n_reps": n_reps,
    }


def run_all_scenarios(
    scenarios: list[ScenarioParams],
    n_reps: int = 2000,
    base_seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run Monte Carlo simulations for all scenarios and return a results table.

    Parameters
    ----------
    scenarios : list[ScenarioParams]
        List of scenario parameter objects.
    n_reps : int
        Number of replications per scenario.
    base_seed : int
        Base random seed.
    verbose : bool
        If True, log progress.

    Returns
    -------
    pd.DataFrame
        Results table with one row per scenario.
    """
    rows = []
    for i, params in enumerate(scenarios):
        if verbose:
            logger.info(
                f"Running scenario {params.label} ({i+1}/{len(scenarios)}): "
                f"T={params.T}, N={params.N}, r_tilde={params.r_tilde}, "
                f"q={params.q}, alpha={params.alpha}"
            )
        # Use different seed per scenario for independence
        seed = base_seed + i * 1000
        result = run_scenario(params, n_reps=n_reps, base_seed=seed, verbose=verbose)

        row = {
            "label": result["label"],
            "T": params.T,
            "N": params.N,
            "r_tilde": params.r_tilde,
            "r_true": params.r_true,
            "k": params.k,
            "q": params.q,
            "alpha": params.alpha,
            "a": params.a,
            "b": params.b,
            "c": params.c,
            "pi": params.pi,
            "delta_1": params.delta_1,
            "delta_2": params.delta_2,
            "R2_Fhat_F": result["r2"],
        }
        for rule in SELECTION_RULES:
            row[f"S2_{rule}"] = result["s2"][rule]
            row[f"r_mean_{rule}"] = result["r_mean"][rule]

        rows.append(row)
        if verbose:
            logger.info(
                f"  -> R^2={result['r2']:.4f}, "
                f"S^2(true)={result['s2']['true']:.4f}, "
                f"S^2(ic3)={result['s2']['ic3']:.4f}"
            )

    return pd.DataFrame(rows)
