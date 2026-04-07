"""
Table 1 scenario definitions for the Monte Carlo study.

Defines scenario families A-F based on the paper's Table 1 description:
  A: Baseline (large N, T, iid, homoskedastic)
  B: Small sample
  C: Overfactoring (k > r_true)
  D: Irrelevant predictors (pi > 0)
  E: Dependent idiosyncratic errors (serial + spatial + GARCH)
  F: Dynamic factors (q > 0) and time-varying loadings (c > 0)

Parameters are calibrated to match the paper's Table 1 structure.
"""

from __future__ import annotations

from src.dgp import ScenarioParams


def get_table1_scenarios() -> list[ScenarioParams]:
    """
    Return the list of Monte Carlo scenarios corresponding to Table 1.

    Each scenario is a ScenarioParams instance with parameters chosen to
    match the paper's scenario families A-F.

    Returns
    -------
    list[ScenarioParams]
        List of scenario parameter objects.
    """
    scenarios = []

    # ------------------------------------------------------------------ #
    # Panel A: Baseline — large N, T, iid, homoskedastic, static factors
    # ------------------------------------------------------------------ #
    # A1: Large N and T
    scenarios.append(ScenarioParams(
        T=100, N=100, r_tilde=1, k=8, q=0,
        alpha=0.0, a=0.0, b=0.0, c=0.0,
        pi=0.0, delta_0=1.0, delta_1=0.0, delta_2=0.0,
        label="A1",
    ))
    # A2: Larger N and T
    scenarios.append(ScenarioParams(
        T=200, N=200, r_tilde=1, k=8, q=0,
        alpha=0.0, a=0.0, b=0.0, c=0.0,
        pi=0.0, delta_0=1.0, delta_1=0.0, delta_2=0.0,
        label="A2",
    ))
    # A3: Multiple factors
    scenarios.append(ScenarioParams(
        T=100, N=100, r_tilde=3, k=8, q=0,
        alpha=0.0, a=0.0, b=0.0, c=0.0,
        pi=0.0, delta_0=1.0, delta_1=0.0, delta_2=0.0,
        label="A3",
    ))

    # ------------------------------------------------------------------ #
    # Panel B: Small sample
    # ------------------------------------------------------------------ #
    # B1: Small T, moderate N
    scenarios.append(ScenarioParams(
        T=40, N=40, r_tilde=1, k=8, q=0,
        alpha=0.0, a=0.0, b=0.0, c=0.0,
        pi=0.0, delta_0=1.0, delta_1=0.0, delta_2=0.0,
        label="B1",
    ))
    # B2: Small T, large N
    scenarios.append(ScenarioParams(
        T=40, N=100, r_tilde=1, k=8, q=0,
        alpha=0.0, a=0.0, b=0.0, c=0.0,
        pi=0.0, delta_0=1.0, delta_1=0.0, delta_2=0.0,
        label="B2",
    ))

    # ------------------------------------------------------------------ #
    # Panel C: Overfactoring (k > r_true, persistent factors)
    # ------------------------------------------------------------------ #
    # C1: Persistent factors (alpha=0.5)
    scenarios.append(ScenarioParams(
        T=100, N=100, r_tilde=1, k=8, q=0,
        alpha=0.5, a=0.0, b=0.0, c=0.0,
        pi=0.0, delta_0=1.0, delta_1=0.0, delta_2=0.0,
        label="C1",
    ))
    # C2: Highly persistent factors (alpha=0.9)
    scenarios.append(ScenarioParams(
        T=100, N=100, r_tilde=1, k=8, q=0,
        alpha=0.9, a=0.0, b=0.0, c=0.0,
        pi=0.0, delta_0=1.0, delta_1=0.0, delta_2=0.0,
        label="C2",
    ))

    # ------------------------------------------------------------------ #
    # Panel D: Irrelevant predictors (pi > 0)
    # ------------------------------------------------------------------ #
    # D1: 20% irrelevant predictors
    scenarios.append(ScenarioParams(
        T=100, N=100, r_tilde=1, k=8, q=0,
        alpha=0.0, a=0.0, b=0.0, c=0.0,
        pi=0.2, delta_0=1.0, delta_1=0.0, delta_2=0.0,
        label="D1",
    ))
    # D2: 50% irrelevant predictors
    scenarios.append(ScenarioParams(
        T=100, N=100, r_tilde=1, k=8, q=0,
        alpha=0.0, a=0.0, b=0.0, c=0.0,
        pi=0.5, delta_0=1.0, delta_1=0.0, delta_2=0.0,
        label="D2",
    ))

    # ------------------------------------------------------------------ #
    # Panel E: Dependent idiosyncratic errors
    # ------------------------------------------------------------------ #
    # E1: Serial dependence (a=0.5)
    scenarios.append(ScenarioParams(
        T=100, N=100, r_tilde=1, k=8, q=0,
        alpha=0.0, a=0.5, b=0.0, c=0.0,
        pi=0.0, delta_0=1.0, delta_1=0.0, delta_2=0.0,
        label="E1",
    ))
    # E2: Spatial dependence (b=0.5)
    scenarios.append(ScenarioParams(
        T=100, N=100, r_tilde=1, k=8, q=0,
        alpha=0.0, a=0.0, b=0.5, c=0.0,
        pi=0.0, delta_0=1.0, delta_1=0.0, delta_2=0.0,
        label="E2",
    ))
    # E3: GARCH heteroskedasticity
    scenarios.append(ScenarioParams(
        T=100, N=100, r_tilde=1, k=8, q=0,
        alpha=0.0, a=0.0, b=0.0, c=0.0,
        pi=0.0, delta_0=0.1, delta_1=0.8, delta_2=0.1,
        label="E3",
    ))
    # E4: Combined serial + spatial + GARCH
    scenarios.append(ScenarioParams(
        T=100, N=100, r_tilde=1, k=8, q=0,
        alpha=0.0, a=0.3, b=0.3, c=0.0,
        pi=0.0, delta_0=0.1, delta_1=0.8, delta_2=0.1,
        label="E4",
    ))

    # ------------------------------------------------------------------ #
    # Panel F: Dynamic factors (q>0) and time-varying loadings (c>0)
    # ------------------------------------------------------------------ #
    # F1: Dynamic factors (q=1)
    scenarios.append(ScenarioParams(
        T=100, N=100, r_tilde=1, k=8, q=1,
        alpha=0.0, a=0.0, b=0.0, c=0.0,
        pi=0.0, delta_0=1.0, delta_1=0.0, delta_2=0.0,
        label="F1",
    ))
    # F2: Time-varying loadings (c=1)
    scenarios.append(ScenarioParams(
        T=100, N=100, r_tilde=1, k=8, q=0,
        alpha=0.0, a=0.0, b=0.0, c=1.0,
        pi=0.0, delta_0=1.0, delta_1=0.0, delta_2=0.0,
        label="F2",
    ))
    # F3: Dynamic factors + time-varying loadings
    scenarios.append(ScenarioParams(
        T=100, N=100, r_tilde=1, k=8, q=1,
        alpha=0.0, a=0.0, b=0.0, c=1.0,
        pi=0.0, delta_0=1.0, delta_1=0.0, delta_2=0.0,
        label="F3",
    ))
    # F4: All robustness features combined
    scenarios.append(ScenarioParams(
        T=100, N=100, r_tilde=1, k=8, q=1,
        alpha=0.5, a=0.3, b=0.3, c=1.0,
        pi=0.2, delta_0=0.1, delta_1=0.8, delta_2=0.1,
        label="F4",
    ))

    return scenarios
