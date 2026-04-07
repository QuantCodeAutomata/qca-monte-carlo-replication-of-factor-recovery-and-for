"""
Tests for the Monte Carlo orchestration module.

Validates:
  - Single replication produces valid output
  - Scenario runner aggregates correctly
  - R^2 improves with larger N and T
  - S^2 is high in baseline scenarios
  - Factor selection counts are within valid range
"""

from __future__ import annotations

import numpy as np
import pytest

from src.dgp import ScenarioParams
from src.monte_carlo import run_single_replication, run_scenario, SELECTION_RULES


class TestRunSingleReplication:
    """Tests for run_single_replication."""

    @pytest.fixture
    def small_params(self) -> ScenarioParams:
        return ScenarioParams(
            T=30, N=20, r_tilde=1, k=5, q=0,
            alpha=0.0, a=0.0, b=0.0, c=0.0,
            pi=0.0, delta_0=1.0, delta_1=0.0, delta_2=0.0,
            label="small",
        )

    def test_output_keys_present(self, small_params: ScenarioParams) -> None:
        """All expected keys should be present in output."""
        rng = np.random.default_rng(42)
        result = run_single_replication(small_params, rng)

        expected_keys = {"r2_num", "r2_den", "sq_diffs", "sq_tilde",
                         "r_selected", "y_tilde", "y_hats"}
        assert set(result.keys()) == expected_keys

    def test_r2_components_non_negative(self, small_params: ScenarioParams) -> None:
        """R^2 numerator and denominator should be non-negative."""
        rng = np.random.default_rng(42)
        result = run_single_replication(small_params, rng)

        assert result["r2_num"] >= -1e-10, f"r2_num={result['r2_num']} < 0"
        assert result["r2_den"] >= -1e-10, f"r2_den={result['r2_den']} < 0"

    def test_r2_num_leq_den(self, small_params: ScenarioParams) -> None:
        """R^2 numerator should be <= denominator."""
        rng = np.random.default_rng(42)
        result = run_single_replication(small_params, rng)

        assert result["r2_num"] <= result["r2_den"] + 1e-10

    def test_all_selection_rules_present(self, small_params: ScenarioParams) -> None:
        """All selection rules should be present in sq_diffs and r_selected."""
        rng = np.random.default_rng(42)
        result = run_single_replication(small_params, rng)

        for rule in SELECTION_RULES:
            assert rule in result["sq_diffs"], f"Rule {rule} missing from sq_diffs"
            assert rule in result["r_selected"], f"Rule {rule} missing from r_selected"

    def test_sq_diffs_non_negative(self, small_params: ScenarioParams) -> None:
        """Squared differences should be non-negative."""
        rng = np.random.default_rng(42)
        result = run_single_replication(small_params, rng)

        for rule in SELECTION_RULES:
            assert result["sq_diffs"][rule] >= 0.0, (
                f"sq_diffs[{rule}]={result['sq_diffs'][rule]} < 0"
            )

    def test_r_selected_in_valid_range(self, small_params: ScenarioParams) -> None:
        """Selected factor counts should be in [0, k]."""
        rng = np.random.default_rng(42)
        result = run_single_replication(small_params, rng)

        k_eff = min(small_params.k, small_params.T - 1, small_params.N)
        for rule in SELECTION_RULES:
            r = result["r_selected"][rule]
            assert 0 <= r <= k_eff, (
                f"r_selected[{rule}]={r} not in [0, {k_eff}]"
            )

    def test_y_hats_finite(self, small_params: ScenarioParams) -> None:
        """All forecasts should be finite."""
        rng = np.random.default_rng(42)
        result = run_single_replication(small_params, rng)

        for rule in SELECTION_RULES:
            assert np.isfinite(result["y_hats"][rule]), (
                f"y_hats[{rule}] is not finite"
            )
        assert np.isfinite(result["y_tilde"])


class TestRunScenario:
    """Tests for run_scenario aggregation."""

    def test_r2_in_unit_interval(self) -> None:
        """Aggregated R^2 should be in [0, 1]."""
        params = ScenarioParams(
            T=30, N=20, r_tilde=1, k=5, q=0,
            alpha=0.0, a=0.0, b=0.0, c=0.0,
            pi=0.0, delta_0=1.0, delta_1=0.0, delta_2=0.0,
            label="test",
        )
        result = run_scenario(params, n_reps=50, base_seed=42)

        assert 0.0 <= result["r2"] <= 1.0, f"R^2={result['r2']} not in [0,1]"

    def test_s2_values_present(self) -> None:
        """S^2 values should be present for all selection rules."""
        params = ScenarioParams(
            T=30, N=20, r_tilde=1, k=5, q=0,
            alpha=0.0, a=0.0, b=0.0, c=0.0,
            pi=0.0, delta_0=1.0, delta_1=0.0, delta_2=0.0,
            label="test",
        )
        result = run_scenario(params, n_reps=50, base_seed=42)

        for rule in SELECTION_RULES:
            assert rule in result["s2"], f"S^2 missing for rule {rule}"
            assert np.isfinite(result["s2"][rule]), (
                f"S^2[{rule}]={result['s2'][rule]} is not finite"
            )

    def test_r2_improves_with_larger_nt(self) -> None:
        """
        R^2 should be higher for larger N and T (asymptotic consistency).
        """
        params_small = ScenarioParams(
            T=30, N=30, r_tilde=1, k=5, q=0,
            alpha=0.0, a=0.0, b=0.0, c=0.0,
            pi=0.0, delta_0=1.0, delta_1=0.0, delta_2=0.0,
            label="small",
        )
        params_large = ScenarioParams(
            T=150, N=150, r_tilde=1, k=5, q=0,
            alpha=0.0, a=0.0, b=0.0, c=0.0,
            pi=0.0, delta_0=1.0, delta_1=0.0, delta_2=0.0,
            label="large",
        )

        result_small = run_scenario(params_small, n_reps=100, base_seed=42)
        result_large = run_scenario(params_large, n_reps=100, base_seed=42)

        assert result_large["r2"] > result_small["r2"], (
            f"R^2 did not improve: small={result_small['r2']:.4f}, "
            f"large={result_large['r2']:.4f}"
        )

    def test_baseline_high_r2(self) -> None:
        """
        In a large baseline scenario, R^2 should be high (>0.7).
        """
        params = ScenarioParams(
            T=200, N=200, r_tilde=1, k=5, q=0,
            alpha=0.0, a=0.0, b=0.0, c=0.0,
            pi=0.0, delta_0=1.0, delta_1=0.0, delta_2=0.0,
            label="large_baseline",
        )
        result = run_scenario(params, n_reps=100, base_seed=42)

        assert result["r2"] > 0.7, (
            f"R^2={result['r2']:.4f} < 0.7 in large baseline scenario"
        )

    def test_r_mean_in_valid_range(self) -> None:
        """Mean selected factor counts should be in [0, k]."""
        params = ScenarioParams(
            T=50, N=40, r_tilde=1, k=5, q=0,
            alpha=0.0, a=0.0, b=0.0, c=0.0,
            pi=0.0, delta_0=1.0, delta_1=0.0, delta_2=0.0,
            label="test",
        )
        result = run_scenario(params, n_reps=50, base_seed=42)

        k_eff = min(params.k, params.T - 1, params.N)
        for rule in SELECTION_RULES:
            r_mean = result["r_mean"][rule]
            assert 0.0 <= r_mean <= k_eff + 0.5, (
                f"r_mean[{rule}]={r_mean:.2f} not in [0, {k_eff}]"
            )

    def test_n_reps_stored(self) -> None:
        """n_reps should be stored in result."""
        params = ScenarioParams(
            T=30, N=20, r_tilde=1, k=5, q=0,
            alpha=0.0, a=0.0, b=0.0, c=0.0,
            pi=0.0, delta_0=1.0, delta_1=0.0, delta_2=0.0,
        )
        result = run_scenario(params, n_reps=50, base_seed=42)
        assert result["n_reps"] == 50
