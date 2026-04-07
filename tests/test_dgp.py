"""
Tests for the Data Generating Process (DGP) module.

Validates:
  - lambda*(R_i²) calibration
  - DGP output shapes and properties
  - GARCH stationarity
  - Factor structure
  - Edge cases
"""

from __future__ import annotations

import numpy as np
import pytest

from src.dgp import ScenarioParams, simulate_dgp, lambda_star


# ------------------------------------------------------------------ #
# Tests for lambda_star calibration
# ------------------------------------------------------------------ #

class TestLambdaStar:
    """Tests for the lambda*(R_i²) calibration function."""

    def test_zero_r2_returns_zero(self) -> None:
        """R_i²=0 should give lambda*=0 (irrelevant predictor)."""
        result = lambda_star(0.0, r_true=1)
        assert result == 0.0

    def test_positive_r2_returns_positive(self) -> None:
        """Positive R_i² should give positive lambda*."""
        result = lambda_star(0.5, r_true=1)
        assert result > 0.0

    def test_monotone_in_r2(self) -> None:
        """lambda* should be increasing in R_i²."""
        r2_vals = [0.1, 0.3, 0.5, 0.7]
        lambda_vals = [lambda_star(r2, r_true=1) for r2 in r2_vals]
        for i in range(len(lambda_vals) - 1):
            assert lambda_vals[i] < lambda_vals[i + 1], (
                f"lambda* not monotone: {lambda_vals}"
            )

    def test_decreasing_in_r_true(self) -> None:
        """lambda* should decrease as r_true increases (more factors share the signal)."""
        r2 = 0.5
        lam1 = lambda_star(r2, r_true=1)
        lam3 = lambda_star(r2, r_true=3)
        assert lam1 > lam3, f"lambda*(r_true=1)={lam1} should > lambda*(r_true=3)={lam3}"

    def test_calibration_baseline(self) -> None:
        """
        Verify calibration: in baseline scenario (alpha=0, iid, homoskedastic),
        the expected R_i² should approximately equal the target R_i².

        For x_it = lambda_i' F_t + e_it with F_t ~ N(0,I), e_it ~ N(0,1):
        E[Var(lambda_i' F_t)] = r_true * lambda*(R_i²)²
        R_i² = r_true * lambda*(R_i²)² / (r_true * lambda*(R_i²)² + 1)
        """
        for r2_target in [0.1, 0.3, 0.5, 0.7]:
            for r_true in [1, 2, 3]:
                lam = lambda_star(r2_target, r_true=r_true)
                signal_var = r_true * lam ** 2
                r2_implied = signal_var / (signal_var + 1.0)
                assert abs(r2_implied - r2_target) < 1e-10, (
                    f"Calibration failed: target={r2_target}, implied={r2_implied}, "
                    f"r_true={r_true}"
                )

    def test_near_one_r2(self) -> None:
        """R_i² close to 1 should give large but finite lambda*."""
        result = lambda_star(0.99, r_true=1)
        assert result > 0 and np.isfinite(result)


# ------------------------------------------------------------------ #
# Tests for ScenarioParams
# ------------------------------------------------------------------ #

class TestScenarioParams:
    """Tests for ScenarioParams validation."""

    def test_valid_params(self) -> None:
        """Valid parameters should not raise."""
        params = ScenarioParams(
            T=100, N=100, r_tilde=1, k=8, q=0,
            alpha=0.0, a=0.0, b=0.0, c=0.0,
            pi=0.0, delta_0=1.0, delta_1=0.0, delta_2=0.0,
            label="test",
        )
        assert params.r_true == 1

    def test_r_true_with_lags(self) -> None:
        """r_true = r_tilde * (1 + q)."""
        params = ScenarioParams(
            T=100, N=100, r_tilde=2, k=8, q=2,
            alpha=0.0, a=0.0, b=0.0, c=0.0,
            pi=0.0, delta_0=1.0, delta_1=0.0, delta_2=0.0,
        )
        assert params.r_true == 6  # 2 * (1 + 2)

    def test_invalid_alpha(self) -> None:
        """alpha >= 1 should raise AssertionError."""
        with pytest.raises(AssertionError):
            ScenarioParams(
                T=100, N=100, r_tilde=1, k=8, q=0,
                alpha=1.0, a=0.0, b=0.0, c=0.0,
                pi=0.0, delta_0=1.0, delta_1=0.0, delta_2=0.0,
            )

    def test_invalid_garch(self) -> None:
        """delta_1 + delta_2 >= 1 should raise AssertionError."""
        with pytest.raises(AssertionError):
            ScenarioParams(
                T=100, N=100, r_tilde=1, k=8, q=0,
                alpha=0.0, a=0.0, b=0.0, c=0.0,
                pi=0.0, delta_0=0.1, delta_1=0.9, delta_2=0.1,
            )

    def test_invalid_pi(self) -> None:
        """pi > 1 should raise AssertionError."""
        with pytest.raises(AssertionError):
            ScenarioParams(
                T=100, N=100, r_tilde=1, k=8, q=0,
                alpha=0.0, a=0.0, b=0.0, c=0.0,
                pi=1.5, delta_0=1.0, delta_1=0.0, delta_2=0.0,
            )


# ------------------------------------------------------------------ #
# Tests for simulate_dgp
# ------------------------------------------------------------------ #

class TestSimulateDGP:
    """Tests for the DGP simulation function."""

    @pytest.fixture
    def baseline_params(self) -> ScenarioParams:
        return ScenarioParams(
            T=50, N=30, r_tilde=1, k=8, q=0,
            alpha=0.0, a=0.0, b=0.0, c=0.0,
            pi=0.0, delta_0=1.0, delta_1=0.0, delta_2=0.0,
            label="baseline",
        )

    def test_output_shapes(self, baseline_params: ScenarioParams) -> None:
        """Output arrays should have correct shapes."""
        rng = np.random.default_rng(42)
        result = simulate_dgp(baseline_params, rng)

        T, N = baseline_params.T, baseline_params.N
        assert result["X"].shape == (T, N), f"X shape: {result['X'].shape}"
        assert result["y"].shape == (T + 1,), f"y shape: {result['y'].shape}"
        assert result["F_true"].shape == (T, baseline_params.r_true), (
            f"F_true shape: {result['F_true'].shape}"
        )

    def test_no_nan_in_output(self, baseline_params: ScenarioParams) -> None:
        """No NaN values should appear in output."""
        rng = np.random.default_rng(42)
        result = simulate_dgp(baseline_params, rng)

        assert not np.any(np.isnan(result["X"])), "NaN in X"
        assert not np.any(np.isnan(result["y"])), "NaN in y"
        assert not np.any(np.isnan(result["F_true"])), "NaN in F_true"

    def test_no_inf_in_output(self, baseline_params: ScenarioParams) -> None:
        """No infinite values should appear in output."""
        rng = np.random.default_rng(42)
        result = simulate_dgp(baseline_params, rng)

        assert not np.any(np.isinf(result["X"])), "Inf in X"
        assert not np.any(np.isinf(result["y"])), "Inf in y"

    def test_reproducibility(self, baseline_params: ScenarioParams) -> None:
        """Same seed should produce identical results."""
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        result1 = simulate_dgp(baseline_params, rng1)
        result2 = simulate_dgp(baseline_params, rng2)

        np.testing.assert_array_equal(result1["X"], result2["X"])
        np.testing.assert_array_equal(result1["y"], result2["y"])

    def test_different_seeds_differ(self, baseline_params: ScenarioParams) -> None:
        """Different seeds should produce different results."""
        rng1 = np.random.default_rng(1)
        rng2 = np.random.default_rng(2)
        result1 = simulate_dgp(baseline_params, rng1)
        result2 = simulate_dgp(baseline_params, rng2)

        assert not np.allclose(result1["X"], result2["X"])

    def test_dynamic_factors_shape(self) -> None:
        """Dynamic factors (q>0) should produce correct r_true dimension."""
        params = ScenarioParams(
            T=50, N=30, r_tilde=2, k=8, q=2,
            alpha=0.0, a=0.0, b=0.0, c=0.0,
            pi=0.0, delta_0=1.0, delta_1=0.0, delta_2=0.0,
        )
        rng = np.random.default_rng(42)
        result = simulate_dgp(params, rng)
        assert result["F_true"].shape == (50, 6)  # r_true = 2*(1+2) = 6

    def test_irrelevant_predictors(self) -> None:
        """With pi=1, all predictors should have zero signal (only noise)."""
        params = ScenarioParams(
            T=100, N=50, r_tilde=1, k=8, q=0,
            alpha=0.0, a=0.0, b=0.0, c=0.0,
            pi=1.0, delta_0=1.0, delta_1=0.0, delta_2=0.0,
        )
        rng = np.random.default_rng(42)
        result = simulate_dgp(params, rng)
        # With pi=1, all R_i^2=0, so lambda*=0, signal=0
        # X should be pure noise (e_it)
        # Check that X has reasonable variance (not zero)
        assert np.std(result["X"]) > 0.1

    def test_garch_variance_positive(self) -> None:
        """GARCH variance should remain positive throughout simulation."""
        params = ScenarioParams(
            T=100, N=20, r_tilde=1, k=8, q=0,
            alpha=0.0, a=0.0, b=0.0, c=0.0,
            pi=0.0, delta_0=0.1, delta_1=0.8, delta_2=0.1,
        )
        rng = np.random.default_rng(42)
        result = simulate_dgp(params, rng)
        # X should be finite (GARCH variance stayed positive)
        assert np.all(np.isfinite(result["X"]))

    def test_time_varying_loadings(self) -> None:
        """Time-varying loadings (c>0) should produce valid output."""
        params = ScenarioParams(
            T=100, N=30, r_tilde=1, k=8, q=0,
            alpha=0.0, a=0.0, b=0.0, c=2.0,
            pi=0.0, delta_0=1.0, delta_1=0.0, delta_2=0.0,
        )
        rng = np.random.default_rng(42)
        result = simulate_dgp(params, rng)
        assert result["X"].shape == (100, 30)
        assert not np.any(np.isnan(result["X"]))

    def test_spatial_dependence(self) -> None:
        """Spatial dependence (b>0) should produce valid output."""
        params = ScenarioParams(
            T=100, N=30, r_tilde=1, k=8, q=0,
            alpha=0.0, a=0.0, b=0.5, c=0.0,
            pi=0.0, delta_0=1.0, delta_1=0.0, delta_2=0.0,
        )
        rng = np.random.default_rng(42)
        result = simulate_dgp(params, rng)
        assert not np.any(np.isnan(result["X"]))

    def test_y_unused_first_element(self, baseline_params: ScenarioParams) -> None:
        """y[0] should be 0 (unused placeholder)."""
        rng = np.random.default_rng(42)
        result = simulate_dgp(baseline_params, rng)
        assert result["y"][0] == 0.0
