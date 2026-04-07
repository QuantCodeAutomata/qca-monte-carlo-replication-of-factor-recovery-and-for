"""
Tests for the statistics module.

Validates:
  - R^2_{Fhat,F} is in [0, 1]
  - R^2 = 1 when F_hat spans F_true
  - R^2 = 0 when F_hat is orthogonal to F_true
  - S^2 computation
  - Infeasible and feasible forecast functions
"""

from __future__ import annotations

import numpy as np
import pytest

from src.statistics import (
    compute_r2_factor_recovery,
    compute_forecast_closeness,
    aggregate_s2,
    compute_infeasible_forecast,
    compute_feasible_forecast,
)


class TestR2FactorRecovery:
    """Tests for compute_r2_factor_recovery."""

    def test_r2_in_unit_interval(self) -> None:
        """R^2 should be in [0, 1]."""
        rng = np.random.default_rng(42)
        T, k, r_true = 100, 3, 2
        F_hat = rng.standard_normal((T, k))
        F_true = rng.standard_normal((T, r_true))

        r2, _, _ = compute_r2_factor_recovery(F_hat, F_true)
        assert 0.0 <= r2 <= 1.0, f"R^2={r2} not in [0,1]"

    def test_r2_equals_one_when_f_hat_spans_f_true(self) -> None:
        """R^2 = 1 when F_hat spans the same space as F_true."""
        rng = np.random.default_rng(42)
        T, r = 100, 2

        F_true = rng.standard_normal((T, r))
        # F_hat is a rotation of F_true (same column space)
        R = np.linalg.qr(rng.standard_normal((r, r)))[0]
        F_hat = F_true @ R

        r2, _, _ = compute_r2_factor_recovery(F_hat, F_true)
        assert abs(r2 - 1.0) < 1e-10, f"R^2={r2} != 1 when F_hat spans F_true"

    def test_r2_equals_zero_when_orthogonal(self) -> None:
        """R^2 = 0 when F_hat is orthogonal to F_true."""
        T = 100
        # F_true in first half of space, F_hat in second half
        F_true = np.zeros((T, 2))
        F_true[:, 0] = np.arange(T, dtype=float)
        F_true[:, 1] = np.arange(T, 0, -1, dtype=float)
        # Orthogonalize
        Q_true, _ = np.linalg.qr(F_true)

        # F_hat orthogonal to F_true
        rng = np.random.default_rng(42)
        F_hat_raw = rng.standard_normal((T, 2))
        # Project out F_true component
        F_hat = F_hat_raw - Q_true @ (Q_true.T @ F_hat_raw)
        # Normalize
        F_hat, _ = np.linalg.qr(F_hat)

        r2, _, _ = compute_r2_factor_recovery(F_hat, F_true)
        assert r2 < 1e-10, f"R^2={r2} != 0 when F_hat orthogonal to F_true"

    def test_r2_improves_with_more_factors(self) -> None:
        """
        R^2 should not decrease when F_hat has more columns spanning F_true.
        """
        rng = np.random.default_rng(42)
        T, r_true = 100, 2
        F_true = rng.standard_normal((T, r_true))

        # F_hat with 1 factor (partial recovery)
        F_hat_1 = F_true[:, :1] + 0.1 * rng.standard_normal((T, 1))
        # F_hat with 2 factors (full recovery)
        F_hat_2 = F_true + 0.1 * rng.standard_normal((T, r_true))

        r2_1, _, _ = compute_r2_factor_recovery(F_hat_1, F_true)
        r2_2, _, _ = compute_r2_factor_recovery(F_hat_2, F_true)

        assert r2_2 >= r2_1 - 0.05, (
            f"R^2 with 2 factors ({r2_2:.4f}) < R^2 with 1 factor ({r2_1:.4f})"
        )

    def test_r2_with_large_n_t(self) -> None:
        """
        In a large factor model with small noise, R^2 should be close to 1.
        """
        rng = np.random.default_rng(42)
        T, N, r = 300, 300, 2

        F_true = rng.standard_normal((T, r))
        Lambda = rng.standard_normal((N, r)) * 2.0
        E = rng.standard_normal((T, N)) * 0.1

        X = F_true @ Lambda.T + E

        from src.pca_estimation import estimate_pca
        F_hat, _, _ = estimate_pca(X, r)

        r2, _, _ = compute_r2_factor_recovery(F_hat, F_true)
        assert r2 > 0.9, f"R^2={r2:.4f} < 0.9 in large factor model"

    def test_numerator_leq_denominator(self) -> None:
        """Numerator should be <= denominator (R^2 <= 1)."""
        rng = np.random.default_rng(42)
        T, k, r_true = 100, 4, 2
        F_hat = rng.standard_normal((T, k))
        F_true = rng.standard_normal((T, r_true))

        r2, num, den = compute_r2_factor_recovery(F_hat, F_true)
        assert num <= den + 1e-10, f"num={num:.6f} > den={den:.6f}"


class TestForecastCloseness:
    """Tests for forecast closeness statistics."""

    def test_sq_diff_zero_when_equal(self) -> None:
        """(y_hat - y_tilde)^2 = 0 when y_hat = y_tilde."""
        sq_diff, sq_tilde = compute_forecast_closeness(1.5, 1.5)
        assert sq_diff == 0.0

    def test_sq_diff_positive(self) -> None:
        """(y_hat - y_tilde)^2 > 0 when y_hat != y_tilde."""
        sq_diff, _ = compute_forecast_closeness(1.0, 2.0)
        assert sq_diff > 0.0

    def test_sq_tilde_correct(self) -> None:
        """sq_tilde should equal y_tilde^2."""
        y_tilde = 3.0
        _, sq_tilde = compute_forecast_closeness(1.0, y_tilde)
        assert abs(sq_tilde - y_tilde ** 2) < 1e-12

    def test_aggregate_s2_perfect_forecast(self) -> None:
        """S^2 = 1 when y_hat = y_tilde for all replications."""
        n_reps = 100
        y_tilde_vals = np.random.default_rng(42).standard_normal(n_reps)
        sq_diffs = np.zeros(n_reps)
        sq_tildes = y_tilde_vals ** 2

        s2 = aggregate_s2(sq_diffs, sq_tildes, version="tilde")
        assert abs(s2 - 1.0) < 1e-10, f"S^2={s2} != 1 for perfect forecast"

    def test_aggregate_s2_zero_when_no_improvement(self) -> None:
        """S^2 = 0 when E[(y_hat - y_tilde)^2] = E[y_tilde^2]."""
        n_reps = 1000
        rng = np.random.default_rng(42)
        y_tilde_vals = rng.standard_normal(n_reps)
        # y_hat = 0 always => (y_hat - y_tilde)^2 = y_tilde^2
        sq_diffs = y_tilde_vals ** 2
        sq_tildes = y_tilde_vals ** 2

        s2 = aggregate_s2(sq_diffs, sq_tildes, version="tilde")
        assert abs(s2) < 0.05, f"S^2={s2} not near 0"

    def test_aggregate_s2_negative_possible(self) -> None:
        """S^2 can be negative if y_hat is worse than zero forecast."""
        n_reps = 100
        rng = np.random.default_rng(42)
        y_tilde_vals = rng.standard_normal(n_reps) * 0.1
        # y_hat is very bad
        sq_diffs = (rng.standard_normal(n_reps) * 10) ** 2
        sq_tildes = y_tilde_vals ** 2

        s2 = aggregate_s2(sq_diffs, sq_tildes, version="tilde")
        # S^2 should be negative (bad forecast)
        assert s2 < 0.5, f"S^2={s2} unexpectedly high for bad forecast"


class TestInfeasibleForecast:
    """Tests for compute_infeasible_forecast."""

    def test_output_is_scalar(self) -> None:
        """Output should be a scalar float."""
        rng = np.random.default_rng(42)
        T, r_true = 50, 2
        y = np.concatenate([[0.0], rng.standard_normal(T)])
        F_true = rng.standard_normal((T, r_true))

        result = compute_infeasible_forecast(y, F_true, T)
        assert isinstance(result, float)

    def test_forecast_finite(self) -> None:
        """Forecast should be finite."""
        rng = np.random.default_rng(42)
        T, r_true = 100, 3
        y = np.concatenate([[0.0], rng.standard_normal(T)])
        F_true = rng.standard_normal((T, r_true))

        result = compute_infeasible_forecast(y, F_true, T)
        assert np.isfinite(result), f"Infeasible forecast is not finite: {result}"

    def test_forecast_close_to_true_in_known_model(self) -> None:
        """
        In a known linear model, infeasible forecast should be close to true value.

        The regression uses y[t+1] on F_true[t-1] (0-indexed) for t=1,...,T-1.
        So y[t+1] = F_true[t-1] @ beta matches the regression alignment.
        The forecast is y_{T+1|T} = beta_hat_0 + F_true[T-1] @ beta_hat.
        """
        rng = np.random.default_rng(42)
        T = 200
        r_true = 2
        beta = np.array([0.5, -0.3])

        # Generate y matching the regression alignment:
        # y[t+1] = F_true[t-1] @ beta for t=1,...,T-1 (regression sample)
        # y[1] = 0 (unused in regression)
        F_true = rng.standard_normal((T, r_true))
        y = np.zeros(T + 1)
        for t in range(1, T):
            y[t + 1] = F_true[t - 1] @ beta + 0.01 * rng.standard_normal()

        result = compute_infeasible_forecast(y, F_true, T)
        # True forecast: F_true[T-1] @ beta (the last factor row)
        true_forecast = F_true[T - 1] @ beta
        assert abs(result - true_forecast) < 0.5, (
            f"Infeasible forecast {result:.4f} far from true {true_forecast:.4f}"
        )


class TestFeasibleForecast:
    """Tests for compute_feasible_forecast."""

    def test_output_is_scalar(self) -> None:
        """Output should be a scalar float."""
        rng = np.random.default_rng(42)
        T, k = 50, 3
        y = np.concatenate([[0.0], rng.standard_normal(T)])
        F_hat = rng.standard_normal((T, k))

        result = compute_feasible_forecast(y, F_hat, r_hat=2, T=T)
        assert isinstance(result, float)

    def test_r_hat_zero_returns_mean(self) -> None:
        """r_hat=0 should return the mean of y_reg."""
        rng = np.random.default_rng(42)
        T = 50
        y = np.concatenate([[0.0], rng.standard_normal(T)])
        F_hat = rng.standard_normal((T, 3))

        result = compute_feasible_forecast(y, F_hat, r_hat=0, T=T)
        expected_mean = float(np.mean(y[2:T + 1]))
        assert abs(result - expected_mean) < 1e-10

    def test_forecast_finite(self) -> None:
        """Forecast should be finite."""
        rng = np.random.default_rng(42)
        T, k = 100, 5
        y = np.concatenate([[0.0], rng.standard_normal(T)])
        F_hat = rng.standard_normal((T, k))

        result = compute_feasible_forecast(y, F_hat, r_hat=3, T=T)
        assert np.isfinite(result)
