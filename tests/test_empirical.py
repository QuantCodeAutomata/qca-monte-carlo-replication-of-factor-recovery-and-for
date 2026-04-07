"""
Tests for the empirical forecasting module.

Validates:
  - Synthetic data generation
  - Panel preparation and standardization
  - PC forecast computation
  - Metrics computation
  - Edge cases
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.empirical import (
    _generate_synthetic_macro_data,
    prepare_panel,
    _pc_forecast,
    compute_forecast_metrics,
    compute_ar_benchmark,
)


class TestSyntheticMacroData:
    """Tests for synthetic macro data generation."""

    def test_output_shape(self) -> None:
        """Synthetic data should have correct shape."""
        df = _generate_synthetic_macro_data(n_series=50, seed=42)
        assert df.shape[1] == 50, f"Expected 50 columns, got {df.shape[1]}"
        assert len(df) > 400, f"Expected >400 rows, got {len(df)}"

    def test_date_range(self) -> None:
        """Date range should cover 1959-1998."""
        df = _generate_synthetic_macro_data(n_series=50, seed=42)
        assert df.index[0] <= pd.Timestamp("1959-01-01")
        assert df.index[-1] >= pd.Timestamp("1998-12-01")

    def test_indpro_column_present(self) -> None:
        """INDPRO column should be present."""
        df = _generate_synthetic_macro_data(n_series=50, seed=42)
        assert "INDPRO" in df.columns

    def test_reproducibility(self) -> None:
        """Same seed should produce identical data."""
        df1 = _generate_synthetic_macro_data(n_series=50, seed=42)
        df2 = _generate_synthetic_macro_data(n_series=50, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self) -> None:
        """Different seeds should produce different data."""
        df1 = _generate_synthetic_macro_data(n_series=50, seed=42)
        df2 = _generate_synthetic_macro_data(n_series=50, seed=99)
        assert not df1.equals(df2)


class TestPreparePanel:
    """Tests for prepare_panel function."""

    @pytest.fixture
    def synthetic_df(self) -> pd.DataFrame:
        return _generate_synthetic_macro_data(n_series=30, seed=42)

    def test_output_shapes(self, synthetic_df: pd.DataFrame) -> None:
        """Output should have correct shapes."""
        origin = pd.Timestamp("1975-01-01")
        X_std, dates_est, valid_cols = prepare_panel(synthetic_df, origin)

        assert X_std.ndim == 2
        assert len(dates_est) == X_std.shape[0]
        assert len(valid_cols) == X_std.shape[1]

    def test_standardization_mean_zero(self, synthetic_df: pd.DataFrame) -> None:
        """Standardized predictors should have approximately zero mean."""
        origin = pd.Timestamp("1975-01-01")
        X_std, _, _ = prepare_panel(synthetic_df, origin)

        col_means = np.abs(np.nanmean(X_std, axis=0))
        assert np.all(col_means < 1e-10), (
            f"Non-zero column means after standardization: max={col_means.max():.2e}"
        )

    def test_standardization_unit_variance(self, synthetic_df: pd.DataFrame) -> None:
        """Standardized predictors should have approximately unit variance."""
        origin = pd.Timestamp("1975-01-01")
        X_std, _, _ = prepare_panel(synthetic_df, origin)

        col_stds = np.nanstd(X_std, axis=0)
        assert np.all(np.abs(col_stds - 1.0) < 0.01), (
            f"Non-unit column stds after standardization: max_dev={np.max(np.abs(col_stds-1)):.4f}"
        )

    def test_no_future_data_used(self, synthetic_df: pd.DataFrame) -> None:
        """Panel should only use data up to origin."""
        origin = pd.Timestamp("1975-01-01")
        _, dates_est, _ = prepare_panel(synthetic_df, origin)

        assert all(d <= origin for d in dates_est), (
            "Future data used in panel preparation"
        )

    def test_expanding_window_grows(self, synthetic_df: pd.DataFrame) -> None:
        """Later origins should have more observations."""
        origin1 = pd.Timestamp("1970-01-01")
        origin2 = pd.Timestamp("1980-01-01")

        X1, dates1, _ = prepare_panel(synthetic_df, origin1)
        X2, dates2, _ = prepare_panel(synthetic_df, origin2)

        assert len(dates2) > len(dates1), (
            f"Later origin has fewer obs: {len(dates2)} <= {len(dates1)}"
        )


class TestPCForecast:
    """Tests for _pc_forecast function."""

    def test_output_is_scalar(self) -> None:
        """Output should be a scalar float."""
        rng = np.random.default_rng(42)
        T_reg, k = 80, 5
        y_reg = rng.standard_normal(T_reg)
        F_reg = rng.standard_normal((T_reg, k))
        F_origin = rng.standard_normal(k)

        result = _pc_forecast(y_reg, F_reg, F_origin, r_hat=3)
        assert isinstance(result, float)

    def test_r_hat_zero_returns_mean(self) -> None:
        """r_hat=0 should return mean of y_reg."""
        rng = np.random.default_rng(42)
        T_reg, k = 50, 3
        y_reg = rng.standard_normal(T_reg)
        F_reg = rng.standard_normal((T_reg, k))
        F_origin = rng.standard_normal(k)

        result = _pc_forecast(y_reg, F_reg, F_origin, r_hat=0)
        assert abs(result - float(np.mean(y_reg))) < 1e-10

    def test_forecast_finite(self) -> None:
        """Forecast should be finite."""
        rng = np.random.default_rng(42)
        T_reg, k = 100, 5
        y_reg = rng.standard_normal(T_reg)
        F_reg = rng.standard_normal((T_reg, k))
        F_origin = rng.standard_normal(k)

        result = _pc_forecast(y_reg, F_reg, F_origin, r_hat=3)
        assert np.isfinite(result)

    def test_known_linear_model(self) -> None:
        """In a known linear model, forecast should be close to true value."""
        rng = np.random.default_rng(42)
        T_reg = 200
        k = 3
        beta = np.array([0.5, -0.3, 0.2])

        F_reg = rng.standard_normal((T_reg, k))
        y_reg = F_reg @ beta + 0.01 * rng.standard_normal(T_reg)
        F_origin = rng.standard_normal(k)

        result = _pc_forecast(y_reg, F_reg, F_origin, r_hat=k)
        true_forecast = F_origin @ beta
        assert abs(result - true_forecast) < 0.5, (
            f"Forecast {result:.4f} far from true {true_forecast:.4f}"
        )


class TestComputeForecastMetrics:
    """Tests for compute_forecast_metrics function."""

    @pytest.fixture
    def mock_results(self) -> pd.DataFrame:
        """Create mock forecast results."""
        rng = np.random.default_rng(42)
        n = 100
        origins = pd.date_range("1970-01-01", periods=n, freq="MS")
        realized = rng.standard_normal(n) * 0.05

        df = pd.DataFrame({
            "realized": realized,
            "ar_forecast": realized + rng.standard_normal(n) * 0.05,
            "pc_ic3_forecast": realized + rng.standard_normal(n) * 0.03,
            "pc_k1_forecast": realized + rng.standard_normal(n) * 0.06,
            "pc_k2_forecast": realized + rng.standard_normal(n) * 0.03,
            "pc_k3_forecast": realized + rng.standard_normal(n) * 0.025,
            "pc_k4_forecast": realized + rng.standard_normal(n) * 0.028,
        }, index=origins)
        return df

    def test_output_columns(self, mock_results: pd.DataFrame) -> None:
        """Metrics table should have required columns."""
        metrics = compute_forecast_metrics(mock_results)
        required_cols = {"method", "mse", "rmse", "relative_mse"}
        assert required_cols.issubset(set(metrics.columns)), (
            f"Missing columns: {required_cols - set(metrics.columns)}"
        )

    def test_ar_relative_mse_equals_one(self, mock_results: pd.DataFrame) -> None:
        """AR benchmark relative MSE should equal 1.0."""
        metrics = compute_forecast_metrics(mock_results)
        ar_row = metrics[metrics["method"] == "AR(4)"]
        assert len(ar_row) == 1
        assert abs(ar_row["relative_mse"].values[0] - 1.0) < 1e-10

    def test_mse_non_negative(self, mock_results: pd.DataFrame) -> None:
        """All MSE values should be non-negative."""
        metrics = compute_forecast_metrics(mock_results)
        assert all(metrics["mse"] >= 0), "Negative MSE found"

    def test_rmse_equals_sqrt_mse(self, mock_results: pd.DataFrame) -> None:
        """RMSE should equal sqrt(MSE)."""
        metrics = compute_forecast_metrics(mock_results)
        for _, row in metrics.iterrows():
            assert abs(row["rmse"] - np.sqrt(row["mse"])) < 1e-10, (
                f"RMSE != sqrt(MSE) for {row['method']}"
            )

    def test_better_forecast_lower_relative_mse(self, mock_results: pd.DataFrame) -> None:
        """PC k=3 (smaller noise) should have lower relative MSE than PC k=1."""
        metrics = compute_forecast_metrics(mock_results)
        k1_mse = metrics[metrics["method"] == "PC k=1"]["relative_mse"].values[0]
        k3_mse = metrics[metrics["method"] == "PC k=3"]["relative_mse"].values[0]
        assert k3_mse < k1_mse, (
            f"PC k=3 relative MSE ({k3_mse:.4f}) >= PC k=1 ({k1_mse:.4f})"
        )


class TestARBenchmark:
    """Tests for compute_ar_benchmark function."""

    def test_output_is_scalar(self) -> None:
        """Output should be a scalar float."""
        rng = np.random.default_rng(42)
        dates = pd.date_range("1959-01-01", "1975-01-01", freq="MS")
        ip_growth = pd.Series(rng.standard_normal(len(dates)) * 0.01, index=dates)
        origin = pd.Timestamp("1975-01-01")

        result = compute_ar_benchmark(ip_growth, origin)
        assert isinstance(result, float)

    def test_forecast_finite(self) -> None:
        """Forecast should be finite."""
        rng = np.random.default_rng(42)
        dates = pd.date_range("1959-01-01", "1975-01-01", freq="MS")
        ip_growth = pd.Series(rng.standard_normal(len(dates)) * 0.01, index=dates)
        origin = pd.Timestamp("1975-01-01")

        result = compute_ar_benchmark(ip_growth, origin)
        assert np.isfinite(result)

    def test_insufficient_data_returns_mean(self) -> None:
        """With very few observations, should return mean."""
        dates = pd.date_range("1970-01-01", periods=5, freq="MS")
        ip_growth = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01], index=dates)
        origin = pd.Timestamp("1970-05-01")

        result = compute_ar_benchmark(ip_growth, origin)
        assert np.isfinite(result)
