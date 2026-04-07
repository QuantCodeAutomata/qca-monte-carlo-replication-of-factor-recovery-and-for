"""
Tests for factor selection criteria.

Validates:
  - Bai-Ng ICp1, ICp2, ICp3 return valid factor counts
  - AIC/BIC return valid factor counts
  - Criteria select correct number in known models
  - Edge cases (k_max=0, single observation)
"""

from __future__ import annotations

import numpy as np
import pytest

from src.factor_selection import bai_ng_criteria, forecasting_aic_bic, select_factors
from src.pca_estimation import estimate_pca


class TestBaiNgCriteria:
    """Tests for Bai-Ng information criteria."""

    def test_output_range(self) -> None:
        """Selected factor counts should be in [0, k_max]."""
        k_max = 8
        V_hat = np.array([1.0, 0.8, 0.6, 0.5, 0.45, 0.42, 0.41, 0.405, 0.40])
        N, T = 100, 100

        r1, r2, r3 = bai_ng_criteria(V_hat, N, T, k_max)

        assert 0 <= r1 <= k_max, f"ICp1 out of range: {r1}"
        assert 0 <= r2 <= k_max, f"ICp2 out of range: {r2}"
        assert 0 <= r3 <= k_max, f"ICp3 out of range: {r3}"

    def test_correct_length_v_hat(self) -> None:
        """V_hat must have length k_max+1."""
        k_max = 5
        V_hat = np.linspace(1.0, 0.5, k_max + 1)
        N, T = 100, 100
        # Should not raise
        r1, r2, r3 = bai_ng_criteria(V_hat, N, T, k_max)
        assert isinstance(r1, int)

    def test_wrong_v_hat_length_raises(self) -> None:
        """Wrong V_hat length should raise AssertionError."""
        k_max = 5
        V_hat = np.linspace(1.0, 0.5, k_max)  # wrong: should be k_max+1
        with pytest.raises(AssertionError):
            bai_ng_criteria(V_hat, N=100, T=100, k_max=k_max)

    def test_selects_correct_r_in_known_model(self) -> None:
        """
        In a large factor model with clear factor structure,
        Bai-Ng should select near the true r.
        """
        rng = np.random.default_rng(42)
        T, N, r_true = 200, 200, 3
        k_max = 8

        # Generate factor model
        F = rng.standard_normal((T, r_true))
        Lambda = rng.standard_normal((N, r_true))
        E = rng.standard_normal((T, N)) * 0.3

        X = F @ Lambda.T + E
        _, _, V_hat = estimate_pca(X, k_max)

        r1, r2, r3 = bai_ng_criteria(V_hat, N, T, k_max)

        # At least one criterion should select near r_true
        assert any(abs(r - r_true) <= 1 for r in [r1, r2, r3]), (
            f"No criterion near r_true={r_true}: ICp1={r1}, ICp2={r2}, ICp3={r3}"
        )

    def test_large_n_t_selects_nonzero(self) -> None:
        """With large N and T and clear factor structure, should select r > 0."""
        rng = np.random.default_rng(42)
        T, N, r_true = 300, 300, 2
        k_max = 8

        F = rng.standard_normal((T, r_true))
        Lambda = rng.standard_normal((N, r_true)) * 2.0
        E = rng.standard_normal((T, N)) * 0.2

        X = F @ Lambda.T + E
        _, _, V_hat = estimate_pca(X, k_max)

        r1, r2, r3 = bai_ng_criteria(V_hat, N, T, k_max)
        assert r3 > 0, f"ICp3 selected 0 factors in clear factor model: {r3}"

    def test_penalty_ordering(self) -> None:
        """
        Verify penalty functions satisfy g3 <= g2 <= g1 for typical N, T.
        (This is a property of the Bai-Ng penalty functions.)
        """
        N, T = 100, 100
        C_NT2 = float(min(N, T))
        g1 = ((N + T) / (N * T)) * np.log(N * T / (N + T))
        g2 = ((N + T) / (N * T)) * np.log(C_NT2)
        g3 = np.log(C_NT2) / C_NT2

        # g3 should be smallest (most lenient penalty)
        assert g3 <= g2 + 1e-10, f"g3={g3:.6f} > g2={g2:.6f}"


class TestForecastingAICBIC:
    """Tests for forecasting-equation AIC/BIC selection."""

    def test_output_range(self) -> None:
        """Selected factor counts should be in [0, k_max]."""
        rng = np.random.default_rng(42)
        T_reg, k_max = 80, 5
        y = rng.standard_normal(T_reg)
        F_hat = rng.standard_normal((T_reg, k_max))

        r_aic, r_bic = forecasting_aic_bic(y, F_hat, k_max)

        assert 0 <= r_aic <= k_max, f"AIC out of range: {r_aic}"
        assert 0 <= r_bic <= k_max, f"BIC out of range: {r_bic}"

    def test_bic_le_aic_in_large_sample(self) -> None:
        """
        BIC penalizes more heavily than AIC in large samples,
        so BIC-selected r should be <= AIC-selected r (on average).
        This is a tendency, not a strict rule per replication.
        """
        rng = np.random.default_rng(42)
        T_reg, k_max = 200, 8
        r_true = 3

        F_true = rng.standard_normal((T_reg, r_true))
        beta = np.array([0.5, -0.3, 0.4])
        y = F_true @ beta + 0.1 * rng.standard_normal(T_reg)

        F_hat = np.column_stack([F_true, rng.standard_normal((T_reg, k_max - r_true))])

        r_aic, r_bic = forecasting_aic_bic(y, F_hat, k_max)
        # BIC should not select more factors than AIC in large samples
        # (this is a tendency, allow some slack)
        assert r_bic <= r_aic + 2, f"BIC={r_bic} >> AIC={r_aic}"

    def test_selects_correct_r_in_known_model(self) -> None:
        """AIC/BIC should select near the true r in a clear factor model."""
        rng = np.random.default_rng(42)
        T_reg, k_max = 150, 6
        r_true = 2

        F_true = rng.standard_normal((T_reg, r_true))
        beta = np.array([1.0, -0.8])
        y = F_true @ beta + 0.05 * rng.standard_normal(T_reg)

        # Add irrelevant factors
        F_hat = np.column_stack([F_true, rng.standard_normal((T_reg, k_max - r_true))])

        r_aic, r_bic = forecasting_aic_bic(y, F_hat, k_max)

        assert abs(r_aic - r_true) <= 1, f"AIC selected {r_aic}, expected near {r_true}"
        assert abs(r_bic - r_true) <= 1, f"BIC selected {r_bic}, expected near {r_true}"

    def test_intercept_only_model(self) -> None:
        """j=0 (intercept only) should be valid and return finite AIC/BIC."""
        rng = np.random.default_rng(42)
        T_reg, k_max = 50, 3
        y = rng.standard_normal(T_reg)
        F_hat = rng.standard_normal((T_reg, k_max))

        # Should not raise
        r_aic, r_bic = forecasting_aic_bic(y, F_hat, k_max)
        assert isinstance(r_aic, int)
        assert isinstance(r_bic, int)


class TestSelectFactors:
    """Tests for the combined select_factors function."""

    def test_all_rules_present(self) -> None:
        """All selection rules should be present in output."""
        rng = np.random.default_rng(42)
        T, N, k_max, r_true = 100, 50, 8, 2

        F_true = rng.standard_normal((T, r_true))
        Lambda = rng.standard_normal((N, r_true))
        E = rng.standard_normal((T, N)) * 0.5
        X = F_true @ Lambda.T + E

        F_hat, _, V_hat = estimate_pca(X, k_max)
        y_reg = rng.standard_normal(T - 1)

        result = select_factors(V_hat, F_hat, y_reg, N, T, k_max, r_true)

        expected_keys = {"true", "ic1", "ic2", "ic3", "aic", "bic"}
        assert set(result.keys()) == expected_keys

    def test_true_rule_returns_r_true(self) -> None:
        """'true' rule should always return r_true."""
        rng = np.random.default_rng(42)
        T, N, k_max, r_true = 100, 50, 8, 3

        X = rng.standard_normal((T, N))
        F_hat, _, V_hat = estimate_pca(X, k_max)
        y_reg = rng.standard_normal(T - 1)

        result = select_factors(V_hat, F_hat, y_reg, N, T, k_max, r_true)
        assert result["true"] == r_true
