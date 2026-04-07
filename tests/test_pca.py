"""
Tests for the PCA estimation module.

Validates:
  - Normalization: Lambda_hat' Lambda_hat / N = I_k
  - F_hat = X Lambda_hat / N
  - V_hat is non-increasing
  - Factor recovery in a known factor model
  - Edge cases (T < N, T > N, k=1)
"""

from __future__ import annotations

import numpy as np
import pytest

from src.pca_estimation import estimate_pca


class TestEstimatePCA:
    """Tests for the estimate_pca function."""

    def test_output_shapes_t_gt_n(self) -> None:
        """Test output shapes when T > N."""
        T, N, k = 100, 30, 5
        X = np.random.default_rng(42).standard_normal((T, N))
        F_hat, Lambda_hat, V_hat = estimate_pca(X, k)

        assert F_hat.shape == (T, k), f"F_hat shape: {F_hat.shape}"
        assert Lambda_hat.shape == (N, k), f"Lambda_hat shape: {Lambda_hat.shape}"
        assert V_hat.shape == (k + 1,), f"V_hat shape: {V_hat.shape}"

    def test_output_shapes_t_lt_n(self) -> None:
        """Test output shapes when T < N."""
        T, N, k = 30, 100, 5
        X = np.random.default_rng(42).standard_normal((T, N))
        F_hat, Lambda_hat, V_hat = estimate_pca(X, k)

        assert F_hat.shape == (T, k)
        assert Lambda_hat.shape == (N, k)
        assert V_hat.shape == (k + 1,)

    def test_normalization_lambda(self) -> None:
        """Lambda_hat' Lambda_hat / N should be approximately I_k."""
        T, N, k = 100, 50, 3
        rng = np.random.default_rng(42)
        X = rng.standard_normal((T, N))
        F_hat, Lambda_hat, V_hat = estimate_pca(X, k)

        LTL_over_N = Lambda_hat.T @ Lambda_hat / N
        np.testing.assert_allclose(
            LTL_over_N, np.eye(k), atol=1e-10,
            err_msg="Lambda_hat' Lambda_hat / N != I_k"
        )

    def test_f_hat_equals_x_lambda_over_n(self) -> None:
        """F_hat should equal X Lambda_hat / N."""
        T, N, k = 80, 40, 3
        rng = np.random.default_rng(42)
        X = rng.standard_normal((T, N))
        F_hat, Lambda_hat, V_hat = estimate_pca(X, k)

        F_hat_check = X @ Lambda_hat / N
        np.testing.assert_allclose(
            F_hat, F_hat_check, atol=1e-10,
            err_msg="F_hat != X Lambda_hat / N"
        )

    def test_v_hat_non_increasing(self) -> None:
        """V_hat should be non-increasing in j."""
        T, N, k = 100, 50, 8
        rng = np.random.default_rng(42)
        X = rng.standard_normal((T, N))
        _, _, V_hat = estimate_pca(X, k)

        for j in range(k):
            assert V_hat[j] >= V_hat[j + 1] - 1e-12, (
                f"V_hat not non-increasing at j={j}: {V_hat[j]:.6f} < {V_hat[j+1]:.6f}"
            )

    def test_v_hat_non_negative(self) -> None:
        """V_hat values should be non-negative."""
        T, N, k = 100, 50, 5
        rng = np.random.default_rng(42)
        X = rng.standard_normal((T, N))
        _, _, V_hat = estimate_pca(X, k)

        assert np.all(V_hat >= -1e-12), f"Negative V_hat: {V_hat}"

    def test_factor_recovery_known_model(self) -> None:
        """
        In a known factor model, PC should recover factors well.

        Generate X = F Lambda' + E with large N, T and small noise.
        The column space of F_hat should be close to that of F_true.
        """
        rng = np.random.default_rng(42)
        T, N, r = 200, 200, 2
        F_true = rng.standard_normal((T, r))
        Lambda = rng.standard_normal((N, r)) * 2.0
        E = rng.standard_normal((T, N)) * 0.1  # small noise

        X = F_true @ Lambda.T + E

        F_hat, _, _ = estimate_pca(X, r)

        # Check that F_hat spans approximately the same space as F_true
        # via projection: tr(F_hat' P_F F_hat) / tr(F_hat' F_hat) should be ~1
        Q, _ = np.linalg.qr(F_true, mode="reduced")
        QTF = Q.T @ F_hat
        r2 = np.sum(QTF ** 2) / np.sum(F_hat ** 2)
        assert r2 > 0.95, f"Factor recovery R^2={r2:.4f} < 0.95 in known model"

    def test_k_equals_one(self) -> None:
        """k=1 should work correctly."""
        T, N = 50, 30
        rng = np.random.default_rng(42)
        X = rng.standard_normal((T, N))
        F_hat, Lambda_hat, V_hat = estimate_pca(X, 1)

        assert F_hat.shape == (T, 1)
        assert Lambda_hat.shape == (N, 1)
        assert V_hat.shape == (2,)
        # Normalization
        np.testing.assert_allclose(
            Lambda_hat.T @ Lambda_hat / N, np.array([[1.0]]), atol=1e-10
        )

    def test_v_hat_zero_at_full_rank(self) -> None:
        """V_hat[k] should be small when k = min(T,N) (full rank)."""
        T, N = 20, 20
        rng = np.random.default_rng(42)
        X = rng.standard_normal((T, N))
        k = min(T, N) - 1
        _, _, V_hat = estimate_pca(X, k)
        # V_hat[k] should be much smaller than V_hat[0]
        assert V_hat[k] < V_hat[0] * 0.5, (
            f"V_hat[{k}]={V_hat[k]:.4f} not much smaller than V_hat[0]={V_hat[0]:.4f}"
        )

    def test_standardized_input(self) -> None:
        """PCA should work on standardized (zero-mean, unit-variance) input."""
        T, N, k = 100, 50, 3
        rng = np.random.default_rng(42)
        X = rng.standard_normal((T, N))
        X = (X - X.mean(axis=0)) / X.std(axis=0)

        F_hat, Lambda_hat, V_hat = estimate_pca(X, k)
        assert not np.any(np.isnan(F_hat))
        assert not np.any(np.isnan(Lambda_hat))
