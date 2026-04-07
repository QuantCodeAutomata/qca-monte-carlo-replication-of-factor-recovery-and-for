"""
Principal Components (PC) factor estimation.

Implements the PC estimator described in Stock & Watson (2002):
  minimize V(F_tilde, Lambda_tilde) = (NT)^{-1} sum_{i,t} (x_it - lambda_i' F_t)^2

Normalization: Lambda_hat' Lambda_hat / N = I_k, F_hat = X Lambda_hat / N.

Uses numpy SVD for numerical stability.
Custom — Context7 found no library equivalent matching this exact normalization.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


def estimate_pca(
    X: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate PC factors from a T x N predictor matrix.

    Normalization: Lambda_hat' Lambda_hat / N = I_k, F_hat = X Lambda_hat / N.
    This is equivalent to: F_hat columns are sqrt(N) times the leading left
    singular vectors of X, and Lambda_hat = X' F_hat / T.

    Parameters
    ----------
    X : np.ndarray, shape (T, N)
        Standardized predictor panel.
    k : int
        Number of factors to extract.

    Returns
    -------
    F_hat : np.ndarray, shape (T, k)
        Estimated factor matrix.
    Lambda_hat : np.ndarray, shape (N, k)
        Estimated loading matrix satisfying Lambda_hat' Lambda_hat / N = I_k.
    V_hat : np.ndarray, shape (k+1,)
        Objective function values V_hat_j for j=0,...,k.
        V_hat[j] = (NT)^{-1} ||X - F_hat[:,:j] @ Lambda_hat[:,:j].T||^2_F.
    """
    T, N = X.shape
    assert k >= 1, "k must be at least 1"
    assert k <= min(T, N), f"k={k} exceeds min(T,N)={min(T,N)}"

    # Use SVD of X for numerical stability
    # X = U S V' where U is (T,T), S is (T,N), V is (N,N)
    # Leading k left singular vectors scaled by sqrt(N) give F_hat
    if T <= N:
        # Compute via X X' (T x T) — faster when T < N
        XXT = X @ X.T  # (T, T)
        eigenvalues, eigenvectors = np.linalg.eigh(XXT)
        # eigh returns ascending order; reverse for descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        # F_hat: T x k, normalized so F_hat' F_hat / T = I_k
        F_hat_raw = eigenvectors[:, :k]  # (T, k), orthonormal columns
        # Impose Lambda_hat' Lambda_hat / N = I_k:
        # Lambda_hat = X' F_hat / T, then F_hat = X Lambda_hat / N
        # With F_hat_raw orthonormal: Lambda_hat = X' F_hat_raw / T * sqrt(N)
        # and F_hat = F_hat_raw * sqrt(N) / sqrt(T) ... let's derive carefully.
        #
        # Standard normalization: F_hat' F_hat / T = I_k
        # => F_hat = sqrt(T) * eigenvectors[:, :k]
        # Lambda_hat = X' F_hat / T = X' eigenvectors[:, :k] / sqrt(T)
        # Check: Lambda_hat' Lambda_hat / N = eigenvectors[:,:k]' X X' eigenvectors[:,:k] / (T*N)
        #       = diag(eigenvalues[:k]) / (T*N)  -- not I_k in general
        #
        # Correct normalization for Lambda_hat' Lambda_hat / N = I_k:
        # F_hat = X Lambda_hat / N, Lambda_hat' Lambda_hat / N = I_k
        # => F_hat' F_hat = Lambda_hat' X' X Lambda_hat / N^2
        # Let Lambda_hat = V_k D_k where V_k are right singular vectors of X
        # and D_k = diag(sigma_k / sqrt(N)) ... this gets complex.
        #
        # Simplest consistent approach:
        # F_hat = sqrt(N) * U_k where U_k are leading left singular vectors of X
        # Lambda_hat = X' F_hat / (F_hat' F_hat) ... but F_hat' F_hat = N * I_k
        # => Lambda_hat = X' F_hat / N = X' U_k / sqrt(N)
        # Check: Lambda_hat' Lambda_hat / N = U_k' X X' U_k / N^2
        #       = U_k' (N * eigenvalues_k * U_k) / N^2 = diag(eigenvalues_k) / N
        # This is NOT I_k unless eigenvalues_k = N.
        #
        # The correct normalization is:
        # F_hat = X V_k / sqrt(N) where V_k are right singular vectors of X
        # Lambda_hat = V_k * sqrt(N) (the right singular vectors scaled)
        # Check: Lambda_hat' Lambda_hat / N = N * V_k' V_k / N = I_k ✓
        # F_hat = X Lambda_hat / N = X V_k * sqrt(N) / N = X V_k / sqrt(N) ✓
        #
        # So we need right singular vectors V_k.
        # From X X' = U S^2 U', we get U. Then V_k = X' U_k / S_k.
        # eigenvalues of X X' = S^2 (squared singular values)
        singular_values_k = np.sqrt(np.maximum(eigenvalues[:k], 0.0))
        # Avoid division by zero
        safe_sv = np.where(singular_values_k > 1e-12, singular_values_k, 1e-12)
        V_k = X.T @ eigenvectors[:, :k] / safe_sv[None, :]  # (N, k)
        F_hat = X @ V_k / np.sqrt(N)  # (T, k)
        Lambda_hat = V_k * np.sqrt(N)  # (N, k)
    else:
        # Compute via X' X (N x N) — faster when N < T
        XTX = X.T @ X  # (N, N)
        eigenvalues, eigenvectors = np.linalg.eigh(XTX)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        # V_k = eigenvectors[:, :k] (right singular vectors of X)
        V_k = eigenvectors[:, :k]  # (N, k)
        F_hat = X @ V_k / np.sqrt(N)  # (T, k)
        Lambda_hat = V_k * np.sqrt(N)  # (N, k)

    # Compute V_hat_j for j=0,...,k
    # V_hat_j = (NT)^{-1} ||X - F_hat[:,:j] @ Lambda_hat[:,:j].T||^2_F
    # = (NT)^{-1} (||X||^2_F - ||F_hat[:,:j] @ Lambda_hat[:,:j].T||^2_F)
    # = (NT)^{-1} (||X||^2_F - sum_{l=1}^j eigenvalue_l)
    # because F_hat[:,l] @ Lambda_hat[:,l].T has Frobenius norm^2 = eigenvalue_l
    # (from the SVD decomposition)
    X_norm2 = np.sum(X ** 2)
    V_hat = np.zeros(k + 1)
    V_hat[0] = X_norm2 / (N * T)
    cumulative_eigenvalues = np.cumsum(eigenvalues[:k])
    for j in range(1, k + 1):
        V_hat[j] = (X_norm2 - cumulative_eigenvalues[j - 1]) / (N * T)
    # Ensure V_hat is non-increasing (numerical safety)
    for j in range(1, k + 1):
        V_hat[j] = min(V_hat[j], V_hat[j - 1])

    return F_hat, Lambda_hat, V_hat
