"""Heat kernel signature (HKS) via normalized Laplacian eigendecomposition."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


def compute_eigen_decomposition(
    laplacian: sparse.csr_matrix,
    num_eigenvalues: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the smallest eigenpairs of a normalized graph Laplacian.

    Args:
        laplacian: Sparse CSR normalized Laplacian matrix ``L_sym``.
        num_eigenvalues: Number of smallest eigenvalues to compute.

    Returns:
        A tuple of ``(eigenvalues, eigenvectors)`` where ``eigenvalues`` has
        shape ``(k,)`` and ``eigenvectors`` has shape ``(N, k)``.
    """
    laplacian = _validate_laplacian(laplacian)
    num_nodes = laplacian.shape[0]
    num_eigenvalues = int(num_eigenvalues)

    if num_eigenvalues <= 0:
        raise ValueError("num_eigenvalues must be positive.")
    if num_nodes < 2:
        raise ValueError("laplacian must contain at least 2 nodes.")

    k = min(num_eigenvalues, num_nodes - 1)
    eigenvalues, eigenvectors = eigsh(
        laplacian,
        k=k,
        which="SM",
        tol=1e-3,
    )

    order = np.argsort(eigenvalues)
    eigenvalues = np.asarray(eigenvalues[order], dtype=np.float64)
    eigenvectors = np.asarray(eigenvectors[:, order], dtype=np.float64)
    eigenvalues = np.clip(eigenvalues, 0.0, 2.0)
    return eigenvalues, eigenvectors


def compute_adaptive_time_scales(
    eigenvalues: np.ndarray,
    num_scales: int = 3,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute HKS diffusion time scales on a log-uniform grid.

    Time bounds follow the classic HKS convention using the smallest
    non-zero and largest Laplacian eigenvalues.

    Args:
        eigenvalues: Laplacian eigenvalues with shape ``(k,)``.
        num_scales: Number of log-spaced diffusion times to generate.
        eps: Threshold for treating eigenvalues as non-zero.

    Returns:
        One-dimensional array of diffusion time scales with shape ``(num_scales,)``.
    """
    eigenvalues = np.sort(np.asarray(eigenvalues, dtype=np.float64).reshape(-1))
    num_scales = int(num_scales)
    eps = float(eps)
    if num_scales <= 0:
        raise ValueError("num_scales must be positive.")
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be a positive finite value.")

    if eigenvalues.size == 0:
        fallback = np.logspace(-2.0, 2.0, num_scales, dtype=np.float64)
        return np.clip(fallback, eps, None)

    lambda_max = float(eigenvalues[-1])
    valid_lambdas = eigenvalues[eigenvalues > eps]
    if valid_lambdas.size > 0:
        lambda_2 = float(valid_lambdas[0])
    elif eigenvalues.size > 1:
        lambda_2 = float(eigenvalues[1])
    else:
        lambda_2 = float(eigenvalues[0])

    lambda_max = max(lambda_max, eps)
    lambda_2 = max(lambda_2, eps)

    log_ten_span = 4.0 * np.log(10.0)
    t_min = log_ten_span / lambda_max
    t_max = log_ten_span / lambda_2

    t_lower = max(min(t_min, t_max), eps)
    t_upper = max(max(t_min, t_max), t_lower * (1.0 + 1e-12))

    if num_scales == 1:
        return np.array([t_lower], dtype=np.float64)

    time_scales = np.logspace(
        np.log10(t_lower),
        np.log10(t_upper),
        num_scales,
        dtype=np.float64,
    )
    return np.clip(time_scales, eps, None).astype(np.float64, copy=False)


def compute_hks(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    time_scales: np.ndarray | tuple[float, ...] | list[float],
) -> np.ndarray:
    """Compute multi-scale heat kernel signatures.

    For each node ``x`` and time scale ``t``:

        HKS(x, t) = sum_l exp(-lambda_l * t) * phi_l(x)^2

    Args:
        eigenvalues: Eigenvalues with shape ``(k,)``.
        eigenvectors: Eigenvectors with shape ``(N, k)``.
        time_scales: One or more diffusion time scales.

    Returns:
        HKS feature matrix with shape ``(N, num_time_scales)``.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
    eigenvectors = np.asarray(eigenvectors, dtype=np.float64)
    time_scales = np.asarray(time_scales, dtype=np.float64).reshape(-1)

    if eigenvectors.ndim != 2:
        raise ValueError("eigenvectors must have shape (N, k).")
    if eigenvectors.shape[1] != eigenvalues.shape[0]:
        raise ValueError("eigenvectors column count must match eigenvalues length.")
    if time_scales.size == 0:
        raise ValueError("time_scales must contain at least one value.")
    if not np.isfinite(time_scales).all():
        raise ValueError("time_scales must contain only finite values.")

    num_nodes = eigenvectors.shape[0]
    evecs_sq = eigenvectors ** 2
    hks_features = np.zeros((num_nodes, len(time_scales)), dtype=np.float64)

    for index, time_scale in enumerate(time_scales):
        decay = np.exp(-eigenvalues * float(time_scale))
        hks_features[:, index] = evecs_sq @ decay

    return hks_features


def _validate_laplacian(laplacian: sparse.csr_matrix) -> sparse.csr_matrix:
    """Validate Laplacian matrix shape and sparsity format."""
    if not sparse.isspmatrix(laplacian):
        raise TypeError("laplacian must be a scipy sparse matrix.")
    laplacian = laplacian.tocsr()
    if laplacian.ndim != 2 or laplacian.shape[0] != laplacian.shape[1]:
        raise ValueError("laplacian must be a square 2D sparse matrix.")
    if laplacian.shape[0] == 0:
        raise ValueError("laplacian must not be empty.")
    return laplacian
