"""Rank-aware pushforward of Gaussian robot-action statistics.

RESEARCH REFERENCE ONLY: the transport and support-aware distance are classical
linear algebra, not a deployed safety mechanism or a new mathematical theorem.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TransportedGaussian:
    mean: np.ndarray
    covariance: np.ndarray
    basis: np.ndarray
    scales: np.ndarray

    @property
    def rank(self):
        return self.scales.size

    def squared_distance(self, output, *, support_atol=1e-9, support_rtol=1e-9):
        """Mahalanobis distance on image support; fail closed off support."""
        output = np.asarray(output, dtype=np.float64)
        if output.shape != self.mean.shape or not np.isfinite(output).all():
            raise ValueError("invalid observed output")
        delta = output - self.mean
        coordinates = self.basis.T @ delta
        orthogonal = delta - self.basis @ coordinates
        scale = max(1.0, float(np.linalg.norm(delta)))
        if np.linalg.norm(orthogonal) > support_atol + support_rtol * scale:
            raise ValueError("output lies outside transported action support")
        return float(np.sum(np.square(coordinates / self.scales)))


def transport_gaussian(mean, covariance, matrix, offset, *, rank_rtol=1e-12):
    """Map N(mean, cov) through y = matrix @ x + offset, including rank loss."""
    mean = np.asarray(mean, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    offset = np.asarray(offset, dtype=np.float64)
    if mean.ndim != 1 or matrix.ndim != 2:
        raise ValueError("mean must be vector; transform must be matrix")
    n = mean.size
    m = matrix.shape[0]
    if covariance.shape != (n, n) or matrix.shape[1] != n or offset.shape != (m,):
        raise ValueError("incompatible action dimensions")
    if not all(np.isfinite(item).all() for item in (mean, covariance, matrix, offset)):
        raise ValueError("nonfinite parameter")
    if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=1e-12):
        raise ValueError("source covariance is not symmetric")
    try:
        chol = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("source covariance must be positive definite") from exc
    # Singular values of A L are standard deviations in target support.
    basis, values, _ = np.linalg.svd(matrix @ chol, full_matrices=False)
    threshold = values[0] * rank_rtol if values.size else 0.0
    active = values > threshold
    basis = basis[:, active]
    scales = values[active]
    y_mean = matrix @ mean + offset
    y_cov = (basis * scales**2) @ basis.T
    return TransportedGaussian(y_mean, y_cov, basis, scales)


def source_squared_distance(x, mean, cov):
    delta = np.asarray(x) - np.asarray(mean)
    return float(delta @ np.linalg.solve(cov, delta))
