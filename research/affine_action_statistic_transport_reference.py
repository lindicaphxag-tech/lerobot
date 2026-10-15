"""Research reference: affine transport of action normalization statistics.

For y = A x + b, mu_y = A mu_x + b and Sigma_y = A Sigma_x A.T.
Exact Mahalanobis invariance requires invertible A and SPD Sigma.
This is NOT integrated into the LeRobot production pipeline.
"""
import unittest
import numpy as np


def transport(mu, cov, A, b):
    mu, cov, A, b = [np.asarray(v, dtype=np.float64) for v in (mu, cov, A, b)]
    n = mu.size
    if mu.shape != (n,) or cov.shape != (n, n) or A.shape != (n, n) or b.shape != (n,):
        raise ValueError("shape mismatch")
    if not all(np.isfinite(x).all() for x in (mu, cov, A, b)):
        raise ValueError("nonfinite values")
    if np.linalg.matrix_rank(A) != n:
        raise ValueError("noninvertible action transform")
    if not np.allclose(cov, cov.T) or np.linalg.eigvalsh(cov).min() <= 0:
        raise ValueError("covariance must be symmetric positive definite")
    return A @ mu + b, A @ cov @ A.T


class TestAffineTransport(unittest.TestCase):
    def test_composition_and_distance(self):
        rng = np.random.default_rng(20261015)
        for n in (2, 3, 6):
            for _ in range(200):
                M = rng.normal(size=(n, n))
                cov = M @ M.T + np.eye(n) * 0.5
                x, mu = rng.normal(size=n), rng.normal(size=n)
                A = rng.normal(size=(n, n)) + 2 * np.eye(n)
                B = rng.normal(size=(n, n)) + 2 * np.eye(n)
                a, b = rng.normal(size=n), rng.normal(size=n)
                m1, c1 = transport(mu, cov, A, a)
                m2, c2 = transport(m1, c1, B, b)
                mc, cc = transport(mu, cov, B @ A, B @ a + b)
                np.testing.assert_allclose(m2, mc, rtol=1e-9, atol=1e-9)
                np.testing.assert_allclose(c2, cc, rtol=1e-9, atol=1e-9)
                d0 = x - mu
                d1 = A @ x + a - m1
                original = d0 @ np.linalg.solve(cov, d0)
                changed = d1 @ np.linalg.solve(c1, d1)
                np.testing.assert_allclose(changed, original, rtol=1e-7, atol=1e-7)

    def test_rank_deficient_rejected(self):
        with self.assertRaises(ValueError):
            transport(np.zeros(2), np.eye(2), np.diag([1., 0.]), np.zeros(2))


if __name__ == "__main__":
    unittest.main()
