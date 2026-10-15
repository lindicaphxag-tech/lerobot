"""Standalone CPU invariance tests for rank-aware action transport."""
import unittest

import numpy as np

from research.rank_aware_action_transport import source_squared_distance, transport_gaussian


class RankAwareTransportTests(unittest.TestCase):
    def test_projection_and_composition(self):
        rng = np.random.default_rng(20261015)
        for n in (2, 4, 7, 12):
            for i in range(150):
                z = rng.normal(size=(n, n))
                covariance = z @ z.T / n + np.eye(n) * 0.4
                mean, x = rng.normal(size=n), rng.normal(size=n)
                m = i % (n + 1) + 1
                A, b = rng.normal(size=(m, n)), rng.normal(size=m)
                if i % 4 == 0:
                    A[0, :] = 0  # deliberate rank loss
                transformed = transport_gaussian(mean, covariance, A, b)
                result = transformed.squared_distance(A @ x + b)
                reference = source_squared_distance(x, mean, covariance)
                self.assertLessEqual(result, reference + 1e-7)
                self.assertLessEqual(transformed.rank, min(m, n))
                k = i % m + 1
                B, c = rng.normal(size=(k, m)), rng.normal(size=k)
                combined = transport_gaussian(
                    mean, covariance, B @ A, B @ b + c
                )
                np.testing.assert_allclose(
                    B @ transformed.mean + c, combined.mean, atol=1e-9
                )
                np.testing.assert_allclose(
                    B @ transformed.covariance @ B.T,
                    combined.covariance, atol=1e-9,
                )

    def test_off_support_refusal(self):
        gaussian = transport_gaussian(
            np.zeros(3), np.eye(3), np.diag([1.0, 1.0, 0.0]), np.zeros(3)
        )
        self.assertEqual(gaussian.rank, 2)
        self.assertAlmostEqual(gaussian.squared_distance([1, 2, 0]), 5)
        with self.assertRaisesRegex(ValueError, "outside transported action support"):
            gaussian.squared_distance([1, 2, 1e-3])
        constant = transport_gaussian(
            np.zeros(2), np.eye(2), np.zeros((3, 2)), np.ones(3)
        )
        self.assertEqual(constant.rank, 0)
        self.assertEqual(constant.squared_distance(np.ones(3)), 0)
        with self.assertRaises(ValueError):
            constant.squared_distance([1, 1, 1.01])

    def test_correlated_joint_variance(self):
        cov = np.array([[1, 0.95], [0.95, 1]])
        gaussian = transport_gaussian(
            np.zeros(2), cov, np.array([[1.0, 1.0]]), np.zeros(1)
        )
        self.assertAlmostEqual(gaussian.covariance[0, 0], 3.9)
        self.assertAlmostEqual(gaussian.squared_distance([2]), 4 / 3.9)
        self.assertGreater(abs(2 - gaussian.squared_distance([2])), 0.9)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            transport_gaussian(
                np.zeros(2), np.diag([1.0, 0.0]), np.eye(2), np.zeros(2)
            )
        with self.assertRaises(ValueError):
            transport_gaussian(
                np.zeros(2), np.eye(2), np.array([[np.nan, 1]]), np.zeros(1)
            )
        with self.assertRaises(ValueError):
            transport_gaussian(
                np.zeros(2), np.eye(2), np.zeros((1, 3)), np.zeros(1)
            )


if __name__ == "__main__":
    unittest.main()
