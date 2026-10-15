"""Research reference: transport normalization under relative joint actions.

For selected joints r = a - s, transport mu_rel(s) = mu_abs - s.
This is an algebraic invariance check, NOT an integrated LeRobot policy fix.
"""
import unittest
import numpy as np


def relative_action(action, state, mask):
    action = np.asarray(action, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if action.shape != state.shape or state.shape != mask.shape:
        raise ValueError("shape mismatch")
    return action - np.where(mask, state, 0.0)


def transport_mean(mean, scale, state, mask):
    mean = np.asarray(mean, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if mean.shape != scale.shape or mean.shape != state.shape or mean.shape != mask.shape:
        raise ValueError("shape mismatch")
    if not all(np.all(np.isfinite(x)) for x in (mean, scale, state)):
        raise ValueError("nonfinite values")
    if np.any(scale <= 0):
        raise ValueError("nonpositive scale")
    return np.where(mask, mean - state, mean)


class TestActionStatisticTransport(unittest.TestCase):
    def test_equivariance(self):
        rng = np.random.default_rng(20261015)
        for n in (1, 2, 7, 16):
            for _ in range(1000):
                state = rng.normal(size=n)
                action = rng.normal(size=n)
                mean = rng.normal(size=n)
                scale = np.exp(rng.uniform(-4, 3, size=n))
                mask = rng.random(size=n) > 0.3
                relative = relative_action(action, state, mask)
                transported = transport_mean(mean, scale, state, mask)
                np.testing.assert_allclose(
                    (relative - transported) / scale,
                    (action - mean) / scale,
                    atol=1e-12, rtol=1e-12,
                )

    def test_fail_closed(self):
        with self.assertRaises(ValueError):
            transport_mean(np.zeros(2), np.zeros(2), np.ones(2), [True, False])
        with self.assertRaises(ValueError):
            transport_mean(np.zeros(2), np.array([1.0, np.nan]), np.ones(2), [True, False])
        with self.assertRaises(ValueError):
            relative_action(np.zeros(2), np.zeros(3), [True, False])


if __name__ == "__main__":
    unittest.main()
