import unittest

import numpy as np

from queue_models.cancellation import CancellationConfig, assess, cdf, simulate, theory


class CancellationTests(unittest.TestCase):
    def test_one_and_two_orders(self):
        for q, mean, variance in [(1, 5, 25), (2, 7.5, 31.25)]:
            ref = theory(CancellationConfig(q, .2))
            self.assertAlmostEqual(ref["mean"], mean)
            self.assertAlmostEqual(ref["variance"], variance)

    def test_cdf_known_values_and_boundaries(self):
        cfg = CancellationConfig(3, .2)
        self.assertAlmostEqual(float(cdf(cfg, np.log(2) / .2)), 1 / 8)
        np.testing.assert_array_equal(cdf(cfg, [-1, 0, np.inf]), [0, 0, 1])
        self.assertGreater(float(cdf(CancellationConfig(1, 1), 1e-15)), 0)

    def test_fourth_moment_single_exponential(self):
        self.assertAlmostEqual(theory(CancellationConfig(1, 2))["fourth_central"], 9 / 16)

    def test_reproducibility_and_scaling_coupled_draws(self):
        for method in ("maximum", "stages"):
            a = simulate(CancellationConfig(7, .2), 100, 8, method)
            b = simulate(CancellationConfig(7, .4), 100, 8, method)
            np.testing.assert_array_equal(a, simulate(CancellationConfig(7, .2), 100, 8, method))
            np.testing.assert_allclose(a / 2, b)
            self.assertTrue((a > 0).all())

    def test_independent_methods_against_theory(self):
        for i, method in enumerate(("maximum", "stages")):
            cfg = CancellationConfig(10, .4)
            row = assess(cfg, simulate(cfg, 30000, 100 + i, method), .0001)
            self.assertLess(abs(row["mean_z"]), 5)
            self.assertLess(abs(row["variance_z"]), 5)
            self.assertTrue(row["cdf_pass"])

    def test_mean_variance_scaling(self):
        a, b = [theory(CancellationConfig(10, t)) for t in (.2, .4)]
        self.assertAlmostEqual(a["mean"] / 2, b["mean"])
        self.assertAlmostEqual(a["variance"] / 4, b["variance"])
        self.assertAlmostEqual(a["cv"], b["cv"])

    def test_invalid_inputs(self):
        for q, t in [(0, 1), (1.5, 1), (True, 1), (1, 0), (1, -1), (1, np.nan)]:
            with self.assertRaises(ValueError):
                CancellationConfig(q, t)
        for kwargs in ({"samples": 0}, {"batch_size": 0}, {"method": "wrong"}):
            with self.assertRaises(ValueError):
                simulate(CancellationConfig(), **kwargs)
        with self.assertRaises(ValueError):
            assess(CancellationConfig(), [1])


if __name__ == "__main__":
    unittest.main()
