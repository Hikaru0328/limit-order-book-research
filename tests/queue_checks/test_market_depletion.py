import unittest

import numpy as np

from queue_models.cancellation import CancellationConfig, theory as cancellation_theory
from queue_models.market_depletion import DepletionConfig, assess, simulate, stage_theory, theory


class MarketDepletionTests(unittest.TestCase):
    def test_cancellation_limit(self):
        for q in (1, 3, 100):
            a = theory(DepletionConfig(q, .2, 0))
            b = cancellation_theory(CancellationConfig(q, .2))
            for key in a:
                self.assertAlmostEqual(a[key], b[key])

    def test_market_only_erlang(self):
        ref = theory(DepletionConfig(100, 0, 2))
        self.assertEqual(ref["mean"], 50)
        self.assertEqual(ref["variance"], 25)
        self.assertAlmostEqual(ref["cv"], .1)

    def test_last_order_ratio_and_shares(self):
        a, b = [stage_theory(DepletionConfig(100, .2, mu)) for mu in (0, 2)]
        self.assertAlmostEqual(b["variance"][0] / a["variance"][0], 1 / 121)
        self.assertAlmostEqual(b["market_probability"][9], .5)
        for key in ("mean_share", "variance_share"):
            self.assertAlmostEqual(b[key].sum(), 1)

    def test_scaling_and_reproducibility(self):
        for method in ("combined", "competing"):
            a = simulate(DepletionConfig(10, .2, 2), 100, 42, method)
            np.testing.assert_array_equal(a, simulate(DepletionConfig(10, .2, 2), 100, 42, method))
            b = simulate(DepletionConfig(10, .4, 4), 100, 42, method)
            np.testing.assert_allclose(a / 2, b)

    def test_independent_methods_and_degenerate_mechanisms(self):
        for i, (theta, mu) in enumerate(((.2, 0), (0, 2), (.2, 2))):
            for j, method in enumerate(("combined", "competing")):
                cfg = DepletionConfig(10, theta, mu)
                row = assess(cfg, simulate(cfg, 20000, 55 + 2*i+j, method))
                self.assertLess(abs(row["mean_z"]), 5)
                self.assertLess(abs(row["variance_z"]), 5)

    def test_cv_bound_and_large_mu_limit(self):
        for mu in (0, .2, 2, 20, 200):
            self.assertGreaterEqual(theory(DepletionConfig(100, .2, mu))["cv"], .1)
        self.assertAlmostEqual(theory(DepletionConfig(100, .2, 1e7))["cv"], .1, places=10)

    def test_invalid_input(self):
        for args in ((0, .2, 2), (1.5, .2, 2), (1, 0, 0), (1, -1, 2), (1, .2, np.inf)):
            with self.assertRaises(ValueError):
                DepletionConfig(*args)
        with self.assertRaises(ValueError):
            simulate(DepletionConfig(), 1)
        with self.assertRaises(ValueError):
            simulate(DepletionConfig(), method="bad")
        with self.assertRaises(ValueError):
            assess(DepletionConfig(), np.zeros((2, 3)))
