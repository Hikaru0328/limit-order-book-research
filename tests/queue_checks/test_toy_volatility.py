import unittest

import numpy as np

from queue_models.toy_volatility import (
    ToyConfig, event_path, estimate_batches, sample_fixed_count, sample_terminal, theory,
)


class ToyTests(unittest.TestCase):
    def test_reproducibility(self):
        a = sample_terminal(ToyConfig(), 2, 100, np.random.default_rng(42))
        b = sample_terminal(ToyConfig(), 2, 100, np.random.default_rng(42))
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)

    def test_zero_rate_and_horizon(self):
        for config, horizon in [(ToyConfig(rate=0), 3), (ToyConfig(), 0)]:
            for array in sample_terminal(config, horizon, 10, np.random.default_rng(1)):
                np.testing.assert_array_equal(array, 0)

    def test_fixed_count_one_sided_and_alternation(self):
        rng = np.random.default_rng(10)
        np.testing.assert_array_equal(sample_fixed_count(ToyConfig(p_up=1), 100, 50, rng), 100)
        alt = ToyConfig(model="alternating")
        np.testing.assert_array_equal(sample_fixed_count(alt, 100, 50, rng), 0)
        np.testing.assert_array_equal(np.abs(sample_fixed_count(alt, 101, 50, rng)), 1)

    def test_event_path_quadratic_variation(self):
        for model in ("independent", "alternating"):
            t, x, qv = event_path(ToyConfig(delta=2, model=model), 5, np.random.default_rng(3))
            self.assertEqual(t[0], 0)
            self.assertEqual(t[-1], 5)
            self.assertTrue(np.all(np.diff(t) >= 0))
            self.assertEqual(np.sum(np.diff(x)**2), qv[-1])
            if model == "alternating":
                self.assertTrue(set(x).issubset({0, 2}))

    def test_exponential_clock_against_poisson_moments(self):
        rng = np.random.default_rng(84)
        config = ToyConfig(rate=4, p_up=.75)
        ends = np.array([event_path(config, 1, rng)[1][-1] for _ in range(10000)])
        self.assertAlmostEqual(ends.mean(), 2, delta=.1)
        self.assertAlmostEqual(ends.var(ddof=1), 4, delta=.3)

    def test_theory_distinguishes_variance_second_moment_and_qv(self):
        up = theory(ToyConfig(p_up=1), 1)
        self.assertEqual(up["variance"], 10)
        self.assertEqual(up["second_moment"], 110)
        alt = theory(ToyConfig(model="alternating"), 10)
        self.assertAlmostEqual(alt["variance"], .25)
        self.assertEqual(alt["quadratic_variation"], 100)
        self.assertEqual(alt["long_run_variance_rate"], 0)

    def test_mc_uncertainty_check(self):
        for model in ("independent", "alternating"):
            rows = estimate_batches(ToyConfig(model=model), 1, 123, batches=30, per_batch=1000)
            self.assertTrue(all(r["within_4se"] for r in rows), rows)

    def test_invalid_parameters(self):
        for kwargs in ({"rate": -1}, {"delta": 0}, {"p_up": 2},
                       {"rate": float("nan")}, {"model": "missing"}):
            with self.assertRaises(ValueError):
                ToyConfig(**kwargs)
        with self.assertRaises(ValueError):
            theory(ToyConfig(), -1)


if __name__ == "__main__":
    unittest.main()
