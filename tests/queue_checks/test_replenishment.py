import unittest

import numpy as np

from queue_models.replenishment import (Config, mean_reference, boundary_probability,
    finite_exit_mean, finite_reference, simulate, corrected_mean)
from queue_models.market_depletion import DepletionConfig, theory


class ReplenishmentTests(unittest.TestCase):
    def test_no_arrival_and_single_order_closed_forms(self):
        a = mean_reference(Config(arrival=0))
        b = theory(DepletionConfig(3, .2, 1))["mean"]
        self.assertAlmostEqual(a["mean"], b)
        cfg = Config(initial=1, arrival=1, removal=0, theta=.5)
        self.assertAlmostEqual(mean_reference(cfg)["mean"], np.expm1(2), places=9)
        self.assertEqual(mean_reference(cfg, 0)["mean"], 0)

    def test_series_bound_and_recurrence(self):
        cfg = Config()
        loose = mean_reference(cfg, tolerance=1e-5)
        tight = mean_reference(cfg, tolerance=1e-12)
        self.assertLessEqual(abs(tight["mean"]-loose["mean"]), loose["tail_bound"]+1e-12)
        u = [mean_reference(cfg, q)["mean"] for q in (2, 3, 4)]
        self.assertAlmostEqual(cfg.arrival*(u[2]-u[1])+(cfg.removal+3*cfg.theta)*(u[0]-u[1]), -1, places=8)

    def test_boundary_identity_and_monotonicity(self):
        cfg = Config()
        previous = 0
        for upper in (4, 8, 16, 32):
            value = finite_exit_mean(cfg, upper)
            identity = mean_reference(cfg)["mean"] - mean_reference(cfg, upper)["mean"]*boundary_probability(cfg, upper)
            self.assertAlmostEqual(value, identity, places=8)
            self.assertGreaterEqual(value, previous)
            previous = value

    def test_uniformization_against_exponential(self):
        cfg = Config(initial=1, arrival=0)
        rate = cfg.removal+cfg.theta
        for row in finite_reference(cfg, [0, 1, 5], 8):
            t = row["cutoff"]
            self.assertAlmostEqual(row["survival"], np.exp(-rate*t), places=11)
            self.assertAlmostEqual(row["restricted_mean"], -np.expm1(-rate*t)/rate, places=11)

    def test_snapshots_reproducibility_and_absorption(self):
        cfg = Config()
        a = simulate(cfg, [0, 2, 10], 100, 11)
        b = simulate(cfg, [0, 2, 10], 100, 11)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x.duration, y.duration)
            self.assertTrue((x.remaining[x.observed] == 0).all())
            self.assertTrue((x.duration[~x.observed] == x.horizon).all())
        self.assertTrue(np.all(a[1].observed <= a[2].observed))
        self.assertAlmostEqual(corrected_mean(cfg, a[0])["corrected_mean"], mean_reference(cfg)["mean"])

    def test_mc_and_corrected_mean(self):
        cfg = Config(arrival=1)
        sample = simulate(cfg, [5], 10000, 19)[0]
        ref = finite_reference(cfg, [5], 32)[0]
        se = sample.duration.std(ddof=1)/np.sqrt(len(sample.duration))
        self.assertLess(abs(sample.duration.mean()-ref["restricted_mean"]), 5*se+ref["rmst_error"])
        corrected = corrected_mean(cfg, sample)
        self.assertLess(abs(corrected["corrected_mean"]-mean_reference(cfg)["mean"]), 5*corrected["corrected_se"])

    def test_scaling(self):
        a, b = Config(), Config(arrival=4, removal=2, theta=.4)
        self.assertAlmostEqual(mean_reference(a)["mean"]/2, mean_reference(b)["mean"], places=8)
        x = simulate(a, [3, 10], 100, 8)
        y = simulate(b, [1.5, 5], 100, 8)
        for s, t in zip(x, y):
            np.testing.assert_allclose(s.duration/2, t.duration)
            np.testing.assert_array_equal(s.remaining, t.remaining)

    def test_validation(self):
        for kwargs in ({"initial": 0}, {"theta": 0}, {"arrival": -1}, {"removal": np.nan}):
            with self.assertRaises(ValueError):
                Config(**kwargs)
        for cuts in ([2, 1], [-1], [], [1, 1]):
            with self.assertRaises(ValueError):
                simulate(Config(), cuts)
        with self.assertRaises(ValueError):
            boundary_probability(Config(), 3)
        with self.assertRaises(RuntimeError):
            mean_reference(Config(), max_terms=1)
