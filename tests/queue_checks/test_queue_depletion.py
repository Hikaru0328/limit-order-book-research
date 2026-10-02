import math
import unittest

import numpy as np

from queue_models.queue_depletion import QueueConfig, QueueSample, reference, simulate, summarize


class QueueDepletionTests(unittest.TestCase):
    def test_lecture_censoring_example(self):
        sample = QueueSample(np.array([2., 8., 10.]), np.array([True, True, False]),
                             np.array([0, 0, 2]), 10.)
        row = summarize(sample, 10)
        self.assertEqual(row["completed_only_mean"], 5)
        self.assertAlmostEqual(row["restricted_mean"], 20/3)
        self.assertAlmostEqual(row["survival"], 1/3)
        self.assertEqual(summarize(sample, 5)["depleted"], 1)

    def test_no_removal_and_zero_horizon(self):
        for cfg, t in [(QueueConfig(arrival=0, removal=0), 10), (QueueConfig(), 0),
                       (QueueConfig(removal=0), 3)]:
            sample = simulate(cfg, t, 20, 1)
            self.assertFalse(sample.observed.any())
            row = summarize(sample, t)
            self.assertEqual(row["restricted_mean"], t)
            self.assertTrue(math.isnan(row["completed_only_mean"]))
            ref = reference(cfg, [t])[0]
            self.assertAlmostEqual(ref["survival"], 1)
            self.assertAlmostEqual(ref["restricted_mean"], t)

    def test_reproducibility_absorption_and_censoring(self):
        a, b = [simulate(QueueConfig(), 30, 100, 41) for _ in range(2)]
        np.testing.assert_array_equal(a.duration, b.duration)
        self.assertTrue((a.remaining[a.observed] == 0).all())
        self.assertTrue((a.remaining[~a.observed] > 0).all())
        self.assertTrue((a.duration[~a.observed] == 30).all())

    def test_pure_death_reference_against_erlang(self):
        cfg = QueueConfig(arrival=0)
        for row in reference(cfg, [0, .1, 1, 3, 20]):
            t = row["cutoff"]
            x = cfg.removal*t
            expected_survival = math.exp(-x)*sum(x**k/math.factorial(k) for k in range(3))
            expected_rmst = sum(1-math.exp(-x)*sum(x**j/math.factorial(j)
                                for j in range(k+1)) for k in range(3))/cfg.removal
            self.assertAlmostEqual(row["survival"], expected_survival, places=11)
            self.assertAlmostEqual(row["restricted_mean"], expected_rmst, places=11)

    def test_mc_against_independent_reference(self):
        for arrival in (0., 1., 1.9, 2.):
            cfg = QueueConfig(arrival=arrival)
            sample = simulate(cfg, 30, 15000, 91)
            for ref in reference(cfg, [5., 30.]):
                row = summarize(sample, ref["cutoff"])
                self.assertLessEqual(abs(row["restricted_mean"]-ref["restricted_mean"]),
                                     5*row["restricted_se"]+1e-9)
                self.assertLessEqual(row["survival_low"]-1e-10, ref["survival"])
                self.assertGreaterEqual(row["survival_high"]+1e-10, ref["survival"])

    def test_monotone_cutoffs_and_reference_convergence(self):
        cfg = QueueConfig(arrival=2.)
        refs = reference(cfg, [1., 10., 100.])
        more = reference(cfg, [1., 10., 100.], margin=16)
        self.assertTrue(np.all(np.diff([r["survival"] for r in refs]) <= 0))
        self.assertTrue(np.all(np.diff([r["restricted_mean"] for r in refs]) >= 0))
        for a, b in zip(refs, more):
            self.assertAlmostEqual(a["restricted_mean"], b["restricted_mean"], places=9)
            self.assertLess(a["rmst_tail_bound"], 1e-12)

    def test_infinite_and_finite_mean_are_distinct_from_hit_probability(self):
        self.assertEqual(QueueConfig(arrival=0).mean_depletion_time, 1.5)
        self.assertEqual(QueueConfig(arrival=1).mean_depletion_time, 3.)
        self.assertAlmostEqual(QueueConfig(arrival=1.9).mean_depletion_time, 30.)
        critical = QueueConfig(arrival=2.)
        self.assertEqual(critical.eventual_depletion_probability, 1.)
        self.assertTrue(math.isinf(critical.mean_depletion_time))

    def test_invalid_inputs_and_extrapolation(self):
        for kwargs in ({"initial": 0}, {"initial": 1.5}, {"arrival": -1},
                       {"removal": float("nan")}):
            with self.assertRaises(ValueError):
                QueueConfig(**kwargs)
        with self.assertRaises(ValueError):
            simulate(QueueConfig(), -1, 20, 0)
        with self.assertRaises(ValueError):
            summarize(simulate(QueueConfig(), 1, 20, 0), 2)


if __name__ == "__main__":
    unittest.main()
