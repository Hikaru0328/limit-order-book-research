import unittest
import numpy as np

from queue_models.queue_race import event_law, theory, sample_cycles, cycle_estimate, price_paths


class QueueRaceTests(unittest.TestCase):
    def test_probability_and_symmetry(self):
        for a in range(1, 5):
            for b in range(1, 5):
                for p in [0, .2, .5, .9, 1]:
                    self.assertAlmostEqual(event_law(a, b, p)[:, 2].sum(), 1)
                    self.assertAlmostEqual(theory(a,b,p)['p_up'], 1-theory(b,a,1-p)['p_up'])

    def test_closed_form_21(self):
        for p in [0, .3, 1/np.sqrt(2), .9, 1]:
            r, d = 3., 2.
            ref = theory(2,1,p,r,d)
            self.assertAlmostEqual(ref['p_up'], p*p)
            self.assertAlmostEqual(ref['mean_t'], (1+p)/r)
            self.assertAlmostEqual(ref['var_t'], (1+2*p-p*p)/r**2)
            self.assertAlmostEqual(ref['cov_jt'], 2*d*p*p*(1-p)/r)
            self.assertAlmostEqual(ref['drift'], d*r*(2*p*p-1)/(1+p))
        self.assertAlmostEqual(theory(2,1,1,3,2)['variance_rate'], 3)

    def test_poisson_and_scaling(self):
        for p in [.1, .5, .9]:
            self.assertAlmostEqual(theory(1,1,p,3,2)['variance_rate'], 12)
            self.assertAlmostEqual(theory(1,1,p)['cov_jt'], 0)
        a = theory(2,1,.8,2,1)
        b = theory(2,1,.8,6,2)
        self.assertAlmostEqual(b['variance_rate'], 12*a['variance_rate'])
        self.assertAlmostEqual(theory(2,2,.5)['cov_jt'],0)

    def test_sample_and_shuffle(self):
        gen = np.random.default_rng(12)
        j,t,n = sample_cycles(gen, 100_000, 2,1,.8,2,1)
        ref = theory(2,1,.8,2,1)
        self.assertTrue(np.all(n[j>0]==2))
        self.assertLess(abs(np.mean(j>0)-ref['p_up']), .008)
        self.assertLess(abs(t.mean()-ref['mean_t']), .015)
        self.assertLess(abs(t.var()-ref['var_t']), .03)
        self.assertAlmostEqual(cycle_estimate(j,t)['drift'],cycle_estimate(j,gen.permutation(t))['drift'])

    def test_path_endpoints_and_variance(self):
        for p, sign in [(0,-1),(1,1)]:
            x = price_paths(np.random.default_rng(25),5000,[0,10,100],ask=2,bid=1,p=p,rate=2,delta=1)
            self.assertTrue(np.all(x[:,0]==0))
            self.assertTrue(np.all(sign*np.diff(x,axis=1)>=0))
            ref = theory(2,1,p,2,1)
            self.assertLess(abs(x[:,-1].var()/100-ref['variance_rate']), .12)

    def test_invalid(self):
        for kw in [dict(ask=0),dict(bid=1.5),dict(p=-.1),dict(rate=0),dict(delta=float('nan'))]:
            with self.assertRaises(ValueError):
                theory(**kw)
        with self.assertRaises(ValueError):
            price_paths(np.random.default_rng(0),10,[1,0])


if __name__ == '__main__':
    unittest.main()
