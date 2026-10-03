"""Scientific invariants for the population-only projection bridge."""
import unittest
from unittest.mock import patch
import numpy as np
from lob_memory.finite_lob import (event_model,block_population,population_covariances,
                                  target_kernel,direct_event_mori)
from lob_memory.mori import from_covariances
from lob_memory.projection_bridge import (population_study,population_case,factors,
                                         BASES,NULL_TOL,IDENTITY_TOL)


class ProjectionBridge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows,cls.lags,cls.checks=population_study((1,2,4),cap=4)

    def test_stochastic_stationary_chain(self):
        for cap in (4,8):
            *_,p,pi=event_model(cap)
            np.testing.assert_allclose(p.sum(1),1,atol=1e-14,rtol=0)
            np.testing.assert_allclose(pi@p,pi,atol=1e-14,rtol=0)
            self.assertAlmostEqual(pi.sum(),1,places=14)
            self.assertGreater(pi.min(),0)
            self.assertGreaterEqual(p.min(),0)

    def test_complete_null_and_nonzero_compressed_memory(self):
        for r in self.rows:
            if r['basis']=='complete':
                self.assertLess(r['memory_strength']+r['tail_bound'],NULL_TOL)
                self.assertLess(r['covariance_recursion_memory_sum_32'],NULL_TOL)
                self.assertFalse(r['horizon_resolved'])
            elif r['basis']=='baseline':
                self.assertGreater(r['memory_strength'],NULL_TOL)
                self.assertTrue(r['horizon_resolved'])

    def test_independent_direct_event_check_at_public_cap(self):
        pop=block_population(8,1)
        for basis in BASES:
            c=population_covariances(pop,33,basis)
            recursive=from_covariances(c)
            direct=direct_event_mori(8,basis,32)
            np.testing.assert_allclose(direct,recursive,atol=IDENTITY_TOL,rtol=0)

    def test_finite_horizons_and_lag_exports(self):
        for r in self.rows:
            self.assertEqual(r['characteristic_event_horizon'],r['b']*r['characteristic_block_lag'])
            for v in r.values():
                if isinstance(v,(int,float)):self.assertTrue(np.isfinite(v))
            self.assertGreater(r['covariance_min_eigenvalue'],0)
            self.assertLess(r['covariance_condition'],1e12)
        for r in self.lags:
            self.assertEqual(r['event_horizon'],r['b']*r['lag'])
            self.assertEqual(r['target_distance'],r['b']*(r['lag']+1))
            self.assertGreaterEqual(r['target_memory_kernel_norm'],0)
            self.assertTrue(np.isfinite(r['target_memory_kernel_norm']))

    def test_no_random_generator_and_seed_independence(self):
        state=np.random.get_state()
        try:
            with patch('numpy.random.default_rng',side_effect=AssertionError('population used RNG')):
                np.random.seed(11);a=population_case(block_population(4,1),'baseline')
                np.random.seed(29);b=population_case(block_population(4,1),'baseline')
            self.assertEqual(a,b)
        finally:np.random.set_state(state)

    def test_factorization_and_tail_against_extended_recursion(self):
        pop=block_population(4,1)
        for basis in BASES:
            row,_=population_case(pop,basis)
            cutoff=row['memory_lags']
            c=population_covariances(pop,cutoff+33,basis)
            recursion=from_covariances(c)
            f,a,g=factors(pop,basis,c[0]);x=g.copy();seq=[]
            for _ in range(cutoff+33):seq.append(f@x);x=a@x
            np.testing.assert_allclose(seq,recursion,atol=IDENTITY_TOL,rtol=0)
            common=target_kernel(np.array(seq),c[0],4,basis)
            tail=np.linalg.norm(common[cutoff+1:],axis=(1,2)).sum()
            self.assertLessEqual(tail,row['tail_bound']+1e-14)

    def test_common_target_covariance(self):
        for r in self.rows:
            self.assertLess(r['common_target_covariance_error'],1e-13)
            self.assertLess(r['normalized_factor_recursion_max_error'],IDENTITY_TOL)
        self.assertEqual({r['features'] for r in self.rows if r['basis']=='baseline'},{3})
        self.assertEqual({r['features'] for r in self.rows if r['basis']=='polynomial'},{8})
        self.assertEqual({r['features'] for r in self.rows if r['basis']=='complete'},{17})


if __name__=='__main__':unittest.main()
