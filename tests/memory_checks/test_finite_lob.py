import itertools,unittest
import numpy as np
from lob_memory.finite_lob import (event_model,block_population,population_covariances,
    feature_map,basis_observations,target_kernel,target_map)
from lob_memory.toy_lob import simulate_lob,aggregate_records,observables
from lob_memory.mori import from_covariances,covariance_sequence


def exhaustive(cap,b):
    states,target,ret,probs,p,pi=event_model(cap);n=len(states);d=n+1
    weighted=np.zeros((d,n,n));raw=np.zeros((d,d));mean=np.zeros(d)
    for start in range(n):
        for path in itertools.product(range(4),repeat=b):
            state=start;prob=1.;r=0.;ua=ub=0
            for e in path:
                prob*=probs[e];r+=ret[e,state];state=target[e,state]
                ua+=int(e==3);ub+=int(e==2)
            y=np.zeros(d)
            if state<n-1:y[state]=1
            y[-2]=(ua-ub)/(ua+ub) if ua+ub else 0
            y[-1]=r
            weighted[:,start,state]+=prob*y
            raw+=pi[start]*prob*np.outer(y,y);mean+=pi[start]*prob*y
    return weighted,raw,mean

class FiniteLOB(unittest.TestCase):
    def test_stationary_and_stochastic_transition(self):
        for cap in (2,4,8):
            _,_,_,_,p,pi=event_model(cap)
            np.testing.assert_allclose(p.sum(axis=1),1,atol=1e-14)
            np.testing.assert_allclose(pi@p,pi,atol=1e-14)
            self.assertGreater(pi.min(),0)

    def test_dp_matches_independent_exhaustive_paths(self):
        for b in (1,2,3):
            pop=block_population(3,b)
            weighted,second,mean=exhaustive(3,b)
            np.testing.assert_allclose(pop.weighted,weighted,atol=1e-13)
            np.testing.assert_allclose(pop.second,second,atol=1e-13)
            np.testing.assert_allclose(pop.mean,mean,atol=1e-13)
            p=event_model(3)[4]
            np.testing.assert_allclose(pop.transition,np.linalg.matrix_power(p,b),atol=1e-13)

    def test_two_block_cross_covariance_by_path_enumeration(self):
        cap=2;b=2
        states,target,ret,probs,p,pi=event_model(cap)
        joint=np.zeros((3,3));mu=np.zeros(3)
        for start in range(len(states)):
            for path in itertools.product(range(4),repeat=2*b):
                state=start;prob=1.;ys=[]
                for j in range(2):
                    ua=ub=0;r=0.
                    for e in path[j*b:(j+1)*b]:
                        prob*=probs[e];r+=ret[e,state];state=target[e,state]
                        ua+=int(e==3);ub+=int(e==2)
                    qb,qa=states[state]
                    ys.append(np.array([(qb-qa)/(qb+qa),(ua-ub)/(ua+ub) if ua+ub else 0,r]))
                joint+=pi[start]*prob*np.outer(ys[1],ys[0]);mu+=pi[start]*prob*ys[0]
        cov=population_covariances(block_population(cap,b),1)
        np.testing.assert_allclose(cov[1],joint-np.outer(mu,mu),atol=1e-13)

    def test_complete_basis_has_zero_memory_after_aggregation(self):
        for b in (1,2,4):
            c=population_covariances(block_population(4,b),9,'complete')
            k=from_covariances(c)
            np.testing.assert_allclose(k[1:],0,atol=1e-11)

    def test_reduced_basis_has_population_memory(self):
        c=population_covariances(block_population(8,1),9,'baseline')
        k=from_covariances(c)
        self.assertGreater(np.linalg.norm(k[1:]),.01)

    def test_feature_map_matches_centered_sample_features(self):
        record=aggregate_records(simulate_lob(1000,seed=4),4)
        canonical=basis_observations(record,basis='complete')
        for basis in ('baseline','polynomial','complete'):
            y=basis_observations(record,basis=basis)
            mapped=canonical@feature_map(8,basis).T
            np.testing.assert_allclose(y-y.mean(0),mapped-mapped.mean(0),atol=1e-13)
            expected=observables(record)
            predicted=y@target_map(8,basis).T
            np.testing.assert_allclose(expected-expected.mean(0),predicted-predicted.mean(0),atol=1e-13)

    def test_exact_moments_match_independent_simulator(self):
        rec=simulate_lob(180000,seed=284,cap=4,burn=0,stationary_start=True)
        for b in (1,4):
            y=observables(aggregate_records(rec,b))
            empirical=covariance_sequence(y,3)
            exact=population_covariances(block_population(4,b),3)
            np.testing.assert_allclose(empirical,exact,atol=.012)

    def test_direct_projection_matches_population_recursion(self):
        from lob_memory.finite_lob import direct_event_mori
        for basis in ('baseline','polynomial','complete'):
            c=population_covariances(block_population(4,1),9,basis)
            recursive=from_covariances(c)
            direct=direct_event_mori(4,basis,8)
            np.testing.assert_allclose(direct,recursive,atol=1e-10)

    def test_invalid_population_inputs(self):
        with self.assertRaises(ValueError): block_population(8,0)
        with self.assertRaises(ValueError): event_model(1)

if __name__=='__main__':unittest.main()
