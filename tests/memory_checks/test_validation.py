import unittest
import numpy as np
from lob_memory.linear import (stationary_covariance, simulate, aggregate, block_dynamics,
                               population_covariances, lifted_block_model, exact_mori)
from lob_memory.mori import from_covariances, whiten_kernel, covariance_sequence, origin_residuals
from lob_memory.prediction import compare_var
A = np.array([[.55, .25], [.10, .80]])
Q = np.array([[.7, .12], [.12, .4]])
C = np.array([[1., 0.]])

class Validation(unittest.TestCase):
    def test_lyapunov(self):
        s = stationary_covariance(A, Q)
        np.testing.assert_allclose(s, A@s@A.T+Q, atol=1e-13)

    def test_invalid_model(self):
        for a, q in [(np.eye(2), Q), (A, -np.eye(2))]:
            with self.assertRaises(ValueError): stationary_covariance(a, q)

    def test_endpoint_semigroup(self):
        a2, q2 = block_dynamics(A, Q, 2)
        a6, q6 = block_dynamics(a2, q2, 3)
        a, q = block_dynamics(A, Q, 6)
        np.testing.assert_allclose(a6, a, atol=1e-13)
        np.testing.assert_allclose(q6, q, atol=1e-13)

    def test_aggregation_associativity(self):
        x = np.arange(194).reshape(97, 2)
        for mode in ('sum', 'mean', 'endpoint'):
            np.testing.assert_allclose(aggregate(aggregate(x, 2, mode), 4, mode), aggregate(x, 8, mode))

    def test_full_state_no_population_memory(self):
        for b in (1, 2, 8):
            cov = population_covariances(A, Q, np.eye(2), b, 17)
            omega = from_covariances(cov)
            np.testing.assert_allclose(omega[0], np.linalg.matrix_power(A, b), atol=1e-12)
            np.testing.assert_allclose(omega[1:], 0, atol=1e-12)

    def test_endpoint_analytic_matches_correlation(self):
        for b in (1, 2, 4, 16):
            a, q = block_dynamics(A, Q, b)
            analytic = exact_mori(a, q, 1, 16)
            recursive = from_covariances(population_covariances(A, Q, C, b, 17))
            np.testing.assert_allclose(analytic, recursive, atol=1e-12)

    def test_aggregate_lift_matches_independent_covariance_sum(self):
        for mode in ('sum', 'mean'):
            for b in (2, 4, 16):
                a, q = lifted_block_model(A, Q, C, b, mode)
                cov = population_covariances(A, Q, C, b, 17, mode)
                s = stationary_covariance(a, q)
                for k in range(18):
                    np.testing.assert_allclose((np.linalg.matrix_power(a,k)@s)[:1,:1], cov[k], atol=1e-11)
                np.testing.assert_allclose(exact_mori(a,q,1,16), from_covariances(cov), atol=1e-11)

    def test_scalar_ar1_aggregation_generates_memory(self):
        a, q, c = np.array([[.8]]), np.array([[1.]]), np.array([[1.]])
        endpoint = from_covariances(population_covariances(a,q,c,4,10,'endpoint'))
        mean = from_covariances(population_covariances(a,q,c,4,10,'mean'))
        np.testing.assert_allclose(endpoint[1:],0,atol=1e-13)
        self.assertGreater(np.linalg.norm(mean[1:]), .01)

    def test_whitening_coordinate_invariance(self):
        s = stationary_covariance(A,Q)
        kernels = np.array([A, A@A])
        t = np.array([[2., .4], [0., .3]])
        transformed = np.array([t@k@np.linalg.inv(t) for k in kernels])
        n1 = np.linalg.norm(whiten_kernel(kernels,s),axis=(1,2))
        n2 = np.linalg.norm(whiten_kernel(transformed,t@s@t.T),axis=(1,2))
        np.testing.assert_allclose(n1,n2,atol=1e-12)

    def test_finite_sample_recovery(self):
        y = simulate(A,Q,100000,seed=71)[:,:1]
        est = from_covariances(covariance_sequence(y,9))
        exact = exact_mori(A,Q,1,8)
        self.assertLess(np.max(np.abs(est-exact)), .025)

    def test_origin_orthogonality_independent_ensemble(self):
        rng=np.random.default_rng(291)
        x=rng.multivariate_normal(np.zeros(2),stationary_covariance(A,Q),60000)
        ys=[x[:,:1].copy()]
        for _ in range(7):
            x=x@A.T+rng.multivariate_normal(np.zeros(2),Q,len(x))
            ys.append(x[:,:1].copy())
        segments=np.stack(ys,axis=1)
        w=origin_residuals(segments,exact_mori(A,Q,1,6))
        normalized=(w[:,:,0]*segments[:,0,0,None]).mean(axis=0)/np.var(segments[:,0,0])
        self.assertLess(np.abs(normalized).max(),.02)

    def test_prediction_uses_same_rows_and_detects_ar2(self):
        a=np.array([[.3,.5],[1.,0.]])
        q=np.diag([1.,0.])
        train=simulate(a,q,25000,seed=81)[:,:1]
        test=simulate(a,q,10000,seed=82)[:,:1]
        score=compare_var(train,test,2,1)
        self.assertGreater(score['delta_r2'],.1)
        self.assertEqual(score['test_targets'],len(test)-3)

    def test_memory_strength_and_lag_do_not_close_scale_flow(self):
        from lob_memory.rg import memory_strength, characteristic_block_lag
        initial=[]; coarse=[]
        for sign in (1,-1):
            a=np.array([[.3,.25],[.2,sign*.4]])
            q=np.eye(2)-a@a.T
            k=exact_mori(a,q,1,64)
            initial.append([k[0,0,0],memory_strength(k[1:]),characteristic_block_lag(k[1:])])
            ab,qb=block_dynamics(a,q,2)
            coarse.append(memory_strength(exact_mori(ab,qb,1,64)[1:]))
        np.testing.assert_allclose(initial[0],initial[1],atol=1e-13)
        self.assertGreater(coarse[0]/coarse[1],40)

    def test_singular_covariance_rejected(self):
        with self.assertRaises(ValueError): from_covariances(np.zeros((3,2,2)))

if __name__ == '__main__': unittest.main()
