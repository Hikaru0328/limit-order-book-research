import unittest
from dataclasses import replace
import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import expm
from ct_lob.model import Model, Parameters, simulate
from ct_lob.filtering import (Survival, observe, queue_filter, price_filter,
                              snapshot_condition, no_book_event, normalize)


class ContinuousTimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = Model(Parameters(cap=3, refill=2))

    def test_generator_and_stationarity(self):
        m = self.model; l = m.generator
        assert_allclose(l.sum(1), 0, atol=1e-14)
        self.assertGreaterEqual((l - np.diag(np.diag(l))).min(), 0)
        assert_allclose(m.stationary @ l, 0, atol=1e-14)
        assert_allclose(m.transition(3).sum(1), 1, atol=1e-13)

    def test_rates_and_reset_both_sides(self):
        m = self.model; p = m.parameters; i = m.index(3, 1, 1)
        assert_allclose(m.rates[i], [0, p.lam*np.exp(-p.beta), p.mu*np.exp(-p.beta),
                                    p.mu*np.exp(p.beta), 3*p.theta, p.theta, p.kappa])
        for e in (3, 5):
            assert_allclose(m.states[m.targets[i, e]], [2, 2, 1])
            self.assertEqual(m.rewards[i, e], 1)
        i = m.index(1, 3, -1)
        for e in (2, 4): self.assertEqual(m.rewards[i, e], -1)

    def test_fixed_spread(self):
        b, a = self.model.quotes(np.arange(-20, 21))
        assert_allclose(a-b, self.model.parameters.spread)
        self.assertTrue(np.all(a>b))

    def test_reflection_symmetry(self):
        m = self.model; mirror = [m.index(a,b,-z) for b,a,z in m.states]
        assert_allclose(m.generator[np.ix_(mirror,mirror)],m.generator)
        assert_allclose(m.reward_operator[np.ix_(mirror,mirror)],-m.reward_operator)
        assert_allclose(m.stationary[mirror],m.stationary,atol=1e-14)

    def test_flip_rate_factor_two(self):
        m=self.model; z=m.states[:,2]
        assert_allclose(m.generator@z,-2*m.parameters.kappa*z,atol=1e-14)
        assert_allclose(m.transition(3)@z,np.exp(-6*m.parameters.kappa)*z,atol=1e-13)

    def test_marked_moments_against_tilted_generator(self):
        m=self.model; t=1.3; eps=1e-4; one=np.ones(m.n)
        def mgf(a):
            tilted=m.generator.copy()
            for sign in (-1,1):tilted+=np.expm1(a*sign*m.parameters.tick)*m.price_jumps[sign]
            return expm(t*tilted)@one
        g,h=m.price_moments(t)
        assert_allclose(g,(mgf(eps)-mgf(-eps))/(2*eps),atol=1e-8)
        assert_allclose(h,(mgf(eps)+mgf(-eps)-2)/(eps*eps),atol=2e-6)
        self.assertGreaterEqual((h-g*g).min(),-1e-12)

    def test_exact_clock_scaling(self):
        m=self.model
        for s in (.25,3,4):
            n=Model(m.parameters.slowed(s))
            assert_allclose(n.generator,m.generator/s,atol=1e-14)
            assert_allclose(n.stationary,m.stationary,atol=1e-14)
            assert_allclose(n.transition(2*s),m.transition(2),atol=1e-13)
            for a,b in zip(n.price_moments(2*s),m.price_moments(2)):assert_allclose(a,b,atol=1e-12)

    def test_gillespie_clock_coupling(self):
        m=self.model; n=Model(m.parameters.slowed(4)); p=simulate(m,20,901); q=simulate(n,80,901)
        assert_allclose(q.times,4*p.times,atol=1e-11)
        assert_allclose(q.state,p.state);assert_allclose(q.event,p.event);assert_allclose(q.reward,p.reward)

    def test_survival_against_dense_exponential(self):
        m=self.model; d=m.generator-m.price_jumps[-1]-m.price_jumps[1]; engine=Survival(d)
        rng=np.random.default_rng(9)
        for dt in (0,1e-6,.1,2,20):
            p=rng.dirichlet(np.ones(m.n))
            assert_allclose(engine.step(p,dt),normalize(p@expm(dt*d)),atol=2e-12)
        engine.spectral=False
        assert_allclose(engine.step(p,2),normalize(p@expm(2*d)),atol=2e-12)

    def test_analytic_two_state_survival(self):
        p=np.array([.1,.9]);h=np.array([7.,12.]);k=.2;a=np.array([[-k-h[0],k],[k,-k-h[1]]])
        for dt in (0,.01,.2,20):
            assert_allclose(no_book_event(p,h,k,dt),normalize(p@expm(dt*a)),atol=1e-13)

    def test_filters_independent_full_matrix_likelihood(self):
        m=self.model; path=simulate(m,2,42)
        for kind in ('price','queue_price','marked'):
            obs=observe(m,path,kind);p=m.stationary.copy()
            if kind!='price':
                mask=np.all(m.states[:,:2]==obs.initial_queue,axis=1);p=normalize(p*mask)
                d=m.events[6]-np.diag(m.rates.sum(1))
            else:d=m.generator-m.price_jumps[1]-m.price_jumps[-1]
            last=0
            for j,t in enumerate(obs.times):
                p=p@expm((t-last)*d)
                if kind=='price': jump=m.price_jumps[1 if obs.reward[j]>0 else -1]
                else:
                    jump=np.zeros_like(d)
                    codes=[obs.event[j]] if kind=='marked' else range(6)
                    for e in codes:
                        for i in range(m.n):
                            dest=m.targets[i,e]
                            if np.array_equal(m.states[dest,:2],obs.queues[j]) and m.rewards[i,e]==obs.reward[j]:
                                jump[i,dest]+=m.rates[i,e]
                p=normalize(p@jump);last=t
            p=normalize(p@expm((2-last)*d))
            if kind=='price':assert_allclose(price_filter(m,obs,[2])[0],p,atol=2e-12)
            else:assert_allclose(queue_filter(m,obs,[2])[0],p.reshape(-1,2).sum(0),atol=2e-12)

    def test_null_hidden_is_uninformative(self):
        m=Model(replace(self.model.parameters,beta=0));path=simulate(m,30,5);queries=np.arange(0,30,2.)
        for kind in ('queue_price','marked'):assert_allclose(queue_filter(m,observe(m,path,kind),queries),.5,atol=1e-13)
        p=price_filter(m,observe(m,path,'price'),queries)
        assert_allclose(p.reshape(len(queries),-1,2).sum(1),.5,atol=2e-12)
        g,_=m.price_moments(3);assert_allclose(g[::2],g[1::2],atol=1e-13)

    def test_causal_query_and_snapshot_conditioning(self):
        m=self.model;path=simulate(m,30,9);queries=np.arange(1,20.)
        for kind in ('price','queue_price','marked'):
            obs=observe(m,path,kind);fn=price_filter if kind=='price' else queue_filter
            a=fn(m,obs,queries);b=fn(m,obs,[queries[-1]])
            assert_allclose(a[-1],b[0],atol=1e-12)
            # The observations data object never contains a hidden state column.
            self.assertFalse(hasattr(obs,'state'))
            if obs.event is not None:self.assertTrue(np.all(obs.event<6))
        obs=observe(m,path,'price');p=price_filter(m,obs,queries);saved=p.copy()
        q=m.states[path.states_at(queries),:2];post=snapshot_condition(m,p,q)
        assert_allclose(p,saved);assert_allclose(post.sum(1),1)

    def test_filter_clock_scaling(self):
        m=self.model;n=Model(m.parameters.slowed(4));p=simulate(m,20,71);q=simulate(n,80,71);t=np.arange(0,20.)
        for kind in ('price','queue_price','marked'):
            fn=price_filter if kind=='price' else queue_filter
            assert_allclose(fn(m,observe(m,p,kind),t),fn(n,observe(n,q,kind),t*4),atol=2e-11)

    def test_invalid_parameters(self):
        for args in ({'kappa':0},{'theta':-1},{'beta':np.nan},{'refill':1},{'cap':2.5}):
            with self.assertRaises(ValueError):Parameters(**args)


if __name__=='__main__':unittest.main()
