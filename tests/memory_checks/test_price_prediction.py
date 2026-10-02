import itertools,unittest
import numpy as np
from lob_memory.price_prediction import (anchors,price_targets,price_design,reward_moments,
    oracle_prediction,select_ridge,r2,population_price_scores,upper_rank_p)
from lob_memory.toy_lob import simulate_lob
from lob_memory.finite_lob import event_model,block_population

class PricePrediction(unittest.TestCase):
    def test_target_exactly_future_events_and_common_rows(self):
        raw=simulate_lob(512,seed=82,burn=0,stationary_start=True)
        raw['return']=np.arange(512,dtype=float)
        common=anchors(512)
        targets=[]
        for b in (1,2,4,8,16):
            x,y,t=price_design(raw,b,t=common)
            targets.append(y)
            np.testing.assert_array_equal(y,[raw['return'][a:a+32].sum() for a in common])
            np.testing.assert_array_equal(t,common)
        for y in targets[1:]:np.testing.assert_array_equal(y,targets[0])

    def test_future_mutations_cannot_change_features_at_anchor(self):
        raw=simulate_lob(256,seed=83)
        t=np.array([64,80])
        before=price_design(raw,4,t=t)[0]
        changed={k:v.copy() for k,v in raw.items()}
        for v in changed.values():v[80:]+=100
        after=price_design(changed,4,t=t)[0]
        np.testing.assert_array_equal(before,after)

    def test_past_mutations_cannot_change_target(self):
        raw=simulate_lob(256,seed=84);t=np.array([80,96])
        y=price_targets(raw,t)
        raw['return'][:80]+=10
        np.testing.assert_array_equal(y,price_targets(raw,t))

    def test_reward_oracle_matches_exhaustive_paths(self):
        cap=2;horizon=3
        states,next_state,reward,probs,_,_=event_model(cap)
        first,second=reward_moments(horizon,cap)
        for start in range(len(states)):
            m=v=0.
            for path in itertools.product(range(4),repeat=horizon):
                state=start;p=1.;r=0.
                for e in path:p*=probs[e];r+=reward[e,state];state=next_state[e,state]
                m+=p*r;v+=p*r*r
            self.assertAlmostEqual(first[start],m,places=12)
            self.assertAlmostEqual(second[start],v,places=12)

    def test_oracle_depends_on_current_state_only(self):
        raw=simulate_lob(256,seed=85);t=np.array([64,80])
        p=oracle_prediction(raw,t)
        changed={k:v.copy() for k,v in raw.items()}
        changed['return'][:]=999
        np.testing.assert_array_equal(p,oracle_prediction(changed,t))

    def test_ridge_matches_direct_normal_equations(self):
        rng=np.random.default_rng(9);x=rng.normal(size=(200,5));y=x@np.arange(5)+rng.normal(size=200)
        xv=rng.normal(size=(50,5));yv=xv@np.arange(5)
        model=select_ridge(x,y,xv,yv,grid=(.01,))[0]
        self.assertEqual(model.regularization,.01)
        z=(x-x.mean(0))/x.std(0)
        expected=np.linalg.solve(z.T@z/len(z)+.01*np.eye(5),z.T@(y-y.mean())/len(z))
        np.testing.assert_allclose(model.coefficient,expected,atol=1e-12)
        np.testing.assert_allclose(model.x_mean,x.mean(0))

    def test_validation_selects_without_refitting_or_test_access(self):
        rng=np.random.default_rng(11);x=rng.normal(size=(100,3));y=2*x[:,0]
        xv=rng.normal(size=(30,3))+4;yv=2*xv[:,0]
        m=select_ridge(x,y,xv,yv)[0]
        np.testing.assert_allclose(m.x_mean,x.mean(0))
        before=m.coefficient.copy()
        prediction=m.predict(xv)
        r2(yv,prediction);r2(yv+100,prediction)
        np.testing.assert_array_equal(m.coefficient,before)
        self.assertEqual(m.regularization,0.)

    def test_zero_correction_option(self):
        rng=np.random.default_rng(4);x=rng.normal(size=(100,3));xv=rng.normal(size=(40,3))
        model=select_ridge(x,np.ones(100),xv,np.zeros(40),zero_targets=(0,))[0]
        self.assertEqual(model.regularization,'disabled')
        np.testing.assert_array_equal(model.predict(xv),np.zeros(40))

    def test_complete_population_price_history_has_no_gain(self):
        for b in (1,2):
            score=population_price_scores(block_population(3,b),'complete',horizon=4,history=4)
            self.assertAlmostEqual(score['delta_r2'],0,places=10)
            self.assertAlmostEqual(score['r2_current'],score['r2_oracle'],places=10)

    def test_population_linear_scores_are_bounded(self):
        for basis in ('baseline','polynomial'):
            score=population_price_scores(block_population(4,2),basis,horizon=8,history=8)
            self.assertGreaterEqual(score['delta_r2'],-1e-10)
            self.assertLessEqual(score['r2_history'],score['r2_oracle']+1e-10)

    def test_upper_rank_keeps_plus_one_and_ties(self):
        self.assertEqual(upper_rank_p(4,[1,2,3]),.25)
        self.assertEqual(upper_rank_p(2,[1,2,3]),.75)
        self.assertEqual(upper_rank_p(0,[1,2,3]),1.)

    def test_invalid_anchors_and_constant_targets(self):
        with self.assertRaises(ValueError):anchors(20)
        with self.assertRaises(ValueError):r2(np.ones(3),np.ones(3))
        with self.assertRaises(ValueError):select_ridge(np.eye(3),np.ones(3),np.eye(3),np.ones(3),grid=(-1.,))

if __name__=='__main__':unittest.main()
