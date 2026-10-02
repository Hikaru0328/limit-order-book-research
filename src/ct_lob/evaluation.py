"""Reusable evaluation of known-parameter filters on independent paths.

Population moments and Monte Carlo summaries are kept distinct. This does not
change the frozen baseline experiment or its output files.
"""
import numpy as np
from .model import simulate
from .filtering import Survival, observe, price_filter, queue_filter, snapshot_condition

ARMS = ('current_queue', 'price_history_current_queue', 'queue_price_history',
        'marked_history', 'price_only')


def entropy(p):
    p=np.clip(p,1e-15,1-1e-15)
    return -p*np.log2(p)-(1-p)*np.log2(1-p)


class Evaluation:
    def __init__(self, model, horizons=(1.,4.,16.)):
        self.model=model;self.horizons=tuple(horizons);pi=model.stationary
        self.queue_mass=pi.reshape(-1,2).sum(1)
        self.queue_posterior=pi.reshape(-1,2)/self.queue_mass[:,None]
        self.moments={};self.population=[]
        for horizon in self.horizons:
            g,h2=model.price_moments(horizon);mean=float(pi@g);variance=float(pi@h2-mean*mean)
            gq=np.sum(self.queue_posterior*g.reshape(-1,2),axis=1)
            oracle=float(pi@((g-mean)**2)/variance)
            current=float(self.queue_mass@((gq-mean)**2)/variance)
            # Compute a potentially small denominator as a nonnegative error,
            # rather than by subtracting two nearly equal R-squared values.
            gap=float(pi@((g-np.repeat(gq,2))**2)/variance)
            self.moments[horizon]=(g,h2,mean,variance,oracle,current,gq,gap)
            self.population.append(dict(horizon=horizon,variance=variance,oracle_r2=oracle,
                current_queue_r2=current,hidden_oracle_gap=gap,
                queue_entropy_bits=float(self.queue_mass@entropy(self.queue_posterior[:,1]))))
        self.survival=Survival(model.generator-model.price_jumps[-1]-model.price_jumps[1])

    def evaluate(self, seed, duration=1024., interval=2., burn=None):
        m=self.model
        if burn is None:burn=max(50.,10/(2*m.parameters.kappa))
        if duration<=0 or interval<=0 or burn<0:raise ValueError('invalid scoring window')
        times=burn+np.arange(0,duration,interval)
        path=simulate(m,burn+duration+max(self.horizons),seed)
        indexes=path.states_at(times);queues=m.states[indexes,:2];qi=indexes//2
        truth=m.states[indexes,2]==1
        pb=price_filter(m,observe(m,path,'price'),times,self.survival)
        beliefs={'current_queue':self.queue_posterior[qi],
                 'price_history_current_queue':snapshot_condition(m,pb,queues),
                 'queue_price_history':queue_filter(m,observe(m,path,'queue_price'),times),
                 'marked_history':queue_filter(m,observe(m,path,'marked'),times)}
        prob={arm:p[:,1] for arm,p in beliefs.items()};prob['price_only']=pb[:,1::2].sum(1)
        filters=[];predictions=[]
        for arm,p in prob.items():
            p=np.clip(p,1e-15,1-1e-15);ent=float(entropy(p).mean())
            cross=float(-np.where(truth,np.log2(p),np.log2(1-p)).mean())
            filters.append(dict(arm=arm,entropy_bits=ent,mutual_information_bits=1-ent,
                brier=float(((p-truth)**2).mean()),posterior_brier=float((p*(1-p)).mean()),
                calibration_entropy_residual=cross-ent))
        for horizon in self.horizons:
            g,h2,mean,var,oracle,current,gq,gap=self.moments[horizon]
            exact=g[indexes];state_values=g.reshape(-1,2)[qi]
            fitted={arm:np.sum(p*state_values,axis=1) for arm,p in beliefs.items()};fitted['price_only']=pb@g
            target=path.prices_at(times+horizon)-path.prices_at(times)
            for arm,pred in fitted.items():
                gain=float(np.mean((pred-gq[qi])**2)/var) if arm!='price_only' else None
                r2=current+gain if arm!='price_only' else float(np.mean((pred-mean)**2)/var)
                eta=gain/gap if arm!='price_only' and gap>1e-12 else None
                direct_gap=float(np.mean((pred-exact)**2)/var)
                realized=float(1-np.mean((target-pred)**2)/var)
                realized_oracle=float(1-np.mean((target-exact)**2)/var)
                predictions.append(dict(arm=arm,horizon=horizon,conditional_r2=r2,
                    hidden_recovery=eta,incremental_r2=gain,oracle_gap=oracle-r2,
                    direct_risk_r2=oracle-direct_gap,projection_identity_residual=oracle-direct_gap-r2,
                    realized_r2=realized,risk_identity_residual=realized_oracle-realized-direct_gap))
        counts=dict(seed=seed,end=path.end,query_count=len(times),events=len(path.times),
                    price_moves=int(np.sum(path.reward!=0)),hidden_flips=int(np.sum(path.event==6)))
        return filters,predictions,counts


def boundary_diagnostics(model):
    p=model.parameters;pi=model.stationary;q=model.states[:,:2];z=model.states[:,2]
    attempt=p.lam*np.column_stack((np.exp(p.beta*z),np.exp(-p.beta*z)))
    suppressed=np.sum(attempt*(q==p.cap),axis=1)
    return dict(states=model.n,mean_queue=float(pi@q.mean(1)),
        cap_side_fraction=float(pi@(q==p.cap).mean(1)),
        suppressed_lo_fraction=float((pi@suppressed)/(pi@attempt.sum(1))),
        price_move_rate=float(pi@((model.price_jumps[-1]+model.price_jumps[1])@np.ones(model.n))),
        book_event_rate=float(pi@model.rates[:,:6].sum(1)),
        stationary_residual=float(abs(pi@model.generator).max()))
