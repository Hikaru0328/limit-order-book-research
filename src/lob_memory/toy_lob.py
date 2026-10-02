"""Minimal finite-queue, fixed-spread event model; not a calibrated market model.

Each event selects bid/ask and add/cancel/market with state-local probabilities.
Adds at the queue cap remain counted attempts (self transitions). A removal at
size one moves mid by one tick toward the depleted side and refills that queue.
The opposite queue is retained. Spread is fixed at one tick and is not a feature.
"""
import numpy as np


def simulate_lob(n, seed=0, cap=8, burn=2048, stationary_start=False):
    if n < 2 or cap < 2 or burn < 0:
        raise ValueError('require n>=2, cap>=2, burn>=0')
    rng=np.random.default_rng(seed)
    # bid add, ask add, bid cancel, ask cancel, bid market, ask market
    qb=qa=cap
    if stationary_start:
        from .finite_lob import event_model
        states,_,_,_,_,pi=event_model(cap)
        qb,qa=states[rng.choice(len(states),p=pi)]
    events=rng.choice(6,n+burn,p=[.2,.2,.15,.15,.15,.15])
    data={k:np.empty(n) for k in ('qb','qa','bid_removed','ask_removed','return','count')}
    for t,e in enumerate(events):
        bid=e%2==0
        removed=0.; ret=0.
        if e < 2:
            if bid: qb=min(cap,qb+1)
            else: qa=min(cap,qa+1)
        else:
            removed=1.
            if bid:
                qb-=1
                if qb==0: qb=cap; ret=-1.
            else:
                qa-=1
                if qa==0: qa=cap; ret=1.
        if t>=burn:
            j=t-burn
            data['qb'][j]=qb; data['qa'][j]=qa
            data['bid_removed'][j]=removed if bid else 0.
            data['ask_removed'][j]=removed if not bid else 0.
            data['return'][j]=ret; data['count'][j]=1.
    return data


def aggregate_records(records, b):
    from .linear import aggregate
    return {k:aggregate(np.asarray(v)[:,None],b,'endpoint' if k in ('qb','qa') else 'sum')[:,0]
            for k,v in records.items()}


def observables(records):
    """I, D, R; D=0 if no removals. Retain raw totals for later aggregation."""
    qb,qa=records['qb'],records['qa']
    ua,ub=records['ask_removed'],records['bid_removed']
    denom=ua+ub
    depletion=np.divide(ua-ub,denom,out=np.zeros_like(denom),where=denom>0)
    return np.column_stack(((qb-qa)/(qb+qa),depletion,records['return']))
