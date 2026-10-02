"""Exact population moments for the finite-queue toy LOB, including block ratios.

DP sums over all event paths, retaining removal counts and first/second return
moments. It has no sampling error; floating-point roundoff remains.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class BlockPopulation:
    transition: np.ndarray
    stationary: np.ndarray
    mean: np.ndarray
    second: np.ndarray
    weighted: np.ndarray  # weighted[i,s,t] = E[f_i(block) 1(end=t) | start=s]
    cap: int
    block_size: int


def event_model(cap=8):
    if not isinstance(cap, (int,np.integer)) or cap < 2:
        raise ValueError('cap must be an integer >= 2')
    states=np.array([(bid,ask) for bid in range(1,cap+1) for ask in range(1,cap+1)])
    n=len(states)
    targets=np.empty((4,n),int); returns=np.zeros((4,n))
    # cancel and market orders have identical effects and are combined here.
    probabilities=np.array([.2,.2,.3,.3])
    for s,(bid,ask) in enumerate(states):
        for event in range(4):
            b,a=bid,ask
            if event==0: b=min(cap,b+1)
            elif event==1: a=min(cap,a+1)
            elif event==2:
                b-=1
                if b==0: b=cap; returns[event,s]=-1
            else:
                a-=1
                if a==0: a=cap; returns[event,s]=1
            targets[event,s]=(b-1)*cap+a-1
    transition=np.zeros((n,n))
    for e in range(4): transition[np.arange(n),targets[e]]+=probabilities[e]
    system=transition.T-np.eye(n); system[-1]=1
    rhs=np.zeros(n);rhs[-1]=1
    pi=np.linalg.solve(system,rhs)
    if pi.min() < -1e-12 or not np.allclose(pi@transition,pi,atol=1e-12):
        raise ValueError('stationary solution failed')
    return states,targets,returns,probabilities,transition,pi


def block_population(cap=8, b=1):
    """Canonical features: 63 endpoint indicators (cap=8), depletion, return.

    Omit the last endpoint indicator to remove the constant dependence. Feature
    dimension is cap**2+1. D is computed after aggregation, with D=0 for no removals.
    """
    if not isinstance(b,(int,np.integer)) or b<1:
        raise ValueError('b must be a positive integer')
    states,targets,returns,probabilities,p,pi=event_model(cap)
    n=len(states)
    # Axis order: ask removal count, bid removal count, start state, end state.
    shape=(b+1,b+1,n,n)
    mass=np.zeros(shape); first=np.zeros(shape); second=np.zeros(shape)
    mass[0,0]=np.eye(n)
    for step in range(b):
        pn=np.zeros(shape); rn=np.zeros(shape); r2n=np.zeros(shape)
        for e,prob in enumerate(probabilities):
            da=int(e==3); db=int(e==2)
            for s in range(n):
                dest=targets[e,s]; ret=returns[e,s]
                old=(slice(0,step+1),slice(0,step+1),slice(None),s)
                new=(slice(da,step+1+da),slice(db,step+1+db),slice(None),dest)
                m=mass[old];r=first[old]
                pn[new]+=prob*m
                rn[new]+=prob*(r+ret*m)
                r2n[new]+=prob*(second[old]+2*ret*r+ret*ret*m)
        mass,first,second=pn,rn,r2n
    ab,bb=np.indices((b+1,b+1));total=ab+bb
    depletion=np.divide(ab-bb,total,out=np.zeros_like(total,dtype=float),where=total>0)
    pb=mass.sum(axis=(0,1))
    md=np.einsum('ab,abst->st',depletion,mass)
    mr=first.sum(axis=(0,1))
    d=n+1
    weighted=np.zeros((d,n,n))
    for i in range(n-1): weighted[i,:,i]=pb[:,i]
    weighted[-2]=md;weighted[-1]=mr
    mean=np.einsum('s,ist->i',pi,weighted)
    raw=np.zeros((d,d))
    raw[:n-1,:n-1]=np.diag((pi@pb)[:n-1])
    raw[:n-1,-2]=(pi@md)[:n-1];raw[-2,:n-1]=raw[:n-1,-2]
    raw[:n-1,-1]=(pi@mr)[:n-1];raw[-1,:n-1]=raw[:n-1,-1]
    raw[-2,-2]=np.einsum('s,ab,abst->',pi,depletion**2,mass)
    raw[-2,-1]=raw[-1,-2]=np.einsum('s,ab,abst->',pi,depletion,first)
    raw[-1,-1]=np.einsum('s,abst->',pi,second)
    return BlockPopulation(pb,pi,mean,raw,weighted,cap,b)


def feature_map(cap, basis='baseline'):
    """Linear map from canonical centered features to chosen centered basis.

    baseline: I,D,R. polynomial: baseline plus qb,qa,qb²,qa²,qb*qa,
    queues divided by cap. complete: all nonconstant endpoint indicators,D,R.
    """
    states=event_model(cap)[0];n=len(states)
    if basis=='complete': return np.eye(n+1)
    if basis not in ('baseline','polynomial'): raise ValueError('unknown basis')
    bid,ask=states.T
    functions=[(bid-ask)/(bid+ask)]
    if basis=='polynomial':
        qb=bid/cap;qa=ask/cap
        functions += [qb,qa,qb**2,qa**2,qb*qa]
    out=np.zeros((3 if basis=='baseline' else 8,n+1))
    out[0,:n-1]=functions[0][:-1]-functions[0][-1]
    out[1,-2]=1;out[2,-1]=1
    for j,values in enumerate(functions[1:],start=3): out[j,:n-1]=values[:-1]-values[-1]
    return out


def population_covariances(pop,max_lag,basis='baseline'):
    if max_lag<0: raise ValueError('max_lag must be nonnegative')
    t=feature_map(pop.cap,basis)
    mu=t@pop.mean
    weighted=np.einsum('ij,jst->ist',t,pop.weighted)
    left=np.einsum('s,ist->ti',pop.stationary,weighted) # past feature weighted by its endpoint
    right=weighted.sum(axis=2).T  # next feature expectation conditional on start
    result=[t@pop.second@t.T-np.outer(mu,mu)]
    propagated=right.copy()
    for lag in range(1,max_lag+1):
        result.append((left.T@propagated).T-np.outer(mu,mu))
        propagated=pop.transition@propagated
    return np.array(result)


def canonical_observations(records,cap=8):
    from .toy_lob import observables
    state=(records['qb'].astype(int)-1)*cap+records['qa'].astype(int)-1
    onehot=(state[:,None]==np.arange(cap*cap-1)).astype(float)
    return np.column_stack((onehot,observables(records)[:,1:]))


def basis_observations(records,cap=8,basis='baseline'):
    # Omit irrelevant offsets: all covariance estimators center the features.
    if basis=='baseline':
        from .toy_lob import observables
        return observables(records)
    if basis=='polynomial':
        from .toy_lob import observables
        qb=records['qb']/cap;qa=records['qa']/cap
        return np.column_stack((observables(records),qb,qa,qb*qb,qa*qa,qb*qa))
    if basis=='complete': return canonical_observations(records,cap)
    raise ValueError('unknown basis')


def target_map(cap,basis):
    """Map chosen centered features to the common centered target (I,D,R)."""
    if basis=='complete': return feature_map(cap,'baseline')
    return np.eye(3,3 if basis=='baseline' else 8)


def target_kernel(omega,c0,cap=8,basis='baseline'):
    """Whitened history-to-common-target map, comparable under feature reparameterization.

    C_target^(-1/2) T Omega[k] C_features^(1/2). Population covariance is used
    for both exact and estimated coefficients when measuring estimation error.
    """
    t=target_map(cap,basis)
    values,vectors=np.linalg.eigh(c0)
    if values.min()<=0: raise ValueError('feature covariance is not positive definite')
    root=(vectors*np.sqrt(values))@vectors.T
    v,u=np.linalg.eigh(t@c0@t.T)
    if v.min()<=0: raise ValueError('target covariance is not positive definite')
    inverse=(u/np.sqrt(v))@u.T
    return np.array([inverse@t@k@root for k in omega])


def direct_event_mori(cap=8,basis='baseline',memory_lags=32):
    """Independent b=1 MZ projection on the finite (start state, event) chain.

    Apply P K (Q K)^k to functions directly; no covariance recursion or block DP.
    This independently checks projection/order conventions for event observations.
    """
    if memory_lags<0: raise ValueError('memory_lags must be nonnegative')
    states,targets,returns,probs,_,pi=event_model(cap)
    n=len(states)
    end=targets.T.reshape(-1)
    rec=dict(qb=states[end,0],qa=states[end,1],
             ask_removed=np.tile(np.array([0.,0.,0.,1.]),n),
             bid_removed=np.tile(np.array([0.,0.,1.,0.]),n),
             **{'return':returns.T.reshape(-1),'count':np.ones(4*n)})
    y=basis_observations(rec,cap,basis)
    weights=(pi[:,None]*probs).reshape(-1)
    y=y-weights@y
    c0=y.T@(weights[:,None]*y)
    f=y.copy();omega=[]
    for _ in range(memory_lags+1):
        conditional=np.einsum('e,sed->sd',probs,f.reshape(n,4,-1))
        evolved=conditional[end]
        coeff=np.linalg.solve(c0,y.T@(weights[:,None]*evolved))
        omega.append(coeff.T)
        f=evolved-y@coeff
    return np.array(omega)
