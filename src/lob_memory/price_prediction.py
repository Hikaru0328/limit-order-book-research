"""Fixed-event price targets, validation-only ridge, and exact Markov controls."""
from dataclasses import dataclass
import numpy as np
from .finite_lob import event_model,feature_map,population_covariances,basis_observations
from .toy_lob import aggregate_records

SCALES=(1,2,4,8,16)
HORIZON=32
HISTORY=32
RIDGE_GRID=(0.,1e-6,1e-4,.01,1.,100.,10000.)


def anchors(n,horizon=HORIZON,history=HISTORY,stride=16,max_block=16):
    """t counts completed events: features <=t; target consists of t+1..t+H."""
    if min(horizon,stride,max_block)<1 or history<0:
        raise ValueError('invalid event windows')
    start=((history+max_block+stride-1)//stride)*stride
    a=np.arange(start,n-horizon+1,stride,dtype=int)
    if len(a)<2: raise ValueError('too few common anchors')
    return a


def price_targets(records,t,horizon=HORIZON):
    r=np.asarray(records['return'],float)
    t=np.asarray(t,int)
    if len(t)==0 or np.any(t<0) or np.any(t+horizon>len(r)) or horizon<1:
        raise ValueError('invalid target anchors')
    cumulative=np.r_[0.,np.cumsum(r)]
    return cumulative[t+horizon]-cumulative[t]


def price_design(records,b,basis='baseline',t=None,history=HISTORY,horizon=HORIZON,cap=8):
    if b not in SCALES or history%b: raise ValueError('block must divide history')
    if basis not in ('baseline','polynomial'): raise ValueError('complete basis is control-only')
    if t is None:t=anchors(len(records['return']),horizon,history)
    t=np.asarray(t,int)
    if np.any(t%b) or np.any(t<history+b): raise ValueError('unaligned or insufficient history')
    block=aggregate_records(records,b)
    y=basis_observations(block,cap,basis)
    index=t//b-1
    x=np.concatenate([y[index-k] for k in range(history//b+1)],axis=1)
    return x,price_targets(records,t,horizon),t


def reward_moments(horizon=HORIZON,cap=8):
    """Exact conditional first/second moments of the next H-event mid change."""
    if horizon<0: raise ValueError('horizon must be nonnegative')
    states,next_state,rewards,prob,p,pi=event_model(cap)
    mean=np.zeros(len(states));second=np.zeros(len(states))
    for _ in range(horizon):
        new_mean=np.sum(prob[:,None]*(rewards+mean[next_state]),axis=0)
        new_second=np.sum(prob[:,None]*(rewards**2+2*rewards*mean[next_state]+second[next_state]),axis=0)
        mean,second=new_mean,new_second
    return mean,second


def oracle_prediction(records,t,horizon=HORIZON,cap=8):
    mean,_=reward_moments(horizon,cap)
    state=(records['qb'][t-1].astype(int)-1)*cap+records['qa'][t-1].astype(int)-1
    return mean[state]


@dataclass
class RidgeModel:
    x_mean: np.ndarray
    x_scale: np.ndarray
    intercept: float
    coefficient: np.ndarray
    regularization: object
    validation_mse: float

    def predict(self,x):
        return self.intercept+((np.asarray(x)-self.x_mean)/self.x_scale)@self.coefficient


def select_ridge(x,y,xv,yv,grid=RIDGE_GRID,zero_targets=()):
    """Fit once on training, choose separately per output using validation only.

    Objective: mean squared error + lambda*||standardized coefficients||².
    Intercept is unpenalized. The zero-slope train-mean model is included; outputs
    in zero_targets also permit an exactly zero prediction (oracle correction off).
    No refit on train+validation and no test input is accepted by this function.
    """
    x,y,xv,yv=map(lambda a:np.asarray(a,float),(x,y,xv,yv))
    if y.ndim==1:y=y[:,None]
    if yv.ndim==1:yv=yv[:,None]
    if x.ndim!=2 or xv.ndim!=2 or x.shape[1]!=xv.shape[1] or len(x)!=len(y) or len(xv)!=len(yv) or y.shape[1]!=yv.shape[1]:
        raise ValueError('incompatible train/validation shapes')
    if len(x)<2 or len(xv)<2 or any(not np.all(np.isfinite(a)) for a in (x,y,xv,yv)):
        raise ValueError('need finite training/validation data')
    grid=np.asarray(grid,float)
    if grid.ndim!=1 or not len(grid) or np.any(grid<0) or not np.all(np.isfinite(grid)):
        raise ValueError('ridge grid must be finite and nonnegative')
    center=x.mean(0);scale=x.std(0);scale=np.where(scale<1e-12,1.,scale)
    z=(x-center)/scale;zv=(xv-center)/scale;means=y.mean(0)
    gram=z.T@z/len(z);rhs=z.T@(y-means)/len(z)
    values,vectors=np.linalg.eigh(gram);values=np.maximum(values,0.)
    projected=vectors.T@rhs
    candidates=[]
    for lam in grid:
        denom=values+lam
        inverse=np.divide(1.,denom,out=np.zeros_like(denom),where=denom>max(1.,values.max())*1e-12) if lam==0 else 1./denom
        candidates.append(vectors@(inverse[:,None]*projected))
    candidates.append(np.zeros_like(rhs))
    lambdas=list(map(float,grid))+['intercept_only']
    models=[]
    for j in range(y.shape[1]):
        coefs=np.column_stack([c[:,j] for c in candidates])
        predictions=means[j]+zv@coefs
        losses=np.mean((predictions-yv[:,j,None])**2,axis=0)
        best=int(np.argmin(losses));offset=means[j];coef=coefs[:,best];label=lambdas[best];loss=losses[best]
        if j in zero_targets:
            zero_loss=float(np.mean(yv[:,j]**2))
            if zero_loss<=loss:offset=0.;coef=np.zeros_like(coef);label='disabled';loss=zero_loss
        models.append(RidgeModel(center.copy(),scale.copy(),float(offset),coef.copy(),label,float(loss)))
    return models


def r2(y,prediction):
    y=np.asarray(y,float);prediction=np.asarray(prediction,float)
    if y.shape!=prediction.shape:raise ValueError('target/prediction shape mismatch')
    variance=np.mean((y-y.mean())**2)
    if variance<=0:raise ValueError('constant test target')
    return float(1-np.mean((y-prediction)**2)/variance)


def population_price_moments(pop,basis='baseline',horizon=HORIZON,history=HISTORY):
    """Centered covariance G and price cross-covariance c in lag-major order."""
    b=pop.block_size
    if history%b:raise ValueError('history not divisible by block size')
    length=history//b
    t=feature_map(pop.cap,basis)
    cov=population_covariances(pop,length,basis)
    weighted=np.einsum('ij,jst->ist',t,pop.weighted)
    end_weight=np.einsum('s,ist->ti',pop.stationary,weighted)
    mu=t@pop.mean
    expected,second=reward_moments(horizon,pop.cap)
    target_mean=pop.stationary@expected
    variance=pop.stationary@second-target_mean**2
    cross=[];future=expected.copy()
    for k in range(length+1):
        cross.append(end_weight.T@future-mu*target_mean)
        future=pop.transition@future
    blocks=np.block([[cov[j-i] if j>=i else cov[i-j].T for j in range(length+1)] for i in range(length+1)])
    oracle=float(np.sum(pop.stationary*(expected-target_mean)**2)/variance)
    return blocks,np.concatenate(cross),float(variance),oracle


def population_price_scores(pop,basis='baseline',horizon=HORIZON,history=HISTORY):
    """Population best linear predictors of the SAME future scalar price target."""
    blocks,cross,variance,oracle=population_price_moments(pop,basis,horizon,history)
    d=len(feature_map(pop.cap,basis))
    def explained(matrix,rhs):
        sd=np.sqrt(np.diag(matrix));g=matrix/sd[:,None]/sd[None,:];r=rhs/sd
        coef=np.linalg.pinv(g,rcond=1e-11,hermitian=True)@r
        return float(r@coef/variance)
    current=explained(blocks[:d,:d],cross[:d]);hist=explained(blocks,cross)
    return dict(r2_current=current,r2_history=hist,delta_r2=hist-current,r2_oracle=oracle,target_variance=float(variance))


def upper_rank_p(observed,reference):
    reference=np.asarray(reference,float)
    if reference.ndim!=1 or len(reference)<1 or not np.all(np.isfinite(reference)):
        raise ValueError('need finite independent reference statistics')
    return float((1+np.count_nonzero(reference>=observed))/(len(reference)+1))
