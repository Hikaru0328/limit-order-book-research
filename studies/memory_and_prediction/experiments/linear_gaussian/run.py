"""Reproduce population checks, seed ensembles, finite-sample errors and VAR controls.
Run from repo root: PYTHONPATH=src python studies/memory_and_prediction/experiments/linear_gaussian/run.py
"""
from pathlib import Path
import argparse, csv, json, platform, sys
import numpy as np
from lob_memory.linear import (simulate, aggregate, block_dynamics, population_covariances,
                               lifted_block_model, exact_mori)
from lob_memory.mori import from_covariances, covariance_sequence
from lob_memory.prediction import compare_var
from lob_memory.rg import memory_strength, characteristic_block_lag, logscale_beta

A=np.array([[.55,.25],[.10,.80]])
Q=np.array([[.7,.12],[.12,.4]])
SCALES=(1,2,4,8,16)
HISTORY=32  # distance from latest resolved block to oldest predictor, in events
TARGET=16  # fixed endpoint-to-endpoint target distance; block averaging window still varies


def run(output, seeds, samples):
    output.mkdir(parents=True,exist_ok=True)
    import matplotlib
    environment=dict(python=sys.version.split()[0],numpy=np.__version__,matplotlib=matplotlib.__version__,
        platform=platform.platform(),training_seeds=list(range(1000,1000+seeds)),
        test_seeds=list(range(2000,2000+seeds)),convergence_seeds=list(range(3000,3000+seeds)))
    (output/'run_environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    rows=[]; kernel_rows=[]; exact_error=0.
    populations={}
    for observation,c in [('full',np.eye(2)),('partial',np.array([[1.,0.]]))]:
        for mode in ('endpoint','mean'):
            for b in SCALES:
                lags=HISTORY//b
                cov=population_covariances(A,Q,c,b,lags+1,mode)
                omega=from_covariances(cov)
                if mode=='endpoint' or b==1:
                    ab,qb=block_dynamics(A,Q,b)
                else:
                    ab,qb=lifted_block_model(A,Q,c,b,mode)
                exact=exact_mori(ab,qb,len(c),lags)
                exact_error=max(exact_error,float(np.max(np.abs(exact-omega))))
                populations[observation,mode,b]=(omega,cov)
                for k,mat in enumerate(omega):
                    for i in range(len(c)):
                        for j in range(len(c)):
                            kernel_rows.append(dict(observation=observation,mode=mode,b=b,lag=k,
                                event_gap=k*b,target_distance=(k+1)*b,i=i,j=j,coefficient=mat[i,j]))
    for seed in range(seeds):
        train=simulate(A,Q,samples,seed=1000+seed)
        test=simulate(A,Q,samples//2,seed=2000+seed)
        for observation,d in [('full',2),('partial',1)]:
            for mode in ('endpoint','mean'):
                for b in SCALES:
                    yt=aggregate(train[:,:d],b,mode)
                    yv=aggregate(test[:,:d],b,mode)
                    lags=HISTORY//b
                    exact,cov=populations[observation,mode,b]
                    estimated=from_covariances(covariance_sequence(yt,lags+1))
                    for horizon_kind,h in [('one_block',1),('fixed_events',TARGET//b)]:
                        scores=compare_var(yt,yv,lags,h)
                        rows.append(dict(seed=seed,observation=observation,mode=mode,b=b,
                            history_lags=lags,history_events=lags*b,horizon_kind=horizon_kind,
                            target_events=h*b,n_train_blocks=len(yt),
                            population_memory=memory_strength(exact[1:]),
                            estimated_memory=memory_strength(estimated[1:]),
                            population_theta=characteristic_block_lag(exact[1:]),
                            kernel_rmse=float(np.sqrt(np.mean((estimated-exact)**2))),**scores))
    convergence=[]
    for seed in range(seeds):
        x=simulate(A,Q,samples,seed=3000+seed)[:,:1]
        truth=exact_mori(A,Q,1,16)
        for n in sorted(set((samples//16,samples//4,samples))):
            estimated=from_covariances(covariance_sequence(x[:n],17))
            convergence.append(dict(seed=seed,samples=n,kernel_rmse=float(np.sqrt(np.mean((estimated-truth)**2)))))
    for name,data in [('diagnostics.csv',rows),('population_kernels.csv',kernel_rows),('convergence.csv',convergence)]:
        with (output/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    # One-dimensional full-state AR(1): temporal averaging alone creates memory.
    ar={}
    for mode in ('endpoint','mean'):
        cov=population_covariances(np.array([[.8]]),np.array([[1.]]),np.array([[1.]]),4,9,mode)
        ar[mode]=memory_strength(from_covariances(cov)[1:])
    # Counterexample: identical M, theta, Omega0 and C0 at b=1, different b=2 flow.
    closure=[]
    for sign in (1,-1):
        a=np.array([[.3,.25],[.2,sign*.4]])
        q=np.eye(2)-a@a.T  # fixes stationary covariance to identity
        initial=exact_mori(a,q,1,64)
        ab,qb=block_dynamics(a,q,2)
        coarse=exact_mori(ab,qb,1,64)
        closure.append(dict(hidden_sign=sign,markov_b1=float(initial[0,0,0]),
            memory_b1=memory_strength(initial[1:]),theta_b1=characteristic_block_lag(initial[1:]),
            memory_b2=memory_strength(coarse[1:])))
    (output/'closure_counterexample.json').write_text(json.dumps(closure,indent=2)+'\n')
    flow=[]
    for obs in ('full','partial'):
        for mode in ('endpoint','mean'):
            strengths=np.array([memory_strength(populations[obs,mode,b][0][1:]) for b in SCALES])
            for i,beta in enumerate(logscale_beta(np.array(SCALES),strengths)):
                flow.append(dict(observation=obs,mode=mode,b_from=SCALES[i],b_to=SCALES[i+1],finite_step_slope=float(beta)))
    with (output/'finite_step_flow.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flow[0])); w.writeheader(); w.writerows(flow)
    report=dict(seed_count=seeds,training_events=samples,test_events=samples//2,
                scales=SCALES,history_events=HISTORY,fixed_target_events=TARGET,
                max_population_identity_error=exact_error,ar1_b4_memory=ar,
                convergence_mean_rmse={str(n):float(np.mean([r['kernel_rmse'] for r in convergence if r['samples']==n])) for n in sorted({r['samples'] for r in convergence})},
                notes=['Finite memory sums use the same 32-event predictor gap, not infinite memory.',
                       'Seed bands describe repeated simulations, not confidence intervals for market data.',
                       'One-block targets differ across scales; fixed-events holds endpoint separation only.',
                       'VAR coefficients are prediction baselines, not Mori coefficients.',
                       'Full-state means can have memory although full-state endpoints are Markov.'])
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    plot(output,rows,convergence)
    print(json.dumps(report,indent=2))


def plot(output,rows,convergence):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.4))
    for obs,mode,color in [('full','endpoint','#757575'),('partial','endpoint','#2166ac'),('partial','mean','#b2182b')]:
        means=[]; stds=[]; exact=[]; gains=[]
        for b in SCALES:
            r=[r for r in rows if r['observation']==obs and r['mode']==mode and r['b']==b and r['horizon_kind']=='one_block']
            means.append(np.mean([v['estimated_memory'] for v in r])); stds.append(np.std([v['estimated_memory'] for v in r]))
            exact.append(r[0]['population_memory'])
            gains.append(np.mean([v['delta_r2'] for v in r]))
        label=obs+' / '+mode
        axes[0].plot(SCALES,exact,'o-',color=color,label=label)
        axes[0].errorbar(SCALES,means,yerr=stds,fmt='x--',color=color,alpha=.65)
        axes[2].plot(SCALES,gains,'o-',color=color,label=label)
    ns=sorted({r['samples'] for r in convergence})
    rms=[np.mean([r['kernel_rmse'] for r in convergence if r['samples']==n]) for n in ns]
    axes[1].loglog(ns,rms,'o-',color='#2166ac')
    axes[1].set(xlabel='Training events',ylabel='Mean coefficient RMSE',title='Recovery of known kernel')
    axes[0].set(xlabel='Block size b',ylabel='Truncated memory strength',title='Solid: population; dashed: estimate')
    axes[2].set(xlabel='Block size b',ylabel='History R² − current-only R²',title='Independent test: one-block target')
    for ax in (axes[0],axes[2]): ax.set_xscale('log',base=2); ax.legend(fontsize=8)
    for ax in axes: ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output/'validation.svg'); fig.savefig(output/'validation.png', dpi=150); plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,default=Path(__file__).parent/'results')
    p.add_argument('--seeds',type=int,default=5); p.add_argument('--samples',type=int,default=65536)
    args=p.parse_args()
    if args.seeds < 1 or args.samples < 2048: p.error('require seeds >= 1 and samples >= 2048')
    run(args.output,args.seeds,args.samples)
