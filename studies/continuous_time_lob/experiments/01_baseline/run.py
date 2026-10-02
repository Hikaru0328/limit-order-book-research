"""Prespecified one-latent-state clock and recoverability experiment.

Run from this study root. Results are Monte Carlo averages of exact causal
filters, NOT population-exact recoverability or learned-parameter results.
"""
import argparse
import csv
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')

import numpy as np
import scipy
from scipy.linalg import expm
from scipy.stats import t as student_t
from ct_lob.model import Model,Parameters,simulate
from ct_lob.filtering import Survival,observe,queue_filter,price_filter,snapshot_condition,normalize


def write_csv(path, rows):
    if not rows:raise ValueError('empty table')
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def entropy(p):
    p=np.clip(p,1e-15,1-1e-15)
    return -(p*np.log2(p)+(1-p)*np.log2(1-p))


def summarize(rows,keys,metrics):
    groups={}
    for r in rows:groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
    result=[]
    for key,group in groups.items():
        row=dict(zip(keys,key));row['replicas']=len(group)
        for metric in metrics:
            values=[r[metric] for r in group if r[metric] is not None]
            if not values:
                for suffix in ('mean','se','low95','high95'):row[metric+'_'+suffix]=None
                continue
            x=np.array(values);mean=x.mean();se=x.std(ddof=1)/np.sqrt(len(x))
            radius=student_t.ppf(.975,len(x)-1)*se
            for suffix,value in zip(('mean','se','low95','high95'),(mean,se,mean-radius,mean+radius)):
                row[metric+'_'+suffix]=float(value)
        result.append(row)
    return result


def run(output,replicas,duration):
    if replicas<2 or duration<=0:raise ValueError('at least two independent paths and positive duration required')
    output.mkdir(parents=True,exist_ok=True)
    base=Parameters();horizons=(1.,4.,16.);betas=(0.,.5,1.);kappas=(.02,.2,2.)
    config=dict(parameters=asdict(base),
                betas=betas,kappas=kappas,horizons=horizons,replicas=replicas,duration=duration,
                observation_interval=2.,burn_in_rule='max(50, 10/(2*kappa))',
                seed_rule='202610010000 + 100*case_index + replica',
                history='from stationary t=0; burn-in excluded from scoring, finite past retained',
                learning='none; all parameters known',numpy=np.__version__,scipy=scipy.__version__,python=platform.python_version())
    (output/'configuration.json').write_text(json.dumps(config,indent=2))
    population=[];clocks=[];filters=[];predictions=[];paired=[];numerics=[];seeds=[]
    start=time.monotonic()
    for case,(beta,kappa) in enumerate((b,k) for b in betas for k in kappas):
        m=Model(replace(base,beta=beta,kappa=kappa));pi=m.stationary
        queue_pi=pi.reshape(-1,2);queue_mass=queue_pi.sum(1);queue_post=queue_pi/queue_mass[:,None]
        killed=m.generator-m.price_jumps[1]-m.price_jumps[-1];engine=Survival(killed)
        rng=np.random.default_rng(7000+case);expm_errors=[]
        for dt in (.001,.1,1.,4.):
            prior=rng.dirichlet(np.ones(m.n))
            expm_errors.append(float(np.max(abs(engine.step(prior,dt)-normalize(prior@expm(dt*killed))))))
        moment={}
        for h in horizons:
            g,second=m.price_moments(h);mean=pi@g;var=pi@second-mean*mean
            gq=np.sum(queue_post*g.reshape(-1,2),axis=1)
            oracle=float(pi@((g-mean)**2)/var);current=float(queue_mass@((gq-mean)**2)/var)
            moment[h]=(g,second,var,oracle,current,gq)
            population.append(dict(beta=beta,kappa=kappa,horizon=h,variance=float(var),oracle_r2=oracle,
                current_queue_r2=current,hidden_oracle_gap=oracle-current,
                queue_entropy_bits=float(queue_mass@entropy(queue_post[:,1])),
                mean_queue=float(pi@(m.states[:,:2].mean(1))),
                cap_side_fraction=float(pi@((m.states[:,:2]==base.cap).mean(1))),
                price_move_rate=float(pi@((m.price_jumps[-1]+m.price_jumps[1])@np.ones(m.n)))))
        for scale in (.25,1.,4.):
            slow=Model(m.parameters.slowed(scale))
            for h in horizons:
                g,h2,*_=moment[h];sg,sh2=slow.price_moments(h*scale)
                clocks.append(dict(beta=beta,kappa=kappa,scale=scale,horizon=h,
                    generator_max_error=float(abs(slow.generator-m.generator/scale).max()),
                    stationary_max_error=float(abs(slow.stationary-pi).max()),
                    transition_max_error=float(abs(slow.transition(h*scale)-m.transition(h)).max()),
                    price_mean_max_error=float(abs(sg-g).max()),price_second_max_error=float(abs(sh2-h2).max())))
        burn=max(50.,10/(2*kappa));times=burn+np.arange(0,duration,2.)
        for replica in range(replicas):
            seed=202610010000+100*case+replica;path=simulate(m,burn+duration+max(horizons),seed)
            indexes=path.states_at(times);queues=m.states[indexes,:2];qi=indexes//2;truth=m.states[indexes,2]==1
            pb=price_filter(m,observe(m,path,'price'),times,engine)
            beliefs=dict(current_queue=queue_post[qi],price_history_current_queue=snapshot_condition(m,pb,queues),
                         queue_price_history=queue_filter(m,observe(m,path,'queue_price'),times),
                         marked_history=queue_filter(m,observe(m,path,'marked'),times))
            prob={k:v[:,1] for k,v in beliefs.items()};prob['price_only']=pb[:,1::2].sum(1)
            seeds.append(dict(beta=beta,kappa=kappa,replica=replica,seed=seed,events=len(path.times),
                              book_events=int(np.sum(path.event<6)),flips=int(np.sum(path.event==6)),
                              price_moves=int(np.sum(path.reward!=0)),query_count=len(times),end=path.end))
            for arm,p in prob.items():
                p=np.clip(p,1e-15,1-1e-15);ent=float(entropy(p).mean())
                cross=float(-np.where(truth,np.log2(p),np.log2(1-p)).mean())
                filters.append(dict(beta=beta,kappa=kappa,replica=replica,arm=arm,entropy_bits=ent,
                    mutual_information_bits=1-ent,brier=float(((p-truth)**2).mean()),
                    posterior_brier=float((p*(1-p)).mean()),cross_entropy_bits=cross,
                    calibration_entropy_residual=cross-ent,bayes_accuracy=float(np.maximum(p,1-p).mean()),
                    empirical_accuracy=float(((p>=.5)==truth).mean())))
            for h in horizons:
                g,second,var,oracle,current,gq=moment[h];target=path.prices_at(times+h)-path.prices_at(times)
                exact=g[indexes];state_values=g.reshape(-1,2)[qi]
                fitted={a:np.sum(v*state_values,axis=1) for a,v in beliefs.items()};fitted['price_only']=pb@g
                errors={a:float(np.mean((f-exact)**2)) for a,f in fitted.items()}
                for arm,pred in fitted.items():
                    direct_gap=errors[arm]/var;direct_r2=oracle-direct_gap;den=oracle-current
                    # Conditional-expectation projection identity uses the same
                    # observed Q baseline and avoids subtracting two noisy risks.
                    if arm=='price_only':r2=float(np.mean((pred-pi@g)**2)/var)
                    else:r2=current+float(np.mean((pred-gq[qi])**2)/var)
                    oracle_gap=oracle-r2
                    eta=(r2-current)/den if arm!='price_only' and den>1e-12 else None
                    real=float(1-np.mean((target-pred)**2)/var)
                    real_oracle=float(1-np.mean((target-exact)**2)/var)
                    predictions.append(dict(beta=beta,kappa=kappa,horizon=h,replica=replica,arm=arm,
                        oracle_gap=oracle_gap,conditional_r2=r2,hidden_recovery=eta,
                        direct_risk_r2=direct_r2,projection_identity_residual=direct_r2-r2,
                        realized_r2=real,risk_identity_residual=(real_oracle-real)-direct_gap))
                for low,high in [('current_queue','price_history_current_queue'),('price_history_current_queue','queue_price_history'),('queue_price_history','marked_history')]:
                    paired.append(dict(beta=beta,kappa=kappa,horizon=h,replica=replica,comparison=high+' - '+low,
                                       r2_gain=(errors[low]-errors[high])/var))
            if beta==.5 and kappa==.2 and replica==0:
                write_csv(output/'example_posteriors.csv',[dict(time=float(t),bid=int(q[0]),ask=int(q[1]),
                    hidden_direction=int(m.states[i,2]),**{a:float(v[j]) for a,v in prob.items()})
                    for j,(t,q,i) in enumerate(zip(times,queues,indexes))])
        numerics.append(dict(beta=beta,kappa=kappa,spectral_condition=engine.condition,
            spectral_enabled=engine.spectral,expm_max_error=max(expm_errors),fallbacks=engine.fallbacks,
            stationarity_error=float(abs(pi@m.generator).max())))
        print(f'completed beta={beta:g}, kappa={kappa:g}: {replicas} independent paths, elapsed {time.monotonic()-start:.1f}s',flush=True)
        for name,table in [('population',population),('clock_controls',clocks),('filter_runs',filters),('prediction_runs',predictions),('paired_runs',paired),('numerics',numerics),('seeds',seeds)]:write_csv(output/(name+'.csv'),table)
    write_csv(output/'filter_summary.csv',summarize(filters,['beta','kappa','arm'],['entropy_bits','mutual_information_bits','brier','posterior_brier','calibration_entropy_residual','bayes_accuracy']))
    write_csv(output/'prediction_summary.csv',summarize(predictions,['beta','kappa','horizon','arm'],['oracle_gap','conditional_r2','hidden_recovery','direct_risk_r2','projection_identity_residual','realized_r2','risk_identity_residual']))
    write_csv(output/'paired_summary.csv',summarize(paired,['beta','kappa','horizon','comparison'],['r2_gain']))
    print(f'finished: {len(seeds)} paths, {len(predictions)} prediction rows',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path(__file__).parent/'results')
    parser.add_argument('--replicas',type=int,default=16);parser.add_argument('--duration',type=float,default=1024.)
    args=parser.parse_args();run(args.output,args.replicas,args.duration)
