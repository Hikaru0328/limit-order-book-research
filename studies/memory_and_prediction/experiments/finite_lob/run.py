"""Exact finite-LOB population vs sample kernel and projection-basis controls."""
from pathlib import Path
import argparse,csv,json,platform,sys
import numpy as np
from lob_memory.finite_lob import (event_model,block_population,population_covariances,
                                  basis_observations,target_kernel,direct_event_mori)
from lob_memory.toy_lob import simulate_lob,aggregate_records
from lob_memory.mori import from_covariances,covariance_sequence
from lob_memory.rg import memory_strength

BASES=('baseline','polynomial','complete')
SCALES=(1,2,4,8,16)
HISTORY=32


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        w.writeheader();w.writerows(rows)


def run(output,seeds=5,samples=131072,cap=8):
    output.mkdir(parents=True,exist_ok=True)
    populations={};truth_rows=[];lag_rows=[]
    for b in SCALES:
        pop=block_population(cap,b)
        for basis in BASES:
            cov=population_covariances(pop,HISTORY//b+1,basis)
            kernel=from_covariances(cov)
            target=target_kernel(kernel,cov[0],cap,basis)
            populations[b,basis]=(cov,kernel,target)
            truth_rows.append(dict(b=b,basis=basis,features=len(cov[0]),memory_lags=HISTORY//b,
                target_memory=memory_strength(target[1:]),covariance_condition=float(np.linalg.cond(cov[0]))))
            for k in range(1,len(target)):
                lag_rows.append(dict(b=b,basis=basis,lag=k,event_gap=k*b,target_distance=(k+1)*b,
                    target_coefficient_norm=float(np.linalg.norm(target[k]))))
    write_csv(output/'population.csv',truth_rows)
    write_csv(output/'population_lags.csv',lag_rows)
    rows=[];sizes=sorted(set((samples//16,samples//4,samples)))
    for seed in range(seeds):
        raw=simulate_lob(samples,seed=7000+seed,cap=cap,burn=0,stationary_start=True)
        for n in sizes:
            for b in SCALES:
                records=aggregate_records({k:v[:n] for k,v in raw.items()},b)
                for basis in BASES:
                    y=basis_observations(records,cap,basis)
                    c= covariance_sequence(y,HISTORY//b+1)
                    cov,truth,target=populations[b,basis]
                    try:
                        estimate=from_covariances(c)
                        weighted=target_kernel(estimate,cov[0],cap,basis)
                        sample_weighted=target_kernel(estimate,c[0],cap,basis)
                        error=weighted[1:]-target[1:]
                        rows.append(dict(seed=seed,events=n,b=b,basis=basis,blocks=len(y),status='ok',
                            population_strength=memory_strength(target[1:]),
                            estimate_population_metric=memory_strength(weighted[1:]),
                            estimate_sample_metric=memory_strength(sample_weighted[1:]),
                            coefficient_rmse=float(np.sqrt(np.mean(error**2))),
                            error_norm_sum=memory_strength(error),sample_covariance_condition=float(np.linalg.cond(c[0]))))
                    except ValueError as e:
                        rows.append(dict(seed=seed,events=n,b=b,basis=basis,blocks=len(y),status=str(e),
                            population_strength=memory_strength(target[1:]),estimate_population_metric='',
                            estimate_sample_metric='',coefficient_rmse='',error_norm_sum='',sample_covariance_condition=str(np.linalg.cond(c[0]))))
        print(f'Completed seed {seed+1}/{seeds}',flush=True)
    write_csv(output/'sampling.csv',rows)
    summary=[]
    for n in sizes:
        for b in SCALES:
            for basis in BASES:
                selected=[r for r in rows if r['events']==n and r['b']==b and r['basis']==basis and r['status']=='ok']
                values=np.array([r['estimate_population_metric'] for r in selected])
                summary.append(dict(events=n,b=b,basis=basis,valid_seeds=len(selected),
                    population_strength=memory_strength(populations[b,basis][2][1:]),
                    estimate_mean=float(values.mean()) if len(values) else None,
                    estimate_sd=float(values.std()) if len(values) else None,
                    coefficient_rmse_mean=float(np.mean([r['coefficient_rmse'] for r in selected])) if selected else None))
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    errors={basis:float(np.max(np.abs(populations[1,basis][1]-direct_event_mori(cap,basis,HISTORY)))) for basis in BASES}
    failed=[r for r in rows if r['status']!='ok']
    verification=dict(direct_projection_max_coefficient_error=errors,total_sampling_cases=len(rows),
                      failed_sampling_cases=len(failed),failures=failed)
    (output/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    metadata=dict(cap=cap,states=cap*cap,scales=SCALES,history_gap_events=HISTORY,sample_sizes=sizes,
        seeds=list(range(7000,7000+seeds)),stationary_start=True,burn=0,
        bases=dict(baseline=['I','D','R'],polynomial=['I','D','R','qb/cap','qa/cap','(qb/cap)^2','(qa/cap)^2','qb*qa/cap^2'],complete='all but one endpoint indicator, D, R'),
        metric='sum ||C_target^(-1/2) T Omega[k] C_features^(1/2)||_F, k=1..32/b',
        metric_covariance='population for controlled error comparisons; sample alternative also exported',
        python=sys.version.split()[0],numpy=np.__version__,platform=platform.platform(),
        limitations=['Five-seed spread is not a calibrated significance threshold.',
                    'Nested sample prefixes are correlated across sample sizes.',
                    'The complete basis is an exact null control, not a recommended small-sample model.',
                    'No market calibration, alpha evaluation or long-memory claim.'])
    (output/'configuration.json').write_text(json.dumps(metadata,indent=2)+'\n')
    plot(output,truth_rows,summary,sizes)
    print(json.dumps([r for r in summary if r['b']==1],indent=2))


def plot(output,truth,summary,sizes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.3))
    colors=dict(baseline='#2166ac',polynomial='#b2182b',complete='#666666')
    labels=dict(baseline='Baseline (3)',polynomial='Polynomial (8)',complete='Complete (65)')
    for basis in BASES:
        color=colors[basis]
        tr=[r for r in truth if r['basis']==basis]
        axes[0].plot([r['b'] for r in tr],[r['target_memory'] for r in tr],'o-',color=color,label=labels[basis])
        sr=[r for r in summary if r['b']==1 and r['basis']==basis]
        axes[1].loglog([r['events'] for r in sr],[r['coefficient_rmse_mean'] for r in sr],'o-',color=color,label=labels[basis])
        axes[2].errorbar([r['events'] for r in sr],[r['estimate_mean'] for r in sr],
                         yerr=[r['estimate_sd'] for r in sr],fmt='o--',color=color)
        axes[2].axhline(tr[0]['target_memory'],color=color,label=labels[basis])
    axes[0].set(xlabel='Block size b',ylabel='Common-target memory strength',title='Population: no sampling error')
    axes[0].set_xscale('log',base=2)
    axes[1].set(xlabel='Training events',ylabel='Target coefficient RMSE',title='Finite-sample recovery at b=1')
    axes[2].set(xlabel='Training events',ylabel='Common-target memory strength',title='Solid: exact; dashed: estimate (b=1)')
    axes[2].set_xscale('log',base=2)
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(output/'finite_lob.svg');plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'results')
    parser.add_argument('--seeds',type=int,default=5)
    parser.add_argument('--samples',type=int,default=131072)
    args=parser.parse_args()
    if args.seeds<1 or args.samples<131072: parser.error('require seeds>=1 and samples>=131072 for the complete-basis control')
    run(args.output,args.seeds,args.samples)
