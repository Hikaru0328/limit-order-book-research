"""Fixed H=32 price-only forecasts plus matched Markov and oracle controls.

The five observed runs and 99 reference runs are all synthetic Markov series.
Only observed run zero is used for primary Monte Carlo ranks; other observed
runs describe replication. No persistent order-flow mechanism is introduced.
"""
from pathlib import Path
import argparse,csv,json,platform,sys,time
import numpy as np
from lob_memory.price_prediction import (SCALES,HORIZON,HISTORY,RIDGE_GRID,anchors,
    price_design,oracle_prediction,select_ridge,r2,population_price_scores,upper_rank_p)
from lob_memory.toy_lob import simulate_lob
from lob_memory.finite_lob import block_population

BASES=('baseline','polynomial')
EVENTS=(65536,32768,65536)


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def one_run(cohort,rep,events=EVENTS):
    root=(10000 if cohort=='observed' else 1000000)+3*rep
    records=[simulate_lob(n,seed=root+i,burn=0,stationary_start=True) for i,n in enumerate(events)]
    times=[anchors(n) for n in events]
    oracle=[oracle_prediction(r,t) for r,t in zip(records,times)]
    rows=[]
    for basis in BASES:
        d=3 if basis=='baseline' else 8
        for b in SCALES:
            designs=[price_design(r,b,basis,t=t) for r,t in zip(records,times)]
            (xt,yt,_),(xv,yv,_),(xs,ys,_)=designs
            current=select_ridge(xt[:,:d],yt,xv[:,:d],yv)[0]
            history,correction=select_ridge(xt,np.column_stack((yt,yt-oracle[0])),
                xv,np.column_stack((yv,yv-oracle[1])),zero_targets=(1,))
            rc=r2(ys,current.predict(xs[:,:d]));rh=r2(ys,history.predict(xs));ro=r2(ys,oracle[2])
            corrected=r2(ys,oracle[2]+correction.predict(xs))
            rows.append(dict(cohort=cohort,replicate=rep,basis=basis,b=b,horizon_events=HORIZON,
                history_gap_events=HISTORY,oldest_feature_support_events=HISTORY+b,
                train_seed=root,validation_seed=root+1,test_seed=root+2,
                train_targets=len(yt),validation_targets=len(yv),test_targets=len(ys),
                current_lambda=current.regularization,history_lambda=history.regularization,
                oracle_correction_lambda=correction.regularization,
                current_validation_mse=current.validation_mse,history_validation_mse=history.validation_mse,
                r2_current=rc,r2_history=rh,delta_r2=rh-rc,r2_oracle=ro,
                oracle_residual_history_gain=corrected-ro))
    return rows


def summarize(rows,population,null_runs):
    summary=[]
    for basis in BASES:
        for b in SCALES:
            obs=[r for r in rows if r['cohort']=='observed' and r['basis']==basis and r['b']==b]
            ref=[r for r in rows if r['cohort']=='markov_reference' and r['basis']==basis and r['b']==b]
            null=np.array([r['delta_r2'] for r in ref]);og=np.array([r['oracle_residual_history_gain'] for r in ref])
            primary=next(r for r in obs if r['replicate']==0)
            truth=next(r for r in population if r['basis']==basis and r['b']==b)
            gains=np.array([r['delta_r2'] for r in obs])
            summary.append(dict(basis=basis,b=b,population_current_r2=truth['r2_current'],
                population_history_r2=truth['r2_history'],population_delta_r2=truth['delta_r2'],
                primary_current_r2=primary['r2_current'],primary_history_r2=primary['r2_history'],
                primary_delta_r2=primary['delta_r2'],observed_mean_delta=float(gains.mean()),observed_sd_delta=float(gains.std()),
                markov_reference_mean=float(null.mean()),markov_reference_q05=float(np.quantile(null,.05)),
                markov_reference_q95=float(np.quantile(null,.95)),
                upper_rank_p_against_markov=upper_rank_p(primary['delta_r2'],null),
                oracle_primary_gain=primary['oracle_residual_history_gain'],oracle_reference_mean_gain=float(og.mean()),
                oracle_reference_q05=float(np.quantile(og,.05)),oracle_reference_q95=float(np.quantile(og,.95)),
                oracle_upper_rank_p=upper_rank_p(primary['oracle_residual_history_gain'],og)))
    primary_rows=[r for r in rows if r['cohort']=='observed' and r['replicate']==0]
    best=max(primary_rows,key=lambda r:r['delta_r2'])
    reference_max=[max(r['delta_r2'] for r in rows if r['cohort']=='markov_reference' and r['replicate']==i) for i in range(null_runs)]
    oracle_max=[max(r['oracle_residual_history_gain'] for r in rows if r['cohort']=='markov_reference' and r['replicate']==i) for i in range(null_runs)]
    primary_oracle_max=max(r['oracle_residual_history_gain'] for r in primary_rows)
    global_result=dict(primary_run=0,search_cells=len(BASES)*len(SCALES),
        selected_basis=best['basis'],selected_b=best['b'],primary_max_delta_r2=best['delta_r2'],
        markov_max_q95=float(np.quantile(reference_max,.95)),
        family_upper_rank_p=upper_rank_p(best['delta_r2'],reference_max),
        primary_max_oracle_gain=primary_oracle_max,
        oracle_family_upper_rank_p=upper_rank_p(primary_oracle_max,oracle_max),
        rank_resolution=1/(null_runs+1),
        interpretation='Ranks test compatibility with this specified Markov data-generating model; they do not test absence of reduced-state history information.')
    return summary,global_result


def plot(output,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.4))
    for basis,color in [('baseline','#2166ac'),('polynomial','#b2182b')]:
        r=[r for r in summary if r['basis']==basis]
        b=[x['b'] for x in r]
        axes[0].plot(b,[x['population_delta_r2'] for x in r],'o-',color=color,label=basis)
        axes[1].fill_between(b,[x['markov_reference_q05'] for x in r],[x['markov_reference_q95'] for x in r],color=color,alpha=.15)
        axes[1].plot(b,[x['primary_delta_r2'] for x in r],'o-',color=color,label=basis)
        axes[1].plot(b,[x['markov_reference_mean'] for x in r],'--',color=color)
        axes[2].fill_between(b,[x['oracle_reference_q05'] for x in r],[x['oracle_reference_q95'] for x in r],color=color,alpha=.15)
        axes[2].plot(b,[x['oracle_primary_gain'] for x in r],'o-',color=color,label=basis)
    axes[0].set(title='Exact population: price history gain',ylabel='Delta R² (H=32 events)')
    axes[1].set(title='Primary run vs matched Markov reference',ylabel='History minus current-only R²')
    axes[2].set(title='History correction beyond full-state oracle',ylabel='Corrected minus oracle R²')
    for ax in axes:
        ax.set_xscale('log',base=2);ax.set_xlabel('Block size b');ax.axhline(0,color='black',lw=.7);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(output/'price_prediction.svg');plt.close(fig)


def run(output,observed_runs=5,null_runs=99):
    output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    metadata=dict(horizon_events=HORIZON,history_gap_events=HISTORY,scales=SCALES,bases=BASES,
        events=dict(zip(('train','validation','test'),EVENTS)),common_anchor_stride=16,first_anchor=48,
        ridge_objective='train mean squared error + lambda times squared coefficients after training-only standardization',
        ridge_grid=RIDGE_GRID,additional_candidate='training intercept only',oracle_extra_candidate='zero correction',
        observed_runs=observed_runs,primary_run=0,reference_runs=null_runs,observed_seed_base=10000,reference_seed_base=1000000,
        seed_rule='base + 3*replicate + split_index (train=0, validation=1, test=2)',
        stationary_start=True,validation_only_selection=True,refit_train_validation=False,
        complete_basis='control only: exact full-state conditional-mean oracle, never fit 65-feature forecasts',
        reference='same Markov model and entire train/validation/test procedure; can contain genuine reduced-basis history gain',
        uncertainty='independent trajectory replicates; 5%-95% reference quantiles are predictive envelopes, not confidence intervals',
        multiplicity='primary maximum over both bases and all five scales compared with replicate-wise reference maxima',
        python=sys.version.split()[0],numpy=np.__version__,platform=platform.platform())
    (output/'configuration.json').write_text(json.dumps(metadata,indent=2)+'\n')
    population=[]
    for b in SCALES:
        pop=block_population(8,b)
        for basis in BASES:
            population.append(dict(b=b,basis=basis,**population_price_scores(pop,basis)))
    write_csv(output/'population.csv',population)
    rows=[]
    for cohort,count in [('observed',observed_runs),('markov_reference',null_runs)]:
        for i in range(count):
            rows.extend(one_run(cohort,i))
            if (i+1)%10==0 or i+1==count:
                print(f'{cohort}: {i+1}/{count}; elapsed {time.monotonic()-started:.1f}s',flush=True)
                write_csv(output/'runs.csv',rows)
    summary,global_result=summarize(rows,population,null_runs)
    write_csv(output/'summary.csv',summary)
    (output/'global_calibration.json').write_text(json.dumps(global_result,indent=2)+'\n')
    plot(output,summary)
    print(json.dumps(global_result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path(__file__).parent/'results')
    p.add_argument('--observed-runs',type=int,default=5);p.add_argument('--null-runs',type=int,default=99)
    a=p.parse_args()
    if a.observed_runs<1 or a.null_runs<19:p.error('need >=1 observed and >=19 independent reference runs')
    run(a.output,a.observed_runs,a.null_runs)
