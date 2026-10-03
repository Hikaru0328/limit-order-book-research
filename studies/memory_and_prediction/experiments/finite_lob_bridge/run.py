"""Exact finite-LOB projection bridge; only publish outputs after verification."""
from pathlib import Path
import argparse
import csv
import json
import os
import tempfile
import numpy as np
from lob_memory.projection_bridge import (population_study,BASES,SCALES,NULL_TOL,
    IDENTITY_TOL,STATIONARY_TOL,TAIL_TOL,HORIZON_TOL)


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        w.writeheader();w.writerows(rows)


def plot(output,rows,lags):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#2166ac','#b2182b','#666666']
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    for basis,color in zip(BASES,colors):
        r=[v for v in lags if v['b']==1 and v['basis']==basis and v['lag']<=64]
        ax.plot([v['lag'] for v in r],[v['target_memory_kernel_norm'] for v in r],label=basis.title(),color=color)
    # symlog retains actual zeros and signed roundoff without inventing a floor.
    ax.set_yscale('symlog',linthresh=NULL_TOL)
    ax.axhline(NULL_TOL,color='#999999',ls=':',label='Numerical-null tolerance')
    ax.set(xlabel='Mori memory lag k (b=1)',ylabel='Common-target kernel Frobenius norm',
           title='Markov queue state; memory after incomplete projection')
    ax.text(.98,.97,'Exact population; first 64 lags shown',transform=ax.transAxes,ha='right',va='top',fontsize=9)
    ax.legend(loc='upper right',bbox_to_anchor=(1,.89),fontsize=9);ax.grid(alpha=.2)
    for ext in ('svg','png'):fig.savefig(output/f'projection_memory.{ext}',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for basis,color in zip(BASES,colors):
        r=[v for v in rows if v['basis']==basis]
        axes[0].plot([v['b'] for v in r],[v['memory_strength'] for v in r],'o-',label=basis.title(),color=color)
        valid=[v for v in r if v['horizon_resolved']]
        if valid:axes[1].plot([v['b'] for v in valid],[v['characteristic_event_horizon'] for v in valid],'o-',label=basis.title(),color=color)
    axes[0].set(ylabel='Total common-target memory strength',title='Population memory, converged lag sums')
    axes[1].set(ylabel='Characteristic event horizon b × theta',title='Horizons only for resolved memory')
    axes[1].text(.02,.97,'Complete basis: numerical null;\nhorizon not interpretable',transform=axes[1].transAxes,va='top',fontsize=9)
    for ax in axes:
        ax.set_xscale('log',base=2);ax.set_xticks(sorted({r['b'] for r in rows}))
        ax.set_xticklabels([str(x) for x in sorted({r['b'] for r in rows})]);ax.set_xlabel('Block size b');ax.grid(alpha=.2);ax.legend(loc='best')
    for ext in ('svg','png'):fig.savefig(output/f'memory_flow.{ext}',dpi=160)
    plt.close(fig)


def run(output,quick=False):
    scales=(1,) if quick else SCALES
    rows,lags,controls=population_study(scales)
    # Repeat primary population result with a changed global RNG state. It must be identical.
    state=np.random.get_state()
    try:
        np.random.seed(101);a=population_study((1,))[0]
        np.random.seed(907);b=population_study((1,))[0]
    finally:np.random.set_state(state)
    seed_independent=a==b
    if not seed_independent:raise ArithmeticError('Population result depends on RNG seed')
    primary=[r for r in rows if r['b']==1]
    strength={r['basis']:r['memory_strength'] for r in primary}
    report=dict(status='passed',model='existing finite queue chain; no hidden state',cap=8,
        states=64,scales=list(scales),population_only=True,seed_independence_identical=seed_independent,
        null_tolerance=NULL_TOL,identity_tolerance=IDENTITY_TOL,stationary_tolerance=STATIONARY_TOL,
        memory_tail_tolerance=TAIL_TOL,event_horizon_interval_tolerance=HORIZON_TOL,
        controls=controls,baseline_nonzero=strength['baseline']>NULL_TOL,
        ordering_supported_b1=strength['baseline']>=strength['polynomial']>=strength['complete'],
        ordering_by_scale={str(s):all(x>=y for x,y in zip([r['memory_strength'] for r in rows if r['b']==s],[r['memory_strength'] for r in rows if r['b']==s][1:])) for s in scales},
        complete_max_lag_norm=max(r['maximum_lag_norm'] for r in rows if r['basis']=='complete'),
        complete_max_covariance_recursion_sum_32=max(r['covariance_recursion_memory_sum_32'] for r in rows if r['basis']=='complete'),
        complete_max_strength_upper_bound=max(r['memory_strength']+r['tail_bound'] for r in rows if r['basis']=='complete'),
        max_direct_recursion_error=max(r['direct_event_recursion_max_error'] for r in rows),
        max_factor_recursion_error=max(r['factor_recursion_max_error'] for r in rows),
        python_float='float64',numpy=np.__version__,
        caveats=['Geometric tail bounds exclude roundoff.',
                 'Null-basis horizon ratios remain in CSV but are flagged unresolved; they are not plotted.',
                 'Feature expansion need not generally reduce this metric.',
                 'No sampled coefficients or VAR fits enter headline figures.'])
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    # Compute and verify everything before changing the output directory.
    with tempfile.TemporaryDirectory(dir=output.parent) as folder:
        stage=Path(folder)
        write_csv(stage/'population_lags.csv',lags);write_csv(stage/'summary.csv',rows)
        (stage/'verification.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        plot(stage,rows,lags)
        output.mkdir(exist_ok=True)
        for p in stage.iterdir():os.replace(p,output/p.name)
    print(json.dumps(dict(primary=primary,verification=report),indent=2,allow_nan=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick',action='store_true',help='b=1 only; same cap and tolerances')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.quick and args.output is None:parser.error('--quick requires a separate --output directory')
    run(args.output or Path(__file__).parent/'results',args.quick)
