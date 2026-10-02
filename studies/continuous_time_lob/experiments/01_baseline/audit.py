"""Independent saved-result audit and seeded filter/risk replay."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from dataclasses import replace
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')

import numpy as np
from scipy.stats import t as student_t
from ct_lob.model import Model,Parameters,simulate
from ct_lob.filtering import observe,price_filter,queue_filter,snapshot_condition,Survival,normalize

import argparse
REPO = Path(__file__).resolve().parents[4]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=REPO/'results/synthetic/continuous_time')
ROOT = parser.parse_args().output

def read(name):
    with (ROOT/(name+'.csv')).open() as f:return list(csv.DictReader(f))


def main():
    config=json.loads((ROOT/'configuration.json').read_text());reps=config['replicas']
    pop=read('population');runs=read('prediction_runs');fruns=read('filter_runs');seeds=read('seeds')
    assert len(pop)==27 and len(runs)==9*reps*15 and len(fruns)==9*reps*5
    assert len({r['seed'] for r in seeds})==9*reps
    assert all(int(r['query_count'])==int(np.ceil(config['duration']/2)) for r in seeds)
    summary_error=0.
    for stem,keys,metrics in [
        ('filter',['beta','kappa','arm'],['entropy_bits','mutual_information_bits','brier','posterior_brier','calibration_entropy_residual','bayes_accuracy']),
        ('prediction',['beta','kappa','horizon','arm'],['oracle_gap','conditional_r2','hidden_recovery','direct_risk_r2','projection_identity_residual','realized_r2','risk_identity_residual']),
        ('paired',['beta','kappa','horizon','comparison'],['r2_gain'])]:
        source=read(stem+'_runs');summary=read(stem+'_summary')
        for s in summary:
            group=[r for r in source if all(r[k]==s[k] for k in keys)];assert len(group)==reps
            for key in metrics:
                values=[float(r[key]) for r in group if r[key]!='']
                if not values:
                    assert s[key+'_mean']=='';continue
                a=np.array(values);mean=a.mean();se=a.std(ddof=1)/np.sqrt(reps)
                expected=[mean,se,mean-student_t.ppf(.975,reps-1)*se,mean+student_t.ppf(.975,reps-1)*se]
                observed=[float(s[key+'_'+suffix]) for suffix in ('mean','se','low95','high95')]
                summary_error=max(summary_error,float(np.max(abs(np.array(expected)-observed))))
    assert summary_error<1e-12
    clock_errors={key:max(float(r[key]) for r in read('clock_controls')) for key in read('clock_controls')[0] if key.endswith('_error')}
    assert max(clock_errors.values())<2e-10
    expm_error=max(float(r['expm_max_error']) for r in read('numerics'));assert expm_error<1e-10
    null=max(abs(float(r['entropy_bits'])-1) for r in fruns if float(r['beta'])==0);assert null<1e-10
    assert all(r['hidden_recovery']=='' for r in runs if float(r['beta'])==0 or r['arm']=='price_only')
    assert all(float(r['hidden_recovery'])==0 for r in runs if float(r['beta'])>0 and r['arm']=='current_queue')
    # Posterior calibration and risk equalities are stochastic diagnostics, not
    # assertions that finite-sample differences must vanish identically.
    def max_z(rows,metric):
        values=[]
        for r in rows:
            se=float(r[metric+'_se']);mean=float(r[metric+'_mean'])
            values.append(abs(mean)/se if se>1e-12 else 0. if abs(mean)<1e-10 else float('inf'))
        return max(values)
    diagnostics=dict(entropy_calibration_max_se=max_z(read('filter_summary'),'calibration_entropy_residual'),
                     projection_identity_max_se=max_z(read('prediction_summary'),'projection_identity_residual'),
                     realized_risk_identity_max_se=max_z(read('prediction_summary'),'risk_identity_residual'))
    replay_errors=[];sensitivity=[]
    for beta,kappa in ((.5,.2),(1.,.02)):
        m=Model(replace(Parameters(),beta=beta,kappa=kappa))
        seedrow=next(r for r in seeds if float(r['beta'])==beta and float(r['kappa'])==kappa and r['replica']=='0')
        path=simulate(m,float(seedrow['end']),int(seedrow['seed']));burn=max(50.,10/(2*kappa))
        times=burn+np.arange(0,config['duration'],2.);ix=path.states_at(times);q=m.states[ix,:2];qi=ix//2
        piq=m.stationary.reshape(-1,2);piq=piq/piq.sum(1)[:,None]
        price_obs=observe(m,path,'price');engine=Survival(m.generator-m.price_jumps[1]-m.price_jumps[-1])
        pb=price_filter(m,price_obs,times,engine)
        beliefs={'current_queue':piq[qi],'price_history_current_queue':snapshot_condition(m,pb,q),
                 'queue_price_history':queue_filter(m,observe(m,path,'queue_price'),times),
                 'marked_history':queue_filter(m,observe(m,path,'marked'),times)}
        for horizon in (1.,4.,16.):
            g,h=m.price_moments(horizon);var=m.stationary@h-(m.stationary@g)**2
            base_row=next(r for r in pop if float(r['beta'])==beta and float(r['kappa'])==kappa and float(r['horizon'])==horizon)
            gq=np.sum(piq*g.reshape(-1,2),axis=1)
            preds={a:np.sum(p*g.reshape(-1,2)[qi],axis=1) for a,p in beliefs.items()};preds['price_only']=pb@g
            for arm,pred in preds.items():
                r2=float(np.mean((pred-m.stationary@g)**2)/var) if arm=='price_only' else float(base_row['current_queue_r2'])+float(np.mean((pred-gq[qi])**2)/var)
                direct=float(base_row['oracle_r2'])-float(np.mean((pred-g[ix])**2)/var)
                saved=next(r for r in runs if float(r['beta'])==beta and float(r['kappa'])==kappa and float(r['horizon'])==horizon and r['replica']=='0' and r['arm']==arm)
                replay_errors.extend([abs(r2-float(saved['conditional_r2'])),abs(direct-float(saved['direct_risk_r2']))])
        # Deliberately opposing initial beliefs; these do not enter main results.
        probe=np.array([burn,burn+10,burn+100])
        for arm in ('price','queue_price','marked'):
            obs=observe(m,path,arm)
            if arm=='price':
                p0=normalize(m.stationary*(m.states[:,2]==-1));p1=normalize(m.stationary*(m.states[:,2]==1))
                a=price_filter(m,obs,probe,engine,p0)[:,1::2].sum(1);b=price_filter(m,obs,probe,engine,p1)[:,1::2].sum(1)
            else:
                a=queue_filter(m,obs,probe,[1.,0.])[:,1];b=queue_filter(m,obs,probe,[0.,1.])[:,1]
            sensitivity.append(dict(beta=beta,kappa=kappa,arm=arm,max_posterior_difference=float(abs(a-b).max())))
    assert max(replay_errors)<1e-12
    report=dict(status='passed',summary_max_error=summary_error,clock_errors=clock_errors,
        matrix_exponential_max_error=expm_error,null_entropy_max_error=null,replayed_paths=2,
        replayed_prediction_cells=30,replay_max_error=max(replay_errors),
        stochastic_diagnostics=diagnostics,initial_belief_sensitivity=sensitivity,
        independent_paths=len(seeds),prediction_rows=len(runs),filter_rows=len(fruns),
        source_hashes={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ('src/ct_lob','tests/continuous_time_checks','studies/continuous_time_lob/experiments/01_baseline')
            for p in sorted((REPO/folder).glob('**/*.py'))},
        result_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.glob('*.csv'))})
    (ROOT/'validation_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
