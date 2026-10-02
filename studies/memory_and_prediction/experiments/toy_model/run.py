"""Small LOB smoke experiment. Run from repository root with PYTHONPATH=src."""
from pathlib import Path
import csv,json
import numpy as np
from lob_memory.toy_lob import simulate_lob,aggregate_records,observables
from lob_memory.mori import covariance_sequence,from_covariances,whiten_kernel
from lob_memory.prediction import compare_var
from lob_memory.rg import memory_strength


def run():
    rows=[]
    for seed in range(5):
        train=simulate_lob(65536,seed=5000+seed)
        test=simulate_lob(32768,seed=6000+seed)
        for b in (1,2,4,8,16):
            yt=observables(aggregate_records(train,b)); yv=observables(aggregate_records(test,b))
            cov=covariance_sequence(yt,32//b+1)
            kernel=from_covariances(cov)
            white=whiten_kernel(kernel,cov[0])
            for label,h in [('one_block',1),('fixed_events',16//b)]:
                scores=compare_var(yt,yv,32//b,h)
                rows.append(dict(seed=seed,b=b,target_kind=label,target_events=h*b,
                    covariance_condition=float(np.linalg.cond(cov[0])),
                    memory_raw=memory_strength(kernel[1:]),memory_white=memory_strength(white[1:]),**scores))
    output=Path(__file__).parent/'results'; output.mkdir(exist_ok=True)
    (output/'configuration.json').write_text(json.dumps(dict(training_seeds=list(range(5000,5005)),test_seeds=list(range(6000,6005)),training_events=65536,test_events=32768,burn=2048,queue_cap=8,scales=[1,2,4,8,16],features=['imbalance','depletion','block_mid_change']),indent=2)+'\n')
    with (output/'diagnostics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary={str(b):dict(memory_white_mean=float(np.mean([r['memory_white'] for r in rows if r['b']==b and r['target_kind']=='one_block'])),
               delta_r2_mean=float(np.mean([r['delta_r2'] for r in rows if r['b']==b and r['target_kind']=='one_block']))) for b in (1,2,4,8,16)}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': run()
