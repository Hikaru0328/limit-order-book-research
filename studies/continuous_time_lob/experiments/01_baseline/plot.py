"""Representative beta=0.5; intervals refer to independent path means."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse, json
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[4]/'results/synthetic/continuous_time')
root = parser.parse_args().output
replicas = json.loads((root/'configuration.json').read_text())['replicas']
def read(name):
    with (root/(name+'.csv')).open() as f:return list(csv.DictReader(f))

filters=read('filter_summary');pred=read('prediction_summary');pop=read('population')
arms=['current_queue','price_history_current_queue','queue_price_history','marked_history']
labels=['Current queues','Price history + current queues','Queue / price history','Marked event history']
colors=['#687782','#ce9b1c','#188b8d','#ad2366'];kappas=[.02,.2,2.]
fig,axs=plt.subplots(2,2,figsize=(12,8.2),layout='constrained')
for arm,label,color in zip(arms,labels,colors):
    rows=[next(r for r in filters if float(r['beta'])==.5 and float(r['kappa'])==k and r['arm']==arm) for k in kappas]
    y=np.array([float(r['entropy_bits_mean']) for r in rows]);lo=np.array([float(r['entropy_bits_low95']) for r in rows]);hi=np.array([float(r['entropy_bits_high95']) for r in rows])
    axs[0,0].errorbar(kappas,y,yerr=np.maximum(0,np.array([y-lo,hi-y])),label=label,color=color,marker='o',capsize=3)
    rows=[next(r for r in pred if float(r['beta'])==.5 and float(r['kappa'])==k and r['horizon']=='4.0' and r['arm']==arm) for k in kappas]
    y=np.array([float(r['hidden_recovery_mean']) for r in rows])*100;lo=np.array([float(r['hidden_recovery_low95']) for r in rows])*100;hi=np.array([float(r['hidden_recovery_high95']) for r in rows])*100
    axs[0,1].errorbar(kappas,y,yerr=np.maximum(0,np.array([y-lo,hi-y])),label=label,color=color,marker='o',capsize=3)
    axs[1,0].plot(kappas,[float(r['conditional_r2_mean']) for r in rows],label=label,color=color,marker='o')
axs[0,0].set(ylabel='Residual hidden entropy (bits)',title='Hidden-direction uncertainty',ylim=(0,1.04))
axs[0,1].set(ylabel='Hidden price-information recovery (%)',title='Same current-queue baseline; H=4')
oracle=[next(float(r['oracle_r2']) for r in pop if r['beta']=='0.5' and float(r['kappa'])==k and r['horizon']=='4.0') for k in kappas]
axs[1,0].plot(kappas,oracle,'k--',label='Full-state oracle')
axs[1,0].set(ylabel='Price prediction R-squared',title='Absolute predictability; H=4')
for ax in axs.flat[:3]:ax.set_xscale('log');ax.set_xlabel('Hidden flip rate kappa');ax.grid(alpha=.2)
example=read('example_posteriors');example=example[:60];t=np.array([float(r['time']) for r in example]);t-=t[0]
axs[1,1].step(t,[(int(r['hidden_direction'])+1)/2 for r in example],where='post',color='k',alpha=.3,label='True direction at sample times')
for arm,label,color in zip(arms[1:],labels[1:],colors[1:]):axs[1,1].plot(t,[float(r[arm]) for r in example],label=label,color=color,lw=1.4)
axs[1,1].set(xlabel='Time since first scored sample',ylabel='Posterior P(z=+1)',title='Illustration only: kappa=0.2',ylim=(-.04,1.04));axs[1,1].grid(alpha=.2)
axs[0,0].legend(fontsize=8,loc='lower right');axs[1,0].legend(fontsize=8);axs[1,1].legend(fontsize=7,loc='upper right')
fig.suptitle(f'Continuous-time fixed-spread LOB | beta=0.5 | {replicas} independent paths per model\nKnown parameters; exact causal filters; Monte Carlo averages with 95% intervals',fontsize=12)
fig.savefig(root/'baseline.svg');fig.savefig(root/'baseline.png',dpi=150);plt.close(fig)
