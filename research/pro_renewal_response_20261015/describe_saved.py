"""Post-result descriptive diagnostics of saved values, not refitting or new ranking."""
import argparse,json
from pathlib import Path
import numpy as np


def describe(results,destination):
    d=Path(results);report={'scope':'Post-result saved-parameter descriptions only; not causal attribution, new candidate or new score','modes':{}}
    for mode in ['RENEWAL','CLOCK','MARKOV','RANK4','RANK6','FIXED6','DIRECT','GRU']:
        logs=[json.loads(p.read_text()) for p in sorted((d/'fits').glob(mode+'_*.json'))]
        r={'fits':len(logs),'epochs':sum(x['epochs'] for x in logs),'epoch0_fits':sum(x['selected_epoch']==0 for x in logs),'allocated_parameters':sorted(set(x['parameter_count'] for x in logs)),'sum_fit_wall_seconds':sum(x['seconds'] for x in logs)}
        if mode in ['RENEWAL','CLOCK']:
            bs=[];res=[]
            for p in sorted((d/'fits').glob(mode+'_*.npz')):
                with np.load(p,allow_pickle=False) as z:
                    bs.append(z['detail__beta'].ravel());res.append(float(abs(z['detail__occupancy'].sum(-1)-1).max()))
            b=np.concatenate(bs);r.update(beta_abs_mean=float(np.abs(b).mean()),beta_abs_max=float(np.abs(b).max()),mass_error_max=max(res))
        report['modes'][mode]=r
    Path(destination).write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--out',required=True);a=p.parse_args();describe(a.results,a.out)
