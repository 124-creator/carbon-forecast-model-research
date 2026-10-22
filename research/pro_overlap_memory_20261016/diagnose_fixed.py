"""Post-result fixed-parameter magnitude diagnostic, not another trained model.

Reports readout-weight and output perturbation magnitudes only. Does not select
parameters or score an alternate method. Always label this analysis post hoc.
"""
from pathlib import Path
import argparse,json
import numpy as np
import core
from experiment import jwrite

def run(out):
    out=Path(out);p,_,_=core.load_prefix(out/'approved_prefix.csv');r=core.reference(p);memory=core.MemoryData(p,r)
    effects=[];wdiff=[];betas=[];entropy=[]
    for path in sorted((out/'fits').glob('OVERLAP_*.npz')):
        with np.load(path,allow_pickle=False) as z:
            ids=z['origins'];w=z['w'];pi=z['pi'];mask=z['valid_mask'];beta=z['beta']
            # DIAGONAL formula: inverse(1+1/pi), normalized. No fitted model changed.
            un=mask*pi/(1+pi);ind=un/un.sum(-1,keepdims=True)
            diff=w-ind[:,None,:]
            effect=memory.local[ids,None]*beta[None,:]*np.einsum('nhk,nkh->nh',diff,memory.values['regular'][ids])
            effects.append(effect);wdiff.append(np.mean(abs(diff),axis=-1));betas.append(beta)
            entropy.extend((-np.sum(pi*np.log(np.maximum(pi,1e-300)),axis=1)).tolist())
    e=np.concatenate(effects);d=np.concatenate(wdiff)
    result=dict(scope='Post-result only, fixed OVERLAP parameters. No retraining, no alternate-model score, no parameter selection.',
      candidate_fit_archives=len(effects),origin_seed_pairs=len(e),mean_abs_weights_delta_by_h=d.mean(0).tolist(),
      raw_price_correction_RMS_delta_by_h=np.sqrt(np.square(e).mean(0)).tolist(),max_abs_output_change=float(abs(e).max()),
      mean_abs_beta=float(abs(np.stack(betas)).mean()),mean_pi_entropy=float(np.mean(entropy)),uniform12_entropy=float(np.log(12.)),
      h1_fixed_readout_equivalence_max_error=float(abs(e[:,0]).max()),
      interpretation='Finite intervention magnitude does not identify a market causal mechanism or establish the unique cause of prediction failure.')
    jwrite(out/'POSTHOC_FIXED_DIAGNOSTIC.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();run(a.out)
