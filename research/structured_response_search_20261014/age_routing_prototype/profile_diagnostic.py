# Secondary nonselective profile diagnostics; all checkpoints remain unchanged.
import json
from pathlib import Path
import numpy as np
import torch
from core import Route
from run import dataset,masks,loss
ROOT=Path(__file__).parent
rows=json.loads((ROOT/'RESULTS.json').read_text());out=[]
for row in rows:
    if row['method']!='full':continue
    tag=f"{row['family']}_{row['seed']}_full";a=np.load(ROOT/f"{row['family']}_{row['seed']}_synthetic.npz");y=a['y'];scale=np.std(np.diff(y[:1200]));cap=np.quantile(np.abs(np.diff(y[:1200]))/scale,.995);o,d,f,t,m=dataset(y,scale,cap);ms=masks(o,m)
    model=Route('full');ck=torch.load(ROOT/(tag+'.pt'),weights_only=True);model.load_state_dict(ck['selected'])
    with torch.no_grad():
        p,e,T,q=model(d,f,True);qstat={'min':float(q.min()),'max':float(q.max()),'mean':float(q.mean()),'persistent_fraction_gt_99pct':float((q>.99).to(q.dtype).mean())};kernel=model.kernel().tolist();profile=[]
        for shift in [-1,0,1]:
          for bias in [-1,0,1]:
            model.load_state_dict(ck['selected']);model.v[0]+=shift;model.g[-1].bias.add_(bias);pred=model(d,f);profile.append({'kernel_first_logit_shift':shift,'gate_bias_shift':bias,'validation_loss':float(loss(pred,t,ms['val'])),'forecast_rms_difference':float((pred-p).square().mean().sqrt())})
        model.load_state_dict(ck['last']);lastpred=model(d,f)
        last_selected_rms=float((lastpred-p).square().mean().sqrt())
    out.append({'tag':tag,'selected_kernel':kernel,'q_summary_not_calibrated_probability':qstat,'profile_secondary_not_selected':profile,'last_vs_selected_forecast_rms':last_selected_rms,'no_cross_initialization_claim':True})
(ROOT/'PROFILE_DIAGNOSTICS.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print('profile rows',len(out))
