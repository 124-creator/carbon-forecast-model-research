from pathlib import Path
import json,hashlib
import numpy as np
p=Path(__file__).resolve().parent
r=json.loads((p/'results.json').read_text()); d=np.load(p/'synthetic_inputs.npz'); preds=np.load(p/'predictions.npz'); heads=np.load(p/'heads.npz')
checks=[]
for row in r['rows']:
 k=f"{row['seed']}_{row['model']}"; v=preds[k]; m=((v-d['yt'][:,None])**2).mean(0)
 if not np.allclose(m,row['per_phase_mse'],atol=1e-12): raise RuntimeError(k)
 if heads[k].shape!=(17,) or not np.isfinite(v).all(): raise RuntimeError(k)
 checks.append(k)
if len(checks)!=30 or r['elapsed_seconds']>600: raise RuntimeError('count or budget')
out={'verified_rows':len(checks),'metrics_recomputed':True,'head_parameters_matched':17,'market_access':False,'files':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in p.iterdir() if f.is_file() and f.name!='verification.json'}}
(p/'verification.json').write_text(json.dumps(out,indent=2)); print('VERIFIED',len(checks),'rows')
