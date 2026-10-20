import json,hashlib,sys
from pathlib import Path
import numpy as np
O=Path(__file__).resolve().parent
import run as r
req=r.req
z=np.load(O/'PREDICTIONS_LOCAL_ONLY.npz',allow_pickle=False);receipt=json.loads((O/'RUN_RECEIPT.json').read_text());results=json.loads((O/'RESULTS.json').read_text());logs=json.loads((O/'TRAINING_LOGS.json').read_text());checks=json.loads((O/'FUTURE_FX_REFITS.json').read_text())
req(len(z['origins'])==451 and receipt['ecb_rows']==1025,'counts')
old=np.load(O.parent/'coal_training/PREDICTIONS_LOCAL_ONLY.npz',allow_pickle=False);req(np.array_equal(z['MLE'],old['MLE']),'baseline exact')
for row in results:
 key='MLE' if row['name']=='MLE480' else row['name'];s,_=r.c.base.score(z[key],z['truth'],z['years']);req(abs(s-row['score'])<1e-12,'recomputed score')
for lag in [2,3,5]:
 v=z[f'valid_lag{lag}'];req(np.array_equal(z[f'FX_lag{lag}'][~v],z['MLE'][~v]),'missing fallback')
for a in logs:req(a['alpha'] in [.1,10.] and a['validation_n']==80 and a['inner_last_target']<a['validation_first'] and a['outer_last_target']<a['end'],'purge')
req(len(checks)==9 and all(a['future_rows']>0 and a['parameter_diff']==a['prediction_diff']==0 and a['alpha_same'] for a in checks),'actual future refits')
req(receipt['all_missing_fallback_diff']==receipt['price_prediction_diff']==0,'fallback/baseline')
for cur,h in receipt['input_hashes'].items():req(hashlib.sha256((O.parent/'fx_probe/raw_LOCAL_ONLY'/f'{cur}.csv').read_bytes()).hexdigest()==h,'source hash')
req(hashlib.sha256((O/'CONTRACT.md').read_bytes()).hexdigest()==receipt['contract_sha256'],'contract')
X,v=r.features(z['dates'],np.array(['2018-01-01']*6,dtype='datetime64[D]'),np.full(6,np.nan),2);req(not v.any(),'allmissing independent')
report=dict(passed=True,optimized=bool(sys.flags.optimize),origins=451,training_refits=len(logs),future_actual_refits=len(checks),price_prediction_diff=0,allmissing_diff=0)
r.dump('VERIFIED_OPT.json' if sys.flags.optimize else 'VERIFIED_NORMAL.json',report);print(json.dumps(report))
