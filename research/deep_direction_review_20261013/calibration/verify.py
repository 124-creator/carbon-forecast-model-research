"""Independent metric and exact Stage F score-contract verification."""
import io,json,zipfile,hashlib
from pathlib import Path
import numpy as np
D=Path(__file__).resolve().parent
with np.load(D/'PREDICTIONS_LOCAL_ONLY.npz',allow_pickle=False) as a:
    with zipfile.ZipFile('D:/默认下载/carbon_stageF_20261013_complete.zip') as z:
        raw=z.read('carbon_stageF_20261013/outputs/market_first/evaluation_contract.npz')
    with np.load(io.BytesIO(raw),allow_pickle=False) as b:
        checks={k:bool(np.array_equal(a[k],b[k])) for k in ['origins','dates','true_changes']}
    rows=json.loads((D/'ALL_RESULTS.json').read_text(encoding='utf-8'))
    deviations=[]
    for r in rows:
        cells=[]
        for year in ['2020','2021']:
            mask=np.char.startswith(a['dates'].astype('U10'),year)
            for h in range(3):
                cells.append(float(np.sum((a[r['name']][mask,h]-a['true_changes'][mask,h])**2)/np.sum(a['true_changes'][mask,h]**2)))
        deviations.append(abs(float(np.mean(cells))-r['score']))
    checks['independent_scores']=max(deviations)<1e-12
    checks['configuration_count_15']=len(rows)==15
logs=json.loads((D/'TRAINING_LOGS.json').read_text(encoding='utf-8'))
checks['ridge_target_maturity']=all(l['max_training_target']<l['fit_end_exclusive'] for l in logs if l['training_label_kind']=='complete_1_5_20_targets') if all('training_label_kind' in l for l in logs) else all(l['max_training_target']<l['fit_end_exclusive'] for l in logs if l.get('training_label_kind')=='complete_1_5_20_targets')
receipt=dict(checks=checks,max_score_deviation=max(deviations),archive_contract_sha256=hashlib.sha256(raw).hexdigest(),contract_read_scope='Only evaluation_contract.npz; no archived market CSV or old model weights',ordinary_and_optimized_test_logs=['TEST_GREEN.log','TEST_OPTIMIZED.log'])
(D/'VERIFICATION.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
(D/'PUBLICATION_EXCLUSIONS.json').write_text(json.dumps(dict(exclude=['PREDICTIONS_LOCAL_ONLY.npz'],reason='Local derived price information; publish aggregates/code/logs only after parent review',git_commit=False,git_push=False),indent=2),encoding='utf-8')
print(json.dumps(receipt,indent=2))
if not all(checks.values()):raise RuntimeError('Verification failure')
(D/'MANIFEST.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in D.iterdir() if p.is_file() and p.name not in ['MANIFEST.json','VERIFY.log','VERIFY_OPTIMIZED.log']},indent=2),encoding='utf-8')
