"""Read-only saved-artifact replay. No training or market access."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import csv, json, hashlib, importlib.util
from pathlib import Path
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parent

def require(ok,msg):
    if not ok: raise RuntimeError(msg)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    spec=importlib.util.spec_from_file_location('matched_saved',ROOT/'run.py')
    if spec is None or spec.loader is None: raise RuntimeError('source import unavailable')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    rows=list(csv.DictReader((ROOT/'metrics.csv').open(encoding='utf-8')))
    keys=[tuple(r[k] for k in ('seed','condition','model','market','horizon')) for r in rows]
    require(len(rows)==144 and len(set(keys))==144,'metric count/duplicates')
    counts=Counter(r['condition'] for r in rows)
    require(dict(counts)=={'shared':72,'no_shared':72},'condition counts')
    for r in rows:
        with np.load(ROOT/f"seed_{r['seed']}_{r['condition']}"/f"{r['model']}_predictions.npz") as a:
            hi=list(module.b.H).index(int(r['horizon'])); m=int(r['market'])
            actual=float(np.mean((a['prediction'][:,m,hi]-a['target'][:,m,hi])**2))
            require(np.isclose(actual,float(r['mse']),rtol=1e-12,atol=0),'metric mismatch')
    replay=0
    for seed in module.b.SEEDS:
        for shared in (True,False):
            cond='shared' if shared else 'no_shared'; folder=ROOT/f'seed_{seed}_{cond}'
            prices,_=module.b.generate(seed,shared)
            with np.load(ROOT.parent/folder.name/'observations.npz') as obs:
                require(np.array_equal(prices,obs['prices']),'synthetic observable mismatch')
            for method in module.METHODS:
                with np.load(folder/f'{method}_predictions.npz') as saved:
                    ts=saved['origins']; target=np.stack([prices[t+module.b.H] for t in ts]).transpose(0,2,1)
                    require(np.array_equal(target,saved['target']),'target mismatch')
                    for market in range(3):
                        data=module.design(prices,seed,method,market)
                        require(np.array_equal(ts,data['test'][2]),'origin mismatch')
                        with np.load(folder/f'{method}_market{market}_checkpoint.npz') as cp:
                            x=data['test'][0]; delta=np.column_stack([np.ones(len(x)),(x-cp['mu'])/cp['sd']])@cp['coef']
                            pred=prices[ts,market,None]+delta
                            require(np.allclose(pred,saved['prediction'][:,market],rtol=1e-12,atol=1e-12),'checkpoint replay mismatch')
                        replay+=1
    v=json.loads((ROOT/'verification.json').read_text(encoding='utf-8'))
    checks=v['future_refits']; ck=[(c['seed'],c['condition'],c['method'],c['market']) for c in checks]
    require(len(checks)==48 and len(set(ck))==48,'historical future evidence count')
    require(all(c['exact_refit_and_origin_prediction_equal'] for c in checks),'historical refit failure')
    provenance=json.loads((ROOT/'provenance.json').read_text(encoding='utf-8'))
    sources={'matched_run':ROOT/'run.py','joint_run':ROOT.parent/'run.py','generator':module.j.SOURCE,'protocol':ROOT/'PROTOCOL.md'}
    require(sha(sources['joint_run'])==provenance['joint_sha256'],'joint source hash')
    require(sha(sources['generator'])==provenance['generator_sha256'],'generator source hash')
    require(sha(sources['protocol'])==provenance['protocol_sha256'],'protocol hash')
    original_checks=0
    for relative,expected in provenance['original_hashes'].items():
        require(sha(ROOT.parent/relative)==expected,'original hash mismatch '+relative); original_checks+=1
    aggregate={c:{m:float(np.mean([float(r['mse']) for r in rows if r['condition']==c and r['model']==m])) for m in module.METHODS} for c in counts}
    for c in aggregate:
        for m,value in aggregate[c].items(): require(np.isclose(value,v['aggregate'][c][m],rtol=1e-12,atol=0),'aggregate mismatch')
    print(json.dumps({'execution':'saved metrics and checkpoints replay only; no ridge/refit/train calls','metric_rows':len(rows),'condition_metric_cells':dict(counts),'checkpoint_replay_units':replay,'historical_future_refit_records':len(checks),'future_refits_executed_this_verification':0,'historical_evidence_limit':'48 unique recorded booleans plus source implementing two six-lambda ridge calls; historical execution not independently re-executed','source_hashes':{k:{'path':str(p),'sha256':sha(p)} for k,p in sources.items()},'original_artifact_hashes_checked':original_checks,'aggregate':aggregate,'status':'passed'},indent=2))
if __name__=='__main__': main()
