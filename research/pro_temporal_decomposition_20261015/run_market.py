"""Execute frozen factorial pilot: prefix-only data, versioned resume, numeric outputs."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse, concurrent.futures as cf, hashlib, json, platform, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
import scipy, sklearn
from model import MODES, SEEDS, read_prefix, fit_model, forecast, numeric_state
ROOT=Path(__file__).resolve().parent

def dump(path,obj):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def task(data,out,mode,seed,block):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    p,dates,_=read_prefix(data);end=459+40*block
    orig=np.array([t for t in range(end,min(end+40,930)) if dates[t][:4]==dates[t+20][:4]],int)
    key=f'{mode}_s{seed}_b{block:02d}';out=Path(out);logpath=out/'fits'/f'{key}.json'
    if logpath.exists() and (out/'fits'/f'{key}.npz').exists():
        rec=json.loads(logpath.read_text())
        if rec['mode']!=mode or rec['seed']!=seed or rec['end_exclusive']!=end:raise ValueError('resume mismatch')
        with np.load(out/'fits'/f'{key}.npz',allow_pickle=False) as z:
            if not np.array_equal(z['origins'],orig) or not np.isfinite(z['prediction']).all():raise ValueError('resume arrays')
        return dict(key=key,resumed=True,seconds=rec['wall_seconds'],epochs=rec['final_epoch_count'])
    m,sd,log=fit_model(p,end,mode,seed)
    pred,diag=forecast(m,sd,p,orig);log['prediction_diagnostics']=diag
    saved={'origins':orig,'prediction':pred,'scale':np.array(sd)}
    saved.update({'parameter__'+k:v for k,v in numeric_state(m).items()})
    np.savez_compressed(out/'fits'/f'{key}.npz',**saved);dump(logpath,log)
    return dict(key=key,resumed=False,seconds=log['wall_seconds'],epochs=log['final_epoch_count'],selected=log['selected_epoch'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default=str(ROOT/'data/guangzhou_prefix950.csv'))
    ap.add_argument('--out',required=True);ap.add_argument('--workers',type=int,default=3);ap.add_argument('--resume',action='store_true')
    a=ap.parse_args();out=Path(a.out)
    if out.exists() and not a.resume:raise FileExistsError('use new output directory; resume is explicit')
    (out/'fits').mkdir(parents=True,exist_ok=True)
    p,dates,audit=read_prefix(a.data)
    source={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ('model.py','test_model.py','run_market.py')}
    if (out/'SOURCE_HASHES.json').exists() and json.loads((out/'SOURCE_HASHES.json').read_text())!=source:raise ValueError('source changed since prior fits')
    dump(out/'SOURCE_HASHES.json',source);dump(out/'DATA_AUDIT.json',audit)
    dump(out/'ENVIRONMENT.json',dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,platform=platform.platform(),cuda_used=False,workers=a.workers,threads_per_worker=1))
    ids=np.array([t for t in range(459,930) if dates[t][:4]==dates[t+20][:4]],int)
    truth=np.array([p[t+1:t+21]-p[t] for t in ids]);years=np.array([x[:4] for x in dates[ids]])
    if len(ids)!=451:raise ValueError('origins mismatch')
    np.savez_compressed(out/'TARGETS.npz',origins=ids,truth=truth,years=years,dates=dates[ids])
    start=time.perf_counter();done=[]
    tasks=[(a.data,str(out),m,s,b) for b in range(12) for s in SEEDS for m in MODES]
    try:
        with cf.ProcessPoolExecutor(max_workers=a.workers) as pool:
            pending={pool.submit(task,*args):args[2:] for args in tasks}
            for fut in cf.as_completed(pending):
                rec=fut.result();done.append(rec)
                dump(out/'STATUS.json',dict(stage='neural_fit',completed=len(done),expected=288,elapsed_seconds=time.perf_counter()-start,last=rec))
                print(f"{len(done):03d}/288 {rec['key']} epochs={rec['epochs']} seconds={rec['seconds']:.2f}",flush=True)
    except Exception:
        dump(out/'FAILURE.json',dict(completed=len(done),traceback=traceback.format_exc()));raise
    arrays={};logs=[]
    for m in MODES:
        for s in SEEDS:
            pred=np.full(truth.shape,np.nan)
            for b in range(12):
                key=f'{m}_s{s}_b{b:02d}'
                with np.load(out/'fits'/f'{key}.npz',allow_pickle=False) as z:
                    at=np.searchsorted(ids,z['origins']);pred[at]=z['prediction']
                logs.append(json.loads((out/'fits'/f'{key}.json').read_text()))
            if not np.isfinite(pred).all():raise ValueError('missing forecasts')
            arrays[f'{m}_s{s}']=pred
    np.savez_compressed(out/'NEURAL_PREDICTIONS.npz',**arrays)
    dump(out/'FIT_INDEX.json',[{k:r[k] for k in ('mode','seed','end_exclusive','parameters','selected_epoch','final_epoch_count','wall_seconds','cpu_seconds','forward_sample_visits','theta_initial_digest','final_digest')} for r in logs])
    dump(out/'STATUS.json',dict(stage='neural_complete',completed=288,elapsed_seconds=time.perf_counter()-start,formal_sum_fit_seconds=sum(r['wall_seconds'] for r in logs),formal_sum_cpu_seconds=sum(r['cpu_seconds'] for r in logs)))
    print('NEURAL COMPLETE',flush=True)
if __name__=='__main__':main()
