#!/usr/bin/env python3
"""Read-only liquidity runner checks: no fitting and no market predictions."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / 'source'))
import run_liquidity_extension as extension
import run_model_search as common


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def hook_identity():
    # Called in a fresh worker. No data preparation, model fitting, or prediction.
    return {
        'prepare': common.prepare.__module__ + '.' + common.prepare.__name__,
        'make_model': common.make_model.__module__ + '.' + common.make_model.__name__,
        'freeze': common.freeze.__module__ + '.' + common.freeze.__name__,
        'markets': common.MARKETS,
        'families': common.FAMILIES,
        'auxiliary_root': str(extension.AUXILIARY_ROOT.resolve()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--auxiliary-root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a new receipt path')
    extension.AUXILIARY_ROOT = args.auxiliary_root.resolve()
    os.environ['CARBON_VOLUME_AUX_DIR'] = str(extension.AUXILIARY_ROOT)
    source_paths = [STUDY/'source'/name for name in [
        'run_model_search.py','innovation_models.py',
        'run_liquidity_extension.py','liquidity_models.py']]
    before = {p.name:sha(p) for p in source_paths}
    assert before['run_model_search.py'] == '26347433fe6416298e4969da618cbba291999f3a66131172bc8f2e510f3633bf'
    assert before['innovation_models.py'] == 'cca83f45474c5110f180ae8b7763131f33596a664939c4ebd34c32db24b4dc1e'
    identity = hook_identity()
    expected = {
        'prepare':'run_liquidity_extension.prepare_liquidity',
        'make_model':'run_liquidity_extension.make_liquidity_model',
        'freeze':'run_liquidity_extension.freeze_liquidity',
        'markets':['GUANGZHOU','HUBEI'],
        'families':['liquidity_full','liquidity_readout','shock_size_only',
                    'liquidity_unconstrained','gru','lstm'],
        'auxiliary_root':str(args.auxiliary_root.resolve()),
    }
    assert identity == expected
    workers = {}
    for method in ['fork','spawn']:
        if method in mp.get_all_start_methods():
            with ProcessPoolExecutor(max_workers=1,mp_context=mp.get_context(method)) as pool:
                workers[method] = pool.submit(hook_identity).result(timeout=45)
            assert workers[method] == expected
    records=[]
    for market in expected['markets']:
        path=args.auxiliary_root/(market+'_auxiliary.csv')
        aux=pd.read_csv(path)
        assert aux.date.is_unique
        assert aux.valid_volume.isin([0,1]).all()
        valid=aux.valid_volume.to_numpy()==1
        vol=aux.volume.to_numpy(float)
        assert np.isfinite(vol[valid]).all() and (vol[valid]>=0).all()
        assert (aux.price_match.to_numpy()[valid]==1).all()
        assert np.isnan(vol[~valid]).all()
        logs=[math.log1p(float(v)) if ok else None for v,ok in zip(vol,valid)]
        context=np.zeros((len(aux),2),dtype=np.float32)
        for t,value in enumerate(logs):
            prior=sorted(v for v in logs[max(0,t-60):t] if v is not None)
            if value is not None and len(prior)>=10:
                k=len(prior);j=k//2
                median=prior[j] if k%2 else (prior[j-1]+prior[j])/2
                mean=math.fsum(prior)/k
                std=math.sqrt(math.fsum((v-mean)**2 for v in prior)/k)
                context[t]=[max(-5,min(5,(value-median)/max(std,.25))),1]
        for year in [2022,2023]:
            base=extension.BASE_PREPARE(market,year)
            data=common.prepare(market,year)
            assert np.array_equal(pd.to_datetime(aux.date).to_numpy(dtype='datetime64[D]'),data['dates'])
            assert np.array_equal(base['x'],data['x'][:,:,:4])
            assert np.isfinite(data['x']).all()
            for key in base['masks']:
                assert np.array_equal(base['masks'][key],data['masks'][key])
            assert np.array_equal(base['y'],data['y'])
            independent=np.stack([context[t-63:t+1] for t in data['origins']])
            discrepancy=float(np.max(np.abs(independent-data['x'][:,:,4:])))
            assert discrepancy<=1e-6
            oo=data['origins'];m=data['masks']
            assert oo[m['train']][-1]+20 < oo[m['val']][0]
            assert oo[m['val']][-1]+20 < oo[m['test']][0]
            assert oo[m['final']][-1]+20 < oo[m['test']][0]
            records.append({'market':market,'year':year,'input_shape':list(data['x'].shape),
                'auxiliary_sha256':sha(path),'valid_volume_rows':int(valid.sum()),
                'usable_context_rows':int(context[:,1].sum()),
                'maximum_context_discrepancy':discrepancy,
                'counts':data['manifest']['counts'],
                'last_training_label':data['manifest']['last_train_label'],
                'first_validation_origin':data['manifest']['first_validation_origin'],
                'last_validation_label':data['manifest']['last_validation_label'],
                'first_test_origin':data['manifest']['first_test_origin'],
                'last_final_label':str(data['dates'][oo[m['final']][-1]+20])})
    assert before=={p.name:sha(p) for p in source_paths}
    result={'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'status':'PASS_READ_ONLY_PREFLIGHT','market_fits':0,'market_predictions_scored':0,
        'scope':'Fixed auxiliary files, independent scalar context reconstruction, inherited split identities and worker hooks; no formal experiment launch',
        'source_sha256':before,'audit_script_sha256':sha(__file__),
        'parent_hooks':identity,'worker_hooks':workers,'records':records,
        'limitations':['Core data preparation is reused for base/extension identity checks.',
                       'Model mathematics is covered by the separate synthetic model audit.',
                       'Input validity is checked here against price_match; runner trusts the recorded valid_volume flag.',
                       'Author volume publication time, units and redistribution license remain unverified.',
                       'This is a new development version after partial v1 results, not a fresh holdout.']}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'status':result['status'],'out':str(args.out),
                      'worker_methods':list(workers),'cells':len(records),'market_fits':0}))


if __name__=='__main__':
    main()
