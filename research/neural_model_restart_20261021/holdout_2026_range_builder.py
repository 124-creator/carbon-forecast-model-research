"""Static train<=2024 / validation2025 / authorized holdout2026 data builder.

Default is development-only. Test requires an explicit, already-frozen selection
JSON with allow_test=true, source_hashes, selected_candidate, and the preprocessing
SHA256 from this builder's development report. This module never chooses or fits
a model. No selection file is created. --check-only uses fake OHLC dates/data.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import types
import numpy as np
import pandas as pd
import round17_shared_precision as S17

B = S17.B
HERE = Path(__file__).resolve().parent
ASSETS = S17.ASSETS
HORIZONS = B.HORIZONS
EPS = B.EPS
VAL_START = np.datetime64('2025-01-01')
TEST_START = np.datetime64('2026-01-01')
TEST_END = np.datetime64('2027-01-01')
FEATURES = ['logrange1','logrange5','logrange22','r1','r5','r22','intraday',
            'logovernightvar','logreturnvar','logvolume','volume_ratio','range_ratio']


def source_paths():
    return dict(builder=Path(__file__).resolve(), R14=Path(B.R14.__file__).resolve(),
        B16=Path(B.__file__).resolve(), proxy_frame=Path(S17.PROXY.__file__).resolve(),
        R17=Path(S17.__file__).resolve(), R18=HERE/'round18_frozen_moments.py',
        prefix=HERE/'kalman_prefix_utils.py', reference_models=B.REFERENCE,
        reference_results=S17.PROXY.REFERENCE_RESULTS,
        **{f'csv_{a}': B.SOURCES[a] for a in ASSETS})


def validate_selection_payload(payload, required, hash_function=B.sha256):
    if payload.get('allow_test') is not True:
        raise PermissionError('Test requires explicit allow_test=true')
    if not payload.get('selected_candidate'):
        raise PermissionError('An already-frozen selected_candidate is required')
    digest = payload.get('preprocessing_sha256', '')
    if len(digest) != 64 or any(c not in '0123456789abcdefABCDEF' for c in digest):
        raise PermissionError('Test requires the frozen development preprocessing_sha256')
    entries = payload.get('source_hashes', {})
    if not set(required).issubset(entries):
        raise PermissionError(f'Missing required source hashes: {sorted(set(required)-set(entries))}')
    for name, entry in entries.items():
        path = Path(entry['path']).resolve()
        if name in required and path != Path(required[name]).resolve():
            raise PermissionError(f'Selection source path changed: {name}')
        if hash_function(path).lower() != entry['sha256'].lower():
            raise PermissionError(f'Selection source hash changed: {name}')
    return payload


def authorize_test(selection_path):
    # This runs BEFORE opening any CSV, calibration, checkpoint, or NPZ.
    if selection_path is None:
        raise PermissionError('Test disabled: supply an explicit frozen selection JSON')
    path = Path(selection_path)
    if not path.is_file():
        raise PermissionError('Frozen selection JSON does not exist; no test data opened')
    payload = json.loads(path.read_text(encoding='utf-8'))
    return validate_selection_payload(payload, source_paths())


def load_scoped_frames(development_only, authorization):
    if not development_only and authorization is None:
        raise PermissionError('Test data cannot enter a helper before authorization')
    frames = {}
    for asset in ASSETS:
        raw = pd.read_csv(B.SOURCES[asset], parse_dates=['date'])
        # Immediate scope cut: no feature/target/statistic is made from raw first.
        if development_only:
            raw = raw.loc[raw.date < pd.Timestamp(TEST_START)].copy().reset_index(drop=True)
        else:
            raw = raw.loc[raw.date < pd.Timestamp(TEST_END)].copy().reset_index(drop=True)
        assert raw.date.is_unique and raw.date.is_monotonic_increasing
        if development_only:
            assert raw.date.max() < pd.Timestamp(TEST_START)
        frames[asset] = raw
    return frames


class FramePandas:
    """Private I/O facade: frozen helpers can only see prefiltered DataFrames."""
    def __init__(self, frames):
        self.frames = {Path(B.SOURCES[a]).resolve(): frames[a] for a in ASSETS}
    def read_csv(self, path, *args, **kwargs):
        frame = self.frames[Path(path).resolve()]
        assert frame.date.max() < pd.Timestamp(TEST_START)
        return frame.copy().reset_index(drop=True)
    def __getattr__(self, name):
        return getattr(pd, name)


class TextFixture:
    def __init__(self, value):
        self.text = json.dumps(value)
    def read_text(self, **kwargs):
        return self.text


def clone_function(function, namespace):
    # Reuse unchanged bytecode with PRIVATE globals. No module/global monkeypatch.
    out = types.FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
    out.__kwdefaults__ = function.__kwdefaults__
    return out


def legacy_development_reference(frames, bundle=None):
    assert all(d.date.max() < pd.Timestamp(TEST_START) for d in frames.values())
    actual_source, actual_reserved = B.R14.SOURCE, B.RESERVED
    facade = FramePandas(frames)
    raw16, old14 = {}, {}
    def prepare(asset, grid):
        rspace = dict(vars(B.R14)); rspace.update(pd=facade, SOURCE=B.SOURCES[asset])
        original_build = clone_function(B.R14.build, rspace)
        def build():
            result = original_build(); old14[asset] = result[0]
            return result
        private_r14 = types.SimpleNamespace(SOURCE=B.SOURCES[asset], build=build)
        bspace = dict(vars(B)); bspace.update(pd=facade, R14=private_r14)
        if bundle is not None:
            bspace['REFERENCE'] = TextFixture({'grid512': bundle['model']})
        result = clone_function(B.prepare, bspace)(asset, grid)
        raw16[asset] = result[0]
        return result
    space = dict(vars(S17))
    space.update(pd=facade, B=types.SimpleNamespace(prepare=prepare, SOURCES=B.SOURCES, RESERVED=B.RESERVED))
    if bundle is not None:
        space['PROXY'] = types.SimpleNamespace(proxy_frame=S17.PROXY.proxy_frame,
            REFERENCE_RESULTS=TextFixture({'resolution_sensitivity': {'mean_P_by_train_resolution': {'512': bundle['c']}}}))
    reference, audit = clone_function(S17.prepare_pair, space)()
    assert B.R14.SOURCE == actual_source and B.RESERVED == actual_reserved
    return reference, audit, raw16, old14


def observations(d):
    # Exact B16 operations, without altering its pre2026 guard or any globals.
    px = d[['open','high','low','close']].to_numpy(float)
    assert np.isfinite(px).all() and (px > 0).all()
    assert (d.high >= d[['open','close']].max(axis=1)).all()
    assert (d.low <= d[['open','close']].min(axis=1)).all()
    r = np.log(d.close.to_numpy(float) / d.open.to_numpy(float))
    u = np.log(d.high.to_numpy(float) / d.open.to_numpy(float))
    low = np.log(d.low.to_numpy(float) / d.open.to_numpy(float))
    q = u-low; observed = q > EPS
    eta = np.divide(r,q,out=np.zeros(len(d)),where=observed)
    xi = np.divide(u+low-r,q,out=np.zeros(len(d)),where=observed)
    assert (np.abs(eta)+np.abs(xi) <= 1+1e-6).all()
    a = np.abs(eta)
    z = np.divide(np.abs(xi),1-a,out=np.zeros(len(d)),where=observed & ((1-a)>EPS))
    assert (z>=0).all() and (z<=1+1e-6).all(); z=np.clip(z,0,1)
    proxy = q**2/(4*np.log(2)); log_proxy=np.zeros(len(d))
    log_proxy[observed] = np.log(proxy[observed])
    volume=d.volume.to_numpy(float)
    assert np.isfinite(volume).all() and (volume>=0).all()
    with np.errstate(divide='ignore',invalid='ignore'):
        volume_ratio=np.log(volume/pd.Series(volume).rolling(22).mean().to_numpy())
    return dict(proxy=proxy,log_proxy=log_proxy,observed=observed,zero_range=~observed,
                even_raw=np.column_stack([a,z,volume_ratio]))


def contexts(d):
    # Literal R14 arithmetic/order. Do not replace range by q**2 here.
    r=np.log(d.adj_close).diff()
    rng=np.log(d.high/d.low).pow(2)/(4*np.log(2))
    intra=np.log(d.close/d.open)
    factor=d.adj_close/d.close
    overnight=np.log(d.open/d.close.shift())+np.log(factor/factor.shift())
    e5=rng.ewm(span=5,adjust=False).mean().clip(lower=EPS)
    e22=rng.ewm(span=22,adjust=False).mean().clip(lower=EPS)
    e63=rng.ewm(span=63,adjust=False).mean().clip(lower=EPS)
    with np.errstate(divide='ignore',invalid='ignore'):
        f=pd.DataFrame({'logrange1':np.log(rng.clip(lower=EPS)),
            'logrange5':np.log(rng.rolling(5).mean().clip(lower=EPS)),
            'logrange22':np.log(rng.rolling(22).mean().clip(lower=EPS)),
            'r1':r,'r5':r.rolling(5).sum(),'r22':r.rolling(22).sum(),
            'intraday':intra,'logovernightvar':np.log(overnight.pow(2).rolling(5).mean().clip(lower=EPS)),
            'logreturnvar':np.log(r.pow(2).rolling(22).mean().clip(lower=EPS)),
            'logvolume':np.log(d.volume),'volume_ratio':np.log(d.volume/d.volume.rolling(22).mean()),
            'range_ratio':np.log(e5/e63)})
    assert list(f.columns)==FEATURES
    return f.to_numpy().astype(np.float32), rng.to_numpy(), e5.to_numpy().astype(np.float32), \
        e22.to_numpy().astype(np.float32).astype(float), e63.to_numpy().astype(np.float32)


def preprocessing_snapshot(reference,audit):
    out=dict(F17_volume=audit['normalization'],c512=audit['proxy_c512'],assets={})
    for asset in ASSETS:
        D=reference[asset]; a=audit['assets'][asset]; bank=D['bank_index']
        out['assets'][asset]=dict(F16_mean=a['neural_mean'],F16_std=a['neural_std'],
            initial_mu=D['initial_mu'],bank_rows=bank.tolist(),
            bank_dates_sha256=hashlib.sha256(D['date'][bank].astype('datetime64[ns]').tobytes()).hexdigest(),
            bank_F17_sha256=hashlib.sha256(D['F17'][bank].tobytes()).hexdigest())
    text=json.dumps(out,sort_keys=True,separators=(',',':'),ensure_ascii=True)
    return out,hashlib.sha256(text.encode('utf-8')).hexdigest()


def assemble_asset(d,asset,model,reference,audit):
    ref=reference[asset]; a=audit['assets'][asset]
    obs=observations(d); ctx,range_raw,short,ewma,long=contexts(d)
    n=len(d); dates=d.date.to_numpy()
    cut=int(np.searchsorted(dates,VAL_START)); cut26=int(np.searchsorted(dates,TEST_START))
    means=np.asarray(a['neural_mean']); stds=np.asarray(a['neural_std'])
    even=(obs['even_raw']-means)/stds; even[~np.isfinite(even)]=0
    recovered=even*stds+means; raw17=recovered.copy()
    raw17[:,:2]=np.clip(raw17[:,:2],0,1); raw17[obs['zero_range'],:2]=0
    with np.errstate(divide='ignore',invalid='ignore'):
        raw17[:,2]=np.log(d.volume.to_numpy(float)/d.volume.rolling(22).mean().to_numpy(float))
    normal=audit['normalization']; F17=raw17.copy(); F17[:,:2]=2*raw17[:,:2]-1
    volume=np.where(np.isfinite(raw17[:,2]),raw17[:,2],normal['pooled_mean'])
    F17[:,2]=(volume-normal['pooled_mean'])/normal['pooled_std']
    assert np.isfinite(F17).all()
    cell=B.reference_cells(obs['even_raw'][:,0],obs['even_raw'][:,1],model)
    origin=[]; split=[]
    finite=np.isfinite(ctx).all(axis=1)
    for t in range(100,n-22):
        if not finite[t]:continue
        if t+22<cut:s=0
        elif t>=cut and t+22<cut26:s=1
        elif t>=cut26:s=2
        else:continue  # Never add 2025 origins whose labels mature in2026.
        origin.append(t);split.append(s)
    origin=np.asarray(origin,dtype=int);split=np.asarray(split,dtype=int)
    y=np.column_stack([np.array([obs['proxy'][t+1:t+h+1].mean() for t in origin]) for h in HORIZONS])
    tr=np.flatnonzero(split==0);va=np.flatnonzero(split==1);te=np.flatnonzero(split==2)
    assert (origin[tr]+22<cut).all() and (origin[va]+22<cut26).all()
    assert (origin[te]+22<n).all() and (dates[origin[te]]>=TEST_START).all()
    assert (dates[origin[te]+22]<TEST_END).all()
    cell_b=np.asarray(model['b_by_cell'],dtype=float)[cell]-np.log(audit['proxy_c512'])
    cell_R=np.asarray(model['R_by_cell'],dtype=float)[cell]
    D=dict(n=n,cut=cut,test_cut=cut26,date=dates,origin=origin,split=split,y=y,tr=tr,va=va,te=te,
        log_proxy=obs['log_proxy'],observed=obs['observed'],zero_range=obs['zero_range'],
        even=even,F16=even,F17=F17,cell=cell,cell_b=cell_b,cell_R=cell_R,
        initial_mu=ref['initial_mu'],global_b=ref['global_b'],global_R=ref['global_R'],constant_R=ref['constant_R'],
        bank_index=ref['bank_index'].copy(),ewma=ewma[origin],ewma22_all=ewma,
        short=short[origin],long=long[origin],context12_all=ctx,context12=ctx[origin],x=ctx[origin],
        range_context_raw=range_raw,range_context_floor=np.maximum(range_raw,EPS),proxy_raw=obs['proxy'],
        baseline_x_F16=np.column_stack([ctx[origin],even[origin]]),
        baseline_x_F17=np.column_stack([ctx[origin],F17[origin]]),
        label_dates=np.column_stack([dates[origin+h] for h in HORIZONS]))
    D['baseline_x']=D['baseline_x_F17']  # retain R17/R18 compatibility
    np.testing.assert_array_equal(origin[:len(ref['origin'])],ref['origin'])
    np.testing.assert_array_equal(tr,ref['tr']);np.testing.assert_array_equal(va,ref['va'])
    np.testing.assert_array_equal(F17[D['bank_index']],ref['F17'][ref['bank_index']])
    assert (dates[D['bank_index']]<VAL_START).all()
    return D


def strict_prefix_checks(datasets,reference,raw16,old14,frames,check_npz):
    report={}
    for asset in ASSETS:
        D,ref=datasets[asset],reference[asset];n=ref['n'];m=len(ref['origin'])
        fields={}
        def same(name,actual,expected):
            np.testing.assert_array_equal(actual,expected,err_msg=f'{asset}: {name}')
            fields[name]=dict(exact=True,shape=list(np.shape(actual)))
        same('old12_context_all_exported_R14_rows',D['context12_all'][old14[asset]['t']],old14[asset]['x'])
        same('old12_context_common',D['context12'],ref['baseline_x'][:,:12]) if m==len(D['origin']) else same('old12_context_common',D['context12'][:m],ref['baseline_x'][:,:12])
        for name in ('even','F17','cell','cell_b','cell_R','log_proxy','observed','zero_range'):
            same(name,D[name][:n],ref[name])
        for name in ('origin','y','ewma'):
            same(name,D[name][:m],ref[name])
        same('baseline_F16',D['baseline_x_F16'][:m],raw16[asset]['baseline_x'])
        same('baseline_F17',D['baseline_x_F17'][:m],ref['baseline_x'])
        same('training_indices',D['tr'],ref['tr']);same('validation_indices',D['va'],ref['va'])
        same('bank_indices',D['bank_index'],ref['bank_index'])
        assert D['initial_mu']==ref['initial_mu']
        same('raw_proxy',D['proxy_raw'][:n],B.observation_arrays(frames[asset])['proxy'])
        npz_check=None
        if check_npz:
            path=HERE/f'bridge_precision_proxy_round16_{asset}_grid512_development_validation_predictions.npz'
            with np.load(path,allow_pickle=False) as saved:
                assert (saved['date']<TEST_START).all() and (saved['label_end']<TEST_START).all()
                query=D['origin'][D['va']]
                same('saved2025_date',D['date'][query],saved['date'])
                same('saved2025_origin',query,saved['origin'])
                same('saved2025_target',D['y'][D['va']],saved['target'])
                same('saved2025_label_end',D['date'][query+22],saved['label_end'])
                same('saved_horizons',np.array(HORIZONS),saved['horizons'])
            npz_check=dict(path=str(path),sha256=B.sha256(path),predictions_loaded=False)
        report[asset]=dict(fields=fields,initial_mu_exact=True,legacy_observation_rows=n,
            context_scope='all context rows exported by original R14, including all horizons; common origins also checked',
            development_origins=m,test_origins=len(D['te']),reference_npz=npz_check)
    return report


def assemble_pair(frames,bundle,check_npz=False,expected_preprocessing=None):
    dev={a:d.loc[d.date<pd.Timestamp(TEST_START)].copy().reset_index(drop=True) for a,d in frames.items()}
    reference,audit,raw16,old14=legacy_development_reference(dev,bundle)
    snapshot,digest=preprocessing_snapshot(reference,audit)
    if expected_preprocessing is not None and digest.lower()!=expected_preprocessing.lower():
        raise PermissionError('Frozen preprocessing/bank digest changed; test labels not constructed')
    datasets={a:assemble_asset(frames[a],a,bundle['model'],reference,audit) for a in ASSETS}
    matching=strict_prefix_checks(datasets,reference,raw16,old14,dev,check_npz)
    return datasets,dict(preprocessing=snapshot,preprocessing_sha256=digest,prefix_matching=matching,
        initializer='original pre2025 initialization retained; learned state parameters are NOT selected here',
        normalization='authoritative original pre2025 values; test checks the frozen digest before constructing test labels')


def build_pair(*,development_only=True,selection_path=None,check_npz=False):
    authorization=None if development_only else authorize_test(selection_path)
    frames=load_scoped_frames(development_only,authorization)
    models=json.loads(B.REFERENCE.read_text(encoding='utf-8'))
    results=json.loads(S17.PROXY.REFERENCE_RESULTS.read_text(encoding='utf-8'))
    bundle=dict(model=models['grid512'],c=float(results['resolution_sensitivity']['mean_P_by_train_resolution']['512']))
    datasets,audit=assemble_pair(frames,bundle,check_npz,
        None if development_only else authorization['preprocessing_sha256'])
    if development_only:assert all(len(D['te'])==0 for D in datasets.values())
    tensors=S17.make_tensors(datasets)
    audit.update(development_only=development_only,test_authorized=authorization is not None,
        selected_candidate=None if authorization is None else authorization['selected_candidate'],
        protocol='static train<=2024,2025 epoch selection,2026 test; no2025 refit',
        model_selection_performed=False,model_predictions_constructed=False,
        test_labels_constructed=not development_only,
        baseline_alias='baseline_x=F17 as in R18; baseline_x_F16 explicitly available for R16 HAR/HGB')
    return datasets,tensors,audit


def synthetic_bundle():
    return dict(c=.94,model=dict(label='synthetic_grid512',edges_a=[0,.2,.4,.6,.8,1],edges_z=[0,.2,.4,.6,.8,1],
        global_b=-.24,global_R_ddof1=.35,shape_bias_constant_R_train_residual_mean_square=.27,
        b_by_cell=np.linspace(-.6,.3,25).tolist(),R_by_cell=np.linspace(.2,.36,25).tolist()))


def synthetic_frames():
    frames={}
    for j,asset in enumerate(ASSETS):
        rng=np.random.default_rng(410+j);dates=pd.bdate_range('2023-01-02','2026-04-30');n=len(dates)
        O=100*np.exp(np.cumsum(rng.normal(0,.003,n)));body=rng.normal(0,.01,n)
        C=O*np.exp(body);H=O*np.exp(np.maximum(body,0)+rng.uniform(.002,.02,n))
        L=O*np.exp(np.minimum(body,0)-rng.uniform(.002,.02,n))
        C[250]=O[250];H[250]=O[250];L[250]=O[250]
        frames[asset]=pd.DataFrame(dict(date=dates,open=O,high=H,low=L,close=C,adj_close=C*.97,
                                       volume=rng.integers(1000,100000,n)))
    return frames


def synthetic_checks():
    try:build_pair(development_only=False)
    except PermissionError:pass
    else:raise AssertionError('Test was not rejected without a selection JSON')
    touched=[]
    try:validate_selection_payload(dict(allow_test=False),{'x':Path('unused')},lambda p:touched.append(p))
    except PermissionError:pass
    else:raise AssertionError('allow_test=false accepted')
    assert not touched
    frames=synthetic_frames();bundle=synthetic_bundle()
    full,audit=assemble_pair(frames,bundle)
    changed={a:d.copy() for a,d in frames.items()}
    for a in ASSETS:
        mask=changed[a].date>=pd.Timestamp('2026-03-01')
        changed[a].loc[mask,'high']*=1.05;changed[a].loc[mask,'low']/=1.05
        changed[a].loc[mask,'volume']*=3
    altered,other=assemble_pair(changed,bundle,expected_preprocessing=audit['preprocessing_sha256'])
    assert audit['preprocessing_sha256']==other['preprocessing_sha256']
    for a in ASSETS:
        D=full[a];E=altered[a];cut=int(np.searchsorted(D['date'],np.datetime64('2026-03-01')))
        for name in ('context12_all','even','F17','log_proxy','ewma22_all','cell','cell_b','cell_R'):
            np.testing.assert_array_equal(D[name][:cut],E[name][:cut])
        mature=D['origin']+22<cut
        np.testing.assert_array_equal(D['y'][mature],E['y'][mature])
        assert len(D['te']) and (D['label_dates'][D['te'],-1]<=D['date'][-1]).all()
        assert not ((D['date'][D['origin'][D['va']]]>=VAL_START)&(D['label_dates'][D['va'],-1]>=TEST_START)).any()
        t=D['origin'][D['te'][0]]
        for j,h in enumerate(HORIZONS):
            assert D['y'][D['te'][0],j]==D['proxy_raw'][t+1:t+h+1].mean()
        np.testing.assert_array_equal(D['tr'],E['tr']);np.testing.assert_array_equal(D['va'],E['va'])
        assert D['bank_index'].max()<D['cut']
    return dict(real_market_files_opened=False,real2026_targets_constructed=False,
        synthetic2026_only=True,test_without_selection_rejected_before_IO=True,
        allow_test_false_rejected_before_hash_IO=True,future_mutation_preserves_past_features=True,
        matured_targets_unchanged=True,targets_only_future1_through_h=True,
        common_h22_maturity=True,no2025_labels_cross2026=True,pre2025_bank_and_normalizers_fixed=True,
        frozen_module_globals_unchanged=True,all_legacy_prefix_checks_exact=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    mode=ap.add_mutually_exclusive_group()
    mode.add_argument('--development-only',action='store_true')
    mode.add_argument('--test',action='store_true')
    ap.add_argument('--selection',type=Path)
    ap.add_argument('--check-only',action='store_true')
    ap.add_argument('--report',type=Path,default=HERE/'holdout_2026_range_builder_development_validation.json')
    args=ap.parse_args()
    if args.check_only:
        print(json.dumps(synthetic_checks(),indent=2),flush=True);return
    development_only=not args.test
    if development_only and args.selection is not None:
        raise ValueError('Selection is only consumed with explicit --test')
    if args.report.exists():raise FileExistsError(f'Existing report protected: {args.report}')
    sources=source_paths();before={k:B.sha256(p) for k,p in sources.items() if not k.startswith('csv_')}
    datasets,_,audit=build_pair(development_only=development_only,selection_path=args.selection,check_npz=True)
    for k,expected in before.items():assert B.sha256(sources[k])==expected
    report=dict(status='DEVELOPMENT PREFIX VALIDATION' if development_only else 'AUTHORIZED TEST DATA BUILD ONLY',
        audit=audit,checks=synthetic_checks() if development_only else None,
        source_hashes={k:dict(path=str(p.resolve()),sha256=B.sha256(p)) for k,p in sources.items()},
        retained={a:dict(n=D['n'],last_retained_date=str(pd.Timestamp(D['date'][-1]).date()),
            train_origins=len(D['tr']),validation_origins=len(D['va']),test_origins=len(D['te'])) for a,D in datasets.items()},
        selection_file_created=False,model_training_performed=False,model_predictions_constructed=False)
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=report['status'],report=str(args.report.resolve()),
        preprocessing_sha256=audit['preprocessing_sha256'],retained=report['retained'],
        test_labels_constructed=audit['test_labels_constructed'],strict_prefix_checks='ALL EXACT'),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
