"""GRN-only pre2026 data builder, not a model trainer or holdout reader.

Default --check-only uses synthetic frames and the synthetic HB reference only.
--build-development opens the one approved 1581-row pre2026 GRN CSV, simulation
calibration JSONs and identity/split metadata; never the GRN2026 raw/labels/scores.
Exports the HB/S17 observation/target schema with a real GRN key, standalone
GRN pre2025 normalizers, fixed hurdle/volume-AR preprocessing and te always empty.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import types
import warnings
import numpy as np
import pandas as pd
import torch
import holdout_2026_range_builder as HB
import round21_gamma_observation_filter as R21

B, S17, R20 = HB.B, HB.S17, R21.R20
HERE = Path(__file__).resolve().parent
ACQ = HERE/'data_acquisition_round13'
ASSETS = ('GRN',)
HORIZONS = (1, 5, 22)
SOURCE = ACQ/'GRN_ETN_USD_available_history_20190918_20251231_round17.csv'
SOURCE_SHA = '52cba5fd28beb71995ce4b0092cd7b0b41ece97aa1b28c0d369d90cd04c57e9b'
IDENTITY = ACQ/'GRN_regulatory_identity_round17_webfetch_excerpt.json'
ACTIONS = ACQ/'GRN_corporate_actions_round17_vendor_unmodified.json'
CUT = np.datetime64('2025-01-01')
END = np.datetime64('2026-01-01')
EPS = B.EPS


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def module_globals_signature():
    return (tuple(HB.ASSETS), tuple(S17.ASSETS), tuple(R20.ASSETS), tuple(R21.ASSETS),
            tuple(sorted(B.SOURCES.items())), B.R14.SOURCE, B.RESERVED)


def source_paths():
    # Explicit allowlist; do not inherit HB.source_paths()'s two ETF CSVs.
    return dict(builder=Path(__file__).resolve(), HB=Path(HB.__file__), S17=Path(S17.__file__),
        B16=Path(B.__file__), proxy=Path(S17.PROXY.__file__), R20=Path(R20.__file__),
        R21=Path(R21.__file__), simulation_reference=B.REFERENCE,
        simulation_moments=S17.PROXY.REFERENCE_RESULTS,
        GRN_pre2026_csv=SOURCE, GRN_identity=IDENTITY, GRN_vendor_splits=ACTIONS)


def validate_frame(d):
    if not d.date.is_unique or not d.date.is_monotonic_increasing or d.date.isna().any():
        raise ValueError('Invalid or duplicated GRN dates')
    if len(d) <= 250 or not (d.date.to_numpy() < END).all():
        raise PermissionError('Only pre2026 observations allowed, before any feature/target creation')
    px = d[['open','high','low','close','adj_close']].to_numpy(float)
    if not np.isfinite(px).all() or not (px > 0).all():
        raise ValueError('Missing/nonpositive OHLC/adj_close cannot be imputed or deleted')
    if not (d.high >= d[['open','close']].max(axis=1)).all() or not (d.low <= d[['open','close']].min(axis=1)).all():
        raise ValueError('OHLC envelope failure')
    volume = d.volume.to_numpy(float)
    if np.isinf(volume).any() or (volume[np.isfinite(volume)] < 0).any():
        raise ValueError('Infinite/negative raw volume; missing volume alone may be flagged')
    cut = int(np.searchsorted(d.date.to_numpy(), CUT))
    if not 100 < cut < len(d)-100:
        raise ValueError('Insufficient pre25/2025 rows')
    return cut


def observations(d):
    """Literal HB price arithmetic; permits missing volume only, flagged separately.

    Missing OHLC is rejected. Real zero range stays zero in proxy_raw. The zero
    log_proxy placeholder is exactly HB convention and never replaces raw P0.
    """
    validate_frame(d)
    r = np.log(d.close.to_numpy(float)/d.open.to_numpy(float))
    u = np.log(d.high.to_numpy(float)/d.open.to_numpy(float))
    low = np.log(d.low.to_numpy(float)/d.open.to_numpy(float))
    q = u-low; observed = q > EPS
    eta = np.divide(r,q,out=np.zeros(len(d)),where=observed)
    xi = np.divide(u+low-r,q,out=np.zeros(len(d)),where=observed)
    assert (np.abs(eta)+np.abs(xi) <= 1+1e-6).all()
    a = np.abs(eta)
    z = np.divide(np.abs(xi),1-a,out=np.zeros(len(d)),where=observed & ((1-a)>EPS))
    assert (z >= 0).all() and (z <= 1+1e-6).all(); z=np.clip(z,0,1)
    proxy=q**2/(4*np.log(2)); log_proxy=np.zeros(len(d))
    log_proxy[observed]=np.log(proxy[observed])
    volume=d.volume.to_numpy(float)
    with np.errstate(divide='ignore',invalid='ignore'):
        ratio=np.log(volume/pd.Series(volume).rolling(22).mean().to_numpy())
    return dict(proxy=proxy,log_proxy=log_proxy,observed=observed,zero_range=~observed,
                even_raw=np.column_stack([a,z,ratio]))


def context_arrays(d):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        return HB.contexts(d)


def common_origins(d, context, cut):
    finite=np.isfinite(context).all(axis=1)
    rows=[]; split=[]
    for t in range(100,len(d)-22):
        if not finite[t]: continue
        if t+22 < cut: s=0
        elif t >= cut: s=1
        else: continue
        rows.append(t);split.append(s)
    rows=np.asarray(rows,dtype=int);split=np.asarray(split,dtype=int)
    return rows,np.flatnonzero(split==0),np.flatnonzero(split==1),finite


def construct(frame,bundle):
    """Pure in-memory construction. No filesystem/model fitting here."""
    stamp=module_globals_signature();d=frame.copy(deep=True).reset_index(drop=True)
    cut=validate_frame(d);obs=observations(d);ctx,_,_,_,_=context_arrays(d)
    origin,tr,va,ctx_finite=common_origins(d,ctx,cut)
    if not len(tr)>100 or not len(va)>100:
        raise ValueError('Insufficient common mature train/validation origins')
    model,c=bundle['model'],float(bundle['c'])
    if not np.isfinite(c) or c<=0:raise ValueError('Invalid simulation-only c512')
    pre=(np.arange(len(d))<cut)&obs['observed']
    means=[];stds=[]
    for j in range(3):
        mask=pre&np.isfinite(obs['even_raw'][:,j])
        if not mask.any():raise ValueError('No valid GRN pre25 normalization values')
        values=obs['even_raw'][mask,j];means.append(float(values.mean()));stds.append(float(values.std())+1e-5)
    means=np.asarray(means);stds=np.asarray(stds)
    even=(obs['even_raw']-means)/stds;even[~np.isfinite(even)]=0
    recovered=even*stds+means;raw17=recovered.copy()
    raw17[:,:2]=np.clip(raw17[:,:2],0,1);raw17[obs['zero_range'],:2]=0
    raw17[:,2]=obs['even_raw'][:,2]
    initial_mu=float((obs['log_proxy'][pre]-model['global_b']).mean())+float(np.log(c))
    ref=dict(n=len(d),cut=cut,date=d.date.to_numpy(),origin=origin,tr=tr,va=va,
        observed=obs['observed'],initial_mu=initial_mu,global_b=float(model['global_b'])-float(np.log(c)),
        global_R=float(model['global_R_ddof1']),constant_R=float(model['shape_bias_constant_R_train_residual_mean_square']),
        baseline_x=np.column_stack([ctx[origin],even[origin]]))
    # Unchanged S17 array operations, in a PRIVATE global namespace. No asset alias.
    install=HB.clone_function(S17.install_features,{**vars(S17),'ASSETS':ASSETS,
        'B':types.SimpleNamespace(CUT_DATE=CUT)})
    normal=install({'GRN':ref},{'GRN':raw17})
    normal.update(asset_weight=1.0,scope='STANDALONE_GRN_PRE2025',
        rule='GRN alone; finite volume-logratio on pre25 resolvable-range rows only; invalid feature filled with fixed GRN pre25 mean',
        legacy_pooled_key_names_only=True,original_two_ETF_pooled_stats_used=False)
    prep=dict(assets={'GRN':dict(neural_mean=means.tolist(),neural_std=stds.tolist())},
              normalization=normal,proxy_c512=c)
    assembler=HB.clone_function(HB.assemble_asset,{**vars(HB),'observations':observations})
    D=assembler(d,'GRN',model,{'GRN':ref},prep)
    D['instrument']='GRN'
    volume=d.volume.to_numpy(float);ratio=obs['even_raw'][:,2]
    flags=dict(first21=np.arange(D['n'])<21,raw_volume_missing=~np.isfinite(volume),
        raw_volume_zero=np.isfinite(volume)&(volume==0),volume_logratio_invalid=~np.isfinite(ratio),
        volume_feature_imputed=~np.isfinite(ratio),quote_H_equals_L=d.high.to_numpy(float)==d.low.to_numpy(float),
        context12_nonfinite=~ctx_finite)
    D['volume_logratio_raw']=ratio.copy()
    for key,value in flags.items():D[key]=np.asarray(value,dtype=bool)
    assert D['n']==len(frame) and len(D['te'])==0 and D['test_cut']==D['n']
    assert (D['date']<END).all() and (D['label_dates']<END).all()
    assert (D['origin']>=100).all() and (D['origin']+22<D['n']).all()
    assert (D['date'][D['origin'][D['tr']]+22]<CUT).all()
    assert (D['date'][D['origin'][D['va']]]>=CUT).all()
    assert (D['bank_index']<cut).all() and (D['bank_index']>=21).all()
    assert D['observed'][D['bank_index']].all() and np.isfinite(D['F17']).all()
    np.testing.assert_array_equal(D['proxy_raw'],obs['proxy'])
    for j,h in enumerate(HORIZONS):
        exact=np.array([obs['proxy'][t+1:t+h+1].mean() for t in D['origin']])
        np.testing.assert_array_equal(D['y'][:,j],exact)
    if np.isfinite(volume).all():
        legacy=HB.observations(d)
        for key in ('proxy','log_proxy','observed','zero_range','even_raw'):
            np.testing.assert_array_equal(obs[key],legacy[key])
    fixed=fixed_preprocessing({'GRN':D})
    assert fixed['GRN']['scale_n']==cut and fixed['GRN']['hurdle']['n']==cut
    assert module_globals_signature()==stamp
    def summarize(ids):
        return dict(rows=len(ids),origin_first=str(pd.Timestamp(D['date'][D['origin'][ids[0]]]).date()),
            origin_last=str(pd.Timestamp(D['date'][D['origin'][ids[-1]]]).date()),
            label_end_first=str(pd.Timestamp(D['date'][D['origin'][ids[0]]+22]).date()),
            label_end_last=str(pd.Timestamp(D['date'][D['origin'][ids[-1]]+22]).date()),
            zero_targets_by_h={str(h):int((D['y'][ids,j]==0).sum()) for j,h in enumerate(HORIZONS)})
    eligible=np.arange(100,D['n']-22)
    audit=dict(asset='GRN',assets=['GRN'],rows=D['n'],cut=cut,test_cut=D['test_cut'],
        first_date=str(d.date.iloc[0].date()),last_date=str(d.date.iloc[-1].date()),
        train=summarize(D['tr']),validation=summarize(D['va']),test_origins=0,
        geometry_feature_order=['2*a-1','2*z-1','standalone_pre25_normalized_log_volume_ratio'],
        target='raw Parkinson mean of following1/5/22 retained bars, including real zero-range bars; not true QV',
        proxy_zero_count=int((D['proxy_raw']==0).sum()),zero_range_by_EPS=int(D['zero_range'].sum()),
        quote_zero_range_with_positive_volume=int((flags['quote_H_equals_L']&np.isfinite(volume)&(volume>0)).sum()),
        proxy_raw_floored_or_imputed=False,OHLC_missing_imputed_or_rows_deleted=False,
        zero_range_meaning='no resolvable daily quote range; neither no trading nor zero true volatility',
        raw_volume_missing=int(flags['raw_volume_missing'].sum()),raw_volume_zero=int(flags['raw_volume_zero'].sum()),
        flags={k:dict(all_rows=int(v.sum()),pre25=int(v[:cut].sum()),validation2025=int(v[cut:].sum())) for k,v in flags.items()},
        first21_explanation='rolling22 volume denominator unavailable on first21 rows; mask can overlap other flags',
        origin_exclusions=dict(first100_lookback_rows=100,final22_unmatured_rows=22,
            eligible_nonfinite_context=int((~ctx_finite[eligible]).sum()),
            eligible_purge_boundary=int(((eligible<cut)&(eligible+22>=cut)&ctx_finite[eligible]).sum()),
            rule='HB origin finite-context filter preserved; rejected origins do not delete observations/zero-range target contributions'),
        normalization=normal,F16_mean=means.tolist(),F16_std=stds.tolist(),
        bank=dict(n=len(D['bank_index']),first_row=int(D['bank_index'][0]),last_row=int(D['bank_index'][-1]),
            first_date=str(pd.Timestamp(D['date'][D['bank_index'][0]]).date()),
            last_date=str(pd.Timestamp(D['date'][D['bank_index'][-1]]).date()),
            volume_imputed_bank_rows=int(flags['volume_feature_imputed'][D['bank_index']].sum()),
            rule='pre25 resolvable-range rows with index>=21 only'),
        initial_mu=initial_mu,proxy_c512=c,fixed_preprocessing=fixed,
        Gamma_or_neural_training_performed=False,hurdle_and_AR_are_fixed_preprocessing_only=True,
        iver='schema-compatible GRN data; no model or forecast created',
        interface_gaps=['R21.make_tensors delegates to S17 hardcoded ASSETS; use this generic make_tensors or a private namespace wrapper.',
            'R21.preprocessing calls R20.fit_volume_ar hardcoded assets; use this fixed_preprocessing private clone.',
            'R21 training/evaluation/main loops and default assets remain two-ETF; future wrapper must explicitly pass assets=(GRN,) and adapt private evaluation loops.',
            'StandaloneGRN normalization differs from original two-ETF pooledstats; direct checkpoint transfer requires a separately declared mapping/normalization contract.'])
    return {'GRN':D},audit


def fixed_preprocessing(datasets):
    if tuple(datasets)!=ASSETS:raise ValueError('GRN-only preprocessing requires real GRN key')
    private_ar=HB.clone_function(R20.fit_volume_ar,{**vars(R20),'ASSETS':ASSETS})
    private_pre=HB.clone_function(R21.preprocessing,{**vars(R21),
        'R20':types.SimpleNamespace(fit_volume_ar=private_ar)})
    return private_pre(datasets)


def make_tensors(datasets):
    if tuple(datasets)!=ASSETS:raise ValueError('Expected GRN only, never an ETF alias')
    return {a:{**B.tensors(D),'F17':torch.tensor(D['F17'],dtype=R21.DTYPE),
        'proxy_raw':torch.tensor(R21.raw_proxy(D),dtype=R21.DTYPE)} for a,D in datasets.items()}


def load_bundle():
    return dict(model=read_json(B.REFERENCE)['grid512'],
        c=float(read_json(S17.PROXY.REFERENCE_RESULTS)['resolution_sensitivity']['mean_P_by_train_resolution']['512']))


def load_frame():
    if sha256(SOURCE)!=SOURCE_SHA:raise PermissionError('Approved1581-row pre2026 GRN snapshot changed')
    d=pd.read_csv(SOURCE,parse_dates=['date'])
    if len(d)!=1581 or str(d.date.iloc[-1].date())!='2025-12-31':
        raise PermissionError('Unexpected approved development window')
    validate_frame(d)
    return d


def build_pair(*,development_only=True):
    if development_only is not True:raise PermissionError('This builder cannot open2026 or produce test data')
    datasets,audit=construct(load_frame(),load_bundle())
    return datasets,make_tensors(datasets),audit


def invariant_checks(d,bundle):
    datasets,audit=construct(d,bundle);D=datasets['GRN'];cut=D['cut']
    altered=d.copy(deep=True)
    altered.loc[cut:,'high']*=1.1;altered.loc[cut:,'low']*=.9;altered.loc[cut:,'volume']*=7
    changed,newaudit=construct(altered,bundle)
    assert audit['normalization']==newaudit['normalization']
    assert audit['F16_mean']==newaudit['F16_mean'] and audit['F16_std']==newaudit['F16_std']
    assert audit['fixed_preprocessing']==newaudit['fixed_preprocessing']
    np.testing.assert_array_equal(D['bank_index'],changed['GRN']['bank_index'])
    np.testing.assert_array_equal(D['F17'][:cut],changed['GRN']['F17'][:cut])
    np.testing.assert_array_equal(D['y'][D['tr']],changed['GRN']['y'][changed['GRN']['tr']])
    return dict(pre25_normalizer_hurdle_AR_scale_bank_validation_mutation_invariant=True,
        pre25_F17_and_training_targets_exact=True,raw_proxy_zero_preserved=True,
        original_modules_ASSETS_SOURCES_unchanged=True,all_targets_pre2026_and_common_h22_mature=True,
        te_always_empty=True,no_model_training_or_forecasts=True)


def synthetic_frame():
    rng=np.random.default_rng(20261022);dates=pd.bdate_range('2023-01-02','2025-12-31');n=len(dates)
    O=50*np.exp(np.cumsum(rng.normal(0,.003,n)));C=O*np.exp(rng.normal(0,.002,n))
    H=np.maximum(O,C)*1.02;L=np.minimum(O,C)*.98;V=np.exp(rng.normal(5,1,n))
    for i in (40,120,300):H[i]=L[i]=C[i]=O[i]
    V[120]=0.;V[125]=0.;V[160]=np.nan
    return pd.DataFrame(dict(date=dates,open=O,high=H,low=L,close=C,adj_close=C,volume=V))


def synthetic_checks():
    d=synthetic_frame();bundle=HB.synthetic_bundle();checks=invariant_checks(d,bundle)
    data,audit=construct(d,bundle);D=data['GRN']
    assert D['proxy_raw'][120]==0 and D['raw_volume_zero'][120]
    assert D['proxy_raw'][40]==0 and not D['raw_volume_zero'][40]
    assert D['raw_volume_missing'][160] and D['volume_feature_imputed'][160] and D['F17'][160,2]==0
    assert D['first21'].sum()==21 and D['volume_feature_imputed'][:21].all()
    assert D['F17'][125,2]==0 and D['proxy_raw'][125]>0
    for value in (0.,np.nan):
        invalid=d.copy();invalid.loc[120,'low']=value
        try:construct(invalid,bundle)
        except ValueError:pass
        else:raise AssertionError('Bad OHLC filled or accepted')
    future=d.copy();future.loc[len(future)-1,'date']=pd.Timestamp('2026-01-02')
    try:construct(future,bundle)
    except PermissionError:pass
    else:raise AssertionError('2026 admitted')
    tensors=make_tensors(data);assert tuple(tensors)==('GRN',) and tensors['GRN']['F17'].shape==(D['n'],3)
    checks.update(missing_or_zero_volume_feature_imputed_from_pre25_only=True,
        first21_flag_separate=True,zero_quote_range_not_equal_zero_volume=True,
        missing_or_zero_OHLC_rejected_no_repair=True,synthetic2026_date_rejected_before_features=True,
        real_market_CSV_or_calibration_opened=False,Gamma_model_instantiated=False)
    return checks


def save_exclusive(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def run_development(tag):
    paths=source_paths();before={k:dict(path=str(p),sha256=sha256(p)) for k,p in paths.items()}
    d=load_frame();bundle=load_bundle();datasets,audit=construct(d,bundle)
    checks=invariant_checks(d,bundle);T=make_tensors(datasets)
    identity=read_json(IDENTITY);actions=read_json(ACTIONS)
    if identity['verified_from_pricing_supplement']['cusip']!='06747C322':raise PermissionError('Wrong ETN identity')
    assert all(ev['event_date_exchange_local']<'2026-01-01' for ev in actions)
    D=datasets['GRN'];arrays={'assets':np.array(ASSETS),'horizons':np.array(HORIZONS)}
    for key,value in D.items():
        if isinstance(value,np.ndarray):arrays['GRN_'+key]=value
        elif isinstance(value,(int,float,str)):arrays['GRN_'+key]=np.array(value)
    arrays['GRN_raw_open']=d.open.to_numpy(float);arrays['GRN_raw_high']=d.high.to_numpy(float)
    arrays['GRN_raw_low']=d.low.to_numpy(float);arrays['GRN_raw_close']=d.close.to_numpy(float)
    arrays['GRN_raw_adj_close']=d.adj_close.to_numpy(float);arrays['GRN_raw_volume']=d.volume.to_numpy(float)
    out=HERE/(tag+'.npz');report=HERE/(tag+'_audit.json')
    if out.exists() or report.exists():raise FileExistsError('Exclusive GRN builder outputs already exist')
    with out.open('xb') as f:np.savez_compressed(f,**arrays)
    with np.load(out,allow_pickle=False) as saved:
        for key,value in arrays.items():np.testing.assert_array_equal(saved[key],value)
        assert saved['GRN_te'].size==0 and (saved['GRN_date']<END).all()
    after={k:dict(path=str(p),sha256=sha256(p)) for k,p in paths.items()};assert before==after
    audit.update(created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_hashes=before,checks=checks,output_npz=dict(path=str(out),sha256=sha256(out)),
        tensor_shapes={a:{k:list(v.shape) for k,v in tt.items()} for a,tt in T.items()},
        data_IO='Only approved1581-row GRN pre2026 CSV; no other market CSV, no2026 raw/labels/scores, no model checkpoints or results.',
        simulation_calibration_IO='Existing Brownian grid512 and simulation-train c512 JSON only; no real-market fit chosen here.',
        identity=identity['verified_from_pricing_supplement'],vendor_actions_unchanged=actions,
        source_limitations=['First-sale2019-09-09/expectedlisting09-10 precede availablefirstbar09-18; missing head not filled.',
            'Vendor quote OHLC and split-adjustment/volume units not independently reconciled to exchange/issuer corporate actions.',
            'ETN is Barclays unsecured obligation with issuer credit/bail-in risk; not a fund directly holding allowances.',
            'Underlying EU carbon futures materially overlap existing funds; not independent macro-market.'],
        old_source_files_modified=False,model_training_performed=False,model_predictions_or_scores_constructed=False,
        heldout2026_file_opened=False,standalone_normalization_not_twoETF_transfer=True)
    save_exclusive(report,audit)
    print(json.dumps(dict(asset='GRN',n=D['n'],cut=D['cut'],train_origins=len(D['tr']),validation_origins=len(D['va']),
        test_origins=len(D['te']),last_date=audit['last_date'],proxy_zeros=audit['proxy_zero_count'],
        bank_size=audit['bank']['n'],normalization_scope=audit['normalization']['scope'],
        csv_sha256=SOURCE_SHA,npz=str(out),npz_sha256=sha256(out),audit=str(report),audit_sha256=sha256(report),
        source_sha256=sha256(__file__),checks=checks),ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group()
    group.add_argument('--check-only',action='store_true');group.add_argument('--build-development',action='store_true')
    parser.add_argument('--tag',default='grn_range_development');args=parser.parse_args()
    import re
    if not re.fullmatch(r'grn_range_development[A-Za-z0-9_-]*',args.tag):raise ValueError('New GRN artifact basename only')
    if args.build_development:run_development(args.tag)
    else:
        checks=synthetic_checks();result=dict(checks=checks,created_at_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha256(__file__))
        save_exclusive(HERE/(args.tag+'_purechecks.json'),result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
