"""Independent classical GK/RS global-measurement-noise baselines.
Only pre2026 OHLC is parsed. Four deterministic seed11 B16 raw_constant fits.
2025 selects epochs as in the frozen R16 raw comparator; target remains Parkinson.
"""
import sys
sys.dont_write_bytecode = True
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
from datetime import datetime,timezone
import argparse,copy,hashlib,json,math,re,time
import numpy as np

HERE=Path(__file__).resolve().parent
SYNTH=HERE/'brownian_reference_round16'/'brownian_reference_samples.npz'
EPS=1e-12
DDOF=1
METHODS=('GK','RS')
ASSETS=('KRBN','KCCA')
LOG2=math.log(2)
CLOSE_COEF=2*LOG2-1


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(value):
    a=np.ascontiguousarray(value)
    return dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes()).hexdigest())


def safe(x):
    if isinstance(x,dict):return {str(k):safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [safe(v) for v in x]
    if isinstance(x,np.ndarray):return safe(x.tolist())
    if isinstance(x,(np.integer,np.bool_)):return x.item()
    if isinstance(x,np.floating):return float(x)
    return x


def write_new_json(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(safe(value),f,ensure_ascii=False,indent=2,allow_nan=False)


def estimates_from_log_ohlc(high,low,close):
    q=high-low
    return {'GK':.5*q*q-CLOSE_COEF*close*close,
            'RS':high*(high-close)+low*(low-close)}


def observations(values):
    values=np.asarray(values,dtype=float)
    assert np.isfinite(values).all()
    assert values.min()>=-1e-14,'Materially negative classical estimator; do not repair'
    mask=values>EPS
    logvalue=np.zeros(len(values),dtype=float)
    logvalue[mask]=np.log(values[mask])
    # Missing/zero estimator is NEVER a log(EPS) pseudo-observation.
    assert np.all(logvalue[~mask]==0)
    return logvalue,mask


def calibrate_train_only():
    z=np.load(SYNTH,allow_pickle=False)
    keys=['grid512_train__'+k for k in ('P','eta','xi','H','L','C')]
    P,eta,xi,H,L,C=[z[k] for k in keys]
    assert len(P)==60000
    cP=float(P.mean());assert abs(cP-.9427735381258473)<1e-14
    geom={'GK':P*(4*LOG2)*(.5-CLOSE_COEF*eta**2),
          'RS':P*(2*LOG2)*(1+xi**2-eta**2)}
    direct=estimates_from_log_ohlc(H,L,C)
    result={}
    for method in METHODS:
        np.testing.assert_allclose(geom[method],direct[method],rtol=1e-11,atol=1e-13)
        logvalue,mask=observations(geom[method])
        logdirect,maskdirect=observations(direct[method])
        np.testing.assert_array_equal(mask,maskdirect)
        b=float(logvalue[mask].mean());R=float(logvalue[mask].var(ddof=DDOF))
        bias=b-math.log(cP)
        assert R>0 and abs(float((logvalue[mask]-bias).mean())-math.log(cP))<1e-12
        result[method]=dict(n_synthetic_train=len(P),positive_gt_EPS=int(mask.sum()),
            skipped_le_EPS=int((~mask).sum()),negative_roundoff_count=int((geom[method]<0).sum()),
            raw_min=float(geom[method].min()),b_log_estimator=b,global_R=R,variance_ddof=DDOF,
            observation_bias_for_log_cP_sigma2=bias,
            frame_formula='b_estimator - log(cP), NOT b_estimator-log(c_estimator)',
            positive_conditioning='global moments of positive estimator only; zero/tiny values skipped')
    return dict(source=str(SYNTH),sha256=hash_file(SYNTH),keys_opened=keys,
        samples='grid512 TRAIN only; no heldout or grid2048 array opened',
        cP=cP,log_cP=math.log(cP),EPS=EPS,estimators=result),direct


def synthetic_checks(B,cal):
    # Flat bar, monotone endpoints, near-unit eta, and a generic valid bar.
    H=np.array([0.,1.,(1+.999999)/2,.4]);L=np.array([0.,0.,(-1+.999999)/2,-.3]);C=np.array([0.,1.,.999999,.1])
    e=estimates_from_log_ohlc(H,L,C)
    for method in METHODS:
        refl=estimates_from_log_ohlc(-L,-H,-C)[method]
        np.testing.assert_allclose(e[method],refl,rtol=0,atol=1e-15)
        scaled=estimates_from_log_ohlc(3*H,3*L,3*C)[method]
        np.testing.assert_allclose(scaled,9*e[method],rtol=1e-12,atol=1e-14)
    gx,gm=observations(e['GK']);rx,rm=observations(e['RS'])
    assert not gm[0] and not rm[0] and gm[1] and not rm[1]
    assert gm[2] and rm[2],'Near-unit geometry must not be dropped by angular threshold'
    n=12
    D=dict(n=n,initial_mu=-8.,log_proxy=np.full(n,-8.),even=np.zeros((n,3)),
        cell_b=np.zeros(n),cell_R=np.ones(n),observed=np.ones(n,dtype=bool),global_b=0.,global_R=.3,constant_R=.3,
        y=np.full((2,3),.001),ewma=np.full(2,.001))
    D['observed'][5]=False;D['log_proxy'][5]=0
    net=B.PrecisionFilter('raw_constant',D['initial_mu'],11)
    with B.torch.no_grad():
        m,P,K,R=B.filter_prefix(net,D,B.tensors(D),n)
        assert float(K[5])==0
        theta=net.parameters_bounded();mu,phi,Q,_=theta
        B.torch.testing.assert_close(m[5],mu+phi*(m[4]-mu),rtol=1e-12,atol=1e-12)
        B.torch.testing.assert_close(P[5],phi**2*P[4]+Q,rtol=1e-12,atol=1e-12)
    return dict(market_files_opened=False,same_path_OHLC_direct_vs_geometry=True,
        calibration_only_512train=True,cP_from_512train_P_only=True,reflection_and_degree2_homogeneity=True,
        flat_bar_GK_RS_skipped=True,monotone_nonzero_range_RS_skipped=True,
        near_unit_geometry_retained_when_estimator_positive=True,zero_RS_not_logEPS=True,
        missing_update_K_zero_and_prediction_step_only=True,frame_bias_parentheses_checked=True,
        variance_ddof=DDOF)


def strict_pre2026_prefix(source,target):
    """Unbuffered date gate: on first >=2026 row, read ONLY its10-byte date.
    No2026 price/volume/adjclose bytes are read or parsed, including for hashes.
    """
    n=0;first=last=None;boundary=None;previous=None;end_offset=None
    with Path(source).open('rb',buffering=0) as src,Path(target).open('xb') as dst:
        header=src.readline();assert header.split(b',')[0]==b'date';dst.write(header)
        while True:
            date=src.read(10)
            if not date:break
            assert re.fullmatch(rb'\d{4}-\d{2}-\d{2}',date),date
            if date>=b'2026-01-01':
                boundary=date.decode();end_offset=src.tell()-10;break
            assert previous is None or previous<date
            rest=src.readline();assert rest.startswith(b',') and rest.endswith(b'\n')
            dst.write(date+rest);n+=1;previous=date
            first=date.decode() if first is None else first;last=date.decode()
        if end_offset is None:end_offset=src.tell()
        assert dst.tell()==end_offset
    assert Path(target).stat().st_size==end_offset
    return dict(original_path=str(source),development_prefix_path=str(target),rows=n,first=first,last=last,
        boundary_date_prefix_only=boundary,no2026_OHLC_bytes_read=True,
        verbatim_source_byte_prefix=True,prefix_bytes=end_offset,source_prefix_end_byte_offset=end_offset,
        prefix_sha256=hash_file(target),original_full_CSV_rehashed=False,
        note='Original full-file hash not recomputed because that would read reserved2026 OHLC bytes')


def scores(pred,y):
    assert pred.shape==y.shape and np.isfinite(pred).all() and (pred>0).all() and (y>0).all()
    L=np.log(pred)+y/pred;N=L-np.log(y)-1
    return dict(raw_QLIKE=float(L.mean()),normalized_QLIKE=float(N.mean()),MSE=float(((pred-y)**2).mean()),
                per_h={str(h):dict(n=len(y),raw_QLIKE=float(L[:,j].mean()),normalized_QLIKE=float(N[:,j].mean()),
                                  MSE=float(((pred[:,j]-y[:,j])**2).mean())) for j,h in enumerate((1,5,22))})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--check-only',action='store_true');args=ap.parse_args()
    import bridge_precision_round16 as B
    B.torch.set_num_threads(1);B.torch.set_num_interop_threads(1)
    cal,_=calibrate_train_only();checks=synthetic_checks(B,cal)
    if args.check_only:
        print(json.dumps(dict(checks=checks,calibration=cal),ensure_ascii=False,indent=2));return
    paths={k:HERE/('round18_classical_measurements'+v) for k,v in dict(manifest='_manifest.json',results='_results.json',predictions='_validation_predictions.npz').items()}
    reportpath=HERE/'ROUND18_CLASSICAL_MEASUREMENTS_ZH.md'
    assert not any(p.exists() for p in list(paths.values())+[reportpath]),'Refuse existing output'
    inputdir=HERE/'round18_classical_measurements_inputs';inputdir.mkdir(exist_ok=True)
    prefix={};datasets={};frames={};audit={}
    source_hashes={str(p):hash_file(p) for p in [Path(__file__).resolve(),Path(B.__file__).resolve(),Path(B.R14.__file__).resolve(),B.REFERENCE,SYNTH]}
    extpaths=[HERE/'round18_frozen_moments_development_results.json',HERE/'round18_frozen_moments_development_validation_predictions.npz']
    for a in ASSETS:extpaths.extend([HERE/f'bridge_precision_proxy_round16_{a}_grid512_development_results.json',HERE/f'bridge_precision_proxy_round16_{a}_grid512_development_validation_predictions.npz'])
    for p in extpaths:source_hashes[str(p)]=hash_file(p)
    r18=np.load(HERE/'round18_frozen_moments_development_validation_predictions.npz',allow_pickle=False)
    original_declared_hashes={'KRBN':'ff99c14e84153717be0d72506b0c8b581605495288849823e4bf219ab1462f11','KCCA':'17c93aeca744b91b041a419413fae105c80aca8c9b214c625c7ab6b47d2cae68'}
    for a in ASSETS:
        original=B.SOURCES[a];p=inputdir/f'{a}_pre2026.csv';prefix[a]=strict_pre2026_prefix(original,p);prefix[a]['original_acquisition_declared_hash']=original_declared_hashes[a]
        source_hashes[str(p)]=hash_file(p)
        try:B.SOURCES[a]=p;D,audit[a]=B.prepare(a,512)
        finally:B.SOURCES[a]=original
        assert prefix[a]['rows']==D['n'] and D['date'].max()<np.datetime64('2026-01-01')
        frame=B.pd.read_csv(p,parse_dates=['date']);assert len(frame)==D['n'] and frame.date.max()<B.pd.Timestamp('2026-01-01')
        px=frame[['open','high','low','close']].to_numpy(float);assert np.isfinite(px).all() and (px>0).all()
        H=np.log(px[:,1]/px[:,0]);L=np.log(px[:,2]/px[:,0]);C=np.log(px[:,3]/px[:,0]);assert (H>=np.maximum(0,C)-1e-14).all() and (L<=np.minimum(0,C)+1e-14).all()
        values=estimates_from_log_ohlc(H,L,C)
        # Verify literal financial formulas rather than only a shared helper.
        literal_g=.5*np.log(px[:,1]/px[:,2])**2-CLOSE_COEF*np.log(px[:,3]/px[:,0])**2
        np.testing.assert_allclose(values['GK'],literal_g,rtol=1e-10,atol=1e-14)
        va=D['va'];np.testing.assert_array_equal(D['date'][D['origin'][va]],r18[a+'_date']);np.testing.assert_array_equal(D['y'][va],r18[a+'_target'])
        datasets[a]=D;frames[a]=values
    meta=dict(created_utc=datetime.now(timezone.utc).isoformat(),status='PRE2026 DEVELOPMENT;2025 per-asset epoch selection;2026 prices unread',
        protocol=dict(methods=METHODS,seed=11,epochs=100,chunk=128,underlying_filter_mode='raw_constant',parameters_per_fit=4,
            target='UNCHANGED future mean Parkinson proxy; not trueIV or returnRV',variance_ddof=DDOF,
            selection='B16.train_one per-asset2025 epoch0..100, patience15 min20 improvement1e-5; same as old raw comparator',
            only_D_fields_changed=['log_proxy','observed','global_b','global_R','initial_mu'],
            frame='log(cP*sigma²); observation bias=b_estimator-log(cP); cP from512 TRAIN P only',
            no_theta_filter='No angular/near-unit filtering; update iff estimator>EPS. All original label rows/origins retained.'),
        formulas=dict(GK='.5*log(H/L)^2-(2log2-1)*log(C/O)^2',RS='log(H/O)*(log(H/O)-log(C/O))+log(L/O)*(log(L/O)-log(C/O))'),
        calibration=cal,checks=checks,sources_hashes=source_hashes,development_prefix_audit=prefix,prepare_audit=audit,
        fits={},observation_audit={},comparisons={},environment=dict(python=sys.executable,numpy=np.__version__,torch=B.torch.__version__),
        original_CSV_full_hash_note='Declared acquisition hash retained, never rehashed or parsed reserved rows',
        outputs={k:str(v) for k,v in paths.items()})
    write_new_json(paths['manifest'],meta)
    arrays=dict(horizons=np.array(B.HORIZONS));started=time.time();allpred={}
    for a in ASSETS:
        D=datasets[a];va=D['va'];basekeys={k:array_hash(D[k]) for k in D if isinstance(D[k],np.ndarray)}
        arrays[a+'_date']=D['date'][D['origin'][va]];arrays[a+'_origin']=D['origin'][va];arrays[a+'_label_end']=D['date'][D['origin'][va]+22];arrays[a+'_target']=D['y'][va]
        old=np.load(HERE/f'bridge_precision_proxy_round16_{a}_grid512_development_validation_predictions.npz',allow_pickle=False)
        allpred[a]={'R18_shared_mean':r18[a+'_shared_mean'],'Parkinson_raw_KF':old['raw_constant'],'Parkinson_fixed_KF':old['fixed_precision']}
        for method in METHODS:
            Dc=copy.deepcopy(D);logvalue,observed=observations(frames[a][method]);cfg=cal['estimators'][method]
            Dc['log_proxy']=logvalue;Dc['observed']=observed;Dc['global_b']=cfg['observation_bias_for_log_cP_sigma2'];Dc['global_R']=cfg['global_R']
            pre=(np.arange(D['n'])<D['cut'])&observed;assert pre.any();Dc['initial_mu']=float((logvalue[pre]-Dc['global_b']).mean())
            allowed={'log_proxy','observed','global_b','global_R','initial_mu'}
            for k in D:
                if k not in allowed:
                    if isinstance(D[k],np.ndarray):np.testing.assert_array_equal(Dc[k],D[k])
                    else:assert Dc[k]==D[k]
            assert np.array_equal(Dc['y'],D['y']) and np.array_equal(Dc['ewma'],D['ewma'])
            # Observation audit reports estimator zeros, not discarded targets.
            tag=a+'_'+method;meta['observation_audit'][tag]=dict(n_rows=D['n'],updates=int(observed.sum()),skipped=int((~observed).sum()),
                skipped_pre2025=int((~observed[:D['cut']]).sum()),skipped2025=int((~observed[D['cut']:]).sum()),
                zero_or_small_with_nonzero_P_range=int(np.sum((~observed)&D['observed'])),
                observed_at_common_origins=int(observed[D['origin'][va]].sum()),common_origins_unchanged=len(va),
                initial_mu=Dc['initial_mu'],global_b=Dc['global_b'],global_R=Dc['global_R'],
                target_hash_before=array_hash(D['y']),target_hash_after_copy=array_hash(Dc['y']),no_logEPS_observations=True)
            print('FITTING',tag,'updates',int(observed.sum()),'skip',int((~observed).sum()),flush=True)
            pred,aux,fit=B.train_one(Dc,'raw_constant',seed=11,epochs=100,chunk=128)
            assert fit['parameters']==4
            for k,v in basekeys.items():assert array_hash(D[k])==v,'Original prepared data mutated'
            np.testing.assert_array_equal(Dc['y'],D['y']);np.testing.assert_array_equal(Dc['origin'],D['origin'])
            assert np.all(aux['K'][~observed[D['origin'][va]]]==0)
            fit.update(measurement=method,fit_executed=True,deterministic_single_seed=True,target_unchanged=True)
            meta['fits'][tag]=fit;arrays[tag]=pred;arrays[tag+'_observed_at_origin']=observed[D['origin'][va]]
            for k,v in aux.items():arrays[tag+'_'+k]=v
            allpred[a][method+'_global_KF']=pred
            print('DONE',tag,'bestepoch',fit['best_epoch'],'knownQLIKE',fit['returned_validation_QLIKE_known_scale'],flush=True)
    joint={};comparisons={}
    for a in ASSETS:
        y=arrays[a+'_target'];comparisons[a]={m:scores(v,y) for m,v in allpred[a].items()}
        comparisons[a]['mean_minus_classical']={method:{str(h):dict(QLIKE=comparisons[a]['R18_shared_mean']['per_h'][str(h)]['raw_QLIKE']-comparisons[a][method+'_global_KF']['per_h'][str(h)]['raw_QLIKE'],MSE=comparisons[a]['R18_shared_mean']['per_h'][str(h)]['MSE']-comparisons[a][method+'_global_KF']['per_h'][str(h)]['MSE']) for h in B.HORIZONS} for method in METHODS}
    for method in ['R18_shared_mean','Parkinson_raw_KF','Parkinson_fixed_KF','GK_global_KF','RS_global_KF']:
        joint[method]={key:float(np.mean([comparisons[a][method][key] for a in ASSETS])) for key in ['raw_QLIKE','normalized_QLIKE','MSE']}
    meta.update(comparisons=comparisons,joint_equal_asset_horizon=joint,elapsed_seconds=time.time()-started,
        interpretation='Classical OHLC measurement+global-noise log-variance filter for identical future Parkinson target, not return-GARCH;2025 epoch-selected development only')
    for p,s in source_hashes.items():assert hash_file(p)==s,'Source/prefix changed during fit'
    with paths['predictions'].open('xb') as f:np.savez_compressed(f,**arrays)
    meta['predictions_sha256']=hash_file(paths['predictions'])
    write_new_json(paths['results'],meta)
    lines=['# Round18传统GK/RS测量+全局噪声滤波基准','','仅新文件，四组asset×measurement拟合，seed11确定性复用不作独立重复。目标始终是原future mean Parkinson proxy，不是GK/RS目标、真实IV或return-GARCH。2025仍用于每asset选epoch，2026价格完全未读。','',
        '## 隔离与校准','','原B16.prepare和R14.build先整表读取再截2026，因此本实验使用新pre2026前缀副本：源文件无缓冲逐行只先读10字节日期，遇首2026日期立即停止、不读该行其余字段。B16.SOURCES只在内存临时指向副本并恢复，源代码/冻结文件未改。原完整CSV不重新hash，既有归档hash作为声明保留；实际prefix及全部代码/模拟/结果源hash已记录。','',
        'GK=.5 log(H/L)^2−(2ln2−1)log(C/O)^2；RS=log(H/O)[log(H/O)−log(C/O)]+log(L/O)[log(L/O)−log(C/O)]。同Brownian路径的P/eta/xi代数式与直接HLC计算一致。只打开grid512 TRAIN六个数组，没有使用heldout或2048。','',
        f"cP={cal['cP']:.15f}，状态框架log(cP*sigma²)。观测bias严格为b_estimator−log(cP)，不使用cG/cRS。R是positive log估计量残差样本方差，ddof=1。estimator<=EPS(1e-12)时skip，不把zero RS替换成logEPS；不按theta阈值删除行，原target/origin/tr/va/ewma保持不变。",'',
        '| 测量 | synthetic正值数 | skip数 | b(log估计量) | bias=b−logcP | R(ddof1) |','|---|---:|---:|---:|---:|---:|']
    for method in METHODS:
        c=cal['estimators'][method];lines.append(f"| {method} | {c['positive_gt_EPS']} | {c['skipped_le_EPS']} | {c['b_log_estimator']:.9f} | {c['observation_bias_for_log_cP_sigma2']:.9f} | {c['global_R']:.9f} |")
    lines+=['','## 四组拟合与skip','','| asset/测量 | epoch/run | 4全局参数 | 更新/总行 | 2025skip | 原共同起点 |','|---|---|---:|---|---:|---:|']
    for a in ASSETS:
        for method in METHODS:
            tag=a+'_'+method;f=meta['fits'][tag];o=meta['observation_audit'][tag];lines.append(f"| {tag} | {f['best_epoch']}/{f['epochs_run']} | {f['parameters']} | {o['updates']}/{o['n_rows']} | {o['skipped2025']} | {o['common_origins_unchanged']} |")
    lines+=['','## joint六格等权','','| 模型 | raw QLIKE | normalized QLIKE | MSE | R18mean减该模型QLIKE |','|---|---:|---:|---:|---:|']
    for m,v in joint.items():lines.append(f"| {m} | {v['raw_QLIKE']:.9f} | {v['normalized_QLIKE']:.9f} | {v['MSE']:.6e} | {joint['R18_shared_mean']['raw_QLIKE']-v['raw_QLIKE']:+.9f} |")
    lines+=['','## 每asset/h直接比较','','| asset/h | GK QLIKE | RS QLIKE | R18mean QLIKE | mean−GK | mean−RS |','|---|---:|---:|---:|---:|---:|']
    for a in ASSETS:
        for h in B.HORIZONS:
            c=comparisons[a];g=c['GK_global_KF']['per_h'][str(h)]['raw_QLIKE'];rs=c['RS_global_KF']['per_h'][str(h)]['raw_QLIKE'];nn=c['R18_shared_mean']['per_h'][str(h)]['raw_QLIKE'];lines.append(f'| {a}/{h} | {g:.6f} | {rs:.6f} | {nn:.6f} | {nn-g:+.6f} | {nn-rs:+.6f} |')
    lines+=['','所有date/Y与R18归档逐元素一致。模型逐asset在2025选epoch，未调整超参追求输赢，未作真实IV识别或因果观测噪声结论。NN若不能超过GK/RS，应收窄为已知OHLC测量改进；若超过，也仍需普通NN控制和独立封存期确认。完整MSE/拟合历史/参数/校准/源hash在round18_classical_measurements_results.json，预测在同名前缀validation_predictions.npz。','']
    with reportpath.open('x',encoding='utf-8') as f:f.write('\n'.join(lines))
    print('JOINT',json.dumps(joint,ensure_ascii=False),flush=True)
    print('FINISHED seconds',round(meta['elapsed_seconds'],1),'2026 PRICE UNREAD',flush=True)


if __name__=='__main__':main()
