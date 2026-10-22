"""GRN independent Gamma runner: reviewed-code handoff, no training by default.

This delivery runs synthetic purechecks and ONE pre2026 no-optimizer init sanity.
A later explicit --run-development uses unchanged R21 HP/architecture via private
FunctionType globals and the hash-bound Fast class. No GRN2026 I/O route exists.
Primary scores use every nonnegative raw target, including real zeros. Normalized
QLIKE is undefined when a scored group contains y=0; never floor/delete raw y.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
import argparse
import ast
import copy
from datetime import datetime,timezone
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
import textwrap
import types
import numpy as np
import pandas as pd
import torch
import grn_range_development_builder as G
import round21_gamma_observation_filter as S21
import round21_fast_filter as FAST

HERE=Path(__file__).resolve().parent
ASSETS=('GRN',)
HORIZONS=(1,5,22)
BASES=('global_dense','shape_dense','fixed_mixture')
SEEDS=(11,22,33)
FAST_SHA='99df4c242379eac9c695f660fd770d02433cc8d4f039d202f16d61a2dd824e7e'
G_SHA='78e55425a0e6a9a04476dbb4dbaa2cb0a1f210e268c3d8a6cfb6ee18f085eb12'
DATA=HERE/'grn_range_development.npz'
AUDIT=HERE/'grn_range_development_audit.json'
DATA_SHA='acd4a33c31a49c62563a7eae2dcaa236e27a1a0432c3dab42f015d8969d47a3f'
AUDIT_SHA='4d12814c03a0202b63ba1045005806c3d231f952e2427a15acc5e831cf603e10'
HP=dict(max_epochs=80,patience=15,min_epochs=20,Adam_lr=.01,gradient_clip=2.,
        grid_size=256,grid_log_bounds=[-8,8],GH=16,sensor_NLL_weight=.01,
        kappa_penalty=.005,epoch0_in_selection=True,improvement_threshold=1e-5)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def finite_json(value):
    """Reject invalid metadata rather than silently replacing failed calculations."""
    json.dumps(value,ensure_ascii=False,allow_nan=False)
    return value


def save_json(path,value):
    finite_json(value)
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def assert_assets(model,datasets,T):
    if tuple(model.assets)!=ASSETS or tuple(datasets)!=ASSETS or tuple(T)!=ASSETS:
        raise ValueError('Actual GRN key and explicit assets=(GRN,) required')
    if tuple(model.processes)!=ASSETS:raise ValueError('Wrong process identity')


def checked_objective(model,datasets,T):
    assert_assets(model,datasets,T)
    result=S21.objective(model,datasets,T)
    if not torch.isfinite(result[0]):raise FloatingPointError('Nonfinite all-target training objective')
    return result


def checked_evaluate(model,datasets,T):
    assert_assets(model,datasets,T)
    pred,joint,scores,aux=S21.evaluate(model,datasets,T)
    if not math.isfinite(joint) or not all(math.isfinite(x) for x in scores.values()):
        raise FloatingPointError('Nonfinite all-target validation QLIKE')
    for a in ASSETS:
        value=pred[a];y=datasets[a]['y'][datasets[a]['va']]
        if value.shape!=y.shape or not np.isfinite(value).all() or not (value>0).all():
            raise FloatingPointError('Invalid point forecasts')
    return pred,joint,scores,aux


def private_train_fit():
    """Clone unchanged R21 bytecode, never edit source or its module globals."""
    space=dict(vars(S21))
    space.update(make_tensors=G.make_tensors,GammaVarianceFilter=FAST.FastGammaVarianceFilter,
                 objective=checked_objective,evaluate=checked_evaluate)
    space['train_epoch']=G.HB.clone_function(S21.train_epoch,space)
    train=G.HB.clone_function(S21.train_fit,space)
    assert train.__code__ is S21.train_fit.__code__ and train.__globals__ is not vars(S21)
    assert train.__globals__['GammaVarianceFilter'] is FAST.FastGammaVarianceFilter
    return train


def source_contract():
    if sha(FAST.__file__)!=FAST_SHA:raise PermissionError('Reviewed Fast source changed')
    if sha(G.__file__)!=G_SHA:raise PermissionError('Reviewed GRN builder changed')
    if tuple(S21.BASES)!=BASES or tuple(S21.SEEDS)!=SEEDS or tuple(S21.HORIZONS)!=HORIZONS:
        raise PermissionError('R21 modes/seeds/horizons changed')
    assert (S21.MAX_EPOCHS,S21.PATIENCE,S21.MIN_EPOCHS,S21.LR,S21.GRID_SIZE,S21.GH_SIZE)==(80,15,20,.01,256,16)
    for function in (S21.objective,S21.evaluate):
        tree=ast.parse(textwrap.dedent(inspect.getsource(function)))
        assert any(isinstance(n,ast.For) and isinstance(n.iter,ast.Attribute) and n.iter.attr=='assets'
            and isinstance(n.iter.value,ast.Name) and n.iter.value.id=='model' for n in ast.walk(tree))
    # Existing raw metrics and objectives have no log(y); normalized reporting below is new and guarded.
    FAST.assert_only_row_access_changed()
    train=private_train_fit()
    assert train.__globals__['make_tensors'] is G.make_tensors
    return dict(Fast_sha256=FAST_SHA,GRN_builder_sha256=G_SHA,
        objective_evaluate_loop_model_assets=True,private_train_fit_bytecode_unchanged=True,
        architecture_and_HP_unchanged=True,old_S17_metrics='raw log(v)+y/v and MSE, no log(y)',
        R21_train_evaluate='B.qlike(v,y,positive_known_EWMA); all nonnegative targets included')


def source_hashes():
    source_contract()
    paths=dict(G.source_paths())
    paths.update(GRN_data_npz=DATA,GRN_audit=AUDIT,runner=Path(__file__),Fast=Path(FAST.__file__),
        gamma_reference=Path(S21.REF.__file__),gamma_JSON=HERE/'sampling_density_gamma_calibration.json',
        gamma_NPZ=HERE/'sampling_density_gamma_calibration.npz',
        log_reference=Path(S21.REF.LOGREF.__file__))
    return {k:dict(path=str(Path(p).resolve()),sha256=sha(p)) for k,p in paths.items()}


def unchanged(before):
    for name,info in before.items():
        if sha(info['path'])!=info['sha256']:raise PermissionError('Source changed: '+name)


def load_grn():
    source_contract()
    if sha(DATA)!=DATA_SHA or sha(AUDIT)!=AUDIT_SHA:raise PermissionError('Approved GRN development artifacts changed')
    audit=read_json(AUDIT)
    for name,info in audit['source_hashes'].items():
        if sha(info['path'])!=info['sha256']:raise PermissionError('GRN builder dependency changed: '+name)
    with np.load(DATA,allow_pickle=False) as arrays:
        assert arrays['assets'].tolist()==['GRN']
        D={k[4:]:v.copy() if v.ndim else v.item() for k in arrays.files if k.startswith('GRN_') for v in [arrays[k]]}
    assert D['instrument']=='GRN' and D['n']==1581 and D['cut']==1331
    assert len(D['tr'])==1158 and len(D['va'])==228 and not len(D['te'])
    assert (D['date']<np.datetime64('2026-01-01')).all() and (D['label_dates']<np.datetime64('2026-01-01')).all()
    assert (D['date'][D['origin'][D['tr']]+22]<np.datetime64('2025-01-01')).all()
    assert np.isfinite(D['y']).all() and (D['y']>=0).all() and (D['proxy_raw']==0).sum()==342
    assert ((D['y'][D['va']]==0).sum(axis=0)==[33,0,0]).all()
    assert D['F17'].shape==(1581,3) and np.isfinite(D['F17']).all()
    fixed=copy.deepcopy(audit['fixed_preprocessing'])  # no new hurdle, scale or AR estimation
    assert tuple(fixed)==ASSETS and fixed['GRN']['scale_n']==1331 and fixed['GRN']['hurdle']['n']==1331
    datasets={'GRN':D};T=G.make_tensors(datasets)
    validate_schema(D)
    return datasets,T,fixed,audit


def validate_schema(D):
    """All real output labels/origin dates must exactly match the approved G NPZ."""
    with np.load(DATA,allow_pickle=False) as z:
        for key in ('date','origin','y','tr','va','te','label_dates','F17','proxy_raw','ewma','bank_index'):
            np.testing.assert_array_equal(D[key],z['GRN_'+key])
        q=D['origin'][D['va']];q0=z['GRN_origin'][z['GRN_va']]
        np.testing.assert_array_equal(D['date'][q],z['GRN_date'][q0])
        np.testing.assert_array_equal(D['y'][D['va']],z['GRN_y'][z['GRN_va']])
    return dict(asset='GRN',F3_shape=list(D['F17'].shape),F3_order=['2*a-1','2*z-1','GRN_standalone_pre25_volumeZ'],
        dtype='float64',validation_rows=len(D['va']),zero_targets_by_h=[int(x) for x in (D['y'][D['va']]==0).sum(0)],
        dates_targets_exact_G_npz=True,te_empty=True)


def normalized_summary(v,y):
    y=np.asarray(y,float);v=np.asarray(v,float);positive=y>0;zeros=int((~positive).sum())
    subset=float(np.mean(np.log(v[positive])-np.log(y[positive])+y[positive]/v[positive]-1)) if positive.any() else None
    return dict(value=None if zeros else subset,
        status='undefined_contains_zero_targets' if zeros else 'defined_all_targets_positive',
        n=int(y.size),zero_targets=zeros,positive_only_subset=dict(n=int(positive.sum()),
            zero_targets_excluded=zeros,value=subset,usage='descriptive_only; never primary, epoch selection or percentage rescue'))


def safe_metrics(predictions,D):
    """No floor/drop/log of raw zero targets. Primary always uses all n observations."""
    y=np.asarray(D['y'][D['va']],float);ewma=np.asarray(D['ewma'][D['va']],float)
    assert np.isfinite(y).all() and (y>=0).all() and np.isfinite(ewma).all() and (ewma>0).all()
    baseline=np.log(ewma)[:,None]+y/ewma[:,None]
    out={}
    for name,values in predictions.items():
        v=np.asarray(values,float)
        if v.shape!=y.shape or not np.isfinite(v).all() or not (v>0).all():raise ValueError('Invalid positive forecasts')
        raw=np.log(v)+y/v;known=raw-np.log(ewma)[:,None];mse=(v-y)**2
        per={str(h):dict(n=len(y),zero_targets=int((y[:,j]==0).sum()),raw_QLIKE=float(raw[:,j].mean()),
            knownscale_QLIKE=float(known[:,j].mean()),MSE=float(mse[:,j].mean()),
            raw_loss_difference_vs_EWMA22=float((raw[:,j]-baseline[:,j]).mean()),
            normalized_QLIKE=normalized_summary(v[:,j],y[:,j])) for j,h in enumerate(HORIZONS)}
        out[name]=dict(primary_all_targets=dict(n_rows=len(y),n_cells=y.size,
            raw_QLIKE=float(raw.mean()),knownscale_QLIKE=float(known.mean()),MSE=float(mse.mean()),
            zero_targets=int((y==0).sum()),raw_loss_difference_vs_EWMA22=float((raw-baseline).mean())),
            per_h=per,normalized_QLIKE_aggregate=normalized_summary(v,y))
    finite_json(out)
    return out


def state_fingerprint(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def parameter_contract(model,neural):
    assert tuple(model.assets)==ASSETS
    count=sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert count==(43 if neural else 4)
    assert all(not p.requires_grad for p in model.processes['GRN'].parameters()) if neural else True
    theta=model.processes['GRN'].bounded()
    if not all(math.isfinite(v) for v in theta.values()):raise FloatingPointError('Nonfinite theta')
    return dict(trainable_parameters=count,theta=theta,theta_frozen=neural)


def exact_NN0_gate(base_model,base_predictions,datasets,T,fixed,reference,seed):
    theta={a:base_model.processes[a].bounded() for a in ASSETS}
    neural=FAST.FastGammaVarianceFilter(datasets,fixed,reference,'neural_density',seed,
        frozen_theta=theta,assets=ASSETS)
    parameter_contract(neural,True)
    nesting=checked_evaluate(neural,datasets,T)[0]
    for a in ASSETS:np.testing.assert_array_equal(nesting[a],base_predictions[a])
    return neural,dict(seed=seed,exact=True,max_absolute_error=0.,theta_source='independent fixed_mixture selected theta',
                       raw_targets_not_modified=True)


def no_update_init(datasets,T,fixed,reference,real=False):
    """Exactly one init case: base fixed mixture + nested seed11 NN; NO optimizer."""
    stamp=G.module_globals_signature()
    base=FAST.FastGammaVarianceFilter(datasets,fixed,reference,'fixed_mixture',11,assets=ASSETS)
    binfo=parameter_contract(base,False)
    base_before=state_fingerprint(base)
    with torch.no_grad():base_loss,base_info=checked_objective(base,datasets,T)
    base_pred=checked_evaluate(base,datasets,T)[0]
    nn,gate=exact_NN0_gate(base,base_pred,datasets,T,fixed,reference,11)
    # This theta is only the INITIAL fixed-mixture theta, never called selected/fit.
    gate['theta_source']='UNTRAINED fixed_mixture initial theta, structural sanity only'
    before=state_fingerprint(nn);nn.zero_grad(set_to_none=True)
    loss,details=checked_objective(nn,datasets,T);loss.backward()
    gradients={name:dict(norm=float(p.grad.norm()),max_abs=float(p.grad.abs().max()))
        for name,p in nn.named_parameters() if p.grad is not None}
    assert gradients and all(math.isfinite(x['norm']) for x in gradients.values())
    head_weight=float(nn.prior_net[-1].weight.grad.norm());head_bias=float(nn.prior_net[-1].bias.grad.norm())
    assert head_weight>0 and head_bias>0
    assert state_fingerprint(nn)==before and state_fingerprint(base)==base_before
    assert G.module_globals_signature()==stamp
    D=datasets['GRN'];metrics=safe_metrics({'fixed_mixture_INIT':base_pred['GRN']},D)
    result=dict(mode='ACTUAL_GRN_INIT_NO_UPDATE' if real else 'SYNTHETIC_INIT_NO_UPDATE',
        initial_base=binfo,neural=parameter_contract(nn,True),train_objective=float(loss.detach()),
        initial_base_train_objective=float(base_loss),train_knownscale_QLIKE=float(details['data'].detach()),
        train_sensor_NLL=float(details['sensor'].detach()),train_kappa_penalty=float(details['kappa_penalty'].detach()),
        head_gradient_weight_norm=head_weight,head_gradient_bias_norm=head_bias,gradient_norms=gradients,
        NN0_exact_fixed_mixture=gate,optimizer_constructed=False,optimizer_steps=0,weight_or_buffer_updated=False,
        backward_calls=1,seed_cases=1,epoch_selection_performed=False,model_training_performed=False,
        validation_metrics_INIT_ONLY_NOT_SELECTED=metrics,
        model_or_N_prior_or_hurdle_hyperparameters_changed=False)
    return finite_json(result)


def synthetic_data():
    rng=np.random.default_rng(921);n=180;cut=128
    P=np.exp(-7+rng.normal(0,.35,n));P[[12,35,105,140]]=0.;observed=P>0
    log=np.zeros(n);log[observed]=np.log(P[observed])
    F=np.column_stack([np.full(n,-.2),np.full(n,-.4),rng.normal(0,1,n)]);F[~observed,:2]=-1
    origin=np.r_[np.arange(100,cut-22),np.arange(cut,n-22)]
    tr=np.flatnonzero(origin+22<cut);va=np.flatnonzero(origin>=cut)
    y=np.column_stack([[P[t+1:t+h+1].mean() for t in origin] for h in HORIZONS])
    dates=np.r_[pd.bdate_range(end='2024-12-31',periods=cut).to_numpy(),pd.bdate_range('2025-01-02',periods=n-cut).to_numpy()]
    D=dict(n=n,cut=cut,date=dates,origin=origin,tr=tr,va=va,te=np.array([],int),y=y,
        proxy_raw=P,observed=observed,F17=F,bank_index=np.flatnonzero(observed&(np.arange(n)<cut)&(np.arange(n)>=21)),
        even=F.copy(),log_proxy=log,cell_b=np.full(n,-.2),cell_R=np.full(n,.4),ewma=np.full(len(origin),P[:cut].mean()))
    datasets={'GRN':D}
    return datasets,G.make_tensors(datasets),G.fixed_preprocessing(datasets),S21.synthetic_reference()


def purechecks():
    contract=source_contract();data,T,fixed,reference=synthetic_data()
    result=no_update_init(data,T,fixed,reference,real=False)
    for mode in BASES:
        model=FAST.FastGammaVarianceFilter(data,fixed,reference,mode,11,assets=ASSETS)
        parameter_contract(model,False)
    D=data['GRN'];v=np.full_like(D['y'][D['va']],.001)
    metrics=safe_metrics({'test':v},D)['test']
    assert metrics['normalized_QLIKE_aggregate']['value'] is None
    assert metrics['normalized_QLIKE_aggregate']['status']=='undefined_contains_zero_targets'
    assert metrics['primary_all_targets']['n_rows']==len(D['va'])
    expected=np.mean(np.log(v)+D['y'][D['va']]/v)
    assert metrics['primary_all_targets']['raw_QLIKE']==float(expected)
    copy_y=D['y'].copy();safe_metrics({'test':v},D);np.testing.assert_array_equal(D['y'],copy_y)
    return dict(contract=contract,synthetic_init=result,
        checks=dict(three_bases_4_parameters_each=True,NN43_parameters_and_frozen_theta=True,
            NN0_exact_fixed_mixture=True,head_gradient_nonzero=True,all_zero_targets_in_primary=True,
            normalized_QLIKE_zero_group_null=True,no_raw_target_floor_or_drop=True,
            private_training_clone_prepared_but_not_called=True,no_real_data_or_calibration_IO=True,
            no_optimizer_or_weight_updates=True),source_sha256=sha(__file__),created_at_utc=utc())


def actual_init_sanity(tag):
    before=source_hashes();stamp=G.module_globals_signature()
    datasets,T,fixed,audit=load_grn();reference=S21.REF.load_gamma_reference()
    result=no_update_init(datasets,T,fixed,reference,real=True)
    schema=validate_schema(datasets['GRN'])
    metric=result['validation_metrics_INIT_ONLY_NOT_SELECTED']['fixed_mixture_INIT']
    assert metric['per_h']['1']['zero_targets']==33
    assert metric['per_h']['1']['normalized_QLIKE']['value'] is None
    assert metric['normalized_QLIKE_aggregate']['value'] is None
    assert G.module_globals_signature()==stamp;unchanged(before)
    result.update(created_at_utc=utc(),source_hashes=before,HP=HP,F3_schema=schema,
        GRN_fixed_preprocessing=fixed,normalization=audit['normalization'],
        heldout2026_opened=False,formal_development_training_started=False,
        future_protocol='Train and select on pre2026 first; freeze separate confirmation protocol before any GRN2026 opening.')
    path=HERE/(tag+'_actual_init_sanity.json');save_json(path,result)
    print(json.dumps(dict(report=str(path),sha256=sha(path),source_sha256=sha(__file__),
        mode=result['mode'],theta=result['initial_base']['theta'],train_objective=result['train_objective'],
        head_gradient_weight_norm=result['head_gradient_weight_norm'],head_gradient_bias_norm=result['head_gradient_bias_norm'],
        NN0_exact=result['NN0_exact_fixed_mixture']['exact'],F3_schema=schema,
        optimizer_steps=0,model_training_performed=False),ensure_ascii=False,indent=2),flush=True)


def run_development(tag):
    """Prepared for later main-thread execution. NEVER called in this delivery."""
    before=source_hashes();stamp=G.module_globals_signature();datasets,T,fixed,audit=load_grn()
    reference=S21.REF.load_gamma_reference();train=private_train_fit()
    paths={m:HERE/(tag+'_'+m+'_GRN.pt') for m in BASES}
    paths.update({f'neural_density_seed{s}':HERE/(tag+f'_neural_density_seed{s}.pt') for s in SEEDS})
    output=HERE/(tag+'_validation_predictions.npz');report=HERE/(tag+'_results.json')
    for path in [*paths.values(),output,report]:
        if path.exists():raise FileExistsError('Exclusive GRN output exists: '+str(path))
    pred={};fits={};checkpoints={};nesting={};arrays={'horizons':np.array(HORIZONS),'assets':np.array(ASSETS)}
    base_model=None;base_predictions=None
    def persist(key,model,info):
        count=parameter_contract(model,key.startswith('neural'))['trainable_parameters']
        assert info['trainable_parameters']==count
        metadata=finite_json(dict(mode=key,fit=info,GRN_preprocessing=fixed,source_hashes=before,
            assets=list(ASSETS),HP=HP,training_scope='GRN pre2025 only; selected on2025; GRN2026 closed',
            score='all-target raw/knownscale QLIKE; y0 included; no normalizedQLIKE selection'))
        with paths[key].open('xb') as f:torch.save(dict(state_dict=model.state_dict(),metadata=metadata),f)
        checkpoints[key]=dict(path=str(paths[key]),sha256=sha(paths[key]),selected_epoch=info['best_joint_epoch'])
    def add_aux(key,aux):
        for field,value in aux['GRN'].items():
            array=np.asarray(value)
            if not np.isfinite(array).all():raise FloatingPointError('Nonfinite diagnostic output: '+field)
            arrays['GRN_'+key+'_'+field]=array
    for mode in BASES:
        model,prediction,aux,info=train(datasets,fixed,reference,mode,11,assets=ASSETS)
        pred[mode]=prediction['GRN'];fits[mode]=info;persist(mode,model,info);add_aux(mode,aux)
        if mode=='fixed_mixture':base_model=model;base_predictions=prediction
        print('GRN_BASE_COMPLETE',mode,'epoch',info['best_joint_epoch'],flush=True)
    theta={a:base_model.processes[a].bounded() for a in ASSETS}
    for seed in SEEDS:
        key=f'neural_density_seed{seed}'
        zero,gate=exact_NN0_gate(base_model,base_predictions,datasets,T,fixed,reference,seed)
        nesting[key]=gate;del zero
        # Explicit GRN assets prevents the old default tuple captured at definition time.
        model,prediction,aux,info=train(datasets,fixed,reference,'neural_density',seed,
            frozen_theta=theta,assets=ASSETS)
        pred[key]=prediction['GRN'];fits[key]=info;persist(key,model,info);add_aux(key,aux)
        assert model.processes['GRN'].bounded()==theta['GRN']
        print('GRN_NN_COMPLETE',seed,'epoch',info['best_joint_epoch'],flush=True)
    pred['neural_density_ensemble']=np.mean([pred[f'neural_density_seed{s}'] for s in SEEDS],axis=0)
    D=datasets['GRN'];pred['EWMA22']=np.repeat(D['ewma'][D['va'],None],3,axis=1);validate_schema(D)
    q=D['origin'][D['va']]
    arrays.update(GRN_date=D['date'][q],GRN_origin=q,GRN_label_end=D['date'][q+22],
        GRN_target=D['y'][D['va']],GRN_ewma=D['ewma'][D['va']])
    arrays.update({'GRN_'+name:value for name,value in pred.items()})
    unchanged(before);assert G.module_globals_signature()==stamp
    with output.open('xb') as f:np.savez_compressed(f,**arrays)
    # Scores are produced after immutable predictions; no target edits and no log(y0).
    scores=safe_metrics(pred,D)
    result=dict(status='GRN INDEPENDENT REUSED2025 DEVELOPMENT; not fresh external confirmation',
        created_at_utc=utc(),source_hashes=before,HP=HP,assets=list(ASSETS),builder_audit_sha256=AUDIT_SHA,
        fixed_preprocessing=fixed,normalization=audit['normalization'],F3_schema=validate_schema(D),
        fits=fits,checkpoints=checkpoints,NN0_nesting_gate=nesting,
        predictions=dict(path=str(output),sha256=sha(output)),scores=scores,
        primary='raw/knownscale QLIKE on all228 targets per horizon including33 h1 zeros; MSE on same rows',
        normalized_QLIKE_policy='null if any zeros; positive subset descriptive only, not selection or percentage',
        target_floor_or_drop_performed=False,heldout2026_opened=False,
        hypergrid_or_new_hurdle_or_N_prior_added=False,GRN_transfer_from_ETF_performed=False,
        future_confirmation='GRN2026 remains unopened; freeze independent protocol after2025 selection before opening.')
    unchanged(before);save_json(report,result)
    print(json.dumps(dict(results=str(report),sha256=sha(report),predictions=str(output),
        source_sha256=sha(__file__),GRN2026_opened=False),indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--check-only',action='store_true');modes.add_argument('--actual-init-sanity',action='store_true')
    modes.add_argument('--run-development',action='store_true')
    parser.add_argument('--tag',default='grn_gamma_development');args=parser.parse_args()
    if not re.fullmatch(r'grn_gamma_development[A-Za-z0-9_-]*',args.tag):raise ValueError('Independent GRN output tag only')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if args.run_development:run_development(args.tag)
    elif args.actual_init_sanity:actual_init_sanity(args.tag)
    else:
        result=purechecks();path=HERE/(args.tag+'_purechecks.json');save_json(path,result)
        print(json.dumps(dict(report=str(path),sha256=sha(path),source_sha256=sha(__file__),checks=result['checks']),indent=2),flush=True)


if __name__=='__main__':main()
