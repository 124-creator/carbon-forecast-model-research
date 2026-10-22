"""Frozen R21 inference only. DEFAULT: --development-replay, no2026 processing.

Separate --used-2026-diagnostic requires an independent explicit manifest; the
period was already used and is NOT independent confirmation. Proposed series is
fixed R21 neural-density ensemble, never selected/replaced using that period.
--record-prospective is a separate real-clock-gated, label-free point logger over
the already accepted 2026-10-21 snapshots. Neither optional mode runs by default.
No fitting, optimizer, preprocessing estimation, epoch selection or old rewrite.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
import round21_gamma_observation_filter as S21
from round21_fast_filter import FastGammaVarianceFilter
import round21_fast_filter as FAST
import round20_holdout_runner as RR
import prospective_forecast_once as P

HB,R20,ASSETS,SEEDS,DTYPE=S21.HB,S21.R20,S21.ASSETS,S21.SEEDS,S21.DTYPE
HERE=Path(__file__).resolve().parent
RESULTS=HERE/'round21_gamma_development_results.json'
SAVED_PREDICTIONS=HERE/'round21_gamma_development_validation_predictions.npz'
PROPOSED='R21_neural_density_ensemble'
BASE_NAMES=tuple('R21_'+m for m in S21.BASES)
R21_NAMES=BASE_NAMES+tuple(f'R21_neural_density_seed{s}' for s in SEEDS)+(PROPOSED,)
R20_NAMES=tuple(RR.ENSEMBLE_NAMES)+tuple(f'R20_{v}_seed{s}' for v in R20.VARIANTS for s in SEEDS)+tuple(f'C18_GRU_seed{s}' for s in SEEDS)
DEV_REPORT=HERE/'round21_frozen_diagnostic_development_replay.json'
DEV_PREDICTIONS=HERE/'round21_frozen_diagnostic_development_predictions.npz'
USED_REPORT=HERE/'round21_frozen_diagnostic_used2026_report.json'
USED_PREDICTIONS=HERE/'round21_frozen_diagnostic_used2026_predictions.npz'
ACCEPTED_PREFIX='prospective_forecast_once_origin20261021'
ACCEPTED_JSON=HERE/(ACCEPTED_PREFIX+'.json')
ACCEPTED_COMMIT=HERE/(ACCEPTED_PREFIX+'.commit.json')
ACCEPTED_JSON_SHA='222289dbb2028e790cecbd4fd7a9a34194e47aa806c70c5a539e0a1f5471fcdd'
ACCEPTED_COMMIT_SHA='1e6a7be9cf323a2db48bf9b7d8ee043bcf5b1dda92ea25b536fa6ac49112d53b'
NEW_PROSPECTIVE_PREFIX='round21_frozen_diagnostic_origin20261021'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def source_paths(scope='development'):
    sources=dict(HB.source_paths())
    sources.update(runner=Path(__file__),R21=Path(S21.__file__),fast_filter=Path(FAST.__file__),
        R21_results=RESULTS,R21_predictions=SAVED_PREDICTIONS,
        R20=Path(R20.__file__),metrics_runner=Path(RR.__file__),prospective_writer=Path(P.__file__),
        Gamma_reference=Path(S21.REF.__file__),Gamma_JSON=HERE/'sampling_density_gamma_calibration.json',
        Gamma_NPZ=HERE/'sampling_density_gamma_calibration.npz')
    for mode in S21.BASES:
        for a in ASSETS:sources[f'checkpoint_{mode}_{a}']=HERE/f'round21_gamma_development_{mode}_{a}.pt'
    for seed in SEEDS:sources[f'checkpoint_neural_density_seed{seed}']=HERE/f'round21_gamma_development_neural_density_seed{seed}.pt'
    if scope=='used2026':
        sources.update(R20_used_period_report=RR.TEST_REPORT,R20_used_period_predictions=RR.TEST_PREDICTIONS,
            R21_development_replay=DEV_REPORT,R21_development_predictions=DEV_PREDICTIONS)
    elif scope=='prospective':
        sources.update(accepted_input_entry=ACCEPTED_JSON,accepted_input_commit=ACCEPTED_COMMIT,
            accepted_input_manifest=P.INPUT_MANIFEST,
            **{f'accepted_snapshot_{a}':HERE/f'{ACCEPTED_PREFIX}_{a}_input_snapshot.csv' for a in ASSETS})
    elif scope!='development':raise ValueError(scope)
    return {k:Path(v).resolve() for k,v in sources.items()}


def hashes(scope='development'):
    return {k:dict(path=str(v),sha256=digest(v)) for k,v in source_paths(scope).items()}


def unchanged(before):
    for k,v in before.items():
        if digest(v['path'])!=v['sha256']:raise PermissionError('Frozen input changed: '+k)


def epoch_bindings(archive):
    out={f'{mode}_{a}':archive['fits'][mode][a]['best_joint_epoch'] for mode in S21.BASES for a in ASSETS}
    out.update({f'neural_density_seed{r["seed"]}':r['best_joint_epoch'] for r in archive['fits']['neural_density']})
    return out


def observation_only(D,stop=None):
    stop=D['n'] if stop is None else stop
    out={k:D[k][:stop].copy() for k in ('date','observed','F17','proxy_raw')}
    out.update(n=stop,cut=D['cut'],bank_index=D['bank_index'].copy())
    assert stop>=out['cut'] and out['bank_index'].max()<out['cut']
    assert not {'y','origin','tr','va','te','label_dates'}&set(out)
    return out


def inference_tensors(datasets):
    return {a:{k:torch.tensor(D[k],dtype=DTYPE) for k in ('F17','proxy_raw')} for a,D in datasets.items()}


def state_digest(model):
    h=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        h.update(key.encode());h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def frozen_context(model,datasets,fixed):
    for a in model.assets:
        D=datasets[a];bank=D['bank_index']
        if D['cut']!=fixed[a]['scale_n']:raise PermissionError('Original pre25 scale window changed')
        for key,value in ((f'bank_index_{a}',torch.tensor(bank,dtype=torch.long)),
                          (f'bank_v_{a}',torch.tensor(D['F17'][bank,2],dtype=DTYPE)),
                          (f'scale_{a}',torch.tensor(fixed[a]['scale'],dtype=DTYPE)),
                          (f'hurdle_{a}',torch.tensor([fixed[a]['hurdle'][k] for k in ('intercept','slope')],dtype=DTYPE)),
                          (f'ar_{a}',torch.tensor([fixed[a]['volume_AR'][k] for k in ('mu','rho','residual_std')],dtype=DTYPE))):
            if not torch.equal(getattr(model,key),value):raise PermissionError('Frozen preprocessing buffer changed: '+key)
    assert not any(p.requires_grad for p in model.parameters())


def load_models(datasets,archive):
    fixed=archive['preprocessing'];reference=S21.REF.load_gamma_reference()
    modes={};binding=epoch_bindings(archive);paths=source_paths()
    for key,row in archive['checkpoints'].items():
        expected=paths['checkpoint_'+key]
        if Path(row['path']).resolve()!=expected or digest(expected)!=row['sha256']:
            raise PermissionError('Checkpoint path/hash changed: '+key)
        own=torch.load(expected,weights_only=True,map_location='cpu');meta=own['metadata'];fit=meta['fit']
        if meta['preprocessing']!=fixed or fit['best_joint_epoch']!=binding[key]:
            raise PermissionError('Checkpoint fit/preprocessing metadata changed')
        if meta['source_hashes']!=archive['source_hashes']:
            raise PermissionError('Checkpoint/source archive provenance differs')
        mode=fit['mode'];assets=tuple(fit['theta'])
        if mode=='neural_density':
            seed=fit['seed'];frozen=archive['fixed_mixture_theta_for_NN'];name=f'R21_neural_density_seed{seed}'
        else:
            seed=11;frozen=None;name='R21_'+mode
        model=FastGammaVarianceFilter(datasets,fixed,reference,mode,seed,frozen,assets)
        # Verify constructed bank/AR/hurdle/reference against saved buffers BEFORE
        # loading; never hide a context mismatch by overwriting it with weights.
        initial=dict(model.named_buffers())
        for field,value in initial.items():
            if field.startswith('processes.'):continue
            if not torch.equal(value,own['state_dict'][field]):
                raise PermissionError('Constructed context differs from checkpoint: '+key+'/'+field)
        model.load_state_dict(own['state_dict'],strict=True);model.eval();model.requires_grad_(False)
        frozen_context(model,datasets,fixed)
        if {a:model.processes[a].bounded() for a in assets}!=fit['theta']:
            raise PermissionError('Learned bounded theta differs from saved fit')
        modes[key]=dict(model=model,name=name,checkpoint=dict(path=str(expected),sha256=row['sha256']),
            selected_epoch=binding[key],mode=mode,seed=seed)
    if len(modes)!=9:raise PermissionError('Require six base and three NN checkpoints')
    return modes


@torch.no_grad()
def predict_models(state,datasets,queries):
    # Inference never receives y, future target dates or a fabricated query row.
    Q={a:observation_only(D) for a,D in datasets.items()};T=inference_tensors(Q)
    predictions={a:{} for a in ASSETS};diagnostics={}
    for key,entry in state['models'].items():
        model=entry['model'];before=state_digest(model);frozen_context(model,Q,state['archive']['preprocessing'])
        diagnostics[key]={}
        for a in model.assets:
            q=np.asarray(queries[a],dtype=int)
            assert len(q) and (q>=0).all() and (q<Q[a]['n']).all()
            f=model.filter_asset(a,Q[a],T[a],Q[a]['n'])
            p,daily,factor=model.forecast(a,T[a],q,f)
            assert torch.isfinite(p).all() and (p>0).all()
            predictions[a][entry['name']]=p.numpy()
            diagnostics[key][a]=dict(c_ref=float(f['c_ref']),max_boundary_mass=float(f['boundary_mass'].max()),
                max_posterior_weighted_clip_mass=float(f['posterior_weighted_clip_mass'].max()),
                max_abs_posterior_weighted_mean_error=float(f['posterior_weighted_mean_error'].abs().max()))
        if state_digest(model)!=before:raise PermissionError('Inference changed checkpoint state')
    for a in ASSETS:
        predictions[a][PROPOSED]=np.mean([predictions[a][f'R21_neural_density_seed{s}'] for s in SEEDS],axis=0)
        assert set(predictions[a])==set(R21_NAMES)
    return predictions,diagnostics


def development_gate():
    before=hashes();archive=read_json(RESULTS)
    unchanged(archive['source_hashes'])
    datasets,_,audit=HB.build_pair(development_only=True,check_npz=True)
    assert all(not len(D['te']) and (D['date']<HB.TEST_START).all() for D in datasets.values())
    if audit['preprocessing_sha256']!=RR.PREPROCESSING_SHA256 or audit['preprocessing']!=archive['builder']['preprocessing']:
        raise PermissionError('Original normalizer/bank preprocessing changed')
    state=dict(dev=datasets,archive=archive,models=load_models(datasets,archive),hashes=before,audit=audit)
    queries={a:D['origin'][D['va']] for a,D in datasets.items()}
    pred,diagnostics=predict_models(state,datasets,queries);gates={}
    with np.load(SAVED_PREDICTIONS,allow_pickle=False) as saved:
        for a in ASSETS:
            RR.archive_dates(datasets[a],saved,a+'_')
            for name,value in pred[a].items():
                gates.setdefault(name,{})[a]=RR.assert_matching(value,saved[a+'_'+name.removeprefix('R21_')])
    unchanged(before)
    state.update(predictions=pred,gates=gates,diagnostics=diagnostics)
    return state


def write_array_predictions(path,pred,datasets,split):
    arrays=dict(horizons=np.array(S21.HORIZONS))
    for a in ASSETS:
        D=datasets[a];ids=D[split];q=D['origin'][ids]
        arrays.update({a+'_date':D['date'][q],a+'_origin':q,a+'_label_end':D['date'][q+22],
                       a+'_target':D['y'][ids],a+'_ewma':D['ewma'][ids]})
        arrays.update({a+'_'+name:value for name,value in pred[a].items()})
    with Path(path).open('xb') as handle:np.savez_compressed(handle,**arrays)


def run_development():
    P.reject_existing([DEV_REPORT,DEV_PREDICTIONS]);state=development_gate()
    write_array_predictions(DEV_PREDICTIONS,state['predictions'],state['dev'],'va')
    # No scoring required to prove exact replay; freeze the already proposed series.
    report=dict(status='R21 FROZEN 2025 DEVELOPMENT REPLAY ONLY',proposed_series=PROPOSED,
        source_hashes=state['hashes'],epochs=epoch_bindings(state['archive']),
        gates=state['gates'],diagnostics=state['diagnostics'],preprocessing_sha256=state['audit']['preprocessing_sha256'],
        prediction_file=dict(path=str(DEV_PREDICTIONS),sha256=digest(DEV_PREDICTIONS)),
        all_checkpoint_state_frozen=True,training_or_preprocessing_fit=False,selection_created=False,
        real2026_inputs_processed=False,used2026_diagnostic_executed=False,prospective_record_executed=False,
        label_free_filter_forecast=True,fast_source_recorded=True)
    unchanged(state['hashes'])
    with DEV_REPORT.open('x',encoding='utf-8') as handle:json.dump(report,handle,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(status=report['status'],source_sha256=digest(__file__),
        max_absolute_replay_error=max(row['max_absolute_error'] for g in state['gates'].values() for row in g.values()),
        development_report_sha256=digest(DEV_REPORT),predictions_sha256=digest(DEV_PREDICTIONS)),indent=2),flush=True)


def validate_used_flags(payload):
    for key in ('allow_test','used_period_diagnostic','not_independent_confirmation'):
        if payload.get(key) is not True:raise PermissionError('Independent diagnostic manifest requires '+key+'=true')
    if payload.get('proposed_series')!=PROPOSED:raise PermissionError('Proposed series is fixed before used-period access')
    if payload.get('preprocessing_sha256')!=RR.PREPROCESSING_SHA256:raise PermissionError('Preprocessing binding changed')
    if not isinstance(payload.get('checkpoint_epochs'),dict):raise PermissionError('Freeze all checkpoint epochs')
    return payload


def authorize_used_period(path):
    # Flags are validated before any source/archive/CSV/checkpoint I/O.
    if path is None:raise PermissionError('Explicit independent used-period manifest required')
    payload=validate_used_flags(read_json(path))
    # A PRIVATE builder compatibility label. No R20 selection is read/changed,
    # no new model selection occurs, and this script creates no manifest.
    auth=dict(payload,selected_candidate=PROPOSED)
    HB.validate_selection_payload(auth,source_paths('used2026'))
    return auth


def run_used_period(path):
    auth=authorize_used_period(path);manifest_hash=digest(path)
    P.reject_existing([USED_REPORT,USED_PREDICTIONS])
    before=hashes('used2026');state=development_gate()
    if auth['checkpoint_epochs']!=epoch_bindings(state['archive']):raise PermissionError('Frozen epochs differ')
    # Reuse unchanged original build bytecode under PRIVATE authorization globals.
    # Scope is the already used period. No R20 selection file or module mutation.
    space=dict(vars(HB));space['authorize_test']=lambda supplied:auth
    build=HB.clone_function(HB.build_pair,space)
    datasets,_,audit=build(development_only=False,selection_path=path,check_npz=True)
    queries={a:D['origin'][D['te']] for a,D in datasets.items()}
    pred,diagnostics=predict_models(state,datasets,queries)
    with np.load(RR.TEST_PREDICTIONS,allow_pickle=False) as old:
        np.testing.assert_array_equal(old['horizons'],np.array(S21.HORIZONS))
        for a in ASSETS:
            D=datasets[a];ids=D['te'];q=D['origin'][ids]
            if len(q)!=179:raise AssertionError('Expected exactly the same 179 old-period origins')
            for key,value in (('date',D['date'][q]),('origin',q),('label_end',D['date'][q+22]),('target',D['y'][ids]),('ewma',D['ewma'][ids])):
                np.testing.assert_array_equal(old[a+'_'+key],value)
            for name in R20_NAMES:pred[a][name]=old[a+'_'+name].copy()
    if digest(path)!=manifest_hash:raise PermissionError('Diagnostic manifest changed during inference')
    unchanged(before);write_array_predictions(USED_PREDICTIONS,pred,datasets,'te')
    comparisons=list(R20_NAMES)+list(BASE_NAMES)
    score=RR.score_predictions(pred,datasets,'te',dict(selected_candidate=PROPOSED,comparisons=comparisons,
        criteria=dict(descriptive_only=True,used_period_diagnostic=True,not_independent_confirmation=True)))
    unchanged(before)
    report=dict(status='USED 2026 DIAGNOSTIC; NOT INDEPENDENT CONFIRMATION',proposed_series=PROPOSED,
        used_period_diagnostic=True,not_independent_confirmation=True,fresh_confirmation=False,
        source_hashes=before,manifest=dict(path=str(Path(path).resolve()),sha256=manifest_hash),
        development_gates=state['gates'],scores=score,diagnostics=diagnostics,builder=audit,
        retained_strong_MEM=['MEM_stationary','MEM_integrated'],all_old29_series_retained=True,
        targets_dates_labelends_exactly_old179=True,model_or_epoch_selection=False,
        optimizer_or_preprocessing_fit=False,scored_only_after_exclusive_prediction_write=True,
        predictions=dict(path=str(USED_PREDICTIONS),sha256=digest(USED_PREDICTIONS)),
        limitation='R21 was proposed after2026 had already been used; this is descriptive reuse. Future confirmation requires another sample.')
    with USED_REPORT.open('x',encoding='utf-8') as handle:json.dump(report,handle,ensure_ascii=False,indent=2,allow_nan=False)


def validate_accepted_metadata(entry,receipt,now):
    P.checked_time(now)
    if not receipt.get('accepted') or receipt.get('mode')!='PROSPECTIVE' or not receipt.get('is_forward'):
        raise PermissionError('Original snapshot receipt was not accepted prospective')
    created=datetime.fromisoformat(entry['created_at_utc']);completed=datetime.fromisoformat(receipt['completed_at_utc'])
    if not created<=completed<P.DEADLINE or completed>now:raise PermissionError('Invalid original receipt chronology')
    if entry.get('origin')!={a:P.LIVE_ORIGIN for a in ASSETS}:raise PermissionError('Only accepted10/21 input origin allowed')
    if receipt.get('json_sha256')!=ACCEPTED_JSON_SHA:raise PermissionError('Accepted JSON receipt binding changed')
    return datetime.fromisoformat(entry['known_input_asof_utc'])


def accepted_snapshot_frames(now):
    # Do not open the old prediction NPZ or use any R20 forecast values.
    if digest(P.INPUT_MANIFEST)!=P.INPUT_MANIFEST_SHA:
        raise PermissionError('Accepted acquisition manifest changed')
    if digest(ACCEPTED_JSON)!=ACCEPTED_JSON_SHA or digest(ACCEPTED_COMMIT)!=ACCEPTED_COMMIT_SHA:
        raise PermissionError('Accepted input metadata/receipt bytes changed')
    entry=read_json(ACCEPTED_JSON);receipt=read_json(ACCEPTED_COMMIT)
    fetched=validate_accepted_metadata(entry,receipt,now);P.checked_time(now,[fetched])
    P.assert_sources_unchanged(entry['source_hashes'])
    frames={};snapshots={}
    for a in ASSETS:
        meta=entry['input_snapshots'][a];expected=source_paths('prospective')['accepted_snapshot_'+a]
        if Path(meta['path']).resolve()!=expected or digest(expected)!=meta['sha256']:
            raise PermissionError('Accepted snapshot path/hash changed')
        frame=pd.read_csv(expected,parse_dates=['date'],float_precision='round_trip')
        if tuple(frame.columns)!=P.COLUMNS or len(frame)!=meta['rows']:raise PermissionError('Snapshot schema/rows changed')
        if not frame.date.is_unique or not frame.date.is_monotonic_increasing:raise PermissionError('Invalid snapshot date order')
        if str(frame.date.iloc[-1].date())!=P.LIVE_ORIGIN:raise PermissionError('Snapshot extends beyond accepted origin')
        HB.observations(frame);frames[a]=frame;snapshots[a]=meta
    return frames,snapshots,[fetched]


def assemble_snapshots(frames,state):
    # Same private-frame preparation used by prospective writer/HB: original
    # normalizers and bank, no globals modified and no future query labels.
    model=read_json(S21.B.REFERENCE)['grid512'];reference=read_json(S21.S17.PROXY.REFERENCE_RESULTS)
    bundle=dict(model=model,c=float(reference['resolution_sensitivity']['mean_P_by_train_resolution']['512']))
    datasets,audit=HB.assemble_pair(frames,bundle,check_npz=True,expected_preprocessing=RR.PREPROCESSING_SHA256)
    for a,D in datasets.items():
        if str(pd.Timestamp(D['date'][-1]).date())!=P.LIVE_ORIGIN:raise PermissionError('Last known bar changed')
        assert D['n']-1 not in D['origin']
        assert (D['origin']+22<D['n']).all() and (D['label_dates']<=D['date'][-1]).all()
        for key in ('F17','proxy_raw','observed'):
            np.testing.assert_array_equal(D[key][:state['dev'][a]['n']],state['dev'][a][key])
    return datasets,audit


def record_prospective():
    started=P.checked_time(P.utc_now())  # FIRST operation; no expired/backdated I/O
    prefix=NEW_PROSPECTIVE_PREFIX;claim=HERE/(prefix+'.claim.json')
    P.reject_existing([claim]+[HERE/(prefix+s) for s in ('.json','.npz','.commit.json')])
    before=hashes('prospective');state=development_gate();P.checked_time(P.utc_now())
    frames,snapshots,fetched=accepted_snapshot_frames(P.utc_now())
    datasets,audit=assemble_snapshots(frames,state)
    queries={a:np.array([D['n']-1]) for a,D in datasets.items()}
    point,diagnostics=predict_models(state,datasets,queries)
    predictions={a:{name:P.positive(value[0]) for name,value in models.items()} for a,models in point.items()}
    created=P.checked_time(P.utc_now(),fetched)
    if created<started:raise PermissionError('Host clock rolled backwards')
    unchanged(before)
    with claim.open('x',encoding='utf-8') as handle:
        json.dump(dict(started_at_utc=P.iso(started),claimed_at_utc=P.iso(P.utc_now()),script_sha256=digest(__file__)),handle,indent=2)
    entry=P.build_entry(predictions,{a:P.LIVE_ORIGIN for a in ASSETS},'PROSPECTIVE',created,before,state['gates'],
        dict(proposed_series=PROPOSED,not_independent_confirmation=True,used_period_diagnostic=False,
            input_snapshots=snapshots,accepted_input_json_sha256=ACCEPTED_JSON_SHA,
            accepted_input_commit_sha256=ACCEPTED_COMMIT_SHA,known_input_asof_utc=P.iso(max(fetched)),
            query_observation_index={a:int(q[0]) for a,q in queries.items()},
            preprocessing_sha256=RR.PREPROCESSING_SHA256,checkpoint_epochs=epoch_bindings(state['archive']),
            gamma_diagnostics=diagnostics,private_frame_assembly=audit,
            old_R20_prediction_arrays_opened=False,old_prediction_files_modified=False,
            prior_accepted_baselines='existing R20/MEM record remains immutable; forecasts not copied or re-read',
            future_target=None,future_score=None,record_scope='new R21 series only; separate immutable actuals later'))
    entry['script_sha256']=digest(__file__);entry['receipt_writer_sha256']=digest(P.__file__)
    unchanged(before)
    return P.write_outputs(prefix,entry,predictions,prospective=True,fetch_times=fetched)


def _purechecks():
    for payload in ({},dict(allow_test=True,used_period_diagnostic=False,not_independent_confirmation=True)):
        try:validate_used_flags(payload)
        except PermissionError:pass
        else:raise AssertionError('Used period flags bypassed')
    touched=[]
    with patch('pandas.read_csv',side_effect=lambda *a,**k:touched.append('csv')):
        try:run_used_period(None)
        except PermissionError:pass
        else:raise AssertionError('Used diagnostic ran without its own manifest')
    assert not touched
    with patch.object(P,'utc_now',return_value=P.DEADLINE):
        try:record_prospective()
        except PermissionError:pass
        else:raise AssertionError('Deadline equality accepted')
    D=S21.synthetic_pair();fixed=S21.preprocessing(D);reference=S21.synthetic_reference()
    theta={a:S21.VarianceProcess().bounded() for a in ASSETS}
    model=FastGammaVarianceFilter(D,fixed,reference,'neural_density',11,theta);model.eval();model.requires_grad_(False)
    before=state_digest(model)
    label_free={a:observation_only(d) for a,d in D.items()};T=inference_tensors(label_free)
    with torch.no_grad():
        for a in ASSETS:
            q=np.array([label_free[a]['n']-1]);f=model.filter_asset(a,label_free[a],T[a],label_free[a]['n'])
            p,_,_=model.forecast(a,T[a],q,f);P.positive(p.numpy()[0])
    assert state_digest(model)==before and not any('y' in t for t in T.values())
    broken=copy.deepcopy(label_free);broken[ASSETS[0]]['F17'][D[ASSETS[0]]['bank_index'][0],2]+=.1
    try:frozen_context(model,broken,fixed)
    except PermissionError:pass
    else:raise AssertionError('Altered bank was accepted')
    return dict(independent_used_manifest_flags_required=True,no_used_period_data_IO_without_manifest=True,
        prospective_deadline_equality_rejected_before_IO=True,last_observation_query_without_y_or_target_row=True,
        positive_points=True,all_model_state_frozen=True,altered_bank_rejected=True,
        proposed_series_fixed_not_ranked=True,real2026_or_prospective_executed=False)


def purechecks():
    with ExitStack() as stack:
        for name in ('pandas.read_csv','numpy.load','pathlib.Path.read_text','torch.load',
                     'sampling_density_gamma_reference.load_gamma_reference','holdout_2026_range_builder.build_pair'):
            stack.enter_context(patch(name,side_effect=AssertionError('Real I/O forbidden in purechecks')))
        return _purechecks()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    modes=ap.add_mutually_exclusive_group();modes.add_argument('--development-replay',action='store_true')
    modes.add_argument('--check-only',action='store_true');modes.add_argument('--used-2026-diagnostic',action='store_true')
    modes.add_argument('--record-prospective',action='store_true');ap.add_argument('--manifest',type=Path)
    args=ap.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if args.check_only:print(json.dumps(dict(checks=purechecks(),source_sha256=digest(__file__)),indent=2),flush=True)
    elif args.used_2026_diagnostic:run_used_period(args.manifest)
    elif args.record_prospective:
        if args.manifest is not None:raise ValueError('Used-period manifest does not authorize prospective recording')
        print(json.dumps(record_prospective(),indent=2),flush=True)
    else:
        if args.manifest is not None:raise ValueError('Default development-only mode consumes no manifest')
        run_development()


if __name__=='__main__':main()
