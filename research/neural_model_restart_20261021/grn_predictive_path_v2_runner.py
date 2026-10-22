"""GRN R22v2 frozen-Gamma reference transport: code/pure/zero-head handoff.

References gamma and ewma_anchor follow the SAME rule for all observations.
EWMA changes BOTH point-mean and activity references, not one-factor ablation.
Latent Gamma posterior/T/uncertainty/ctx and original observation bank stay fixed.
Point forecasts are reference-mean anchor transports, not identified IV.
Default purechecks. --actual-zerohead is frozen pre2026 inference, no optimizer.
--run-development is a separate later-authorized action. No GRN2026 route.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
import argparse
import copy
import importlib
import json
from pathlib import Path
import time
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
import grn_gamma_development_runner as GG
import round22_predictive_path as V1
import round22_predictive_path_v2 as V2

G,S21,HB,DTYPE=GG.G,GG.S21,V1.HB,V1.DTYPE
HERE=Path(__file__).resolve().parent
ASSETS=('GRN',)
REFERENCES=('gamma','ewma_anchor')
MODES,SEEDS=V1.MODES,V1.SEEDS
CHECKPOINT=HERE/'grn_gamma_development_global_dense_GRN.pt'
GAMMA_RESULTS=HERE/'grn_gamma_development_results.json'
GAMMA_PREDICTIONS=HERE/'grn_gamma_development_validation_predictions.npz'
STRONG_PREDICTIONS=HERE/'grn_strong_baselines_development_validation_predictions.npz'
ACTUAL_REPORT=HERE/'grn_predictive_path_v2_actual_zerohead.json'
ACTUAL_ARRAYS=HERE/'grn_predictive_path_v2_actual_zerohead.npz'
TRANSPORT_NOTE='Conditional reported-range mean anchor transport; risk factor/emission scale not separately identified IV'


def ewma_point_all(D):
    """Exact HB.contexts/R14 order, including float32->float64 widening.

    range_context_raw is frozen literal log(high/low)^2/(4log2), which need
    not be bitwise proxy_raw. No new CSV, proxy substitution or future filling.
    """
    literal=pd.Series(np.asarray(D['range_context_raw'],float)).ewm(span=22,adjust=False).mean().clip(lower=HB.EPS)
    recovered=literal.to_numpy().astype(np.float32).astype(float)
    np.testing.assert_array_equal(recovered,D['ewma22_all'])
    np.testing.assert_array_equal(recovered[D['origin']],D['ewma'])
    assert np.isfinite(recovered).all() and (recovered>0).all()
    return np.asarray(D['ewma22_all'],float).copy()


def ewma_active_all(D):
    activity=pd.Series((np.asarray(D['proxy_raw'])>0).astype(float)).ewm(span=22,adjust=False).mean().to_numpy()
    return np.clip(activity,1e-6,1-1e-6)


@torch.no_grad()
def markov_mean(posterior,T,grid):
    # Same multiply/sum reduction as pure core baseline_markov_marginals.
    means=[];mass=posterior
    for _ in range(22):mass=mass@T;means.append((mass*grid).sum(-1))
    return torch.stack(means,1)


@torch.no_grad()
def reference_bundles(D,gamma_cache):
    point=ewma_point_all(D);activity=ewma_active_all(D);result={}
    for code,name in enumerate(REFERENCES):
        cache={k:v.detach().clone() for k,v in gamma_cache.items()}
        if name=='ewma_anchor':
            cache['base_daily']=torch.tensor(np.repeat(point[:,None],22,1),dtype=DTYPE)
            cache['base_active']=torch.tensor(np.repeat(activity[:,None],22,1),dtype=DTYPE)
        # These are deterministic reference metadata/caches, not learned state.
        baseES=markov_mean(cache['posterior'],cache['transition'],cache['grid'])
        cache['anchor_markov_mean']=baseES
        cache['anchor_emission_scale']=cache['base_daily']/(baseES*cache['base_active'])
        cache['reference_code']=torch.tensor(code,dtype=torch.long)
        for key in gamma_cache:
            if name=='ewma_anchor' and key in ('base_daily','base_active'):continue
            assert torch.equal(cache[key],gamma_cache[key]),'Frozen latent cache changed: '+key
        result[name]={'GRN':cache}
    return result


class GRNPredictivePath(V2.PredictivePathV2):
    def __init__(self,caches,mode,seed,assets=ASSETS,kernel=None):
        if tuple(caches)!=ASSETS or tuple(assets)!=ASSETS:raise ValueError('Explicit genuine GRN asset required')
        super().__init__(caches,mode,seed,assets=ASSETS,kernel=kernel)
        self.reference_name=REFERENCES[int(self.cache('GRN','reference_code'))]

    def predict(self,asset,origins):
        value,aux=super().predict(asset,origins)
        q=torch.as_tensor(origins,dtype=torch.long)
        baseES=markov_mean(self.cache(asset,'posterior')[q],self.cache(asset,'transition'),self.cache(asset,'grid'))
        p=self.cache(asset,'base_active')[q];base=self.cache(asset,'base_daily')[q]
        emission=base/(baseES*p)
        if self.mode=='direct_marked_control':
            # Original direct point formula stays untouched. Its diagnostic risk
            # mark uses physical analytic mean; transport its units into the
            # common discrete-Markov frame ONLY for point decomposition.
            unit=baseES/self.cache(asset,'base_analytic_mean')[q]
        else:unit=torch.ones_like(baseES)
        risk=aux['EqS']*unit;joint=aux['EqSZ']*unit;cov=aux['risk_resolution_covariance']*unit
        reconstructed=emission*joint
        torch.testing.assert_close(reconstructed,aux['daily'],rtol=1e-11,atol=1e-14)
        aux.update(anchor_daily=base,anchor_active=p,anchor_markov_mean=baseES,emission_scale=emission,
            risk_unit_transport=unit,transported_risk_mean=risk,transported_joint_mean=joint,
            point_mark_mean=emission*risk,point_joint_mean=reconstructed,
            point_covariance=emission*cov,daily_reconstructed=reconstructed)
        return value,aux


def source_hashes():
    result=GG.source_hashes();result['GRN_gamma_runner']=result.pop('runner')
    paths=dict(runner=Path(__file__),v1=Path(V1.__file__),v2=Path(V2.__file__),
        path_kernel=Path(importlib.import_module('risk_quote_path_tilt').__file__),
        R14=Path(G.B.R14.__file__),GRN_global_checkpoint=CHECKPOINT,GRN_gamma_results=GAMMA_RESULTS,
        GRN_gamma_predictions=GAMMA_PREDICTIONS,strong_baseline_source=HERE/'grn_strong_baselines.py',
        strong_baseline_results=HERE/'grn_strong_baselines_development_results.json',strong_baseline_predictions=STRONG_PREDICTIONS)
    result.update({k:dict(path=str(p.resolve()),sha256=GG.sha(p)) for k,p in paths.items()})
    return result


def verify_strong_EWMA(D):
    value=ewma_point_all(D)
    with np.load(STRONG_PREDICTIONS,allow_pickle=False) as old:
        for key in ('tr','va'):np.testing.assert_array_equal(old['GRN_'+key],D[key])
        for key,expected in (('all_origin',D['origin']),('all_date',D['date']),('all_y',D['y']),('all_ewma',D['ewma'])):
            np.testing.assert_array_equal(old['GRN_'+key],expected)
        q=D['origin'][D['va']]
        for key,expected in (('date',D['date'][q]),('origin',q),('label_end',D['date'][q+22]),('target',D['y'][D['va']])):
            np.testing.assert_array_equal(old['GRN_'+key],expected)
        expected=np.repeat(value[q,None],3,axis=1)
        np.testing.assert_array_equal(old['GRN_EWMA22'],expected)
        for split in ('tr','va'):
            ids=D[split];np.testing.assert_array_equal(value[D['origin'][ids]],old['GRN_all_ewma'][ids])
    return dict(full_observation_EWMA_matches_HB_float32_widening_bitwise=True,
        train_and_validation_EWMA_match_frozen_G_and_strong_archive_bitwise=True,
        strong_validation_EWMA22_bitwise=True,span=22,adjust=False,point_floor='original HB EPS only',
        activity='causal span22 adjustFalse float64 EWMA(rawproxy>0), clip[1e-6,1-1e-6]')


def load_frozen():
    sources=source_hashes();Dset,T,fixed,audit=GG.load_grn();D=Dset['GRN']
    results=GG.read_json(GAMMA_RESULTS);entry=results['checkpoints']['global_dense']
    assert Path(entry['path']).resolve()==CHECKPOINT.resolve() and GG.sha(CHECKPOINT)==entry['sha256']
    own=torch.load(CHECKPOINT,weights_only=True,map_location='cpu');meta=own['metadata']
    if meta.get('assets')!=['GRN'] or 'GRN_preprocessing' not in meta:
        raise PermissionError('Genuine GRN checkpoint metadata required; never two-ETF aliases')
    assert meta['GRN_preprocessing']==fixed and meta['fit']['best_joint_epoch']==entry['selected_epoch']==0
    GG.unchanged(meta['source_hashes'])
    reference=S21.REF.load_gamma_reference()
    model=GG.FAST.FastGammaVarianceFilter(Dset,fixed,reference,'global_dense',11,assets=ASSETS)
    for key,value in model.named_buffers():torch.testing.assert_close(value,own['state_dict'][key],rtol=0,atol=0)
    model.load_state_dict(own['state_dict'],strict=True);model.eval();model.requires_grad_(False)
    assert tuple(model.assets)==ASSETS and tuple(model.processes)==ASSETS
    fingerprint=V1.state_digest(model);original_bank=D['bank_index'].copy()
    gamma=V2.build_caches(Dset,{'GRN':model},assets=ASSETS)['GRN']
    assert V1.state_digest(model)==fingerprint
    np.testing.assert_array_equal(D['bank_index'],original_bank)
    torch.testing.assert_close(gamma['gamma_bank_index'],getattr(model,'bank_index_GRN'),rtol=0,atol=0)
    torch.testing.assert_close(gamma['nn_calibration_bank_index'],torch.tensor(D['origin'][D['tr']],dtype=torch.long),rtol=0,atol=0)
    ewma_gate=verify_strong_EWMA(D);bundles=reference_bundles(D,gamma)
    GG.unchanged(sources)
    return dict(datasets=Dset,fixed=fixed,audit=audit,backbone=model,backbone_fingerprint=fingerprint,
        bundles=bundles,sources=sources,ewma_gate=ewma_gate,checkpoint=dict(path=str(CHECKPOINT),sha256=entry['sha256'],selected_epoch=0))


def bank_metadata(cache):
    return dict(nn_calibration_bank_index=cache['nn_calibration_bank_index'].tolist(),
        nn_calibration_bank_scope=V2.NN_BANK_SCOPE,gamma_observation_bank_index=cache['gamma_bank_index'].tolist(),
        gamma_bank_scope=V2.GAMMA_BANK_SCOPE,risk_mean=cache['risk_mean'].tolist(),risk_std=cache['risk_std'].tolist())


def zero_gate(state):
    D=state['datasets']['GRN'];q=D['origin'][D['va']];gates={};arrays={};scores={}
    with np.load(GAMMA_PREDICTIONS,allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['GRN_date'],D['date'][q]);np.testing.assert_array_equal(saved['GRN_target'],D['y'][D['va']])
        expected_gamma=saved['GRN_global_dense'].copy()
    for name,caches in state['bundles'].items():
        cache=caches['GRN'];expected=expected_gamma if name=='gamma' else np.repeat(D['ewma'][D['va'],None],3,1)
        gates[name]={}
        for mode in MODES:
            model=GRNPredictivePath(caches,mode,11,assets=ASSETS);model.eval();model.requires_grad_(False)
            before=V1.state_digest(model);pred,aux=model.predict('GRN',q);value=pred.numpy()
            np.testing.assert_allclose(value,expected,rtol=1e-11,atol=1e-14)
            assert V1.state_digest(model)==before
            gates[name][mode]=dict(max_absolute_error=float(np.abs(value-expected).max()),same_reference_zerohead=True,
                parameters=sum(p.numel() for p in model.parameters()),optimizer_steps=0)
            prefix=name+'_'+mode;arrays['GRN_'+prefix]=value;scores[prefix]=value
            for key,tensor in aux.items():
                if isinstance(tensor,torch.Tensor):arrays[f'GRN_{prefix}_{key}']=tensor.numpy()
        arrays[f'GRN_{name}_anchor_daily_all']=cache['base_daily'].numpy()
        arrays[f'GRN_{name}_anchor_active_all']=cache['base_active'].numpy()
        arrays[f'GRN_{name}_emission_scale_all']=cache['anchor_emission_scale'].numpy()
    return gates,arrays,GG.safe_metrics(scores,D)


def actual_zerohead():
    for path in (ACTUAL_REPORT,ACTUAL_ARRAYS):
        if path.exists():raise FileExistsError('Exclusive actual sanity exists: '+str(path))
    with patch.object(torch.optim.Optimizer,'__init__',side_effect=AssertionError('No optimizer allowed in actual zerohead')):
        with torch.no_grad():
            state=load_frozen();gates,arrays,scores=zero_gate(state)
    D=state['datasets']['GRN'];cache=state['bundles']['gamma']['GRN'];q=D['origin'][D['va']]
    arrays.update(assets=np.array(ASSETS),horizons=np.array(S21.HORIZONS),GRN_date=D['date'][q],GRN_origin=q,
        GRN_label_end=D['date'][q+22],GRN_target=D['y'][D['va']],GRN_nn_calibration_bank_index=cache['nn_calibration_bank_index'].numpy(),
        GRN_gamma_observation_bank_index=cache['gamma_bank_index'].numpy())
    assert V1.state_digest(state['backbone'])==state['backbone_fingerprint'];GG.unchanged(state['sources'])
    report=dict(status='ACTUAL FROZEN PRE2026 ZEROHEAD ONLY; NO OPTIMIZER OR TRAINING',source_hashes=state['sources'],
        assets=list(ASSETS),checkpoint=state['checkpoint'],EWMA_archive_gate=state['ewma_gate'],zerohead_gates=gates,
        reference_note='EWMA bundle changes BOTH mean anchor and activity reference; not a one-factor ablation',
        interpretation=TRANSPORT_NOTE,emission_rule='base_daily/(sameT baselineES*base_active); reconstruct mean from transported joint mark',
        direct_note='Point formula unchanged; analytic diagnostic risk unit transported to sameT frame only for point decomposition',
        bank_metadata=bank_metadata(cache),metrics=scores,raw_y0_retained=True,GRN2026_opened=False,
        optimizer_constructed=False,optimizer_steps=0,model_training=False,HP_or_lambda_changed=False,
        normalized_QLIKE='null where any y0; positive subset never primary or selection',
        arrays=dict(path=str(ACTUAL_ARRAYS)))
    with ACTUAL_ARRAYS.open('xb') as f:np.savez_compressed(f,**arrays)
    report['arrays']['sha256']=GG.sha(ACTUAL_ARRAYS);GG.save_json(ACTUAL_REPORT,report)
    print(json.dumps(dict(status=report['status'],gates=gates,report_sha256=GG.sha(ACTUAL_REPORT),arrays_sha256=GG.sha(ACTUAL_ARRAYS),runner_sha256=GG.sha(__file__)),indent=2),flush=True)


def synthetic_context():
    data,_,fixed,reference=GG.synthetic_data();D=data['GRN']
    D['range_context_raw']=D['proxy_raw']*(1+1e-5*np.sin(np.arange(D['n'])))
    D['ewma22_all']=pd.Series(D['range_context_raw']).ewm(span=22,adjust=False).mean().clip(lower=HB.EPS).to_numpy().astype(np.float32).astype(float)
    D['ewma']=D['ewma22_all'][D['origin']]
    model=GG.FAST.FastGammaVarianceFilter(data,fixed,reference,'global_dense',11,assets=ASSETS);model.eval();model.requires_grad_(False)
    gamma=V2.build_caches(data,{'GRN':model},assets=ASSETS)['GRN']
    return data,model,reference_bundles(D,gamma)


def _purechecks():
    data,backbone,bundles=synthetic_context();D=data['GRN'];fingerprint=V1.state_digest(backbone);out={}
    baseline=V1.S21.evaluate(backbone,data,G.make_tensors(data))[0]['GRN']
    for reference,caches in bundles.items():
        out[reference]={};cache=caches['GRN'];q=D['origin'][D['va']]
        expected=baseline if reference=='gamma' else np.repeat(D['ewma'][D['va'],None],3,1)
        for mode in MODES:
            model=GRNPredictivePath(caches,mode,11,assets=ASSETS);pred,aux=model.predict('GRN',q)
            np.testing.assert_allclose(pred.detach().numpy(),expected,rtol=1e-11,atol=1e-14)
            with torch.no_grad():model.head[-1].weight.fill_(.03);model.head[-1].bias.fill_(.06)
            value,aux=model.predict('GRN',q)
            torch.testing.assert_close(aux['daily_reconstructed'],aux['daily'],rtol=1e-11,atol=1e-14)
            torch.testing.assert_close(aux['point_joint_mean']-aux['point_mark_mean']*aux['EqZ'],aux['point_covariance'],rtol=1e-8,atol=1e-12)
            total,info=V1.objective(model,data);total.backward()
            assert all(torch.isfinite(info[k]) for k in ('QLIKE','BCE','KL_per_day','output_penalty'))
            assert model.head[-1].weight.grad.abs().sum()>0
            assert sum(p.numel() for p in model.parameters())==(110 if mode=='independent_path' else 137)
            out[reference][mode]=dict(zerohead_correct_reference=True,positive_daily=True,emission_reconstruction=True,point_covariance_identity=True,
                all_losses_finite=True,head_gradient_nonzero=True,parameter_count=sum(p.numel() for p in model.parameters()))
    changed=copy.deepcopy(D);cut=D['cut']+4;changed['range_context_raw'][cut:]*=17
    changed['ewma22_all']=pd.Series(changed['range_context_raw']).ewm(span=22,adjust=False).mean().clip(lower=HB.EPS).to_numpy().astype(np.float32).astype(float)
    changed['ewma']=changed['ewma22_all'][changed['origin']];changed['proxy_raw'][cut:]=0
    np.testing.assert_array_equal(ewma_point_all(changed)[:cut],ewma_point_all(D)[:cut])
    np.testing.assert_array_equal(ewma_active_all(changed)[:cut],ewma_active_all(D)[:cut])
    assert V1.state_digest(backbone)==fingerprint
    metrics=GG.safe_metrics({'positive':np.full_like(D['y'][D['va']],.001)},D)
    assert metrics['positive']['normalized_QLIKE_aggregate']['value'] is None
    assert metrics['positive']['primary_all_targets']['n_rows']==len(D['va'])
    assert V2.PredictivePathV2.__module__=='round22_predictive_path_v2'
    return dict(references=out,anchor_causality_no_futurefill=True,latent_posterior_T_context_uncertainty_unchanged=True,
        NN_and_Gamma_bank_unchanged=True,raw_y0_retained_normalized_null=True,
        mean_and_activity_changed_together_disclosed=True,no_true_IV_claim=True,no_real_data_IO=True,no_real_training=True)


def purechecks():
    with ExitStack() as stack:
        for name in ('pandas.read_csv','numpy.load','pathlib.Path.read_text','torch.load','grn_gamma_development_runner.load_grn',
                     'sampling_density_gamma_reference.load_gamma_reference'):
            stack.enter_context(patch(name,side_effect=AssertionError('Real I/O forbidden in purechecks')))
        return _purechecks()


def private_train_one(caches,data,mode,seed):
    space=V2.private_globals();space['PredictivePath']=GRNPredictivePath
    fit=HB.clone_function(V1.train_one,space)
    return fit(caches,data,mode,seed)


def run_development(reference_name):
    """Later separately authorized pre2026 run; never executed at code handoff."""
    if reference_name not in REFERENCES:raise ValueError(reference_name)
    tag='grn_predictive_path_v2_'+reference_name+'_development'
    output=HERE/(tag+'_results.json');predfile=HERE/(tag+'_validation_predictions.npz')
    paths={(m,s):HERE/f'{tag}_{m}_seed{s}.pt' for m in MODES for s in SEEDS}
    for path in [output,predfile,*paths.values()]:
        if path.exists():raise FileExistsError('Exclusive output exists: '+str(path))
    started=time.time();state=load_frozen();data=state['datasets'];D=data['GRN'];caches=state['bundles'][reference_name]
    with torch.no_grad():gates,_,_=zero_gate(state)
    arrays=dict(assets=np.array(ASSETS),horizons=np.array(S21.HORIZONS));pred={};fits={};checkpoints={};ensemble={}
    for mode in MODES:
        fits[mode]=[];components=[]
        for seed in SEEDS:
            model,value,aux,info=private_train_one(caches,data,mode,seed);fits[mode].append(info)
            name=f'{mode}_seed{seed}';pred[name]=value['GRN'];components.append(aux['GRN'])
            arrays.update({f'GRN_{name}_{k}':v for k,v in aux['GRN'].items()})
            p=paths[(mode,seed)]
            payload=dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                metadata=dict(assets=list(ASSETS),reference_name=reference_name,mode=mode,seed=seed,fit=info,
                    source_hashes=state['sources'],GRN_preprocessing=state['fixed'],bank_metadata=bank_metadata(caches['GRN']),
                    anchor_interpretation=TRANSPORT_NOTE,mean_and_activity_reference_change=True))
            with p.open('xb') as f:torch.save(payload,f)
            torch.load(p,weights_only=True,map_location='cpu');checkpoints[name]=dict(path=str(p),sha256=GG.sha(p))
        name=mode+'_ensemble';pred[name]=np.mean([pred[f'{mode}_seed{s}'] for s in SEEDS],0)
        moments={k:np.mean([a[k] for a in components],0) for k in ('EqZ','point_mark_mean','point_joint_mean','daily')}
        moments['point_covariance']=moments['point_joint_mean']-moments['point_mark_mean']*moments['EqZ']
        arrays.update({f'GRN_{name}_{k}':v for k,v in moments.items()})
        ensemble[mode]=dict(BCE=float(V1.event_bce(torch.tensor(moments['EqZ'],dtype=DTYPE),V1.future_event_labels(D,D['origin'][D['va']],D['n']))),
            mean_component_KL_per_day=float(np.mean([a['KL'].mean()/22 for a in components])) if mode!='direct_marked_control' else None,
            KL_note='mean component KL is not mixture KL',point_covariance_mean=float(moments['point_covariance'].mean()))
    cache=caches['GRN'];q=D['origin'][D['va']]
    arrays.update(GRN_date=D['date'][q],GRN_origin=q,GRN_label_end=D['date'][q+22],GRN_target=D['y'][D['va']],
        GRN_anchor_daily_all=cache['base_daily'].numpy(),GRN_anchor_active_all=cache['base_active'].numpy(),
        GRN_emission_scale_all=cache['anchor_emission_scale'].numpy(),GRN_nn_calibration_bank_index=cache['nn_calibration_bank_index'].numpy(),
        GRN_gamma_observation_bank_index=cache['gamma_bank_index'].numpy())
    arrays.update({'GRN_'+k:v for k,v in pred.items()})
    GG.unchanged(state['sources'])
    with predfile.open('xb') as f:np.savez_compressed(f,**arrays)
    report=dict(status='GRN PRE2026 DEVELOPMENT; GRN2026 CLOSED; not fresh confirmation',reference_name=reference_name,
        source_hashes=state['sources'],reference_note='Both mean anchor and activity reference changed, not one-factor ablation',
        interpretation=TRANSPORT_NOTE,EWMA_gate=state['ewma_gate'],zerohead_gates=gates[reference_name],
        assets=list(ASSETS),backbone_checkpoint=state['checkpoint'],GRN_preprocessing=state['fixed'],bank_metadata=bank_metadata(cache),
        HP=dict(epochs=80,patience=15,min_epochs=20,lr=.01,clip=2,seeds=SEEDS,modes=MODES,BCE_weight=.05,output_penalty=.005,path_KL_per_day_weight=.05),
        lambda_or_hypergrid_changed=False,all_y0_kept=True,primary_selection_scope='2025 only; never GRN2026',
        fits=fits,ensemble_decomposition=ensemble,checkpoints=checkpoints,metrics=GG.safe_metrics(pred,D),
        predictions=dict(path=str(predfile),sha256=GG.sha(predfile)),elapsed_seconds=time.time()-started)
    GG.save_json(output,report)


def main():
    ap=argparse.ArgumentParser(description=__doc__);modes=ap.add_mutually_exclusive_group()
    modes.add_argument('--check-only',action='store_true');modes.add_argument('--actual-zerohead',action='store_true')
    modes.add_argument('--run-development',action='store_true');ap.add_argument('--reference',choices=REFERENCES)
    args=ap.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if args.actual_zerohead:actual_zerohead()
    elif args.run_development:
        if args.reference is None:raise ValueError('Explicit reference_name required for separate exclusive runs')
        run_development(args.reference)
    else:print(json.dumps(dict(checks=purechecks(),source_sha256=GG.sha(__file__)),indent=2),flush=True)


if __name__=='__main__':main()
