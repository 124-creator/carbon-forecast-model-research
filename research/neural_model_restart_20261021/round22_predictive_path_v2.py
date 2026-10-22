"""R22v2: free quote calibration/coupling over unchanged frozen Gamma backbone.

v1 source/results are never modified. a=.25tanh is centered on mature training
forecast origins, b=2tanh and c=.5tanh are NOT centered. All datasets use this
same rule. The NN cohort includes current unresolved/zero observations; it is
separate from the original Gamma positive-observation bank, which stays frozen.
Default CLI is guarded pure synthetic checks; real fitting requires a later
explicit authorization. No GRN2026 or used-period inference is exposed here.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import time
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np
import torch
import round22_predictive_path as V1

S21,HB,DTYPE=V1.S21,V1.HB,V1.DTYPE
ASSETS,SEEDS,MODES=V1.ASSETS,V1.SEEDS,V1.MODES
HERE=Path(__file__).resolve().parent
NN_BANK_SCOPE='ALL mature pre25 common-h22 training forecast origins D.origin[D.tr], including current unresolved/zero ranges'
GAMMA_BANK_SCOPE='Original frozen Gamma observation bank D.bank_index; never replaced by the NN cohort'


def nn_calibration_bank(D):
    tr=np.asarray(D['tr'],dtype=int);origins=np.asarray(D['origin'][tr],dtype=int)
    assert len(origins) and len(np.unique(origins))==len(origins)
    assert (origins>=0).all() and (origins+22<D['cut']).all()
    assert D['cut']<=D['n']
    return torch.tensor(origins,dtype=torch.long)


@torch.no_grad()
def build_asset_cache(D,backbone,asset):
    original_gamma_bank=np.asarray(D['bank_index']).copy();before=V1.state_digest(backbone)
    cache=V1.build_asset_cache(D,backbone,asset)
    gamma_bank=cache['bank_index'].detach().clone()
    nn_bank=nn_calibration_bank(D)
    risk=torch.stack([torch.log(cache['posterior_mean']/cache['mu']),
        torch.log1p(cache['posterior_variance']/cache['posterior_mean'].square())],1)
    mean=risk[nn_bank].mean(0);std=risk[nn_bank].std(0,unbiased=False).clamp_min(1e-5)
    cache['legacy_v1_risk_mean']=cache['risk_mean'].clone()
    cache['legacy_v1_risk_std']=cache['risk_std'].clone()
    cache['risk_mean']=mean;cache['risk_std']=std
    cache['ctx']=torch.cat([cache['ctx'][:,:3],(risk-mean)/std,cache['ctx'][:,5:6]],1)
    cache['gamma_bank_index']=gamma_bank
    cache['nn_calibration_bank_index']=nn_bank
    # The legacy bank_index alias remains the original Gamma bank, not the NN
    # calibration cohort. V2.fields explicitly uses nn_calibration_bank_index.
    assert torch.equal(cache['bank_index'],gamma_bank)
    np.testing.assert_array_equal(D['bank_index'],original_gamma_bank)
    assert V1.state_digest(backbone)==before
    return {k:v.detach().clone() for k,v in cache.items()}


def build_caches(datasets,backbones,assets=None):
    assets=tuple(datasets) if assets is None else tuple(assets)
    return {a:build_asset_cache(datasets[a],backbones[a],a) for a in assets}


class PredictivePathV2(V1.PredictivePath):
    def __init__(self,caches,mode,seed,assets=None,kernel=None):
        super().__init__(caches,mode,seed,assets,kernel)
        for a in self.assets:
            assert torch.equal(self.cache(a,'bank_index'),self.cache(a,'gamma_bank_index'))
            assert len(self.cache(a,'nn_calibration_bank_index'))>0

    def raw(self,features):
        fields=torch.tanh(self.head(features)).reshape(-1,3,self.channels)
        a=.25*fields[:,:,0];b=2*fields[:,:,1]
        c=.5*fields[:,:,2] if self.channels==3 else torch.zeros_like(a)
        return torch.stack([a,b,c],-1)

    def fields(self,asset,origins):
        ctx=self.cache(asset,'ctx');bank=self.cache(asset,'nn_calibration_bank_index')
        bank_raw=self.raw(ctx[bank])
        center_a=bank_raw[:,:,0].mean(0)
        # Only a is centered. Constants b/c are legitimate probability/coupling
        # calibration, not reference moments forced to zero by a positive bank.
        center=torch.stack([center_a,torch.zeros_like(center_a),torch.zeros_like(center_a)],-1)
        delta=self.raw(ctx[origins])-center;reference=bank_raw-center
        daily=delta[:,self.day_bucket,:]
        scale=(self.lengths[self.day_bucket][None,:] if self.mode!='direct_marked_control'
               else daily.new_ones((1,22)))
        return (daily[:,:,0]/scale,daily[:,:,1],daily[:,:,2]/scale),reference,center,delta


def private_globals():
    """No source-module monkeypatching: old functions see a private namespace."""
    space=dict(vars(V1))
    space.update(PredictivePath=PredictivePathV2,build_caches=build_caches,source_hashes=source_hashes)
    return space


def train_one(caches,datasets,mode,seed):
    # Same HP/objective/selection; only the constructor/cache/field convention
    # changes. Each call gets an isolated globals dict, leaving V1 untouched.
    fit=HB.clone_function(V1.train_one,private_globals())
    return fit(caches,datasets,mode,seed)


def load_development():
    loader=HB.clone_function(V1.load_development,private_globals())
    return loader()


def source_hashes():
    out=V1.source_hashes();out['v1_predictive_path']=out.pop('runner')
    path=Path(__file__).resolve();out['runner']=dict(path=str(path),sha256=V1.digest(path))
    return out


def synthetic_context():
    D=S21.synthetic_pair()
    # Ensure the mature forecast cohort visibly includes a current zero row,
    # while the Gamma observation bank remains its original positive rows.
    for a in D:
        D[a]['origin'][D[a]['tr'][1]]=12
        assert not D[a]['observed'][12]
    fixed=S21.preprocessing(D);reference=S21.synthetic_reference();backbones={}
    for a in D:
        model=V1.FastGammaVarianceFilter(D,fixed,reference,'global_dense',11,assets=(a,))
        model.eval();model.requires_grad_(False);backbones[a]=model
    original={a:np.asarray(d['bank_index']).copy() for a,d in D.items()}
    before={a:V1.state_digest(m) for a,m in backbones.items()}
    caches=build_caches(D,backbones)
    for a in D:
        assert V1.state_digest(backbones[a])==before[a]
        np.testing.assert_array_equal(D[a]['bank_index'],original[a])
        torch.testing.assert_close(caches[a]['gamma_bank_index'],getattr(backbones[a],f'bank_index_{a}'),rtol=0,atol=0)
        torch.testing.assert_close(caches[a]['nn_calibration_bank_index'],torch.tensor(D[a]['origin'][D[a]['tr']],dtype=torch.long),rtol=0,atol=0)
        assert (~D[a]['observed'][caches[a]['nn_calibration_bank_index'].numpy()]).any()
    return D,backbones,caches


def constant_context_counterexamples(caches):
    constant=copy.deepcopy(caches)
    for a in constant:
        cache=constant[a]
        cache['ctx']=torch.zeros_like(cache['ctx']);cache['ctx'][:,5]=1
        cache['base_active']=torch.full_like(cache['base_active'],.8)
        cache['base_daily']=cache['base_analytic_mean']*.8*cache['scale']*cache['c512']/cache['c_ref']
    logit_shift=math.log(.95/.05)-math.log(.8/.2)
    raw_bias=math.atanh(logit_shift/2)
    results={}
    for mode in MODES:
        old=V1.PredictivePath(constant,mode,11);new=PredictivePathV2(constant,mode,11)
        with torch.no_grad():
            old.head[-1].bias.reshape(3,old.channels)[:,1]=raw_bias
            new.head[-1].bias.reshape(3,new.channels)[:,1]=raw_bias
        for a in constant:
            origins=constant[a]['nn_calibration_bank_index'][:2]
            oldfields=old.fields(a,origins)[0];newfields=new.fields(a,origins)[0]
            assert oldfields[1].abs().max()<1e-14
            torch.testing.assert_close(newfields[1],torch.full_like(newfields[1],logit_shift),rtol=1e-14,atol=1e-14)
            _,oldaux=old.predict(a,origins);_,newaux=new.predict(a,origins)
            torch.testing.assert_close(oldaux['EqZ'],torch.full_like(oldaux['EqZ'],.8),rtol=1e-13,atol=1e-13)
            torch.testing.assert_close(newaux['EqZ'],torch.full_like(newaux['EqZ'],.95),rtol=1e-13,atol=1e-13)
            # Bank penalty includes a constant b; permission to calibrate does
            # not imply a free/unpenalized intercept or a fitted optimum at1.558.
            ref=new.fields(a,origins)[1][:,:,:new.channels]
            expected=3*logit_shift**2
            torch.testing.assert_close(ref.square().sum((1,2)).mean(),ref.new_tensor(expected),rtol=1e-13,atol=1e-13)
        results[mode]=dict(v1_constant_b_after_center=0.,v2_constant_b=logit_shift,v1_active=.8,v2_active=.95,
            constant_b_output_penalty=.005*3*logit_shift**2)
    old=V1.PredictivePath(constant,'coupled_path',11);new=PredictivePathV2(constant,'coupled_path',11)
    with torch.no_grad():
        old.head[-1].bias.reshape(3,3)[:,2]=math.atanh(.3/.5)
        new.head[-1].bias.reshape(3,3)[:,2]=math.atanh(.3/.5)
    max_cov=0.
    for a in constant:
        q=constant[a]['nn_calibration_bank_index'][:2]
        _,left=old.predict(a,q);_,right=new.predict(a,q)
        assert left['risk_resolution_covariance'].abs().max()<1e-14
        assert (right['risk_resolution_covariance']>0).all()
        max_cov=max(max_cov,float(right['risk_resolution_covariance'].detach().max()))
        _,ref,center,delta=new.fields(a,q)
        assert torch.equal(delta[:,:,0],torch.zeros_like(delta[:,:,0]))
        torch.testing.assert_close(delta[:,:,2],torch.full_like(delta[:,:,2],.3),rtol=1e-14,atol=1e-14)
        assert torch.equal(center[:,1:],torch.zeros_like(center[:,1:]))
    bce_before=-(.95*math.log(.8)+.05*math.log(.2))
    bce_after=-(.95*math.log(.95)+.05*math.log(.05))
    assert bce_after<bce_before
    return dict(b_counterexamples=results,coupled_constant_c=.3,coupled_constant_c_max_covariance=max_cov,
        v1_constant_c_covariance=0.,population_BCE_before=bce_before,population_BCE_after=bce_after,
        counterexample_is_feasibility_not_a_new_fit=True)


def _purechecks():
    D,backbones,caches=synthetic_context()
    counter=constant_context_counterexamples(caches);report={}
    T=S21.make_tensors(D)
    old_gamma={a:V1.state_digest(m) for a,m in backbones.items()}
    gamma_banks={a:D[a]['bank_index'].copy() for a in D}
    for mode in MODES:
        model=PredictivePathV2(caches,mode,11)
        assert sum(p.numel() for p in model.parameters())==(110 if mode=='independent_path' else 137)
        maxzero=0.
        for a in D:
            expected=S21.evaluate(backbones[a],D,T)[0][a]
            q=D[a]['origin'][D[a]['va']];pred,aux=model.predict(a,q)
            np.testing.assert_allclose(pred.detach().numpy(),expected,rtol=1e-11,atol=1e-14)
            maxzero=max(maxzero,float(np.abs(pred.detach().numpy()-expected).max()))
            assert (aux['daily']>0).all()
            torch.testing.assert_close(pred,torch.stack([aux['daily'][:,:h].mean(1) for h in S21.HORIZONS],1))
        # Main QLIKE, not only BCE/regularization, must reach every active head.
        loss=0
        for a in D:
            ids=D[a]['tr'];pred,_=model.predict(a,D[a]['origin'][ids])
            loss=loss+S21.B.qlike(pred,torch.tensor(D[a]['y'][ids],dtype=DTYPE),torch.tensor(D[a]['ewma'][ids],dtype=DTYPE)).mean()/len(D)
        loss.backward()
        grad=model.head[-1].weight.grad.abs().sum(1)
        assert (grad>0).all()
        bias=model.head[-1].bias.grad.reshape(3,model.channels)
        assert (bias[:,1:].abs()>0).all(),'Free b/c biases must have main-loss gradient'
        with torch.no_grad():
            model.head[-1].weight.copy_(torch.linspace(-.06,.09,model.head[-1].weight.numel(),dtype=DTYPE).reshape_as(model.head[-1].weight))
            model.head[-1].bias.fill_(.04)
        for a in D:
            q=D[a]['origin'][D[a]['tr']];daily,reference,center,delta=model.fields(a,q)
            assert torch.equal(center[:,1:],torch.zeros_like(center[:,1:]))
            assert reference[:,:,0].mean(0).abs().max()<1e-14
            assert (delta[:,:,0].abs()<=.5).all() and (delta[:,:,1].abs()<=2).all() and (delta[:,:,2].abs()<=.5).all()
            assert reference[:,:,1].abs().max()>0
            if model.channels==3:assert reference[:,:,2].abs().max()>0
            for day,bucket in ((0,0),(1,1),(5,2)):
                L=1 if mode=='direct_marked_control' else V1.BUCKET_LENGTHS[bucket]
                torch.testing.assert_close(daily[0][:,day],delta[:,bucket,0]/L,rtol=0,atol=0)
                torch.testing.assert_close(daily[1][:,day],delta[:,bucket,1],rtol=0,atol=0)
                torch.testing.assert_close(daily[2][:,day],delta[:,bucket,2]/L,rtol=0,atol=0)
            model.cache(a,'ctx').requires_grad_(True)
        total,info=V1.objective(model,D)
        assert all(torch.isfinite(info[k]) for k in ('QLIKE','BCE','KL_per_day','output_penalty'))
        explicit=.005*sum(model.fields(a,D[a]['origin'][D[a]['tr']])[1][:,:,:model.channels].square().sum((1,2)).mean() for a in D)/len(D)
        torch.testing.assert_close(info['output_penalty'],explicit,rtol=0,atol=0)
        model.zero_grad(set_to_none=True);total.backward()
        for a in D:
            assert (model.cache(a,'ctx').grad[D[a]['cut']:] ==0).all()
            model.cache(a,'ctx').requires_grad_(False)
        changed=copy.deepcopy(D)
        for a in D:
            changed[a]['y'][D[a]['va']]*=100;changed[a]['proxy_raw'][D[a]['cut']:]*=9
        changedloss,_=V1.objective(model,changed)
        torch.testing.assert_close(total.detach(),changedloss.detach(),rtol=0,atol=0)
        buffers={k:v.clone() for k,v in model.named_buffers()};opt=torch.optim.Adam(model.parameters(),lr=V1.LR)
        with patch('torch.autograd.backward',wraps=torch.autograd.backward) as backward:
            with patch.object(opt,'step',wraps=opt.step) as step:
                losses=V1.train_epoch(model,D,opt)
                assert step.call_count==1
        assert backward.call_count==1
        for k,v in model.named_buffers():assert torch.equal(v,buffers[k])
        report[mode]=dict(parameters=sum(p.numel() for p in model.parameters()),zero_global_max_absolute_error=maxzero,
            all_main_head_gradients_nonzero=True,free_b_c_bias_gradients_nonzero=True,
            all_loss_terms_finite=True,val_context_and_future_labels_excluded=True,
            direct_amplitude_not_weakened=True,sum_active_output_penalty_exact=True,one_backward_step=True)
    modified=copy.deepcopy(D)
    for a in D:
        modified[a]['proxy_raw'][D[a]['cut']+3:]*=6;modified[a]['F17'][D[a]['cut']+3:,2]+=5
        modified[a]['y']*=100
    alternate=build_caches(modified,backbones)
    for a in D:
        cutoff=D[a]['cut']+3
        for key in ('ctx','posterior','base_daily','base_active'):
            torch.testing.assert_close(caches[a][key][:cutoff],alternate[a][key][:cutoff],rtol=0,atol=0)
        for key in ('nn_calibration_bank_index','gamma_bank_index','risk_mean','risk_std'):
            torch.testing.assert_close(caches[a][key],alternate[a][key],rtol=0,atol=0)
        assert V1.state_digest(backbones[a])==old_gamma[a]
        np.testing.assert_array_equal(D[a]['bank_index'],gamma_banks[a])
        cohort=caches[a]['nn_calibration_bank_index']
        torch.testing.assert_close(caches[a]['ctx'][cohort,3:5].mean(0),torch.zeros(2,dtype=DTYPE),rtol=0,atol=1e-12)
    namespace=private_globals()
    assert namespace['PredictivePath'] is PredictivePathV2
    assert V1.train_one.__globals__['PredictivePath'] is V1.PredictivePath
    assert V1.load_development.__globals__['build_caches'] is V1.build_caches
    return dict(constant_context_counterexample=counter,modes=report,
        nn_cohort_all_mature_train_origins_includes_current_zero=True,
        NN_bank_and_Gamma_bank_separately_named=True,NN_bank_statistics_trainonly_and_causal=True,
        Gamma_theta_hurdle_AR_reference_bank_unchanged=True,no_v1_module_global_mutation=True,
        no_dataset_quality_or_future_score_branch=True,real_training_executed=False,GRN2026_opened=False)


def purechecks():
    with ExitStack() as stack:
        for name in ('pandas.read_csv','numpy.load','pathlib.Path.read_text','torch.load',
                     'holdout_2026_range_builder.build_pair','sampling_density_gamma_reference.load_gamma_reference'):
            stack.enter_context(patch(name,side_effect=AssertionError('Real I/O forbidden in purechecks')))
        return _purechecks()


def run_development(tag):
    """Future main-authorized execution only, identical policy on every dataset."""
    assert re.fullmatch(r'round22_predictive_path_v2_[A-Za-z0-9_-]+',tag)
    outputs=dict(results=HERE/f'{tag}_results.json',predictions=HERE/f'{tag}_validation_predictions.npz')
    checkpoints={(m,s):HERE/f'{tag}_{m}_seed{s}.pt' for m in MODES for s in SEEDS}
    for p in [*outputs.values(),*checkpoints.values()]:
        if p.exists():raise FileExistsError('Protected output: '+str(p))
    started=time.time();D,caches,audit,hashes,gates=load_development()
    pred={a:{} for a in D};arrays=dict(horizons=np.array(S21.HORIZONS));fits={};checkpoint_info={};ensembles={}
    for mode in MODES:
        fits[mode]=[];component={a:[] for a in D}
        for seed in SEEDS:
            model,values,aux,info=train_one(caches,D,mode,seed);fits[mode].append(info)
            for a in D:
                name=f'{mode}_seed{seed}';pred[a][name]=values[a];component[a].append(aux[a])
                arrays.update({f'{a}_{name}_{k}':v for k,v in aux[a].items()})
            p=checkpoints[(mode,seed)]
            own=dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                metadata=dict(version='R22v2',mode=mode,seed=seed,selected_epoch=info['best_joint_epoch'],source_hashes=hashes,
                    nn_calibration_bank_scope=NN_BANK_SCOPE,gamma_bank_scope=GAMMA_BANK_SCOPE,
                    decoder='a centered .25tanh; b free2tanh; c free.5tanh',fixed_backbone=True))
            with p.open('xb') as handle:torch.save(own,handle)
            torch.load(p,weights_only=True,map_location='cpu');checkpoint_info[f'{mode}_seed{seed}']=dict(path=str(p),sha256=V1.digest(p))
        ensembles[mode]={}
        for a in D:
            name=mode+'_ensemble';pred[a][name]=np.mean([pred[a][f'{mode}_seed{s}'] for s in SEEDS],axis=0)
            moments={k:np.mean([aux[k] for aux in component[a]],axis=0) for k in ('EqS','EqZ','EqSZ','daily')}
            moments['risk_resolution_covariance']=moments['EqSZ']-moments['EqS']*moments['EqZ']
            arrays.update({f'{a}_{name}_{k}':v for k,v in moments.items()})
            ids=D[a]['va'];origins=D[a]['origin'][ids];v=pred[a][name];y=D[a]['y'][ids]
            ensembles[mode][a]=dict(QLIKE=float((np.log(v)+y/v-np.log(D[a]['ewma'][ids])[:,None]).mean()),
                BCE=float(V1.event_bce(torch.tensor(moments['EqZ'],dtype=DTYPE),V1.future_event_labels(D[a],origins,D[a]['n']))),
                mean_component_KL_per_day=float(np.mean([aux['KL'].mean()/22 for aux in component[a]])) if mode!='direct_marked_control' else None,
                KL_note='mean component KL, not ensemble-mixture KL',risk_resolution_covariance_mean=float(moments['risk_resolution_covariance'].mean()))
    scope={}
    for a in D:
        q=D[a]['origin'][D[a]['va']]
        arrays.update({a+'_date':D[a]['date'][q],a+'_origin':q,a+'_label_end':D[a]['date'][q+22],a+'_target':D[a]['y'][D[a]['va']]})
        arrays.update({a+'_'+k:v for k,v in pred[a].items()})
        arrays[a+'_nn_calibration_bank_index']=caches[a]['nn_calibration_bank_index'].numpy()
        arrays[a+'_gamma_observation_bank_index']=caches[a]['gamma_bank_index'].numpy()
        scope[a]=dict(nn_calibration_bank_scope=NN_BANK_SCOPE,nn_calibration_bank_index=caches[a]['nn_calibration_bank_index'].tolist(),
            nn_calibration_current_zero_count=int((~D[a]['observed'][caches[a]['nn_calibration_bank_index'].numpy()]).sum()),
            gamma_bank_scope=GAMMA_BANK_SCOPE,gamma_observation_bank_index=caches[a]['gamma_bank_index'].tolist(),
            risk_mean=caches[a]['risk_mean'].tolist(),risk_std=caches[a]['risk_std'].tolist())
    scores={a:V1.S21.S17.metrics(pred[a],D[a]) for a in D};joint={}
    for name in pred[tuple(D)[0]]:
        asset_scores={}
        for a in D:
            ids=D[a]['va'];v=pred[a][name];y=D[a]['y'][ids]
            asset_scores[a]=dict(known_scale_QLIKE=float((np.log(v)+y/v-np.log(D[a]['ewma'][ids])[:,None]).mean()),
                raw_QLIKE=float((np.log(v)+y/v).mean()),MSE=float(((v-y)**2).mean()))
        joint[name]=dict(asset_scores=asset_scores,**{k:sum(r[k] for r in asset_scores.values())/len(D) for k in ('known_scale_QLIKE','raw_QLIKE','MSE')},
            per_h={str(h):{k:sum(scores[a][name][str(h)][k] for a in D)/len(D) for k in ('mean_QLIKE','loss_difference_vs_EWMA22','MSE')} for h in S21.HORIZONS})
    report=dict(version='R22v2',status='REUSED2025 DEVELOPMENT; any old2026 reuse is not fresh confirmation',sources=hashes,
        config=dict(modes=MODES,seeds=SEEDS,epochs=V1.MAX_EPOCHS,patience=V1.PATIENCE,min_epochs=V1.MIN_EPOCHS,lr=V1.LR,clip=2),
        head_rules='a=.25tanh centered on NN cohort; b=2tanh and c=.5tanh NOT centered; bounds .5/2/.5',
        daily_amplitude='Only path a/c dividedL1/4/17; direct a/c retain full bucket logmark size; b never divided',
        output_penalty='.005 equalasset meanNNbank SUM active9/6 output squares; includes constant b/c',
        loss='equalasset/h knownscale QLIKE+.05daily eventBCE+.005outputPenalty+.05pathKL/22; direct has no pathKL',
        regularization_note='Path KL is part of method, not fully matched regularization against direct control',
        scope=scope,zero_global_gate=gates,backbone_Gamma_bank_theta_AR_hurdle_reference_unchanged=True,
        same_version_for_all_datasets=True,dataset_quality_or_future_score_branch=False,
        builder=audit,fits=fits,scores=scores,joint_scores=joint,ensemble_decomposition=ensembles,checkpoints=checkpoint_info,
        seed_note='same datasets, repeated initialization, not independent datasets',elapsed_seconds=time.time()-started)
    assert source_hashes()==hashes
    with outputs['predictions'].open('xb') as handle:np.savez_compressed(handle,**arrays)
    with outputs['results'].open('x',encoding='utf-8') as handle:json.dump(report,handle,ensure_ascii=False,indent=2,allow_nan=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__);flags=ap.add_mutually_exclusive_group()
    flags.add_argument('--check-only',action='store_true');flags.add_argument('--run-development',action='store_true')
    ap.add_argument('--tag',default='round22_predictive_path_v2_development');args=ap.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if args.run_development:run_development(args.tag)
    else:
        before=V1.digest(V1.__file__);checks=purechecks();assert V1.digest(V1.__file__)==before
        print(json.dumps(dict(checks=checks,v1_source_sha256=before,v2_source_sha256=V1.digest(__file__)),indent=2),flush=True)


if __name__=='__main__':main()
