"""Round16 proxy-mean frame, preserving the frozen filter implementation.
The finite-grid Brownian reference has E[P|sigma^2=1]=c != 1.
Shift the log-risk frame by log(c), equivalently forecasting c*E[sigma^2].
The same simulation-train constant is applied to every mode. No real validation
or test observation estimates this conversion; it is not a neural contribution.
"""
import argparse
import copy
import json
from pathlib import Path
import re
import time
import numpy as np
import torch
import bridge_precision_round16 as B

HERE=Path(__file__).resolve().parent
REFERENCE_RESULTS=HERE/'brownian_reference_round16'/'brownian_reference_results.json'


def proxy_frame(D,c):
    assert np.isfinite(c) and c>0
    out=dict(D)
    offset=float(np.log(c))
    out['cell_b']=D['cell_b']-offset
    out['global_b']=D['global_b']-offset
    out['initial_mu']=D['initial_mu']+offset
    return out


def checks():
    rng=np.random.default_rng(20261022);n=36;c=.9427735381258473
    D=dict(n=n,initial_mu=-8.,log_proxy=-8+rng.normal(0,.3,n),even=rng.normal(size=(n,3)),
           cell_b=np.linspace(-.5,.2,n),cell_R=np.linspace(.2,.35,n),observed=np.ones(n,dtype=bool),
           global_b=-.23,global_R=.35,constant_R=.27,y=np.full((3,3),.0005),ewma=np.full(3,.0005))
    D['observed'][9]=False
    shifted=proxy_frame(D,c)
    for mode in B.MODES:
        left=B.PrecisionFilter(mode,D['initial_mu']);right=B.PrecisionFilter(mode,shifted['initial_mu'])
        with torch.no_grad():
            if mode in B.NEURAL_MODES:
                for net in [left,right]:
                    net.precision_net[-1].weight.fill_(.02)
                    net.precision_net[-1].bias.fill_(.03)
            lm,lP,lK,lR=B.filter_prefix(left,D,B.tensors(D),n)
            rm,rP,rK,rR=B.filter_prefix(right,shifted,B.tensors(shifted),n)
            torch.testing.assert_close(rm,lm+np.log(c),rtol=1e-12,atol=1e-12)
            for a,b in [(rP,lP),(rK,lK),(rR,lR)]:torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
            torch.testing.assert_close(right.forecast(rm,rP),c*left.forecast(lm,lP),rtol=1e-12,atol=1e-12)
    return dict(market_data_opened=False,reference_file_opened=False,
                same_proxy_frame_for_every_mode=True,log_state_translation=True,
                covariance_gain_precision_unchanged=True,forecasts_equal_c_times_latent_mean=True,
                nonzero_neural_output_checked=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--instrument',choices=list(B.SOURCES),default='KRBN')
    ap.add_argument('--reference-grid',choices=[512,2048],type=int,default=512)
    ap.add_argument('--seeds',type=int,nargs='+',default=[11,22,33]);ap.add_argument('--epochs',type=int,default=100)
    ap.add_argument('--chunk',type=int,default=128);ap.add_argument('--tag');ap.add_argument('--check-only',action='store_true')
    args=ap.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if args.check_only:print(json.dumps(checks(),indent=2));return
    tag=args.tag or f'bridge_precision_proxy_round16_{args.instrument}_grid{args.reference_grid}_development'
    assert re.fullmatch(r'bridge_precision_proxy_round16_[A-Za-z0-9_-]+',tag)
    paths={k:HERE/f'{tag}{suffix}' for k,suffix in dict(manifest='_manifest.json',results='_results.json',predictions='_validation_predictions.npz').items()}
    assert not any(p.exists() for p in paths.values())
    sources=dict(runner=Path(__file__).resolve(),filter=Path(B.__file__).resolve(),range_clock=Path(B.R14.__file__).resolve(),
                 reference_models=B.REFERENCE,reference_results=REFERENCE_RESULTS,csv=B.SOURCES[args.instrument])
    hashes={name:dict(path=str(path),sha256=B.sha256(path)) for name,path in sources.items()}
    reference=json.loads(REFERENCE_RESULTS.read_text(encoding='utf-8'))
    c=float(reference['resolution_sensitivity']['mean_P_by_train_resolution'][str(args.reference_grid)])
    original,audit=B.prepare(args.instrument,args.reference_grid);D=proxy_frame(original,c)
    audit.update(state_frame='log of conditional mean Parkinson proxy in the reference model',
                 reference_train_mean_proxy=c,known_log_frame_shift=float(np.log(c)),
                 initial_mu=D['initial_mu'],global_b=D['global_b'])
    meta=dict(config=vars(args),status='DEVELOPMENT ONLY; no 2026 forecasts or targets',audit=audit,hashes=hashes,
        frame='common simulation-train conversion; cell_b/global_b minus log(c), initial_mu plus log(c)',
        frame_note='equivalent to predicting c*exp(latent m+P/2); c is not estimated on market validation; actual diffusion assumptions unverified',
        primary='same QLIKE mean across h1/5/22 common origins; per-mode validation epoch including zero',
        target='future Parkinson proxy mean, not true integrated variance',checks=checks(),fits={},scores={})
    paths['manifest'].write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    pred=B.direct_baselines(D);va=D['va'];start=time.time()
    arrays=dict(date=D['date'][D['origin'][va]],origin=D['origin'][va],target=D['y'][va],horizons=np.array(B.HORIZONS),
                label_end=D['date'][D['origin'][va]+22])
    for mode in B.MODES:
        fits=[];values=[];cache=None
        for seed in args.seeds:
            if mode not in B.NEURAL_MODES and cache is not None:
                value,aux,info=cache[0].copy(),{k:v.copy() for k,v in cache[1].items()},copy.deepcopy(cache[2])
                info.update(seed=seed,fit_executed=False,deterministic_reuse_from_seed=args.seeds[0])
            else:
                value,aux,info=B.train_one(D,mode,seed,args.epochs,args.chunk)
                info.update(fit_executed=True,deterministic_reuse_from_seed=None)
                if mode not in B.NEURAL_MODES:cache=(value.copy(),{k:v.copy() for k,v in aux.items()},copy.deepcopy(info))
            values.append(value);fits.append(info);pred[f'{mode}_seed{seed}']=value
            for name,state in aux.items():arrays[f'{mode}_seed{seed}_{name}']=state
            print(args.instrument,mode,seed,'epoch',info['best_epoch'],'val',round(info['best_validation_QLIKE_known_scale'],7),flush=True)
        pred[mode]=values[0].copy() if mode not in B.NEURAL_MODES else np.mean(values,axis=0)
        meta['fits'][mode]=fits
    y=D['y'][va]
    for j,h in enumerate(B.HORIZONS):
        base=np.log(pred['EWMA22'][:,j])+y[:,j]/pred['EWMA22'][:,j]
        meta['scores'][str(h)]={}
        for name,p in pred.items():
            assert p.shape==y.shape and np.isfinite(p).all() and (p>0).all()
            loss=np.log(p[:,j])+y[:,j]/p[:,j]
            meta['scores'][str(h)][name]=dict(n=len(y),mean_QLIKE=float(loss.mean()),
                loss_difference_vs_EWMA22=float((loss-base).mean()),MSE=float(((p[:,j]-y[:,j])**2).mean()))
        print('h',h,{name:round(row['loss_difference_vs_EWMA22'],7) for name,row in meta['scores'][str(h)].items() if '_seed' not in name},flush=True)
    for name,path in sources.items():assert B.sha256(path)==hashes[name]['sha256']
    meta['elapsed_seconds']=time.time()-start
    paths['results'].write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    arrays.update(pred);np.savez_compressed(paths['predictions'],**arrays)
    print('FINISHED; 2026 UNUSED',round(meta['elapsed_seconds'],1),flush=True)


if __name__=='__main__':main()
