# -*- coding: utf-8 -*-
"""Low-capacity conditional transport decoders on the same honest forest prior.
Compare direct linear and additive radial basis heads; no recurrent encoder is
claimed. Existing sequence/context/cache artifacts are immutable.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from threadpoolctl import threadpool_limits
import prior_readout_transport_round5 as N
R=N.R;B=N.B;P=Path(__file__).resolve().parent
MODELS=['linear_prior','rbf_prior']


class SmallDecoder(nn.Module):
    def __init__(self,variant):
        super().__init__();self.variant=variant
        self.register_buffer('centers',torch.tensor([-2.,-1.,0.,1.,2.]))
        self.encoder=nn.Linear(240 if variant=='rbf_prior' else 40,4)
        nn.init.zeros_(self.encoder.weight);nn.init.zeros_(self.encoder.bias)
    def forward(self,X,idx):
        x=torch.cat([X['x'][idx],X['prior'][idx]],1)
        if self.variant=='rbf_prior':x=torch.cat([x,torch.exp(-.5*(x[:,:,None]-self.centers).square()).flatten(1)],1)
        return self.encoder(x)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--markets',nargs='+',default=list(R.T.B.CUT));ap.add_argument('--tag',default='additive_transport_round5');args=ap.parse_args()
    args.min_epochs=30;args.patience=16;args.epochs=64;args.seeds=[11,22,33,44,55]
    torch.set_num_threads(1);torch.set_num_interop_threads(1);B.activate();R.TransportNet=SmallDecoder
    panel=pd.read_csv(R.T.B.PANEL,parse_dates=['date']);old=np.load(P/'prior_readout_transport_round5_predictions.npz')
    frozen=[P/'prior_readout_transport_round5.py',P/'prior_readout_transport_round5_predictions.npz',P/'forest_transport_round5.py']
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen}
    report={'status':'exploratory low-capacity decoder experiment; all two arms and seeds retained','config':vars(args),
            'models':'linear40->4(164params), additive[40 linear+40x5 Gaussian basis]->4(964params), allzerohead',
            'basis':'fixed standardized centers -2,-1,0,1,2 width1; no data/test knot search',
            'loss':'same exactCRPS plus beta regularization; original normalized selector, rawrelative diagnostic',
            'reference':'honest prefix training u, unchanged fullphase validation/test u; no added external inputs',
            'frozen_before':before,'markets':{}}
    (P/(args.tag+'_manifest.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    arrays={};start=time.perf_counter()
    for market in args.markets:
        d=panel[panel.market==market].sort_values('date').reset_index(drop=True);D=R.S.build(d)
        prepared,audits,ci=N.load_explicit(D,market,'honest');Xi,bi=prepared['inner'];Xf,bf=prepared['final']
        te=D['te'];hi=D['hi'][te];scale=D['scale'][te];key=market+'_'
        for suffix,value in [('origin',D['t'][te]),('horizon',R.T.HOR[hi]),('target',D['y'][te]),('scale',scale)]:
            assert np.array_equal(value,old[key+suffix]);arrays[key+suffix]=value
        pred={};fits={};states={}
        for model in MODELS:
            allops=[];fs=[]
            for seed in args.seeds:
                _,info=B.run_inner(D,Xi,bi,model,seed,args);ep=info['best_epoch']
                net,rf=R.run_fit(D,Xf,bf,model,seed,args,'rf',ep)
                (q,c,p),branch,aa,bb,ops,diag=R.evaluate_seed(net,D,Xf,bf);allops.append(ops)
                info.update(seed=seed,refit=rf,parameter_count=sum(p.numel() for p in net.parameters()),
                            trainable_projection_updated=rf['encoder_parameter_L2_change']>0,geometry=diag)
                fs.append(info);states[f'{model}_seed{seed}']=net.state_dict()
                for suffix,value in [('Q',q),('CRPS',c),('p',p),('a',aa),('b',bb)]:arrays[key+model+f'_seed{seed}_'+suffix]=value
            q=np.empty((len(te),15));c=np.empty(len(te));p=np.empty(len(te))
            for j,bank in enumerate(bf):
                ix=np.flatnonzero(hi==j);qq,cc,pp,_,_=R.mixture(torch.tensor(D['z'][te[ix]]),bank['z'],[o[j] for o in allops])
                q[ix]=qq.numpy()*scale[ix,None];c[ix]=cc.numpy()*scale[ix];p[ix]=pp.numpy()
            pred[model]=(q,c,p);fits[model]=fs
            print(market,model,'epochs',[v['best_epoch'] for v in fs],'elapsed_s',round(time.perf_counter()-start,1),flush=True)
        for model in ['forest_reference_only','rule_qrf','qrf_original','lstm_qr','honest_no_readout','honest_readout']:
            pred[model]=(old[key+model+'_Q'],old[key+model+'_CRPS'] if key+model+'_CRPS' in old else None,
                         old[key+model+'_p'] if key+model+'_p' in old else None)
        cells={}
        for j,h in enumerate(R.T.HOR):
            ix=hi==j;cells[str(h)]={m:R.metrics(D['y'][te[ix]],q[ix],None if c is None else c[ix],None if p is None else p[ix]) for m,(q,c,p) in pred.items()}
        for model,(q,c,p) in pred.items():
            arrays[key+model+'_Q']=q
            if c is not None:arrays[key+model+'_CRPS']=c
            if p is not None:arrays[key+model+'_p']=p
        report['markets'][market]={'fits':fits,'by_horizon':cells,'reference_sha':ci['sha256'],'preparation':audits}
        torch.save({'state_dicts':states,'fits':fits,'cache_info':ci,'preparation':audits},P/f'{args.tag}_{market}_checkpoints.pt')
        (P/(args.tag+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');np.savez_compressed(P/(args.tag+'_predictions.npz'),**arrays)
    report['summary']={}
    for model in MODELS:
        report['summary'][model]={}
        for base in ['forest_reference_only','rule_qrf','qrf_original','lstm_qr','honest_no_readout','honest_readout']:
            report['summary'][model][base]={}
            for metric in ['MQS','CRPS']:
                rr=[c[model][metric]/c[base][metric] for m in report['markets'].values() for c in m['by_horizon'].values() if c[base][metric] is not None]
                if rr:report['summary'][model][base][metric]={'geometric_gain_percent':float(100*(1-np.exp(np.log(rr).mean()))),'wins':int((np.array(rr)<1).sum()),'cells':len(rr)}
    report['frozen_after']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen};assert before==report['frozen_after']
    report['elapsed_seconds']=time.perf_counter()-start
    (P/(args.tag+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report['summary'],indent=2),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    with threadpool_limits(limits=1):main()
