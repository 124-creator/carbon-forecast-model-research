# -*- coding: utf-8 -*-
"""Matched raw-validation epoch selection for all four round5 readout arms.
No inner retraining and no test-based bandwidth/epoch/model selection.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
import prior_readout_transport_round5 as N
R=N.R;P=Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',default='prior_readout_rawselect_round5');a=ap.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);N.activate()
    meta=json.loads((P/'prior_readout_transport_round5.json').read_text(encoding='utf-8'))
    old=np.load(P/'prior_readout_transport_round5_predictions.npz');panel=pd.read_csv(R.T.B.PANEL,parse_dates=['date'])
    args=argparse.Namespace(min_epochs=30,patience=16,epochs=64)
    frozen=[P/'prior_readout_transport_round5.py',P/'prior_readout_transport_round5.json',P/'prior_readout_transport_round5_predictions.npz']
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen}
    report={'status':'secondary matched selector experiment, all four arms retained',
            'selection':'minimum visited inner mean_h log(rawMQS_h/zerohead_reference_rawMQS_h)',
            'inner_retraining_count':0,'frozen_before':before,'markets':{}}
    arrays={};start=time.perf_counter();nrefit=0
    for market in R.T.B.CUT:
        d=panel[panel.market==market].sort_values('date').reset_index(drop=True);D=R.S.build(d)
        prepared={};cacheinfo={}
        for kind in ['honest','insample']:prepared[kind],_,cacheinfo[kind]=N.load_explicit(D,market,kind)
        cp=torch.load(P/f'prior_readout_transport_round5_{market}_checkpoints.pt',map_location='cpu',weights_only=False)
        te=D['te'];hi=D['hi'][te];scale=D['scale'][te];key=market+'_'
        for suffix,value in [('origin',D['t'][te]),('horizon',R.T.HOR[hi]),('target',D['y'][te]),('scale',scale)]:
            assert np.array_equal(value,old[key+suffix]);arrays[key+suffix]=value
        models={};states={};pred={}
        for model in N.MODELS:
            kind='insample' if model=='insample_readout' else 'honest';Xf,bf=prepared[kind]['final']
            allops=[];fits=[]
            for info in meta['markets'][market]['fits'][model]:
                seed=info['seed'];ep=info['best_raw_epoch'];norm=info['best_epoch']
                assert ep==min(info['raw_relative_validation_history'],key=lambda row:row[1])[0]
                if ep==norm:
                    net=N.ReadoutNet(model);net.load_state_dict(cp['state_dicts'][f'{model}_seed{seed}']);reused=True
                else:
                    net,_=R.run_fit(D,Xf,bf,model,seed,args,'rf',ep);nrefit+=1;reused=False
                (q,c,p),branch,aa,bb,ops,diag=R.evaluate_seed(net,D,Xf,bf);allops.append(ops)
                for suffix,value in [('Q',q),('CRPS',c),('p',p),('a',aa),('b',bb)]:arrays[key+model+f'_seed{seed}_'+suffix]=value
                states[f'{model}_seed{seed}']=net.state_dict()
                fits.append({'seed':seed,'normalized_epoch':norm,'raw_epoch':ep,'reused_checkpoint':reused,'geometry':diag})
            q=np.empty((len(te),15));c=np.empty(len(te));p=np.empty(len(te))
            for j,bank in enumerate(bf):
                ix=np.flatnonzero(hi==j)
                qq,cc,pp,_,_=R.mixture(torch.tensor(D['z'][te[ix]]),bank['z'],[o[j] for o in allops])
                q[ix]=qq.numpy()*scale[ix,None];c[ix]=cc.numpy()*scale[ix];p[ix]=pp.numpy()
            pred[model]=(q,c,p);models[model]={'fits':fits,'cache_kind':kind,'cache_sha':cacheinfo[kind]['sha256']}
            print(market,model,'changed',sum(not f['reused_checkpoint'] for f in fits),'elapsed',round(time.perf_counter()-start,1),flush=True)
        for model in ['forest_reference_only','rule_qrf','qrf_original','lstm_qr']:
            pred[model]=(old[key+model+'_Q'],old[key+model+'_CRPS'] if key+model+'_CRPS' in old else None,
                         old[key+model+'_p'] if key+model+'_p' in old else None)
        for model in N.MODELS:
            source='normalized_'+model
            pred[source]=(old[key+model+'_Q'],old[key+model+'_CRPS'],old[key+model+'_p'])
        cells={}
        for j,h in enumerate(R.T.HOR):
            ix=hi==j
            cells[str(h)]={model:R.metrics(D['y'][te[ix]],qq[ix],None if cc is None else cc[ix],None if pp is None else pp[ix])
                           for model,(qq,cc,pp) in pred.items()}
        report['markets'][market]={'models':models,'by_horizon':cells}
        for model,(qq,cc,pp) in pred.items():
            arrays[key+model+'_Q']=qq
            if cc is not None:arrays[key+model+'_CRPS']=cc
            if pp is not None:arrays[key+model+'_p']=pp
        torch.save({'state_dicts':states,'models':models,'cache_info':cacheinfo},P/f'{a.tag}_{market}_checkpoints.pt')
        (P/(a.tag+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        np.savez_compressed(P/(a.tag+'_predictions.npz'),**arrays)
    report['summary']={}
    for model in N.MODELS:
        report['summary'][model]={}
        for base in ['forest_reference_only','rule_qrf','qrf_original','lstm_qr','normalized_'+model,*[m for m in N.MODELS if m!=model]]:
            report['summary'][model][base]={}
            for metric in ['MQS','CRPS']:
                rr=[c[model][metric]/c[base][metric] for m in report['markets'].values() for c in m['by_horizon'].values() if c[base][metric] is not None]
                if rr:report['summary'][model][base][metric]={'geometric_gain_percent':float(100*(1-np.exp(np.log(rr).mean()))),
                                                            'wins':int((np.array(rr)<1).sum()),'cells':len(rr)}
    report['refits_changed_epochs']=nrefit
    report['frozen_after']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen};assert before==report['frozen_after']
    report['elapsed_seconds']=time.perf_counter()-start
    (P/(a.tag+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({m:{b:r for b,r in data.items() if b in ['rule_qrf','forest_reference_only','normalized_'+m]} for m,data in report['summary'].items()},indent=2),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    with threadpool_limits(limits=1):main()
