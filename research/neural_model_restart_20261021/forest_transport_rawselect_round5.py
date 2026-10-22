# -*- coding: utf-8 -*-
"""Secondary epoch-selection experiment: reuse existing inner trajectories.
Only raw-criterion-selected epochs that differ are refitted; old artifacts stay
frozen. Criterion and traces were recorded before this experiment's test scoring.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
import forest_transport_round5 as B
R=B.R;P=Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',default='forest_transport_rawselect_round5');a=ap.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);B.activate()
    meta=json.loads((P/'forest_transport_round5.json').read_text(encoding='utf-8'))
    old=np.load(P/'forest_transport_round5_predictions.npz')
    panel=pd.read_csv(R.T.B.PANEL,parse_dates=['date'])
    args=argparse.Namespace(min_epochs=30,patience=16,epochs=64)
    frozen=[P/'forest_transport_round5.py',P/'forest_transport_round5.json',P/'forest_transport_round5_predictions.npz']
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen}
    report={'status':'secondary validation-criterion experiment; no new inner fits',
            'selection':'minimum visited inner mean_h log(rawMQS_h/zerohead_reference_rawMQS_h)',
            'same':'reference, model, training objective, data, seeds and refit protocol unchanged',
            'frozen_before':before,'markets':{}}
    arrays={};start=time.perf_counter();nrefit=0
    for market in R.T.B.CUT:
        d=panel[panel.market==market].sort_values('date').reset_index(drop=True);D=R.S.build(d)
        prepared,_,ci=B.load_market(D,market);Xf,bf=prepared['final']
        checkpoint=torch.load(P/f'forest_transport_round5_{market}_checkpoints.pt',map_location='cpu',weights_only=False)
        te=D['te'];hi=D['hi'][te];scale=D['scale'][te];key=market+'_'
        for suffix,value in [('origin',D['t'][te]),('horizon',R.T.HOR[hi]),('target',D['y'][te]),('scale',scale)]:
            assert np.array_equal(value,old[key+suffix]);arrays[key+suffix]=value
        allops=[];fits=[];states={}
        for info in meta['markets'][market]['fits']['forest_transport']:
            seed=info['seed'];ep=info['best_raw_epoch'];norm=info['best_epoch']
            assert ep==min(info['raw_relative_validation_history'],key=lambda row:row[1])[0]
            if ep==norm:
                net=R.TransportNet('forest_transport');net.load_state_dict(checkpoint['state_dicts'][f'forest_transport_seed{seed}']);reused=True
            else:
                net,_=R.run_fit(D,Xf,bf,'forest_transport',seed,args,'rf',ep);nrefit+=1;reused=False
            (q,c,p),branch,aa,bb,ops,diag=R.evaluate_seed(net,D,Xf,bf);allops.append(ops)
            for suffix,value in [('Q',q),('CRPS',c),('p',p),('a',aa),('b',bb)]:arrays[key+f'rawselect_seed{seed}_'+suffix]=value
            states[str(seed)]=net.state_dict()
            fits.append({'seed':seed,'normalized_epoch':norm,'raw_epoch':ep,'reused_checkpoint':reused,'geometry':diag})
        q=np.empty((len(te),15));c=np.empty(len(te));p=np.empty(len(te))
        for j,bank in enumerate(bf):
            ix=np.flatnonzero(hi==j)
            qq,cc,pp,_,_=R.mixture(torch.tensor(D['z'][te[ix]]),bank['z'],[o[j] for o in allops])
            q[ix]=qq.numpy()*scale[ix,None];c[ix]=cc.numpy()*scale[ix];p[ix]=pp.numpy()
        pred={'rawselect':(q,c,p)}
        for model,source in [('normalized','forest_transport'),('reference','forest_reference_only'),('rule_qrf','rule_qrf'),('qrf','qrf_original'),('lstm_qr','lstm_qr')]:
            pred[model]=(old[key+source+'_Q'],old[key+source+'_CRPS'] if key+source+'_CRPS' in old else None,
                         old[key+source+'_p'] if key+source+'_p' in old else None)
        cells={}
        for j,h in enumerate(R.T.HOR):
            ix=hi==j
            cells[str(h)]={model:R.metrics(D['y'][te[ix]],qq[ix],None if cc is None else cc[ix],None if pp is None else pp[ix])
                           for model,(qq,cc,pp) in pred.items()}
        report['markets'][market]={'fits':fits,'by_horizon':cells,'reference_sha':ci['sha256']}
        for model,(qq,cc,pp) in pred.items():
            arrays[key+model+'_Q']=qq
            if cc is not None:arrays[key+model+'_CRPS']=cc
            if pp is not None:arrays[key+model+'_p']=pp
        torch.save({'states':states,'fits':fits,'reference_sha':ci['sha256']},P/f'{a.tag}_{market}_checkpoints.pt')
        (P/(a.tag+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        np.savez_compressed(P/(a.tag+'_predictions.npz'),**arrays)
        print(market,'changed',sum(not f['reused_checkpoint'] for f in fits),'elapsed',round(time.perf_counter()-start,1),flush=True)
    report['summary']={}
    for base in ['normalized','reference','rule_qrf','qrf','lstm_qr']:
        report['summary'][base]={}
        for metric in ['MQS','CRPS']:
            rr=[c['rawselect'][metric]/c[base][metric] for m in report['markets'].values() for c in m['by_horizon'].values() if c[base][metric] is not None]
            if rr:report['summary'][base][metric]={'geometric_gain_percent':float(100*(1-np.exp(np.log(rr).mean()))),
                                                   'wins':int((np.array(rr)<1).sum()),'cells':len(rr)}
    report['refits_changed_epochs']=nrefit;report['inner_retraining_count']=0
    report['frozen_after']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen};assert before==report['frozen_after']
    report['elapsed_seconds']=time.perf_counter()-start
    (P/(a.tag+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report['summary'],indent=2),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    with threadpool_limits(limits=1):main()
