# -*- coding: utf-8 -*-
"""Read-only temporal stability of per-origin distributional losses.
Answers: is a gain broad across test time, or driven by a few episodes?
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import neural_restart as B

P=Path(__file__).resolve().parent
MODELS=[('rule_qrf','qrf_round4_probability_control_predictions.npz','qrf_rule_atom'),
        ('qrf_original','qrf_round4_probability_control_predictions.npz','qrf_original'),
        ('lstm_qr','deep_seq_round4_predictions.npz','lstm_qr'),
        ('forest_reference_only','forest_transport_round5_predictions.npz','forest_reference_only'),
        ('forest_transport','forest_transport_round5_predictions.npz','forest_transport'),
        ('honest_no_readout','prior_readout_transport_round5_predictions.npz','honest_no_readout'),
        ('honest_readout','prior_readout_transport_round5_predictions.npz','honest_readout'),
        ('rbf_prior','additive_transport_round5_predictions.npz','rbf_prior'),
        ('linear_prior','additive_transport_round5_predictions.npz','linear_prior'),
        ('ess_shrinkage','ess_shrinkage_round5_predictions.npz','chosen')]
TAUS=np.array(B.TAUS);BASE='rule_qrf'


def pinball(y,q):
    e=y[:,None]-q
    return (2*np.maximum(TAUS*e,(TAUS-1)*e)).mean(1)


def main():
    panel=pd.read_csv(B.PANEL,parse_dates=['date'])
    rows=[];summary={}
    for model,file,key in MODELS:
        path=P/file
        if not path.exists():continue
        source=np.load(path)
        if not all(m+'_'+key+'_Q' in source.files for m in B.CUT):continue
        for market in B.CUT:
            d=panel[panel.market==market].sort_values('date').reset_index(drop=True)
            price=d.price.to_numpy(float);lp=np.log(price);prefix=market+'_'
            t=source[prefix+'origin'];h=source[prefix+'horizon']
            assert np.array_equal(lp[t+h]-lp[t],source[prefix+'target'])
            if model==BASE:base=pinball(source[prefix+'target'],source[prefix+'qrf_rule_atom_Q'])
            else:
                baseline=np.load(P/'qrf_round4_probability_control_predictions.npz')
                base=pinball(baseline[prefix+'target'],baseline[prefix+'qrf_rule_atom_Q'])
                assert np.array_equal(baseline[prefix+'target'],source[prefix+'target'])
            scale=source[prefix+'scale'];y=source[prefix+'target']
            q=source[prefix+key+'_Q'];score=pinball(y,q)
            dates=d.date.to_numpy()[t]
            for horizon in [1,5,10,20]:
                ix=h==horizon
                period=pd.to_datetime(dates[ix]).to_period('Q').astype(str)
                frame=pd.DataFrame({'p':period,'model':score[ix],'base':base[ix]})
                for name,g in frame.groupby('p'):
                    rows.append(dict(market=market,horizon=int(horizon),period=name,model=model,
                                     n=len(g),model_pinball=float(g.model.mean()),base_pinball=float(g.base.mean()),
                                     gain_percent=float(100*(1-g.model.mean()/g.base.mean())),
                                     origin_win_fraction=float((g.model<g.base).mean())))
    df=pd.DataFrame(rows);df.to_csv(P/'TEMPORAL_STABILITY_ROUND5_CELLS.csv',index=False,encoding='utf-8-sig')
    for model in df.model.unique():
        a=df[df.model==model]
        summary[model]={'overall_gain_percent':float(100*(1-np.exp(np.log(a.model_pinball/a.base_pinball).mean()))),
                        'quarters':int(len(a)),'quarters_won':int((a.gain_percent>0).sum()),
                        'mean_origin_win_fraction':float(a.origin_win_fraction.mean()),
                        'cells':int(len(a.groupby(['market','horizon'])))}
        for market in B.CUT:
            m=a[a.market==market]
            summary[model][market]={'gain_percent':float(100*(1-np.exp(np.log(m.model_pinball/m.base_pinball).mean()))),
                                    'quarters_won':int((m.gain_percent>0).sum()),'quarters':int(len(m))}
    report={'base':BASE,'pinball':'per-origin mean over 15 quantiles (MQS numerator), canonical return units',
            'summary':summary,'note':'quarter cells are unweighted within each market/horizon; not a significance test'}
    (P/'TEMPORAL_STABILITY_ROUND5.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    table=pd.DataFrame({m:{'gain%':v['overall_gain_percent'],'quarters_won':f"{v['quarters_won']}/{v['quarters']}",
                           'origin_win_frac':round(v['mean_origin_win_fraction'],3)} for m,v in summary.items()}).T
    print(table.to_string())
    for model in ['honest_no_readout','rbf_prior','forest_reference_only','forest_transport']:
        if model in summary:print(model,{k:round(v['gain_percent'],3) for k,v in summary[model].items() if isinstance(v,dict)})


if __name__=='__main__':main()
