# -*- coding: utf-8 -*-
"""Secondary price-unit evaluation of the same frozen forecast quantiles.
Primary return-unit model rankings are retained; this does not select models or
create CRPS from fifteen quantiles. Point RMSE uses the common median forecast.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import neural_restart as B

P=Path(__file__).resolve().parent
SOURCES=[
 ('qrf','qrf_round4_predictions.npz','qrf_ensemble'),
 ('rule_qrf','qrf_round4_probability_control_predictions.npz','qrf_rule_atom'),
 ('lstm_qr','deep_seq_round4_predictions.npz','lstm_qr'),
 ('uniform_seq_transport','seq_transport_round4_predictions.npz','seq_transport'),
 ('forest_reference','forest_transport_round5_predictions.npz','forest_reference_only'),
 ('forest_transport','forest_transport_round5_predictions.npz','forest_transport'),
 ('forest_rawselect','forest_transport_rawselect_round5_predictions.npz','rawselect'),
 ('honest_readout','prior_readout_transport_round5_predictions.npz','honest_readout'),
 ('honest_no_readout','prior_readout_transport_round5_predictions.npz','honest_no_readout'),
 ('insample_readout','prior_readout_transport_round5_predictions.npz','insample_readout'),
 ('honest_readout_reweight_only','prior_readout_transport_round5_predictions.npz','honest_readout_reweight_only')]


def main():
    panel=pd.read_csv(B.PANEL,parse_dates=['date']);rows=[];missing=[]
    for model,path,key in SOURCES:
        if not (P/path).exists():missing.append(model);continue
        source=np.load(P/path)
        if not all(m+'_'+key+'_Q' in source for m in B.CUT):missing.append(model);continue
        for market in B.CUT:
            d=panel[panel.market==market].sort_values('date').reset_index(drop=True)
            price=d.price.to_numpy(float);lp=np.log(price);prefix=market+'_'
            t=source[prefix+'origin'];h=source[prefix+'horizon'];y=source[prefix+'target']
            assert np.array_equal(lp[t+h]-lp[t],y)
            q=source[prefix+key+'_Q'];now=price[t];actual=price[t+h];qp=now[:,None]*np.exp(q)
            assert np.isfinite(qp).all() and np.all(np.diff(qp,axis=1)>=-1e-8)
            for horizon in [1,5,10,20]:
                ix=h==horizon;error=actual[ix]-qp[ix,7]
                rows.append({'market':market,'horizon':horizon,'model':model,'n':int(ix.sum()),
                    'return_MQS':float(B.score(y[ix],q[ix]).mean()),
                    'price_MQS':float(B.score(actual[ix],qp[ix]).mean()),
                    'price_WIS':float(B.wis(actual[ix],qp[ix]).mean()),
                    'median_price_MAE':float(np.abs(error).mean()),
                    'median_price_RMSE':float(np.sqrt(np.mean(error**2)))})
    df=pd.DataFrame(rows);df.to_csv(P/'PRICE_SCALE_ROUND5_CELLS.csv',index=False,encoding='utf-8-sig')
    base=df[df.model=='rule_qrf'];out=[]
    for model in df.model.unique():
        a=df[df.model==model].merge(base,on=['market','horizon'],suffixes=('_model','_base'))
        assert len(a)==20 and np.array_equal(a.n_model,a.n_base)
        for metric in ['return_MQS','price_MQS','price_WIS','median_price_MAE','median_price_RMSE']:
            rr=a[metric+'_model']/a[metric+'_base']
            out.append({'model':model,'baseline':'rule_qrf','metric':metric,'cells':len(rr),
                        'wins':int((rr<1).sum()),'geometric_gain_percent':float(100*(1-np.exp(np.log(rr).mean())))})
    result=pd.DataFrame(out);result.to_csv(P/'PRICE_SCALE_ROUND5_SUMMARY.csv',index=False,encoding='utf-8-sig')
    report={'status':'secondary price-scale evaluation; historically reused tests',
            'primary_retained':'raw log-return MQS remains original model criterion',
            'mapping':'P_quantile(t+h,tau)=P_t*exp(return_quantile(t+h,tau))',
            'point_metric':'MAE and RMSE both use median price; no unidentified mean from15q',
            'no_price_CRPS_inferred_from15q':True,'incomplete_sources_skipped':missing,'summary':out}
    (P/'PRICE_SCALE_ROUND5.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    table=result.pivot(index='model',columns='metric',values='geometric_gain_percent')
    text=['# 同一预测的价格单位补充结果','','原主要指标仍为log-return MQS；本表仅补充实际价格尺度，不据此切换主要指标或挑模型。',
          '全部价格分位数由P_t*exp(q_return)得到，目标保持原canonical panel。MAE/RMSE均针对共同中位数预测，不把有限15q伪装成完整分布均值或CRPS。',
          '', '下表相对rule-QRF，正值表示该模型误差更低，20个市场×期限格等权几何改善。','',
          '| 模型 | return MQS | price MQS | price WIS | median-price MAE | median-price RMSE |','|---|---:|---:|---:|---:|---:|']
    for model,r in table.iterrows():
        text.append('| '+model+' | '+' | '.join(f'{r[m]:+.4f}%' for m in ['return_MQS','price_MQS','price_WIS','median_price_MAE','median_price_RMSE'])+' |')
    (P/'PRICE_SCALE_ROUND5_REPORT_ZH.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(table.to_string())


if __name__=='__main__':main()
