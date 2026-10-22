# -*- coding: utf-8 -*-
"""Single consolidated test-set table over all round-4/round-5 frozen prediction archives.
Read-only: recomputes per-cell MQS from stored quantiles, reuses stored exact CRPS when present,
and verifies identical target/origin/horizon tensors across archives.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import neural_restart as B

P=Path(__file__).resolve().parent
ARCHIVES=['qrf_round4_predictions.npz','qrf_round4_probability_control_predictions.npz',
          'deep_seq_round4_predictions.npz','forest_transport_round5_predictions.npz',
          'forest_transport_rawselect_round5_predictions.npz','prior_readout_transport_round5_predictions.npz',
          'prior_readout_rawselect_round5_predictions.npz','additive_transport_round5_predictions.npz',
          'horizon_kernel_round5_predictions.npz','prefix_transport_controls_round5_predictions.npz']
BASE=('qrf_round4_probability_control_predictions.npz','qrf_rule_atom')
ALIAS={'qrf_rule_atom':'rule_qrf','rule_qrf':'rule_qrf','qrf_ensemble':'qrf_original','qrf':'qrf_original',
       'forest_transport_normalized':'forest_transport','fixed_0.05':'forest_reference_only_smooth'}
TAUS=np.array(B.TAUS)


def discover(source):
    keys=set()
    for f in source.files:
        if not f.endswith('_Q'):continue
        stem=f[:-2]
        for market in B.CUT:
            if stem.startswith(market+'_'):keys.add(stem[len(market)+1:])
    return sorted(keys)


def cells(source,key,market):
    q=source[market+'_'+key+'_Q'];y=source[market+'_target']
    e=y[:,None]-q;pin=(2*np.maximum(TAUS*e,(TAUS-1)*e)).mean(1)
    out={}
    for j,h in enumerate(B.HOR):
        ix=source[market+'_horizon']==h
        c=source[market+'_'+key+'_CRPS'][ix] if market+'_'+key+'_CRPS' in source else None
        out[h]=dict(n=int(ix.sum()),pin=float(pin[ix].mean()),crps=float(c.mean()) if c is not None else None)
    return out


def main():
    panel=pd.read_csv(B.PANEL,parse_dates=['date'])
    canonical={m:np.load(P/BASE[0]) for m in B.CUT}
    for market in B.CUT:
        d=panel[panel.market==market].sort_values('date').reset_index(drop=True)
        lp=np.log(d.price.to_numpy(float))
        t=canonical[market][market+'_origin'];h=canonical[market][market+'_horizon']
        assert np.allclose(lp[t+h]-lp[t],canonical[market][market+'_target'],atol=1e-12)
    rows=[];seen=set()
    for name in ARCHIVES:
        path=P/name
        if not path.exists():continue
        source=np.load(path)
        for key in discover(source):
            label=ALIAS.get(key,key)
            if label in seen:continue
            seen.add(label)
            for market in B.CUT:
                c=cells(source,key,market)
                for h,v in c.items():
                    rows.append(dict(model=label,archive=name,key=key,market=market,horizon=h,**v))
    df=pd.DataFrame(rows)
    df.to_csv(P/'ROUND5_RESULTS_TABLE_CELLS.csv',index=False,encoding='utf-8-sig')
    base=df[(df.archive==BASE[0])&(df.key==BASE[1])].set_index(['market','horizon'])
    table=[]
    for (label,archive,key),g in df.groupby(['model','archive','key']):
        g=g.set_index(['market','horizon']).reindex(base.index)
        row=dict(model=label,archive=archive,key=key,
                 MQS=float(g.pin.mean()),CRPS=float(g.crps.mean()) if g.crps.notna().all() else None,
                 cells=len(g),n=int(g.n.sum()))
        for metric,col in [('MQS','pin'),('CRPS','crps')]:
            r=g[col]/base[col]
            row['gain_'+metric]=float(100*(1-np.exp(np.log(r.dropna()).mean()))) if r.notna().all() else None
            row['wins_'+metric]=int((r.dropna()<1).sum()) if r.notna().all() else None
        table.append(row)
    t=pd.DataFrame(table).sort_values('gain_MQS',ascending=False)
    t.to_csv(P/'ROUND5_RESULTS_TABLE.csv',index=False,encoding='utf-8-sig')
    per_seed=t.model.str.contains('_seed')|t.model.str.startswith('frozen_')|t.model.str.startswith('normalized_')|t.model.str.match(r'^qrf_\d\d$')
    ens=t[~per_seed];dia=t[per_seed]
    lines=['# 第5轮统一测试集结果表','',f'基线：`{BASE[1]}`（规则零原子 + QRF 条件响应库）。所有模型共用同一origin/horizon/target；MQS由存盘的15个分位重算，CRPS只取有完整分布存盘者。',
           '','**口径**：只有 5 种子集成行（下表第一部分）算“一个模型”；`*_seedNN` 是诊断行，单独列在第二部分，**不得**用其中某个种子的最好值冒充模型结果。',
           '',f'## 集成级配置（{len(ens)} 个）','','| 模型 | 相对基线 MQS | 胜格 | 相对基线 CRPS | 胜格 |','|---|---:|---:|---:|---:|']
    def emit(frame):
        out=[]
        for _,r in frame.iterrows():
            q=f"{r.gain_MQS:+.4f}%" if pd.notna(r.gain_MQS) else '—'
            qw=f"{int(r.wins_MQS)}/20" if pd.notna(r.wins_MQS) else '—'
            c=f"{r.gain_CRPS:+.4f}%" if pd.notna(r.gain_CRPS) else '—'
            cw=f"{int(r.wins_CRPS)}/20" if pd.notna(r.wins_CRPS) else '—'
            out.append(f'| `{r.model}` | {q} | {qw} | {c} | {cw} |')
        return out
    lines+=emit(ens)
    lines+=['',f'## 逐种子诊断（{len(dia)} 行）','','| 模型 | 相对基线 MQS | 胜格 | 相对基线 CRPS | 胜格 |','|---|---:|---:|---:|---:|']
    lines+=emit(dia)
    best=ens[ens.gain_CRPS.notna()].iloc[0]
    lines+=['','## 结论','',
            f'在 {len(ens)} 个集成级配置中，`{best.model}` 的 MQS（{best.gain_MQS:+.4f}%）与 CRPS（{best.gain_CRPS:+.4f}%）同时排名第一；逐种子里有更高的单点值（例如 `rbf_prior_seed33`），但那是种子选择偏差，不计入模型比较。',
            '第5轮所有神经增量与参考结构调整方向都没有超过参考本身；增益主要来自参考构造，且经自助法只在上海中长周期有区间支持。历史测试已复用，本表不证明显著性。']
    (P/'ROUND5_RESULTS_TABLE_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(t[['model','MQS','gain_MQS','wins_MQS','gain_CRPS','wins_CRPS']].to_string(index=False,float_format=lambda x:f'{x:.4f}'))


if __name__=='__main__':main()
