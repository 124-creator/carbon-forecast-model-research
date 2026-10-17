"""Aggregate immutable actual development results; no optimizer or price IO."""
from pathlib import Path
import json
from collections import defaultdict
import numpy as np

ROOT=Path(__file__).resolve().parent

def main():
    for stage in ('D1','D2'):
        if not json.loads((ROOT/(stage+'_RUN_COMPLETE.json')).read_text())['all_completed']:
            raise ValueError('Cannot select models from an incomplete development run')
    rows=[]
    for stage in ('D1','D2'):
        for r in json.loads((ROOT/(stage+'_RESULTS.json')).read_text())['results']:
            r['stage']=stage
            rows.append(r)
    groups=defaultdict(list)
    for r in rows:
        t=r['task'];key=(t['scope'],t['config']['id'],t.get('train_cap'))
        groups[key].append(r)
    summary=[]
    for (scope,config_id,cap), members in groups.items():
        seeds=sorted({r['task']['seed'] for r in members})
        views={v for r in members for v in r['metrics']}
        allcells=[dict(c,seed=r['task']['seed']) for r in members for c in r['metrics']['full']['cells']]
        a=dict(scope=scope,config_id=config_id,config=members[0]['task']['config'],train_cap=cap,
               fits=len(members),seeds=seeds,score=float(np.mean([c['relative_mse'] for c in allcells])),
               per_seed={str(s):float(np.mean([c['relative_mse'] for c in allcells if c['seed']==s])) for s in seeds},
               per_market={m:float(np.mean([c['relative_mse'] for c in allcells if c['market']==m])) for m in sorted({c['market'] for c in allcells})},
               per_year={str(y):float(np.mean([c['relative_mse'] for c in allcells if c['year']==y])) for y in sorted({c['year'] for c in allcells})},
               per_horizon={str(h):float(np.mean([c['relative_mse'] for c in allcells if c['horizon']==h])) for h in [1,5,20]},
               views={v:float(np.mean([c['relative_mse'] for r in members if v in r['metrics'] for c in r['metrics'][v]['cells']])) for v in sorted(views)},
               selected_epoch_zero=sum(r['selected_epoch']==0 for r in members),
               task_ids=[dict(stage=r['stage'],task_id=r['task']['id']) for r in members])
        summary.append(a)
    summary.sort(key=lambda a:a['score'])
    d1=json.loads((ROOT/'D1_RESULTS.json').read_text())
    controls={}
    for family in ('rw','stat'):
        cells=[c for k,v in d1['controls'].items() if 'SHANGHAI_D4' in k and 'GUANGZHOU' in k for c in v[family]['cells']]
        assert len(cells)==36
        controls[family]=dict(score=float(np.mean([c['relative_mse'] for c in cells])),cells=cells)
    by_key={(s['scope'],s['config_id'],s['train_cap']):s for s in summary}
    candidate_pool=[by_key[('pooled',f'kcm_h32_s1_l{lam}',None)] for lam in (1,100)]
    assert all(c['seeds']==[17,29,43] and c['fits']==9 for c in candidate_pool)
    candidate=min(candidate_pool,key=lambda c:(c['score'],c['config']['strength']))
    patch=min([s for s in summary if s['config']['family']=='patchtst' and s['train_cap'] is None],key=lambda c:c['score'])
    ridge=min([s for s in summary if s['config']['family']=='ridge'],key=lambda c:c['score'])
    baseline_pool=[by_key[('pooled','gru_price_h16_s0.1_l0',None)],
                   by_key[('pooled','lstm_price_h32_s1_l0',None)],
                   by_key[('local','gru_price_h16_s1_l0',None)],patch,ridge]
    baseline_pool += [dict(config_id=k,scope='deterministic',config=dict(id=k,family=k),score=v['score']) for k,v in controls.items()]
    baseline=min(baseline_pool,key=lambda c:c['score'])
    curve=[]
    for family,cid,scope in [('shared_price_GRU','gru_price_h16_s0.1_l0','pooled'),
                             ('KCM_lambda1','kcm_h32_s1_l1','pooled'),
                             ('local_price_GRU','gru_price_h16_s1_l0','local')]:
        for cap in (100,400,None):
            members=[r for r in rows if r['task']['config']['id']==cid and r['task']['scope']==scope
                     and r['task'].get('train_cap')==cap and r['task']['cutoff']=='2023-01-01'
                     and (scope!='local' or r['task']['seed']==17)]
            seeds=sorted({r['task']['seed'] for r in members})
            seed_scores={str(s):float(np.mean([c['relative_mse'] for r in members if r['task']['seed']==s for c in r['metrics']['full']['cells']])) for s in seeds}
            curve.append(dict(model=family,per_market_train_cap=cap,seeds=seeds,score=float(np.mean(list(seed_scores.values()))),
                              per_seed=seed_scores,period='2023',common_uncapped_training_scale=True))
    out=dict(development_summary=summary,controls=controls,learning_curve=curve,
             selected_candidate=candidate,selected_best_baseline=baseline,selected_patchtst=patch,selected_ridge=ridge,
             no_confirmation_data_read=True,model_selection_is_development=True)
    (ROOT/'DEVELOPMENT_AGGREGATE.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(candidate={k:candidate[k] for k in ['config_id','score','per_seed']},
          baseline={k:baseline[k] for k in ['config_id','scope','score']},
          patchtst={k:patch[k] for k in ['config_id','score']},learning_curve=curve),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
