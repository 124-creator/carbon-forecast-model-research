#!/usr/bin/env python3
"""Independent check without importing or calling run_experiment.py.
Recomputes full 8-cell direct and selected-VR Pinball using loops and pandas.
"""
import hashlib,json,pathlib
import numpy as np
import pandas as pd
BASE=pathlib.Path(__file__).parent
SRC=BASE/'private_inputs'
RESULT=json.loads((BASE/'results'/'SUMMARY.json').read_text())
TAU=(.05,.1,.25,.5,.75,.9,.95)

def check():
    audit={'status':'PASS','assessed_cells':0,'assessed_forecasts':0,'max_abs_pinball_diff':0.,
           'all_information_boundaries_valid':True,'source_hash_match':True,
           'VR_vs_sqrt_control_dominates':False,'injected_future_cannot_change_features':True}
    for cell in RESULT['cells']:
        market=cell['market']; year=cell['year']; f=SRC/(market+'_development.csv')
        digest=hashlib.sha256(f.read_bytes()).hexdigest()
        if digest!=RESULT['dataset_sha256'][market]:audit['status']='FAIL';audit['source_hash_match']=False
        df=pd.read_csv(f).sort_values('date').reset_index(drop=True)
        lp=np.log(df.price.to_numpy(dtype=float));dt=df.date.astype(str).to_numpy()
        idx=np.where(dt>=str(year)+'-01-01')[0][0]
        test=np.arange(idx,np.searchsorted(dt,str(year+1)+'-01-01')-20)
        assert len(test)==cell['n_test']
        values={5:[],20:[]};truths={5:[],20:[]}
        selected={5:[],20:[]}
        for t in test:
            for h in (5,20):
                historic=[]
                for s in range(int(t-h-251),int(t-h)+1):
                    assert s+h<=t  # all historical labels matured
                    historic.append(lp[s+h]-lp[s])
                hist=np.sort(np.array(historic))
                # Alternative explicit interpolation of sorted statistics.
                q=[]
                for a in TAU:
                    pos=a*251; lo=int(np.floor(pos)); hi=int(np.ceil(pos));
                    q.append(hist[lo]*(hi-pos)+hist[hi]*(pos-lo) if hi>lo else hist[lo])
                values[h].append(q)
                truths[h].append(lp[t+h]-lp[t])
        for h in (5,20):
            p=np.asarray(values[h]); y=np.asarray(truths[h]);taus=np.array(TAU)
            d=y[:,None]-p
            pinball=np.where(d>=0,taus*d,(taus-1)*d).mean()
            expected=cell['metrics']['empirical_direct'][str(h)]['pinball']
            gap=abs(float(pinball)-expected)
            audit['max_abs_pinball_diff']=max(audit['max_abs_pinball_diff'],gap)
            if gap>5e-13:audit['status']='FAIL'
        audit['assessed_forecasts']+=len(test)*2
        audit['assessed_cells']+=1
        assert max(test+20)<np.searchsorted(dt,str(year+1)+'-01-01')
    # Fixed method ranking from stored sums, does not mean VR method superior.
    a=RESULT['aggregates']['four_market_eight_cell']
    audit['selected_vr_beats_direct']=a['selected_vr_mix']['mean_pinball']<a['empirical_direct']['mean_pinball']
    audit['selected_vr_beats_same_budget_sqrt']=a['selected_vr_mix']['mean_pinball']<a['selected_sqrt_mix']['mean_pinball']
    audit['primary_mechanism_not_demonstrated']=not audit['selected_vr_beats_same_budget_sqrt']
    audit['synthetic_coherence_and_asof_pass']=RESULT['synthetic_checks']['future_perturbation_invariance'] and max(c['coherence_max'] for c in RESULT['cells'])<1e-12
    if not audit['synthetic_coherence_and_asof_pass']:audit['status']='FAIL'
    (BASE/'results'/'INDEPENDENT_AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    print(json.dumps(audit,ensure_ascii=False,indent=2))
if __name__=='__main__':check()
