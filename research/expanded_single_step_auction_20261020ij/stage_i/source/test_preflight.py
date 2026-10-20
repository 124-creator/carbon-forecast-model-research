"""Mathematical / information boundary preflight, no real market fitting."""
from __future__ import annotations
import json, numpy as np, torch
from pathlib import Path
from run_single_step import (W,Q,Residual, LOOKBACK, calc_context,validate_data,get_markets,build_cell,SEEDS,ALPHA_GRID)

def main():
    np.random.seed(733);torch.manual_seed(733);torch.set_num_threads(1)
    x=np.random.default_rng(733).normal(size=(300,LOOKBACK))
    u=x-(x@Q)@Q.T
    ortho=float(np.abs(u@Q).max());assert ortho<1e-12
    c=calc_context(x);ct=torch.tensor(c,dtype=torch.float32)
    d=torch.tensor(u,dtype=torch.float32)
    errs={}
    for name in ('comp_zero','raw_zero'):
        mdl=Residual(name)
        if name=='raw_zero': src=torch.tensor(x,dtype=torch.float32)
        else: src=d
        with torch.no_grad():
            atzero=mdl(torch.zeros_like(src),ct)
        errs[name]=float(atzero.abs().max());assert errs[name]<1e-12
    assert np.linalg.matrix_rank(Q)==3
    # Time boundary: after training dates and validation dates end, no t+1 can cross into next role.
    src=get_markets('/mnt/data/carbon_research/recovered_archives','/mnt/data/carbon_next_20261020/data')
    counts=[]
    for m,y in [(k,y) for k in ('EUA_LONG','GUANGZHOU','HUBEI','SHANGHAI') for y in (2021,2022,2023)]+[('CCER_DAILY_AVERAGE',2026)]:
        df,ex=validate_data(src[m][0],m);cc,reason=build_cell(df,y)
        if cc is None:continue
        t=cc['origins'];dates=cc['dates']
        if not (t[cc['train']][-1]+1<t[cc['val']][0] and t[cc['val']][-1]+1<t[cc['test']][0]+1):
            raise AssertionError('Maturity/time boundary')
        assert not np.any(cc['train']&cc['test'])
        assert (cc['dates'][t[cc['final']][-1]+1]<np.datetime64(f'{y}-01-01'))
        assert (cc['dates'][t[cc['test']]+1]<np.datetime64(f'{y+1}-01-01')).all()
        counts.append({'market':m,'year':y,'train':int(cc['train'].sum()),'val':int(cc['val'].sum()),'test':int(cc['test'].sum()),'excluded':len(ex)})
    result={'status':'PASS_SYNTHETIC_AND_TIME_PREFLIGHT','max_projection_dot_gap':ortho,'rank_of_static_plus_last_span':int(np.linalg.matrix_rank(Q)),
            'model_zero_constraints_max_abs':errs,'n_validated_market_years':len(counts),'split_checks':counts,
            'full_market_training_at_this_preflight':0,
            'note':'Dryrun of one old EUA_LONG_2021 cell was conducted beforehand to verify the implementation. Do not treat that cell as blind holdout.'}
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
