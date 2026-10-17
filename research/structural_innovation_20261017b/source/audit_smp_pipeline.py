#!/usr/bin/env python3
"""Synthetic data-boundary and native-risk objective audit, without fitting.

No load_market call, CSV/NPZ read, model checkpoint load, optimizer instance,
or market metric is used. The outer experiment runner is audited separately.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
from datetime import date, timedelta, datetime, timezone
from pathlib import Path

import numpy as np
import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'audits/SMP_PIPELINE_SYNTHETIC.json')
    args=parser.parse_args()
    source=Path(__file__).resolve().parent
    input_paths=[source/n for n in ['data.py','fit.py','models.py','smp_model.py']]
    source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths}
    sys.path.insert(0,str(source))
    import data
    import fit
    import models
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    checks=[];comparisons=0;maxima={}

    def check(name,ok,detail=None):
        if not bool(ok):raise AssertionError((name,detail))
        checks.append({'name':name,'status':'PASS'})

    def close(name,a,b,category='numeric',atol=1e-11,rtol=1e-10):
        nonlocal comparisons
        a=np.asarray(a);b=np.asarray(b)
        check(name+' shape',a.shape==b.shape,(a.shape,b.shape))
        error=float(np.max(np.abs(a-b))) if a.size else 0.
        maxima[category]=max(error,maxima.get(category,0.));comparisons+=a.size
        check(name+' values',np.allclose(a,b,atol=atol,rtol=rtol),error)

    def raises(name,fn):
        try:fn()
        except (ValueError,TypeError,IndexError):check(name,True)
        else:check(name,False)

    # This entire fixture is generated in memory; it is unrelated to quote files.
    n=1200
    ix=np.arange(n)
    changes=.2*np.sin(ix/13.)+.05*np.cos(ix/7.)
    changes[:90]=0.
    changes[180:240]=0.
    prices=100.+np.cumsum(changes)
    dates=np.array([str(date(2018,1,1)+timedelta(days=int(i))) for i in ix])
    days=np.array([date.fromisoformat(d).toordinal() for d in dates])
    cov=np.column_stack([100.+ix/20.,2.+(ix%7)/10.,10.+(ix%5)])
    market=data.Market('SYNTHETIC_ONLY',dates,days,prices,cov,('a','b','c'),'synthetic-no-source-file')
    annual=data.AnnualData(market,2020)
    receipt=annual.support_receipt()
    check('all train H20 labels precede first validation origin',int(annual.train.max())+20<int(annual.validation.min()))
    check('all validation H20 labels precede annual cut',int(annual.validation.max())+20<annual.start)
    check('all scored H20 labels remain in evaluation year',all(dates[t+20]<'2021-01-01' for t in annual.test))
    check('evaluation begins at first observed annual cut',annual.test[0]==annual.start)
    check('evaluation includes all and only same-year H20 origins',np.array_equal(annual.test,np.arange(annual.start,annual.end-20)))
    check('scale/statistics endpoint precedes validation',annual.stats_end==int(annual.validation[0]))

    # An independent full-prefix loop verifies state construction, including the
    # two generic early zero states and persistence through >20 consecutive zeroes.
    last_sign,age=0,0
    expected=[3]
    for t in range(1,n):
        delta=prices[t]-prices[t-1]
        if abs(delta)>1e-10:last_sign,age=(-1 if delta<0 else 1),0
        else:age=min(age+1,2)
        expected.append((last_sign+1)*3+age)
    close('state_path equals full-prefix oracle',annual.states,np.array(expected),category='state')
    check('constant t60 prefix uses state5',annual.at(60)[2]==5)
    check('zero run does not forget last nonzero sign outside L20',annual.states[239]//3==annual.states[179]//3)
    check('all long-prefix states belong to reachable seven',set(annual.states[60:]).issubset({0,1,2,5,6,7,8}))
    constant_x,constant_s,_=annual.at(60)
    check('constant prefix normalization is finite and positive',constant_s>0 and np.isfinite(constant_x).all())

    # Independently rebuild one full token window with the already-fitted train
    # statistics; no future date, quotation, or covariate is needed.
    t=int(annual.test[0])+17
    start=t-19
    scale=max(float(np.sqrt(np.mean(np.diff(prices[t-60:t+1])**2))),annual.floor)
    expected_x=np.column_stack([
        np.diff(prices[t-20:t+1])/scale,
        np.log1p(np.diff(days[t-20:t+1]))/np.log(8.),
        np.full(20,np.log(scale/annual.train_rms)),
        np.array(expected)[start:t+1]//3-1,
        (np.array(expected)[start:t+1]%3)/2.,
        (np.log(cov[start:t+1])-annual.cov_mean)/annual.cov_std,
    ])
    close('at(t) token features rebuilt independently',annual.at(t)[0],np.clip(expected_x,-10,10),category='features')
    close('at(t) known origin scale independently rebuilt',annual.at(t)[1],scale,category='features')

    suffix_checks=[]
    for t in [int(annual.validation[0]),int(annual.test[0]),int(annual.test[-1])]:
        altered=copy.deepcopy(market)
        altered.price[t+1:]+=50.+np.arange(n-t-1)*.2
        altered.covariates[t+1:]*=4.
        other=data.AnnualData(altered,2020)
        close(f'quote/cov suffix does not alter known feature t{t}',annual.at(t)[0],other.at(t)[0],category='future_suffix')
        close(f'quote/cov suffix does not alter known scale t{t}',annual.at(t)[1],other.at(t)[1],category='future_suffix')
        check(f'quote suffix does not alter known state t{t}',annual.at(t)[2]==other.at(t)[2])
        suffix_checks.append(t)
    # Future dates are shifted after the first issued query, preserving its cut.
    t=int(annual.test[0])
    altered=copy.deepcopy(market)
    altered.days[t+1:]+=2
    altered.dates[t+1:]=[str(date.fromordinal(int(d))) for d in altered.days[t+1:]]
    other=data.AnnualData(altered,2020)
    close('future date suffix does not alter issued feature',annual.at(t)[0],other.at(t)[0],category='future_suffix')

    raises('refuse immature H20 labels',lambda:annual.batch([t-19],t,labels=True))
    check('same immature origin is allowed as label-free input',annual.batch([t-19],t,labels=False)['x'].shape[0]==1)
    raises('refuse future input',lambda:annual.batch([t+1],t,labels=False))
    raises('refuse too-short input prefix',lambda:annual.at(59))
    mature=annual.batch([t-20],t,labels=True)
    close('latest fully mature y is delta divided only by scale',mature['y'],mature['delta']/mature['scale'][:,None],category='target_units')
    check('target has no sqrt horizon factor',not np.allclose(mature['y'][:,1:],mature['delta'][:,1:]/mature['scale'][:,None]/np.sqrt(np.arange(2,21))))

    # The native objective is independently differentiated in mean=delta/s.
    train=annual.batch(annual.train,annual.start-1)
    selected=np.arange(0,len(train['x']),max(1,len(train['x'])//11))
    small={k:v[selected] for k,v in train.items()}
    batch=fit.tensor_batch(small)
    denominator=data.risk_denominator(train)
    expected_d=np.mean(train['delta']**2,axis=0)
    floor=max(np.mean(train['scale']**2)*1e-8,1e-16)
    close('training-only RW horizon denominator',denominator,np.maximum(expected_d,floor),category='risk_denominator')
    denominator_t=torch.tensor(denominator)
    prediction=np.cos(np.arange(len(selected)*20).reshape(len(selected),20)/7.)*.1
    shrink=.1
    objective_rows=[]
    for family in ['gru_native','gru_normalized','smp_full','smp_no_aux','smp_one_step','gru_aux']:
        mean=torch.tensor(prediction,requires_grad=True)
        aux={'q_current':torch.full((len(selected),3),1/3), 'b_current':torch.ones((len(selected),2))}
        losses=fit.objective(family,mean,aux,batch,denominator_t,shrink)
        used=[0] if family=='smp_one_step' else list(range(20))
        scales=small['scale'][:,None]
        if family=='gru_normalized':
            weight=1/np.arange(1,21)[None,:]
            sup=np.mean(weight*(prediction-small['y'])**2)
            pen=np.mean(weight*prediction**2)
            derivative=2*weight*(prediction-small['y']+shrink*prediction)/prediction.size
        else:
            raw=prediction[:,used]*scales
            sup=np.mean((raw-small['delta'][:,used])**2/denominator[used])
            pen=np.mean(raw**2/denominator[used])
            derivative=np.zeros_like(prediction)
            derivative[:,used]=2*(raw-small['delta'][:,used]+shrink*raw)*scales/denominator[used]/raw.size
        aw=.05 if family in ['smp_full','smp_one_step','gru_aux'] else 0.
        nll=np.mean(np.log(3.)+(small['event']!=0)*abs(small['y'][:,0])) if aw else 0.
        expected_total=sup+shrink*pen+aw*nll
        close(f'{family} independent objective components',[float(x.detach()) for x in losses],[expected_total,sup,pen,nll],category='risk_objective')
        losses[0].backward()
        close(f'{family} native objective analytic mean derivative',mean.grad.detach().numpy(),derivative,category='risk_objective_gradient')
        if family=='smp_one_step':check('one_step higher horizon mean derivatives exactly zero',bool(torch.count_nonzero(mean.grad[:,1:])==0))
        objective_rows.append({'family':family,'supervised':sup,'penalty':pen,'nll':nll,'total':expected_total,'mean_derivative_check':'PASS'})
    check('auxiliary matched control has identical coefficient',models.auxiliary_weight('gru_aux')==models.auxiliary_weight('smp_full')==.05)
    check('no_aux removes only NLL coefficient',models.auxiliary_weight('smp_no_aux')==0.)

    # Scoring is checked only on synthetic labels, with no market scores read.
    raw_prediction=prediction*small['scale'][:,None]
    actual_metric=data.metrics(raw_prediction,small)
    all_mse=np.mean((raw_prediction-small['delta'])**2,axis=0)
    all_rw=np.mean(small['delta']**2,axis=0)
    close('independent synthetic raw MSE metric',actual_metric['mse'],all_mse,category='metric')
    close('independent synthetic relative MSE metric',actual_metric['relative_mse'],all_mse/all_rw,category='metric')
    close('primary metric averages only h1/5/20',actual_metric['score'],(all_mse/all_rw)[[0,4,19]].mean(),category='metric')
    source_end={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths}
    check('reviewed sources unchanged during synthetic audit',source_end==source_hashes)
    output={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'PASS','scope':'Synthetic data/fit objective only; outer runner and frozen protocol still require independent review',
            'source_sha256':source_hashes,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'environment':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__},
            'checks':len(checks),'numerical_values_compared':int(comparisons),'maximum_absolute_errors':maxima,'check_records':checks,
            'synthetic_support':receipt,'future_suffix_query_indices':suffix_checks,'objective_components':objective_rows,
            'activity':{'market_file_reads':0,'confirmation_reads':0,'checkpoint_or_prediction_file_reads':0,'model_fits':0,'optimizer_instances':0,'optimizer_steps':0,'old_file_writes':0},
            'limits':['No load_market call was made. The Market object was constructed entirely in memory from arbitrary synthetic prices/dates/covariates.',
                      'Data state_path may precompute the whole supplied development array; suffix perturbations establish prefix invariance of the features read at the checked origins.',
                      'Training scalers use the allowed fitted training prefix. Causality tests perturb after the fixed statistics prefix, not the fitted training data themselves.',
                      'The current source supports local per-market annual models. This test makes no cross-market as-of fusion claim.',
                      'Fit-neural optimizer scheduling, deadline/eligibility, test-label isolation in the outer runner, task counts, and selection need the frozen runner/protocol review.']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'output':str(args.output),'status':'PASS','checks':len(checks),'values':int(comparisons),'maximum_absolute_errors':maxima,'source_sha256':source_hashes,'activity':output['activity']},ensure_ascii=False))


if __name__=='__main__':main()
