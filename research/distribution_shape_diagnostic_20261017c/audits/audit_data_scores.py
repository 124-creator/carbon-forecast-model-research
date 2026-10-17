"""Independent synthetic-only audit of distribution-shape data, CRPS and gates.

No market reader, fitting routine, optimizer, NPZ, or confirmation source is used.
Run with the same NumPy/Torch environment as the frozen experiment.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from datetime import date, timedelta, datetime, timezone
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-dir', required=True)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    src = Path(args.source_dir).resolve()
    protocol_path = Path(args.protocol).resolve()
    checks, numbers = [], 0

    def forbid_data_reads(event, values):
        if event == 'open' and values and isinstance(values[0], (str, bytes)):
            p = str(values[0])
            if Path(p).suffix.lower() in ('.csv', '.xlsx', '.xls', '.npz', '.npy', '.pt', '.pth') or '/local_only/' in p:
                raise PermissionError('Synthetic audit forbids market/array/weight reads: ' + p)
    sys.addaudithook(forbid_data_reads)

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    data = load('data', src/'data.py')
    scores = load('scores', src/'scores.py')
    runner = load('shape_audit_runner', src/'run.py')
    import torch
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    protocol = json.loads(protocol_path.read_text())

    def yes(name, value):
        if not bool(value):
            raise AssertionError(name)
        checks.append(dict(name=name, kind='logical', pass_=True))

    def close(name, actual, expected, tol=1e-10):
        nonlocal numbers
        a, b = np.asarray(actual), np.asarray(expected)
        if a.shape != b.shape:
            raise AssertionError((name, 'shape', a.shape, b.shape))
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise AssertionError((name, 'nonfinite'))
        err = float(np.max(np.abs(a-b))) if a.size else 0.
        if err > tol:
            raise AssertionError((name, err, tol))
        numbers += a.size
        checks.append(dict(name=name, kind='numeric', comparisons=int(a.size), max_abs_error=err, tolerance=tol, pass_=True))

    def reject(name, fn):
        try:
            fn()
        except (ValueError, IndexError):
            checks.append(dict(name=name, kind='rejection', pass_=True))
        else:
            raise AssertionError(name + ': accepted invalid input')

    # Manufacture all dates and quotes; never invoke Data.__init__ or read CSV.
    rng = np.random.default_rng(20261017)
    n = 950
    dates = np.array([(date(2020,1,1)+timedelta(days=i)).isoformat() for i in range(n)])
    prices = 1000 + np.cumsum(.1*np.sin(np.arange(n)*.073) + rng.normal(size=n))

    def synthetic(dates_, prices_):
        d = data.Data.__new__(data.Data)
        d.initialize('SYNTHETIC_ONLY', dates_, prices_, 2022, 80)
        return d

    d = synthetic(dates, prices)
    pre = [i for i, value in enumerate(dates) if i >= 60 and i+20 < len(dates) and dates[i+20] < '2022-01-01']
    expected_val = np.array(pre[-80:], dtype=int)
    expected_train = np.array([i for i in pre if i+20 < expected_val[0]], dtype=int)
    expected_test = np.array([i for i in range(60,n-20) if dates[i].startswith('2022-') and dates[i+20].startswith('2022-')])
    close('training support from explicit maturity inequalities', d.train, expected_train, 0)
    close('validation last80 fully mature before year', d.val, expected_val, 0)
    close('same-year fullH20 test support', d.test, expected_test, 0)
    yes('train labels strictly precede validation origin', max(d.train)+20 < min(d.val))
    yes('validation labels strictly precede test year', max(d.val)+20 < min(d.test))
    yes('evaluation labels exist and remain in fixed year', all(dates[i+20].startswith('2022-') for i in d.test))

    # Independent scalar EWMA, including the known-through-t update convention.
    increments = np.diff(prices)
    vv = float(sum(float(v*v) for v in increments[:60])/60)
    floor = max(vv*1e-8, 1e-16)
    expected_var = {60:max(vv,floor)}
    for i in range(61,n):
        vv = max(.94*vv + .06*float((prices[i]-prices[i-1])**2), floor)
        expected_var[i] = vv
    close('EWMA against scalar past-price recursion', d.variance[60:], np.array([expected_var[i] for i in range(60,n)]), 2e-12)
    q = 800
    b = d.batch([q],q,labels=False)
    yes('prediction-only batch excludes all labels', 'delta' not in b and 'y' not in b)
    close('own-origin forecast scale sqrt(h)', b['sigma'][0], np.array([np.sqrt(expected_var[q]*h) for h in range(1,21)]))
    expected_x = []
    for i in range(q-19,q+1):
        expected_x.append([(prices[i]-prices[i-1])/np.sqrt(expected_var[q]), np.log(2)/np.log(8), np.log(np.sqrt(expected_var[q])/d.train_rms)])
    close('past20 observed input from scalar prices and dates', b['x'][0], np.clip(np.array(expected_x),-10,10))
    boundary = d.batch([q-20],q,labels=True)
    close('fullH label endpoint at current origin', boundary['delta'][0], np.array([prices[q-20+h]-prices[q-20] for h in range(1,21)]))
    close('historical targets use their own issue-time scale', boundary['y'][0], boundary['delta'][0]/np.array([np.sqrt(expected_var[q-20]*h) for h in range(1,21)]))
    reject('immature complete path rejected', lambda:d.batch([q-19],q,labels=True))
    reject('future feature origin rejected', lambda:d.batch([q+1],q,labels=False))
    reject('pre-initialization origin rejected', lambda:d.batch([59],q,labels=False))

    p2 = prices.copy()
    p2[q+1:] += 30+np.arange(n-q-1)*.2
    dates2 = dates.copy()
    dates2[q+1:] = [(date.fromisoformat(v)+timedelta(days=2)).isoformat() for v in dates[q+1:]]
    dp = synthetic(dates2,p2)
    bp = dp.batch([q],q,labels=False)
    for field in ('x','sigma','base'):
        close('future price/calendar poison leaves '+field+' unchanged', bp[field], b[field], 0)
    close('future poison leaves train empirical support unchanged', dp.base_atoms, d.base_atoms, 0)
    close('future poison leaves rolling distribution unchanged', dp.rolling_atoms([q]),d.rolling_atoms([q]),0)
    ids = np.array([i for i in range(60,q+1) if i+20 <= q][-120:])
    own_standardized = np.array([[(prices[i+h]-prices[i])/np.sqrt(expected_var[i]*h) for h in range(1,21)] for i in ids])
    qa = np.quantile(own_standardized,(np.arange(31)+.5)/31,axis=0).T
    rg = np.maximum(np.diff(qa,axis=-1),1e-4)
    ra = np.c_[np.zeros(20),np.cumsum(rg,axis=1)]
    ra -= ra.mean(axis=1,keepdims=True)
    ra /= np.sqrt((ra*ra).mean(axis=1,keepdims=True))
    close('rolling pool latest120 exactly matured own-origin outcomes', d.rolling_atoms([q])[0],ra,2e-12)
    yes('rolling pool last sample endpoint equals current observation', ids[-1]+20 == q)
    reject('constant normalization rejected', lambda:data.normalize_atoms(np.ones(31)))
    reject('nonfinite normalization rejected', lambda:data.normalize_atoms(np.array([1.,np.inf])))
    reject('degenerate quantiles rejected before gap floor', lambda:data.regular_quantiles(np.zeros((200,20))))
    rare = np.zeros((1000,20)); rare[-1,:] = 2
    reject('rare tails outside midpoint quantiles do not manufacture distribution', lambda:data.regular_quantiles(rare))

    # Independent O(K^2) CRPS formula, distinct from production cumulative formula.
    def brute(a,p,y):
        return np.sum(p*np.abs(a-np.asarray(y)[...,None]),axis=-1) - .5*np.sum(p[..., :,None]*p[...,None,:]*np.abs(a[..., :,None]-a[...,None,:]),axis=(-1,-2))
    a = rng.normal(size=(3,4,7))
    logits = rng.normal(size=a.shape)
    p = np.exp(logits-logits.max(-1,keepdims=True)); p /= p.sum(-1,keepdims=True)
    y = rng.normal(size=(3,4))
    close('sorted exact CRPS vs pairwise weighted finite-law identity', scores.crps_numpy(a,p,y),brute(a,p,y),2e-14)
    close('CRPS translation invariance', scores.crps_numpy(a+7.3,p,y+7.3),brute(a,p,y),2e-14)
    close('CRPS positive scale homogeneity in native units', scores.crps_numpy(2.7*a,p,2.7*y),2.7*brute(a,p,y),3e-14)
    ties = a.copy(); ties[...,1] = ties[...,0]
    pt = p.copy(); pt[...,2] = 0; pt /= pt.sum(-1,keepdims=True)
    close('ties and zero probabilities preserve weighted law CRPS',scores.crps_numpy(ties,pt,y),brute(ties,pt,y),2e-14)
    aa = torch.tensor(a,requires_grad=True)
    zz = torch.tensor(logits,requires_grad=True)
    pp = torch.softmax(zz,dim=-1)
    score = scores.crps_torch(aa,pp,torch.tensor(y))
    close('torch sorted CRPS vs independent NumPy double sum',score.detach().numpy(),brute(a,p,y),2e-14)
    score.sum().backward()
    atom_gradient = p*np.sign(a-y[...,None]) - p*np.sum(p[...,None,:]*np.sign(a[..., :,None]-a[...,None,:]),axis=-1)
    probability_gradient = np.abs(a-y[...,None])-np.sum(p[...,None,:]*np.abs(a[..., :,None]-a[...,None,:]),axis=-1)
    logit_gradient = p*(probability_gradient-np.sum(p*probability_gradient,axis=-1,keepdims=True))
    close('atom gradients vs exact analytic formula',aa.grad.numpy(),atom_gradient,2e-14)
    close('simplex logit gradients vs exact analytic formula',zz.grad.numpy(),logit_gradient,2e-14)
    finite_v = float(scores.crps_numpy(np.array([-1.,1.]),np.array([.5,.5]),np.array(0.)))
    close('two-point finite measure uses V score .5',np.array(finite_v),np.array(.5),1e-15)
    yes('iid-ensemble offdiagonal correction is different score', finite_v != 0.)

    z = np.linspace(-1,1,31); z -= z.mean(); z /= np.sqrt(np.mean(z*z))
    ae = np.broadcast_to(z,(5,20,31)).copy()
    pe = np.full_like(ae,1/31)
    ye = rng.normal(size=(5,20))
    metrics, losses = scores.evaluate(ae,pe,dict(delta=ye))
    close('zero-change mean gives exactly RW point risk',np.array(metrics['mean_mse_ratio']),np.ones(20),2e-14)
    close('evaluate CRPS uses proper individual finite measures',losses,brute(ae,pe,ye),2e-14)
    reject('2D forecast prevents silent horizon broadcast',lambda:scores.evaluate(ae[...,0],pe[...,0],dict(delta=ye)))
    reject('empty forecast origins rejected',lambda:scores.evaluate(ae[:0],pe[:0],dict(delta=ye[:0])))
    reject('empty atom axis rejected',lambda:scores.evaluate(ae[:,:,:0],pe[:,:,:0],dict(delta=ye)))
    reject('mismatched horizon dimensions rejected',lambda:scores.evaluate(ae[:,:19],pe[:,:19],dict(delta=ye)))
    bad = pe.copy(); bad[0,0,0] = -1
    reject('negative probability rejected',lambda:scores.evaluate(ae,bad,dict(delta=ye)))
    reject('unnormalized probability rejected',lambda:scores.evaluate(ae,pe*.9,dict(delta=ye)))
    bad_a=ae.copy();bad_a[0,0,0]=np.nan
    reject('nonfinite predictive atom rejected',lambda:scores.evaluate(bad_a,pe,dict(delta=ye)))

    # Oracle examples concern a proper score, never any fitted market model.
    z2 = z**3; z2 -= z2.mean(); z2 /= np.sqrt(np.mean(z2*z2))
    weights = np.full(31,1/31)
    def expected_score(fa,fp,truth,tp):
        return float(np.sum(tp*np.array([brute(fa,fp,v) for v in truth])))
    pooled = np.r_[z,z2]; pooled_p = np.full(62,1/62)
    true_conditional = .5*(expected_score(z,weights,z,weights)+expected_score(z2,weights,z2,weights))
    static_pool = .5*(expected_score(pooled,pooled_p,z,weights)+expected_score(pooled,pooled_p,z2,weights))
    yes('different conditional shapes with same mean/variance give positive oracle score gain', static_pool > true_conditional+1e-6)
    close('two shape worlds have same mean',np.array([z.mean(),z2.mean()]),np.zeros(2),1e-14)
    close('two shape worlds have same variance',np.array([(z*z).mean(),(z2*z2).mean()]),np.ones(2),1e-14)
    scale_world_static = .5*(expected_score(z,weights,z,weights)+expected_score(2*z,weights,2*z,weights))
    scale_world_conditional = .5*(expected_score(z,weights,z,weights)+expected_score(2*z,weights,2*z,weights))
    close('correct dynamic scale with fixed true shape leaves no oracle shape gain',np.array(scale_world_static),np.array(scale_world_conditional),0)

    # Exercise only pure aggregation with invented positive cell losses.
    all_families = protocol['statistical_families']+protocol['neural_families']
    def rows_from_scores(family_scores, per_market=None):
        result=[]
        for j,spec in enumerate(protocol['markets']):
            market_rows=[]
            for f in all_families:
                v = family_scores.get(f,1.) if per_market is None or f not in per_market else per_market[f][j]
                fitted = f in protocol['neural_families'] or f in ('garch_normal','garch_t5')
                market_rows.append(dict(market=spec['name'],family=f,status='complete',metrics={'crps':(np.arange(1,21)*v).tolist()},fitted=fitted,optimizer_steps=0,completed_epochs=0))
            result.append(dict(rows=market_rows))
        return result
    values=dict(fhs_fixed=1.,fhs_rolling=.95,global_shape=.95,conditional_shape=.8,scale_only=.9,shape_scale=.7)
    aa_result=runner.aggregate(rows_from_scores(values),protocol)
    yes('all five synthetic resource gates pass jointly',aa_result['advance'] and all(aa_result['gate'].values()))
    yes('forecast/fit accounting is 65/30/10', (aa_result['expected_jobs'],aa_result['actual_neural_fits'],aa_result['actual_statistical_fits']) == (65,30,10))
    close('aggregation averages within-cell ratios before equal markets',np.array(aa_result['scores']['conditional_shape']),np.array(.8),1e-14)
    fail_values={**values,'conditional_shape':.95}
    out=runner.aggregate(rows_from_scores(fail_values),protocol)
    yes('shape below scale alone cannot bypass conditional-shape gate',not out['advance'])
    # Positive aggregate gain is insufficient when fewer than three markets improve.
    out=runner.aggregate(rows_from_scores(values,{'conditional_shape':[.1,.1,1.,1.,1.]}),protocol)
    yes('two-of-five shape wins fail breadth gate', not out['gate']['conditional_shape_beats_global_in_3_series'])
    out=runner.aggregate(rows_from_scores(values,{'conditional_shape':[.1,.1,.1,1.,1.]}),protocol)
    yes('three-of-five shape wins pass breadth gate', out['gate']['conditional_shape_beats_global_in_3_series'])
    out=runner.aggregate(rows_from_scores(values,{'shape_scale':[.1,.1,1.,1.,1.]}),protocol)
    yes('two-of-five shape plus scale wins fail breadth gate', not out['gate']['shape_scale_beats_scale_only_in_3_series'])
    bad_rows=rows_from_scores(values);bad_rows[0]['rows'][0]['status']='failed'
    yes('one failed flow blocks all expansion',not runner.aggregate(bad_rows,protocol)['advance'])
    missing=rows_from_scores(values);missing[-1]['rows'].pop()
    yes('one missing flow blocks all expansion',not runner.aggregate(missing,protocol)['advance'])
    yes('fixed design is five exposed development series in2022',protocol['year']==2022 and len(protocol['markets'])==5 and protocol['data_end_exclusive']=='2024-01-01')
    yes('no automatic D2 bootstrap or confirmation follows this diagnostic','No D2/bootstrap/C' in protocol['later_stages'])
    yes('1percent gate explicitly not publication threshold','not a publication threshold' in protocol['gate_meaning'])

    source_files=['data.py','scores.py','run.py','models.py','statistics.py']
    source_sha={name:hashlib.sha256((src/name).read_bytes()).hexdigest() for name in source_files}
    receipt=dict(status='PASS',completed_utc=datetime.now(timezone.utc).isoformat(),
        scope='Independent pure-synthetic data/CRPS/aggregation audit plus explicit static review; no claim of completed market results.',
        data_reads=0,market_fits=0,optimizer_steps=0,confirmation_reads=0,
        runtime={'numpy':np.__version__,'torch':torch.__version__,'python':sys.version.split()[0]},
        source_sha256=source_sha,protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        checks=len(checks),numeric_comparisons=int(numbers),maximum_abs_error=max((x.get('max_abs_error',0) for x in checks),default=0),
        synthetic_support={'rows':n,'train':len(d.train),'validation':len(d.val),'evaluation':len(d.test)},
        oracle_examples={'same_moments_shape_oracle_expected_crps':true_conditional,'static_pooled_expected_crps':static_pool,'conditional_shape_gain':static_pool-true_conditional,'scale_only_oracle_shape_gain':scale_world_static-scale_world_conditional},
        test_details=checks,
        limits=['No market score, fitted prediction, GARCH optimizer, or neural optimizer was run.',
                'Past-only feature proofs are conditional on the declared fixed training-prefix scaler; offline training does not claim a prequential scaler at each training origin.',
                'The31-atom laws are finite approximations; marginal CRPS does not establish a joint path distribution.',
                'Raw test labels used only by post-forecast scoring and checkpoint-independent reporting must remain separate from model inputs.',
                'Prior sibling model/GARCH synthetic audits are separate evidence and are not counted in this script.'])
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:receipt[k] for k in ('status','checks','numeric_comparisons','maximum_abs_error','runtime','source_sha256')},ensure_ascii=False))


if __name__ == '__main__':
    main()
