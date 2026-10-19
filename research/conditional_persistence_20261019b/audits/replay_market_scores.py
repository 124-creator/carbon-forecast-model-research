"""Independent saved-distribution score reconstruction; never fits a model.

No production scoring module is imported. Normal expectations, lognormal
truncated first moments, and finite-distribution pair distances are evaluated
directly. --partial is an incremental technical check, not a complete audit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr, logsumexp

STUDY = Path(__file__).resolve().parents[1]
SQRT2PI = np.sqrt(2 / np.pi)
LOG2PI = np.log(2 * np.pi)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.writing')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def expected_abs_normal(delta, sd):
    z = delta / sd
    return sd * SQRT2PI * np.exp(-z*z/2) + delta * (2*ndtr(z)-1)


def direct_mixture(y, w, mu, sd, unit, p0, target):
    """Direct expectation formulae, independent of production implementation."""
    assert np.all(sd > 0) and np.all(w >= 0)
    np.testing.assert_allclose(w.sum(1), 1, rtol=0, atol=5e-12)
    w = w / w.sum(1, keepdims=True)
    n, k = w.shape
    out = {key: np.empty(n) for key in (
        'normalized_CRPS', 'normalized_NLL', 'price_CRPS', 'price_NLL',
        'mean_price_squared_error', 'rw_mean_price_squared_error', 'mean_price')}
    for a in range(0, n, 48):
        b = min(n, a+48)
        ww, mm, ss = w[a:b], mu[a:b], sd[a:b]
        yy, uu, pp, tt = y[a:b], unit[a:b], p0[a:b], target[a:b]
        term1 = (ww * expected_abs_normal(mm-yy[:,None], ss)).sum(1)
        pair_sd = np.sqrt(ss[:,:,None]**2 + ss[:,None,:]**2)
        pair = expected_abs_normal(mm[:,:,None]-mm[:,None,:], pair_sd)
        out['normalized_CRPS'][a:b] = term1 - .5*np.einsum('ni,nij,nj->n',ww,pair,ww)
        lw = np.full_like(ww, -np.inf)
        np.log(ww, where=ww>0, out=lw)
        ln_density = logsumexp(lw-np.log(ss)-.5*LOG2PI-.5*((yy[:,None]-mm)/ss)**2, axis=1)
        out['normalized_NLL'][a:b] = -ln_density
        out['price_NLL'][a:b] = -ln_density + np.log(uu) + np.log(tt)
        # E|X-t| from the lognormal CDF and its left truncated first moment.
        ell, vv = np.log(pp)[:,None] + uu[:,None]*mm, uu[:,None]*ss
        means = np.exp(ell+.5*vv*vv)
        z = (np.log(tt)[:,None]-ell)/vv
        first_abs = tt[:,None]*(2*ndtr(z)-1) + means*(1-2*ndtr(z-vv))
        # E[min(X_i,X_j)] = E[X_i 1(X_i<=X_j)] + E[X_j 1(X_j<X_i)].
        # Sum over both ordered pairs to avoid copying a special-case kernel.
        joint_sd = np.sqrt(vv[:,:,None]**2+vv[:,None,:]**2)
        truncated_i = means[:,:,None]*ndtr((ell[:,None,:]-ell[:,:,None]-vv[:,:,None]**2)/joint_sd)
        minimum = truncated_i + truncated_i.transpose(0,2,1)
        absolute_pair = means[:,:,None]+means[:,None,:]-2*minimum
        out['price_CRPS'][a:b] = np.sum(ww*first_abs,1)-.5*np.einsum('ni,nij,nj->n',ww,absolute_pair,ww)
        point = (ww*means).sum(1)
        out['mean_price'][a:b] = point
        out['mean_price_squared_error'][a:b] = (point-tt)**2
        out['rw_mean_price_squared_error'][a:b] = (pp-tt)**2
    for value in out.values():
        assert np.isfinite(value).all()
    return out


def direct_atoms(y, atoms, unit, p0, target):
    prices = p0[:,None]*np.exp(unit[:,None]*atoms)
    def crps(truth, values):
        # Independent cumulative-mass expression for half the pair distance.
        ordered = np.sort(values,axis=1)
        mass = np.full(values.shape[1], 1/values.shape[1])
        cumulative_before = np.cumsum(mass)-mass
        pair_half = (ordered*(mass*(2*cumulative_before+mass-1))).sum(1)
        return np.mean(np.abs(values-truth[:,None]),1)-pair_half
    mean = prices.mean(1)
    return {'normalized_CRPS':crps(y,atoms),'price_CRPS':crps(target,prices),
            'mean_price':mean,'mean_price_squared_error':(mean-target)**2,
            'rw_mean_price_squared_error':(p0-target)**2}


def compare(actual, saved, key):
    assert np.shape(actual) == np.shape(saved)
    assert np.isfinite(actual).all() and np.isfinite(saved).all()
    error = np.abs(actual-saved)
    # The absolute term permits harmless cancellation in near-zero price CRPS.
    np.testing.assert_allclose(actual, saved, rtol=2e-10, atol=2e-9,
                               err_msg=f'independent replay: {key}')
    return {'max_absolute_error':float(error.max(initial=0)),
            'max_scaled_error':float(np.max(error/(1+np.abs(saved)),initial=0)),
            'values':int(np.size(actual)), 'mean_recomputed':float(np.mean(actual))}


def one_distribution(z, prefix, col=None, horizon=None):
    if horizon is None:
        y, unit, p0, target = [z[k] for k in ('truth','unit','origin_price','target_price')]
        getscore = lambda k:z['score_'+k]
        atomic = 'atoms' in z
        if atomic:
            atom = z['atoms']
            out = direct_atoms(y,atom,unit,p0,target)
        else:
            w,mu,sd = [z[k] for k in ('weights','means','sigmas')]
            out = direct_mixture(y,w,mu,sd,unit,p0,target)
    else:
        y, unit = z[prefix+'y'][:,col], z[prefix+'unit'][:,col]
        p0,target = z[prefix+'origin_price'],z[prefix+'target_price'][:,col]
        w,mu,sd = [z[prefix+k+f'_h{horizon}'] for k in ('weights','means','sigmas')]
        getscore = lambda k:z[prefix+f'score_h{horizon}_'+k]
        atomic = False
        out = direct_mixture(y,w,mu,sd,unit,p0,target)
    np.testing.assert_allclose(p0*np.exp(unit*y),target,rtol=2e-10,atol=2e-10)
    lower,upper = getscore('lower90'),getscore('upper90')
    out['coverage90'] = ((target>=lower)&(target<=upper)).astype(float)
    out['width90'] = upper-lower
    qlo,qhi = np.log(lower/p0)/unit,np.log(upper/p0)/unit
    out['normalized_width90'] = qhi-qlo
    if atomic:
        ordered = np.sort(atom,axis=1)
        at = np.ceil(np.array([.05,.95])*atom.shape[1]).astype(int)-1
        np.testing.assert_allclose(qlo,ordered[:,at[0]],rtol=2e-9,atol=2e-8)
        np.testing.assert_allclose(qhi,ordered[:,at[1]],rtol=2e-9,atol=2e-8)
        cdf_error = None
    else:
        probs = np.column_stack(((w*ndtr((qlo[:,None]-mu)/sd)).sum(1),
                                  (w*ndtr((qhi[:,None]-mu)/sd)).sum(1)))
        cdf_error = float(np.max(np.abs(probs-np.array([.05,.95]))))
        np.testing.assert_allclose(probs,np.broadcast_to([.05,.95],probs.shape),rtol=0,atol=2e-8)
    checks = {key:compare(val,getscore(key),key) for key,val in out.items()}
    return {'origins':len(y),'atomic':atomic,'quantile_cdf_max_error':cdf_error,'metrics':checks}


def process(path, cache, kind):
    fhash,codehash = digest(path),digest(__file__)
    key = hashlib.sha256(str(path).encode()).hexdigest()
    checkpoint = cache/(key+'.json')
    if checkpoint.exists():
        previous = json.loads(checkpoint.read_text())
        if previous['forecast_sha256'] == fhash and previous['audit_sha256'] == codehash:
            return previous
        raise ValueError('A previously audited score artifact or audit implementation changed')
    with np.load(path,allow_pickle=False) as z:
        if kind == 'persistence':
            checks = {f'{role}_h{h}':one_distribution(z,role+'_',col,h)
                      for role in ('validation','test') for col,h in enumerate((1,5,20))}
        else:
            checks = {'test':one_distribution(z,'')}
    record = {'forecast_path':str(path),'forecast_sha256':fhash,'audit_sha256':codehash,
              'kind':kind,'passed':True,'distributions':checks}
    save(checkpoint,record)
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--persistence-private',type=Path,default=Path('/tmp/carbon-persistence-market-runs'))
    p.add_argument('--gas-private',type=Path,default=Path('/tmp/carbon-gas-normalization-run'))
    p.add_argument('--cache',type=Path,default=Path('/tmp/carbon-score-replay-20261019b'))
    p.add_argument('--partial',action='store_true')
    args=p.parse_args()
    tasks=[]
    # Only finished artifacts, identified by a final receipt, are eligible.
    pub=STUDY/'results/market/persistence'
    for receipt in sorted((pub/'fits').glob('*.json'))+sorted((pub/'references').glob('*.json')):
        r=json.loads(receipt.read_text())
        for rel,sha in r['private_artifacts'].items():
            if rel.endswith('.npz'):
                path=args.persistence_private/rel
                assert digest(path)==sha
                tasks.append((path,'persistence'))
    for receipt in sorted((args.gas_private/'predictions').glob('*/*/*.json')):
        r=json.loads(receipt.read_text());path=receipt.with_suffix('.npz')
        assert digest(path)==r['prediction_sha256']
        tasks.append((path,'gas'))
    assert len({str(p) for p,k in tasks})==len(tasks)
    counts={k:sum(kind==k for path,kind in tasks) for k in ('persistence','gas')}
    if not args.partial and counts != {'persistence':338,'gas':352}:
        raise ValueError(f'Formal audit requires all 690 prediction artifacts, found {counts}')
    records=[]
    for i,(path,kind) in enumerate(tasks):
        records.append(process(path,args.cache,kind))
        if (i+1)%50==0:
            print(json.dumps({'stage':'score_replay','completed':i+1,'available':len(tasks),'partial':args.partial}),flush=True)
    maximums={}; values=0; forecasts=0; qmax=0
    for record in records:
        for dist in record['distributions'].values():
            forecasts+=dist['origins']
            if dist['quantile_cdf_max_error'] is not None:
                qmax=max(qmax,dist['quantile_cdf_max_error'])
            for metric,check in dist['metrics'].items():
                values+=check['values']
                row=maximums.setdefault(metric,{'max_absolute_error':0.,'max_scaled_error':0.,'values':0})
                for key in ('max_absolute_error','max_scaled_error'):
                    row[key]=max(row[key],check[key])
                row['values']+=check['values']
    result={'status':'PARTIAL_TECHNICAL_CHECK' if args.partial else 'PASS_COMPLETE_SCORE_REPLAY',
            'created_utc':datetime.now(timezone.utc).isoformat(),'audit_sha256':digest(__file__),
            'complete':not args.partial,'artifacts':counts,'origin_horizon_distributions':forecasts,
            'individual_metric_values':values,'metrics':maximums,'quantile_cdf_max_error':qmax,
            'fit_or_gradient_steps':0,'production_score_module_imported':False,
            'role':'Root wrote this independent score reconstruction; core scoring was authored by another agent. It is an internal cross-check, not external peer review.',
            'scope':'All saved validation/test candidate mixtures and all saved test GAS/fixed controls; normalized/price CRPS, continuous NLL, analytic price mean/MSE, interval width/coverage and quantile CDFs.',
            'limits':'Does not independently rerun training or prove statistical generalization. Block-bootstrap uncertainty and novelty need separate assessment.'}
    destination=args.cache/'PARTIAL_SUMMARY.json' if args.partial else STUDY/'audits/MARKET_SCORE_REPLAY.json'
    save(destination,result)
    print(json.dumps({k:result[k] for k in ('status','artifacts','origin_horizon_distributions','individual_metric_values')}),flush=True)


if __name__=='__main__':
    main()
