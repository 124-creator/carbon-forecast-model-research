"""Corrected exploratory audit of HAR-D; this is NOT an untouched test set.

Forecasts are issued after close t for mean(r^2[t+1:t+h+1]). At a refit
origin i, a labelled training row s is eligible only if s < i and s+h <= i.
The same availability rule applies to smearing and scale estimation.

Standard variance QLIKE is y/v - log(y/v) - 1 (y>0). Its scale optimum is
c=mean(y/v), rather than the harmonic-target rule in rounds 9--11.
No model is discarded because its persistence is close to one.
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '1'

import hashlib
import json
import math
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from arch import arch_model
import arch
import scipy

HERE = Path(__file__).resolve().parent
# SOURCE is made absolute independently of the shell's working directory.
SOURCE = Path(r'E:\Agent员工\碳价格预测研究\独立数据归档\_legacy_202605_archive\主要实验推进_审计加固版_CLEAN_20260528\08-新实验\01_数据\EUA_原始数据_完整版.csv')
EPS = 1e-12
REFIT = 22
SEED = 20261021
HAR = ['lrv1', 'lrv5', 'lrv22']
SPECS = {
    'HAR': (HAR, True),
    'HAR-D': (HAR + ['g22'], True),
    'HAR_signed_pair': (HAR + ['dn22', 'up22'], True),
    'HAR_ratio_sq': (HAR + ['asym22_sq'], True),
    'HAR_level': (['rv1', 'rv5', 'rv22'], False),
    'HAR_GM_level': (['rv1', 'rv5', 'rv22', 'gm22'], False),
}


def features(raw):
    d = raw.copy().reset_index(drop=True)
    d['r'] = np.log(d['price']).diff()
    r2 = d['r'] ** 2
    for w in (1, 5, 22):
        d[f'rv{w}'] = r2.rolling(w).mean()
        d[f'lrv{w}'] = np.log(d[f'rv{w}'].clip(lower=EPS))
    pp = d['r'].clip(lower=0).pow(2).rolling(22).mean()
    nn = d['r'].clip(upper=0).pow(2).rolling(22).mean()
    d['up22'] = np.log(pp.clip(lower=1e-14))
    d['dn22'] = np.log(nn.clip(lower=1e-14))
    d['g22'] = d['up22'] + d['dn22']
    d['gm22'] = np.sqrt(pp * nn)
    d['asym22_sq'] = (d['dn22'] - d['up22']) ** 2
    # After close t, r_t is observed; the state here forecasts t+1 onward.
    d['EWMA94'] = r2.ewm(alpha=.06, adjust=False, min_periods=1).mean()
    d['HIST22'] = d['rv22']
    for h in (5, 22):
        d[f'y{h}'] = r2.rolling(h).mean().shift(-h)
        d[f'ly{h}'] = np.log(d[f'y{h}'].clip(lower=EPS))
    return d


def load():
    raw = pd.read_csv(SOURCE, usecols=['date', 'eua_close_t'])
    raw = raw.rename(columns={'eua_close_t': 'price'})
    raw['date'] = pd.to_datetime(raw['date'])
    raw['price'] = pd.to_numeric(raw['price'], errors='raise')
    raw = raw.sort_values('date').reset_index(drop=True)
    assert raw['date'].is_unique and raw['date'].notna().all()
    assert np.isfinite(raw['price']).all() and (raw['price'] > 0).all()
    return raw, features(raw)


def qlike(v, y):
    v, y = np.asarray(v), np.asarray(y)
    assert (v > 0).all() and (y > 0).all()
    ratio = y / v
    return ratio - np.log(ratio) - 1.


def fit_direct(d, cols, h, i, log_model):
    X = d[cols].to_numpy(float)
    y = d[f'y{h}'].to_numpy(float)
    target = np.log(np.maximum(y, EPS)) if log_model else y
    ix = np.arange(len(d))
    m = (ix < i) & (ix + h <= i)
    m &= np.isfinite(X).all(axis=1) & np.isfinite(target)
    used = np.flatnonzero(m)
    assert len(used) >= 500
    assert (used + h <= i).all()
    A = np.column_stack([np.ones(len(used)), X[m]])
    b = np.linalg.lstsq(A, target[m], rcond=None)[0]
    smear = float(np.exp(target[m] - A @ b).mean()) if log_model else 1.
    fitted = np.exp(A @ b) * smear if log_model else A @ b
    scale = float(np.mean(y[m] / np.maximum(fitted, EPS)))
    return b, smear, scale, used


def direct(d, cols, h, start_i, log_model, stop=None):
    n = len(d)
    stop = n - h if stop is None else stop
    p = np.full(n, np.nan)
    v = np.full(n, np.nan)
    vc = np.full(n, np.nan)
    checks = []
    X = d[cols].to_numpy(float)
    for i in range(start_i, stop, REFIT):
        b, smear, scale, used = fit_direct(d, cols, h, i, log_model)
        j = min(i + REFIT, stop)
        pp = np.column_stack([np.ones(j-i), X[i:j]]) @ b
        raw = np.exp(pp) * smear if log_model else pp
        vv = np.maximum(raw, EPS)
        p[i:j] = pp if log_model else np.log(vv)
        v[i:j] = vv
        vc[i:j] = vv * scale
        checks.append({'refit_origin': int(i), 'last_train_origin': int(used[-1]),
                       'last_train_label_end': int(used[-1] + h),
                       'n_train': int(len(used)), 'scale': scale,
                       'nonpositive_forecasts': int((raw <= 0).sum())})
    return v, vc, p, checks


def naive_scale(d, col, h, start_i):
    n = len(d)
    v = d[col].to_numpy(float)
    vc = np.full(n, np.nan)
    ix = np.arange(n)
    y = d[f'y{h}'].to_numpy(float)
    for i in range(start_i, n-h, REFIT):
        m = (ix < i) & (ix+h <= i) & np.isfinite(y) & np.isfinite(v)
        m &= v > 0
        c = float(np.mean(y[m] / v[m]))
        vc[i:min(i+REFIT, n-h)] = v[i:min(i+REFIT, n-h)] * c
    return v, vc


def garch_family(d, start_i):
    n = len(d)
    r = d['r'].to_numpy(float) * 100.
    out, diagnostics = {}, {}
    for name, o, dist in [('GARCH', 0, 'normal'), ('GJR', 1, 'normal'),
                          ('GARCH-t', 0, 't')]:
        paths = {h: np.full(n, np.nan) for h in (5, 22)}
        issues, persistences = [], []
        for i in range(start_i, n-5, REFIT):
            # Include the return at close i, available at this forecast origin.
            hist = r[1:i+1]
            assert np.isfinite(hist).all()
            am = arch_model(hist, mean='Zero', vol='GARCH', p=1, o=o, q=1,
                            dist=dist, rescale=False)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                res = am.fit(disp='off', show_warning=False,
                             options={'maxiter': 2000})
            if res.convergence_flag:
                raise RuntimeError(f'{name} convergence failure at {i}: {res.convergence_flag}')
            if caught:
                issues.extend([{'origin': int(i), 'message': str(w.message)} for w in caught])
            par = res.params
            om, al, be = (float(par[x]) for x in ('omega', 'alpha[1]', 'beta[1]'))
            ga = float(par['gamma[1]']) if o else 0.
            phi = al + be + ga/2.
            persistences.append(phi)
            assert om >= 0 and phi >= 0 and np.isfinite(phi)
            for j in range(i, min(i+REFIT, n-5)):
                # Fixed parameters, updated filter using only data <= j.
                aj = arch_model(r[1:j+1], mean='Zero', vol='GARCH', p=1, o=o, q=1,
                                dist=dist, rescale=False)
                fixed = aj.fix(par)
                s2 = float(fixed.conditional_volatility[-1]) ** 2
                e = om + (al + ga*float(r[j] < 0)) * r[j]**2 + be*s2
                future = [e]
                for _ in range(21):
                    future.append(om + phi*future[-1])
                future = np.asarray(future) / 1e4
                for h in (5, 22):
                    if j < n-h:
                        paths[h][j] = float(future[:h].mean())
        out[name] = paths
        diagnostics[name] = {'warnings': issues, 'refits': len(persistences),
                             'min_phi': min(persistences), 'max_phi': max(persistences)}
        print(f'  {name}: {len(persistences)} refits; no excluded origins', flush=True)
    return out, diagnostics


def dm(diff, lag=44):
    diff = np.asarray(diff)
    n = len(diff)
    mu = diff.mean()
    e = diff-mu
    long_var = float(e@e)/n
    for L in range(1, min(lag+1, n)):
        long_var += 2*(1-L/(lag+1))*float(e[L:]@e[:-L])/n
    if long_var <= 0:
        return {'z': None, 'p': None}
    z = float(mu / np.sqrt(long_var/n))
    return {'z': z, 'p': math.erfc(abs(z)/np.sqrt(2))}


def circular_ci(diff, block=44, reps=1999):
    rng = np.random.default_rng(SEED)
    n = len(diff)
    starts = rng.integers(n, size=(reps, int(np.ceil(n/block))))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    idx = idx.reshape(reps, -1)[:, :n]
    return np.quantile(np.asarray(diff)[idx].mean(axis=1), [.025, .975]).tolist()


def holm(pvalues):
    out = np.empty(len(pvalues))
    order = np.argsort(pvalues)
    running = 0.
    for rank, idx in enumerate(order):
        running = max(running, (len(pvalues)-rank)*pvalues[idx])
        out[idx] = min(running, 1.)
    return out.tolist()


def checks(raw, d):
    # Loss direction: mean variance, rather than harmonic variance, is optimal.
    y = np.array([1., 4., 9.])
    arithmetic, harmonic = y.mean(), 1/np.mean(1/y)
    assert qlike(np.full(3, arithmetic), y).mean() < qlike(np.full(3, harmonic), y).mean()
    v = np.array([2., 2., 3.])
    c = np.mean(y/v)
    lc = qlike(v*c, y).mean()
    assert lc < qlike(v*c*.7, y).mean() and lc < qlike(v*c*1.3, y).mean()
    i = int(np.searchsorted(d['date'].to_numpy(), np.datetime64('2022-01-01')))
    # Perturb every future price. Neither past features nor an origin-i forecast
    # may change, even though some unobservable training labels do change.
    perturbed = raw.copy()
    m = len(raw)-i-1
    perturbed.loc[i+1:, 'price'] *= np.exp(.15*np.sin(np.arange(m)*.37))
    dd = features(perturbed)
    for h in (5, 22):
        r2 = d['r'].to_numpy()**2
        assert np.isclose(d[f'y{h}'].iloc[i], np.mean(r2[i+1:i+h+1]))
        for name in ('HAR', 'HAR-D'):
            cols, log = SPECS[name]
            p0 = direct(d, cols, h, i, log, stop=i+1)
            p1 = direct(dd, cols, h, i, log, stop=i+1)
            for k in range(3):
                np.testing.assert_allclose(p0[k][i], p1[k][i], rtol=1e-10, atol=1e-13)
    return {'qlike_arithmetic_mean_optimum': True, 'standard_scale_optimum': True,
            'forward_target_index': True, 'future_price_perturbation_invariant': True,
            'perturbation_origin': str(d['date'].iloc[i].date())}


def main():
    raw, d = load()
    meta = {'source': str(SOURCE), 'sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'rows': len(d), 'first': str(d['date'].min().date()),
            'last': str(d['date'].max().date()), 'arch': arch.__version__,
            'scipy': scipy.__version__, 'numpy': np.__version__, 'pandas': pd.__version__,
            'qlike': 'y/v-log(y/v)-1', 'availability': 's<i and s+h<=i; close t origin',
            'checks': checks(raw, d), 'refit': REFIT,
            'status': 'EXPLORATORY; all these test periods were used in model search',
            'holm_scope': 'four HAR-D vs HAR QLIKE comparisons only, not full research search'}
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    rows, comparisons, forecast_rows, refits, gdiag = [], [], [], [], {}
    for start in ('2019-01-01', '2022-01-01'):
        print('\nSTART '+start, flush=True)
        i = int(np.searchsorted(d['date'].to_numpy(), np.datetime64(start)))
        gfc, diag = garch_family(d, i)
        gdiag[start] = diag
        for h in (5, 22):
            fc, lp = {}, {}
            for name, (cols, log) in SPECS.items():
                v, vc, p, trace = direct(d, cols, h, i, log)
                fc[name], fc[name+'_scaled'], lp[name] = v, vc, p
                for t in trace:
                    refits.append({'window': start, 'h': h, 'model': name, **t})
            for name in ('EWMA94', 'HIST22'):
                v, vc = naive_scale(d, name, h, i)
                fc[name], fc[name+'_scaled'] = v, vc
            for name, vals in gfc.items():
                fc[name] = vals[h]
            mask = np.arange(len(d)) >= i
            mask &= np.isfinite(d[f'y{h}'])
            for v in fc.values():
                mask &= np.isfinite(v) & (v > 0)
            expected = len(d)-h-i
            assert int(mask.sum()) == expected, (start, h, mask.sum(), expected)
            y = d[f'y{h}'].to_numpy()[mask]
            assert (y > 0).all(), 'Use the unnormalised standard score if zero targets exist'
            losses = {name: qlike(v[mask], y) for name, v in fc.items()}
            for name in fc:
                score = float(losses[name].mean())
                logmse = float(np.mean((lp[name][mask]-np.log(y))**2)) if name in lp else None
                rows.append({'window': start, 'h': h, 'model': name, 'n': int(mask.sum()),
                             'qlike': score, 'log_mse_direct': logmse})
            for a, b in [('HAR-D', 'HAR'), ('HAR-D', 'EWMA94'), ('HAR-D', 'HIST22'),
                         ('HAR-D', 'GARCH'), ('HAR-D', 'GJR'), ('HAR-D', 'GARCH-t'),
                         ('HAR-D_scaled', 'HAR_scaled'), ('HAR-D_scaled', 'EWMA94_scaled'),
                         ('HAR_signed_pair', 'HAR-D'), ('HAR_ratio_sq', 'HAR-D')]:
                diff = losses[a] - losses[b]
                ci = circular_ci(diff)
                comparisons.append({'window': start, 'h': h, 'a': a, 'b': b,
                    'gain_percent': float(100*(1-losses[a].mean()/losses[b].mean())),
                    'mean_loss_diff': float(diff.mean()), 'ci44': ci,
                    'ci88': circular_ci(diff, block=88), **dm(diff)})
            print(f'  h={h} n={int(mask.sum())}', flush=True)
            for name in ('HAR', 'HAR-D', 'EWMA94', 'HIST22', 'GARCH', 'GJR', 'GARCH-t',
                         'HAR_scaled', 'HAR-D_scaled', 'EWMA94_scaled'):
                print(f'    {name:19s} {losses[name].mean():.6f}', flush=True)
            for name in fc:
                for j in np.flatnonzero(mask):
                    forecast_rows.append({'window': start, 'h': h, 'date': d['date'].iloc[j],
                                          'model': name, 'y': d[f'y{h}'].iloc[j],
                                          'forecast': fc[name][j]})
    target_comps = [x for x in comparisons if x['a']=='HAR-D' and x['b']=='HAR']
    for row, adj in zip(target_comps, holm([x['p'] for x in target_comps])):
        row['holm_p_four_cells'] = adj
    meta['garch_diagnostics'] = gdiag
    result = {'metadata': meta, 'scores': rows, 'comparisons': comparisons}
    (HERE/'publication_audit_round12_results.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame(rows).to_csv(HERE/'publication_audit_round12_scores.csv', index=False)
    pd.DataFrame(comparisons).to_csv(HERE/'publication_audit_round12_comparisons.csv', index=False)
    pd.DataFrame(forecast_rows).to_csv(HERE/'publication_audit_round12_forecasts.csv', index=False)
    pd.DataFrame(refits).to_csv(HERE/'publication_audit_round12_refits.csv', index=False)
    print('\nHAR-D vs HAR (exploratory, corrected)', flush=True)
    print(pd.DataFrame(target_comps).to_string(index=False), flush=True)
    print('WROTE results, scores, comparisons, forecasts and refit traces.', flush=True)


if __name__ == '__main__':
    main()
