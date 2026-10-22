# -*- coding: utf-8 -*-
"""Round-3 origin-known input probe; fixed budget and no novelty claim."""
from __future__ import annotations
import os
import sys
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
sys.dont_write_bytecode = True
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
import neural_restart as B
import multiscale_round1 as M
HERE = Path(__file__).resolve().parent
PREFIX = 'context_probe_round3'
PARAMS = dict(loss='quantile', learning_rate=.05, max_iter=60, max_leaf_nodes=7,
              min_samples_leaf=25, l2_regularization=1., early_stopping=False, random_state=11)
EXTRA = ['origin_logprice', 'logprice_minus_mean120', 'logprice_minus_mean250',
         'price_range_position250', 'origin_doy_sin1', 'origin_doy_cos1',
         'origin_doy_sin2', 'origin_doy_cos2']
STATE = ['log_sd250', 'zero_share250', 'log1p_stale_age', 'log_sd20_over_sd250',
         'mean_ret10_over_sd250', 'mean_ret60_over_sd250', 'ret_ac1', 'last_ret_over_sd250']
STATE += [f'scale{s}_{stat}' for stat in ['last', 'mean', 'std'] for s in M.SCALES]


def features(D, d, refit):
    X, _ = M.prepare(D, refit)
    seq = D['seq'][:, :, :, 0]
    state = np.concatenate([X['ctx'].numpy(), seq[:, :, -1], seq.mean(2), seq.std(2)], 1)
    lp, p = np.log(d.price.to_numpy(float)), d.price.to_numpy(float)
    dates = pd.DatetimeIndex(d.date)
    cache = {}
    for t in np.unique(D['t']):
        w = p[t-249:t+1]
        span = w.max()-w.min()
        pos = (p[t]-w.min())/span if span > 1e-12 else .5
        phase = 2*np.pi*(dates[t].dayofyear-1)/(366 if dates[t].is_leap_year else 365)
        cache[t] = [lp[t], lp[t]-lp[t-119:t+1].mean(), lp[t]-lp[t-249:t+1].mean(), pos,
                    np.sin(phase), np.cos(phase), np.sin(2*phase), np.cos(2*phase)]
    raw = np.asarray([cache[t] for t in D['t']], np.float64)
    ids = D['indices']['development' if refit else 'train']
    mean, std = raw[ids].mean(0), raw[ids].std(0)+1e-5
    aug = np.concatenate([state, (raw-mean)/std], 1)
    assert np.isfinite(aug).all() and np.array_equal(state, aug[:, :17])
    return {'state_only': state, 'augmented': aug}, dict(extra_mean=mean.tolist(), extra_std=std.tolist())


def summarize(rows):
    rr = np.array([r['MQS_ratio'] for r in rows])
    return dict(cells=len(rows), wins=int((rr < 1).sum()), mean_ratio=float(rr.mean()),
                median_ratio=float(np.median(rr)), geomean_ratio=float(np.exp(np.log(rr).mean())))


def main():
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    panel = pd.read_csv(B.PANEL, parse_dates=['date'])
    report = dict(status='input diagnostic; reused historical test; no novelty claim',
                  fixed_budget=True, validation_used_for_selection=False,
                  final_fit='all and only t+h<cut labels including purged inner boundary',
                  only_origin_known_features=True, no_target_future_date=True,
                  no_legacy_processed_features=True, feature_names=dict(state_only=STATE, added=EXTRA),
                  windows='120/250 observed rows including origin; raw-price range position',
                  normalization='M.prepare context; extra fits same IDs; y/scale from M.extract_data',
                  hgb_params=PARAMS, ridge_alpha=10., quantiles=B.TAUS.tolist(), horizons=M.HOR,
                  quantile_rearrangement='sort per origin identically for both input groups',
                  metric='B.score MQS, twice mean pinball',
                  panel_sha256=hashlib.sha256(B.PANEL.read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), markets={})
    arrays, cells = {'taus': B.TAUS}, []
    with threadpool_limits(limits=2):
        for market in B.CUT:
            d = panel[panel.market == market].sort_values('date').reset_index(drop=True)
            D = M.extract_data(d)
            tr, va, te, dev = [D['indices'][k] for k in ['train', 'validation', 'test', 'development']]
            targets = D['t']+np.array(M.HOR)[D['hi']]
            cut, v0 = D['audit']['cut_index'], D['audit']['validation_start']
            assert targets[tr].max() < v0
            assert D['t'][va].min() >= v0 and targets[va].max() < cut
            assert targets[dev].max() < cut and D['t'][te].min() >= cut
            assert np.array_equal(dev, np.flatnonzero(targets < cut))
            _, inner_norm = features(D, d, False)
            feats, final_norm = features(D, d, True)
            preds = {f'{model}_{group}': np.empty((len(te), len(B.TAUS)))
                     for model in ['hgb', 'ridge'] for group in feats}
            timings = {}
            for hi, h in enumerate(M.HOR):
                fitids = dev[D['hi'][dev] == hi]
                tk = np.flatnonzero(D['hi'][te] == hi)
                testids = te[tk]
                yfit = D['z'][fitids]
                for group, X in feats.items():
                    ts = time.perf_counter()
                    ridge = Ridge(alpha=10.).fit(X[fitids], yfit)
                    rq = np.quantile(yfit-ridge.predict(X[fitids]), B.TAUS)
                    q = ridge.predict(X[testids])[:, None]+rq
                    preds[f'ridge_{group}'][tk] = np.sort(q, axis=1)*D['scale'][testids, None]
                    q = np.empty((len(testids), len(B.TAUS)))
                    for j, tau in enumerate(B.TAUS):
                        model = HistGradientBoostingRegressor(quantile=float(tau), **PARAMS)
                        model.fit(X[fitids], yfit)
                        q[:, j] = model.predict(X[testids])
                    preds[f'hgb_{group}'][tk] = np.sort(q, axis=1)*D['scale'][testids, None]
                    timings[f'h{h}_{group}'] = time.perf_counter()-ts
                for model in ['hgb', 'ridge']:
                    s = float(B.score(D['y'][testids], preds[f'{model}_state_only'][tk]).mean())
                    a = float(B.score(D['y'][testids], preds[f'{model}_augmented'][tk]).mean())
                    cells.append(dict(market=market, horizon=h, model=model, n_train=len(fitids), n_test=len(testids),
                                      state_only_MQS=s, augmented_MQS=a, MQS_ratio=a/s, MQS_change_pct=100*(a/s-1)))
                print(market, 'h', h, 'HGB ratio', round(cells[-2]['MQS_ratio'], 4),
                      'Ridge ratio', round(cells[-1]['MQS_ratio'], 4),
                      'elapsed_s', round(time.perf_counter()-started, 1), flush=True)
            for name, q in preds.items():
                assert np.isfinite(q).all() and np.all(np.diff(q, axis=1) >= -1e-9)
                arrays[f'{market}_{name}'] = q
                arrays[f'{market}_{name}_origin_MQS'] = B.score(D['y'][te], q)
            arrays[f'{market}_origin'] = D['t'][te]
            arrays[f'{market}_origin_date'] = d.date.to_numpy(dtype='datetime64[D]')[D['t'][te]]
            arrays[f'{market}_horizon'] = np.array(M.HOR)[D['hi'][te]]
            arrays[f'{market}_target'] = D['y'][te]
            arrays[f'{market}_scale'] = D['scale'][te]
            report['markets'][market] = dict(audit=D['audit'], train_last_target=int(targets[tr].max()),
                validation_first_origin=int(D['t'][va].min()), validation_last_target=int(targets[va].max()),
                inner_normalization=inner_norm, final_normalization=final_norm, timings_seconds=timings)
            report['elapsed_seconds'] = time.perf_counter()-started
            report['cells'] = cells
            (HERE/f'{PREFIX}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            np.savez_compressed(HERE/f'{PREFIX}_predictions.npz', **arrays)
            pd.DataFrame(cells).to_csv(HERE/f'{PREFIX}_cells.csv', index=False, encoding='utf-8-sig')
    report['summary'] = {}
    for model in ['hgb', 'ridge']:
        rows = [r for r in cells if r['model'] == model]
        report['summary'][model] = dict(overall=summarize(rows),
            by_horizon={str(h): summarize([r for r in rows if r['horizon'] == h]) for h in M.HOR},
            by_market={m: summarize([r for r in rows if r['market'] == m]) for m in B.CUT})
    report['elapsed_seconds'] = time.perf_counter()-started
    (HERE/f'{PREFIX}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 第三轮输入表示探针', '',
             '固定HGB 60轮、7叶、lr=.05、leaf=25、L2=1；Ridge alpha=10。',
             '两组同原点、归一化、purge和全部precut拟合；不按test选设置。',
             '这是输入遗漏对照，不是模型创新；price/calendar联合输入不能单独归因。', '',
             '| 模型 | 胜出格数 | MQS比值中位数 | MQS比值几何均值 |', '|---|---:|---:|---:|']
    for model, summ in report['summary'].items():
        s = summ['overall']
        lines.append(f"| {model} | {s['wins']}/{s['cells']} | {s['median_ratio']:.4f} | {s['geomean_ratio']:.4f} |")
    lines += ['', '比值=增强MQS/状态MQS，低于1更好。', '',
              '| 市场 | h | HGB变化% | Ridge变化% |', '|---|---:|---:|---:|']
    for m in B.CUT:
        for h in M.HOR:
            ss = {r['model']: r for r in cells if r['market'] == m and r['horizon'] == h}
            lines.append(f"| {m} | {h} | {ss['hgb']['MQS_change_pct']:+.2f} | {ss['ridge']['MQS_change_pct']:+.2f} |")
    lines += ['', f"运行耗时 {report['elapsed_seconds']:.1f} 秒。", '',
              '新增：origin logprice、相对过去120/250平均logprice、过去250价格range位置、',
              'origin年内sin/cos一二次谐波。NPZ含逐原点索引/日期/h/y/scale及四组15分位数。']
    (HERE/f'{PREFIX}_report_zh.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(report['summary'], ensure_ascii=False), flush=True)
    print('elapsed_seconds', report['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    main()
