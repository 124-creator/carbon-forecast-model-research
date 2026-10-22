# -*- coding: utf-8 -*-
"""No-training QRF atom-probability control; preserve conditional nonzero marks.

Only qrf_round4_probability_control* is created. Frozen forest/tilt outputs
are read only. Operates on the original five-seed probability-mixture QRF.
"""
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
from threadpoolctl import threadpool_limits
import empirical_tilt_round3 as T

P = Path(__file__).resolve().parent
TAG = 'qrf_round4_probability_control'
MARKETS = list(T.B.CUT)


def replace_probability(support, w, p):
    support, w = support.copy(), w.copy()
    dummy = not np.any(np.abs(support) <= 1e-12)
    if dummy:
        zero_pos = int(np.searchsorted(support, 0.))
        support = np.insert(support, zero_pos, np.float32(0.))
        w = np.insert(w, zero_pos, np.float32(0.), axis=1)
    iszero = np.abs(support) <= 1e-12
    isnz = ~iszero
    assert isnz.any() and iszero.any()
    nz, zz = w[:, isnz].astype(np.float64), w[:, iszero].astype(np.float64)
    nm, zm = nz.sum(1), zz.sum(1)
    nf, zf = nm <= 0., zm <= 0.
    conditional_mark = np.divide(nz, nm[:, None], out=np.zeros_like(nz), where=nm[:, None] > 0)
    conditional_zero = np.divide(zz, zm[:, None], out=np.zeros_like(zz), where=zm[:, None] > 0)
    conditional_mark[nf] = 1./isnz.sum()
    conditional_zero[zf] = 1./iszero.sum()
    new = np.zeros_like(w)
    new[:, isnz] = (conditional_mark*p[:, None]).astype(np.float32)
    new[:, iszero] = (conditional_zero*(1-p[:, None])).astype(np.float32)
    actual_nonzero = new[:, isnz].sum(1, dtype=np.float64)
    actual_zero = new[:, iszero].sum(1, dtype=np.float64)
    reconditioned = new[:, isnz].astype(np.float64)/actual_nonzero[:, None]
    mark_error = float(np.max(np.abs(reconditioned-conditional_mark)))
    mass_error = float(np.max(np.abs(new.sum(1, dtype=np.float64)-1.)))
    p_error = float(np.max(np.abs(actual_nonzero-p)))
    zero_error = float(np.max(np.abs(actual_zero-(1-p))))
    assert np.isfinite(new).all() and np.all(new >= 0)
    assert mass_error < 2e-6 and p_error < 2e-6 and zero_error < 2e-6 and mark_error < 1e-7
    return support, new, dict(dummy_zero_added=dummy, zero_mass_fallback_queries=int(zf.sum()),
        nonzero_mass_fallback_queries=int(nf.sum()), nonzero_mark_preservation_max_error=mark_error,
        total_mass_max_error=mass_error, p_rule_max_error=p_error, zero_mass_max_error=zero_error,
        original_zero_support_count=int(iszero.sum())-(1 if dummy else 0),
        support_size=len(support))


def independent_crps(y, w, support):
    w, support = w.astype(np.float64), support.astype(np.float64)
    cw, dz = w.cumsum(1), np.diff(support)
    left = np.clip(np.minimum(support[None, 1:], y[:, None])-support[None, :-1], 0., dz)
    loss = (cw[:, :-1]**2*left+(cw[:, :-1]-1)**2*(dz[None, :]-left)).sum(1)
    return loss+np.maximum(support[0]-y, 0.)+np.maximum(y-support[-1], 0.)


def evaluate_distribution(z, scale, support, w):
    q, c = np.empty((len(w), 15)), np.empty(len(w))
    error = 0.
    for start in range(0, len(w), 128):
        stop = min(start+128, len(w))
        wt, st = torch.tensor(w[start:stop]), torch.tensor(support)
        yy = torch.tensor(np.asarray(z[start:stop], np.float32))
        q[start:stop] = T.quantiles(wt, st).numpy()*scale[start:stop, None]
        c[start:stop] = T.crps(yy, wt, st).numpy()*scale[start:stop]
        expected = independent_crps(yy.numpy().astype(np.float64), w[start:stop], support)*scale[start:stop]
        error = max(error, float(np.max(np.abs(expected-c[start:stop]))))
        assert np.allclose(c[start:stop], expected, atol=3e-7, rtol=3e-5)
    assert np.isfinite(q).all() and np.isfinite(c).all() and c.min() > -1e-7
    assert np.all(np.diff(q, axis=1) >= -1e-8)
    return q, c, error


def score_rows(rows, market, D, name, q, crps, p):
    y = D['y'][D['te']]
    for hi, h in enumerate(T.HOR):
        mask = D['hi'][D['te']] == hi
        yh, qh = y[mask], q[mask]
        rows.append(dict(market=market, horizon=int(h), model=name, n=int(mask.sum()),
            MQS=float(T.B.score(yh, qh).mean()), CRPS=float(crps[mask].mean()),
            WIS=float(T.B.wis(yh, qh).mean()), MAE=float(np.abs(yh-qh[:, 7]).mean()),
            coverage80=float(np.mean((yh >= qh[:, 2])&(yh <= qh[:, 12]))),
            coverage95=float(np.mean((yh >= qh[:, 0])&(yh <= qh[:, 14]))),
            Brier=float(np.mean((p[mask]-(np.abs(yh) > 1e-12))**2))))


def compare(rows):
    pairs = [('qrf_rule_atom', 'qrf_original'), ('qrf_rule_atom', 'tilt_full_5seed'),
             ('qrf_original', 'tilt_full_5seed'), ('qrf_rule_atom', 'uniform_rule')]
    lookup = {(r['market'], r['horizon'], r['model']): r for r in rows}
    output = []
    for model, base in pairs:
        for metric in ['MQS', 'CRPS', 'WIS', 'MAE']:
            selected = [r for r in rows if r['model'] == model]
            for group_type, group in [('overall', 'all'), *[('market', m) for m in MARKETS], *[('horizon', int(h)) for h in T.HOR]]:
                rr = [r for r in selected if group_type == 'overall' or (group_type == 'market' and r['market'] == group) or (group_type == 'horizon' and r['horizon'] == group)]
                ratios = np.array([r[metric]/lookup[(r['market'], r['horizon'], base)][metric] for r in rr])
                output.append(dict(model=model, baseline=base, metric=metric, grouping=group_type, group=group,
                    cells=len(ratios), wins=int((ratios < 1).sum()), median_ratio=float(np.median(ratios)),
                    geometric_gain_pct=float(100*(1-np.exp(np.log(ratios).mean())))))
    return output


def main():
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    frozen_names = ['qrf_round4.py', 'qrf_round4_summarize.py', 'qrf_round4.json',
        'qrf_round4_predictions.npz', 'qrf_round4_cells.csv', 'qrf_round4_summary.csv',
        'qrf_round4_report_zh.md', 'qrf_round4_verification.json',
        'empirical_tilt_round3.py', 'empirical_tilt_round3_followup.py',
        'empirical_tilt_round3_followup.json', 'empirical_tilt_round3_followup_predictions.npz']
    frozen_names += [f'qrf_round4_{m}_weights.npz' for m in MARKETS]
    frozen = {name: hashlib.sha256((P/name).read_bytes()).hexdigest() for name in frozen_names}
    qrf = np.load(P/'qrf_round4_predictions.npz')
    tilt = np.load(P/'empirical_tilt_round3_followup_predictions.npz')
    panel = pd.read_csv(T.B.PANEL, parse_dates=['date'])
    report = dict(status='no-training probability-control diagnostic; same frozen five-seed distributions',
        operation='preserve QRF conditional nonzero mark weights; set p_nonzero=.8*local+.2*precut_nonzero_rate',
        normalization='conditional nonzero and zero weights separately normalized per query; all original supports retained',
        fallback='append dummy0 if no zero support; uniform over zero records if old zero mass0; uniform over all nonzero records if old nonzero mass0',
        seeds=[11, 22, 33, 44, 55], training_calls=0, no_test_selection=True,
        precision='same float32 supports/weights and normalized target as frozen QRF; independent float64 CDF check',
        frozen_sha256_before=frozen, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), markets={})
    rows, arrays = [], {'taus': T.B.TAUS}
    with threadpool_limits(limits=2):
        for market in MARKETS:
            d = panel[panel.market == market].sort_values('date').reset_index(drop=True)
            D = T.add_context(d, T.E.build(d))
            te, key = D['te'], market+'_'
            assert np.array_equal(D['t'][te], qrf[key+'origin']) and np.array_equal(D['t'][te], tilt[key+'origin'])
            assert np.array_equal(T.HOR[D['hi'][te]], qrf[key+'horizon']) and np.array_equal(T.HOR[D['hi'][te]], tilt[key+'horizon'])
            assert np.array_equal(D['y'][te], qrf[key+'target']) and np.array_equal(D['y'][te], tilt[key+'target'])
            assert np.array_equal(D['scale'][te], qrf[key+'scale']) and np.array_equal(D['scale'][te], tilt[key+'scale'])
            arrays[key+'origin'], arrays[key+'horizon'] = D['t'][te], T.HOR[D['hi'][te]]
            arrays[key+'target'], arrays[key+'scale'] = D['y'][te], D['scale'][te]
            arrays[key+'origin_date'] = qrf[key+'origin_date']
            weights = np.load(P/f'qrf_round4_{market}_weights.npz')
            Q, C, PRO = np.empty((len(te), 15)), np.empty(len(te)), np.empty(len(te))
            audit = {}
            for hi, h in enumerate(T.HOR):
                positions = weights[f'test_positions_h{h}']
                assert np.array_equal(positions, np.flatnonzero(D['hi'][te] == hi))
                refit = D['rf'][D['hi'][D['rf']] == hi]
                prior = float(np.mean(np.abs(D['y'][refit]) > 1e-12))
                local = torch.tensor(D['local'][te[positions]], dtype=torch.float32)
                probability = (.8*local+.2*torch.full_like(local, prior)).numpy()
                p_frozen_error = float(np.max(np.abs(probability-tilt[key+'full_ensemble_p'][positions])))
                assert p_frozen_error < 2e-7
                support, new, details = replace_probability(weights[f'support_h{h}'], weights[f'ensemble_weights_h{h}'], probability.astype(np.float64))
                q, c, error = evaluate_distribution(D['z'][te[positions]], D['scale'][te[positions]], support, new)
                Q[positions], C[positions], PRO[positions] = q, c, probability
                details.update(precut_nonzero_rate=prior, p_rule_vs_frozen_tilt_max_error=p_frozen_error,
                    independent_CDF_CRPS_max_error=error, n_queries=len(positions))
                audit[str(h)] = details
            score_rows(rows, market, D, 'qrf_rule_atom', Q, C, PRO)
            arrays[key+'qrf_rule_atom_Q'], arrays[key+'qrf_rule_atom_CRPS'], arrays[key+'qrf_rule_atom_p'] = Q, C, PRO
            for name, source, oldkey in [('qrf_original', qrf, 'qrf_ensemble'),
                ('tilt_full_5seed', tilt, 'full_ensemble'), ('uniform_rule', tilt, 'uniform_rule')]:
                q, c, p = [source[key+oldkey+'_'+part] for part in ['Q', 'CRPS', 'p']]
                score_rows(rows, market, D, name, q, c, p)
                arrays[key+name+'_Q'], arrays[key+name+'_CRPS'], arrays[key+name+'_p'] = q, c, p
            report['markets'][market] = dict(audit=audit, strict_canonical_alignment=True)
            print('completed', market, 'elapsed_s', round(time.perf_counter()-started, 2), flush=True)
    after = {name: hashlib.sha256((P/name).read_bytes()).hexdigest() for name in frozen_names}
    assert after == frozen
    report.update(frozen_sha256_after=after, cells=rows, summary=compare(rows), elapsed_seconds=time.perf_counter()-started)
    (P/f'{TAG}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame(rows).to_csv(P/f'{TAG}_cells.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(report['summary']).to_csv(P/f'{TAG}_summary.csv', index=False, encoding='utf-8-sig')
    np.savez_compressed(P/f'{TAG}_predictions.npz', **arrays)
    summary = [s for s in report['summary'] if s['grouping'] == 'overall']
    lines = ['# QRF零概率控制：无重训诊断', '',
        '使用原五seed QRF ensemble，逐query保留条件非零mark权重；将非零总概率替换为与tilt完全相同的p_rule。',
        'p_rule=.8×origin历史滚动非零率+.2×precut非零率；未改变mark排序/相对权重，未选择市场，训练调用为0。',
        '无零support时追加dummy0；有零support但原零质量0时零records均匀分配；原非零质量0时全部非零records均匀fallback。', '',
        '| 模型 | 基线 | MQS改善% | 胜格 | 完整CRPS改善% | 胜格 |', '|---|---|---:|---:|---:|---:|']
    for model, base in [('qrf_rule_atom', 'qrf_original'), ('qrf_rule_atom', 'tilt_full_5seed'), ('qrf_original', 'tilt_full_5seed'), ('qrf_rule_atom', 'uniform_rule')]:
        dd = {s['metric']: s for s in summary if s['model'] == model and s['baseline'] == base}
        q, c = dd['MQS'], dd['CRPS']
        lines.append(f"| {model} | {base} | {q['geometric_gain_pct']:+.4f} | {q['wins']}/20 | {c['geometric_gain_pct']:+.4f} | {c['wins']}/20 |")
    lines += ['', '改善为1−20格得分比值的几何均值，正值更好。', '',
        '| 市场 | h | 原QRF MQS | rule-QRF MQS | tilt MQS | 原QRF CRPS | rule-QRF CRPS | tilt CRPS |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for market in MARKETS:
        for h in T.HOR:
            dd = {r['model']: r for r in rows if r['market'] == market and r['horizon'] == int(h)}
            a, b, c = dd['qrf_original'], dd['qrf_rule_atom'], dd['tilt_full_5seed']
            lines.append(f"| {market} | {h} | {a['MQS']:.7f} | {b['MQS']:.7f} | {c['MQS']:.7f} | {a['CRPS']:.7f} | {b['CRPS']:.7f} | {c['CRPS']:.7f} |")
    lines += ['', '所有比较严格相同canonical origin/h/target/scale；原QRF与tilt取冻结的五seed概率混合结果。',
        '零概率替换会改变最优分位位置，但不改变条件非零mark形状；这是分布组件诊断，不是重新训练后模型优劣的因果识别。',
        '所有原QRF源/JSON/NPZ/weights及读取的冻结tilt文件SHA前后一致。',
        f"计算耗时 {report['elapsed_seconds']:.2f} 秒；每格fallback及概率/mark/CRPS审计见同前缀JSON。"]
    (P/f'{TAG}_report_zh.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
