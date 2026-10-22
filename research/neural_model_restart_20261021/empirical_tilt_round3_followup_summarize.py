# -*- coding: utf-8 -*-
"""Verify saved probability mixtures and summarize without retraining."""
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parent
TAG = 'empirical_tilt_round3_followup'
SEEDS = [11, 22, 33, 44, 55]
VARIANTS = ['full', 'state_only', 'horizon_only']
MARKETS = ['CEA', 'GUANGZHOU', 'SHANGHAI', 'HUBEI', 'EU']


def integrated_crps(y, w, support):
    # Independent CDF integral on each interval, split exactly at the observed y.
    w = w.astype(np.float64)
    support = support.astype(np.float64)
    cumulative = np.cumsum(w, axis=1)
    dz = np.diff(support)
    left = np.clip(np.minimum(support[None, 1:], y[:, None])-support[None, :-1], 0., dz)
    right = dz[None, :]-left
    integral = (cumulative[:, :-1]**2*left+(cumulative[:, :-1]-1)**2*right).sum(1)
    return integral+np.maximum(support[0]-y, 0.)+np.maximum(y-support[-1], 0.)


def main():
    report = json.loads((P/f'{TAG}.json').read_text(encoding='utf-8'))
    assert set(report['markets']) == set(MARKETS)
    assert report['frozen_sha256_before'] == report['frozen_sha256_after']
    pred = np.load(P/f'{TAG}_predictions.npz')
    verification = {'method': 'independent interval integral of squared CDF error; no training repeated', 'markets': {}}
    for market in MARKETS:
        weights = np.load(P/f'{TAG}_{market}_weights.npz')
        key = market+'_'
        y = pred[key+'target']
        scale = pred[key+'scale']
        z = (y/scale).astype(np.float32).astype(np.float64)
        diagnostics = dict(max_mixture_weight_error=0., max_CRPS_integral_error=0.,
                           max_horizon_only_conditional_mark_variation=0.,
                           max_rule_probability_variation=0.)
        for h in [1, 5, 10, 20]:
            rows = weights[f'test_positions_h{h}']
            support = weights[f'support_h{h}']
            zero = int(weights[f'zero_position_h{h}'])
            assert support[zero] == 0 and np.all(np.diff(support) >= 0)
            for variant in VARIANTS:
                ensemble = weights[f'{variant}_ensemble_weights_h{h}']
                seed_weights = [weights[f'{variant}_seed{s}_weights_h{h}'] for s in SEEDS]
                err = float(np.max(np.abs(np.mean(seed_weights, axis=0)-ensemble)))
                diagnostics['max_mixture_weight_error'] = max(diagnostics['max_mixture_weight_error'], err)
                assert np.allclose(np.mean(seed_weights, axis=0), ensemble, atol=3e-8, rtol=1e-6)
                for seed, w in [(str(s), sw) for s, sw in zip(SEEDS, seed_weights)]+[('ensemble', ensemble)]:
                    assert np.isfinite(w).all() and np.all(w >= 0)
                    assert np.allclose(w.sum(1), 1., rtol=0, atol=2e-6)
                    crps = integrated_crps(z[rows], w, support)*scale[rows]
                    name = key+variant+('_ensemble' if seed == 'ensemble' else '_seed'+seed)
                    stored = pred[name+'_CRPS'][rows]
                    error = float(np.max(np.abs(crps-stored)))
                    diagnostics['max_CRPS_integral_error'] = max(diagnostics['max_CRPS_integral_error'], error)
                    assert np.allclose(crps, stored, atol=2e-7, rtol=3e-5)
                    assert np.allclose(1-w[:, zero], pred[name+'_p'][rows], atol=2e-7, rtol=0)
                    rule = pred[key+'uniform_rule_p'][rows]
                    p_error = float(np.max(np.abs(pred[name+'_p'][rows]-rule)))
                    diagnostics['max_rule_probability_variation'] = max(diagnostics['max_rule_probability_variation'], p_error)
                    if variant == 'horizon_only':
                        marks = np.delete(w, zero, axis=1)
                        marks = marks/marks.sum(1, keepdims=True)
                        variation = float(np.max(np.abs(marks-marks[:1])))
                        diagnostics['max_horizon_only_conditional_mark_variation'] = max(diagnostics['max_horizon_only_conditional_mark_variation'], variation)
                        assert variation < 1e-6
        verification['markets'][market] = diagnostics
    (P/f'{TAG}_verification.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = pd.DataFrame(report['summary'])
    cells = pd.DataFrame(report['cells'])
    lines = ['# 经验mark神经重加权：五seed同容量组件验证', '',
             '五市场、h=1/5/10/20、seed=11/22/33/44/55；各seed独立内层选epoch，再从初始seed全precut refit。',
             'max64epoch，epoch0可选，沿用冻结实现的purge、标准化、因果maturity mask、同bank和完整weighted-ECDF CRPS。',
             '三版本均1407参数：full使用36列；state-only将最后8列price/calendar归一化后置0；',
             'horizon-only仅保留normalized h第26列。三者p均为相同的0.8local+0.2precut频率规则。', '',
             '## 五seed概率混合结果', '',
             '每个ensemble先在同support上平均五seed完整权重，再求分位数和完整CRPS；下表收益为20格比值几何均值的改善百分比，正值更好。', '',
             '| 版本 | 基线 | MQS收益% | 胜格 | exact CRPS收益% | 胜格 |', '|---|---|---:|---:|---:|---:|']
    for variant in VARIANTS:
        for base in ['uniform_rule', 'uniform_static', 'legacy_empirical', 'gaussian', 'hgb_augmented', 'hgb_state_only', 'horizon_only', 'state_only']:
            ss = summary[(summary.model == variant)&(summary.seed == 'ensemble')&(summary.baseline == base)]
            if ss.empty:
                continue
            bymetric = {r['metric']: r for r in ss.to_dict('records')}
            q = bymetric['MQS']
            c = bymetric.get('CRPS')
            lines.append(f"| {variant} | {base} | {q['geometric_gain_pct']:+.3f} | {q['wins']}/20 | "+
                         (f"{c['geometric_gain_pct']:+.3f} | {c['wins']}/20 |" if c else '— | — |'))
    lines += ['', '## 逐seed稳定性（相对uniform_rule）', '',
              '| 版本 | seed | MQS收益% | 胜格 | exact CRPS收益% | 胜格 |', '|---|---:|---:|---:|---:|---:|']
    for variant in VARIANTS:
        for seed in map(str, SEEDS):
            ss = summary[(summary.model == variant)&(summary.seed == seed)&(summary.baseline == 'uniform_rule')]
            dd = {r['metric']: r for r in ss.to_dict('records')}
            q, c = dd['MQS'], dd['CRPS']
            lines.append(f"| {variant} | {seed} | {q['geometric_gain_pct']:+.3f} | {q['wins']}/20 | {c['geometric_gain_pct']:+.3f} | {c['wins']}/20 |")
    lines += ['', '## 各市场与期限的ensemble原始分数', '',
              '| 市场 | h | uniform_rule MQS | full MQS | state MQS | h-only MQS | full CRPS | state CRPS | h-only CRPS |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for market in MARKETS:
        for h in [1, 5, 10, 20]:
            dd = {r['model']: r for r in cells[(cells.market == market)&(cells.horizon == h)&cells.seed.isin(['ensemble', 'fixed'])].to_dict('records')}
            lines.append(f"| {market} | {h} | {dd['uniform_rule']['MQS']:.7f} | {dd['full']['MQS']:.7f} | {dd['state_only']['MQS']:.7f} | {dd['horizon_only']['MQS']:.7f} | {dd['full']['CRPS']:.7f} | {dd['state_only']['CRPS']:.7f} | {dd['horizon_only']['CRPS']:.7f} |")
    lines += ['', '## 审计与解释边界', '',
              '- 每seed的Q/CRPS/p均保留；每市场weights NPZ含同support、各h在canonical预测表的位置、逐seed及ensemble完整概率weights和beta。',
              '- 独立CDF区间积分复核所有逐seed及ensemble的完整CRPS；检查混合权重、零atom、概率和与h-only条件mark恒定性。',
              '- 原冻结脚本、JSON、NPZ前后SHA256一致；未重新训练HGB。',
              '- HGB按(origin,h)一一重排；origin/h完全一致，canonical收益转其float32存储类型后完全一致。分数统一以canonical原始收益重算。',
              '- tilt使用128观测lookback，HGB使用250；这些强baseline比较包含输入/信息差，不将差异归因于神经结构。',
              '- legacy empirical的MQS用样本插值分位数，完整CRPS用全部ECDF；uniform_static的MQS/CRPS来自同一离散分布。',
              '- HGB只有15分位数，故只比较raw MQS，不把分位网格近似称exact CRPS。',
              '- price/calendar在本实验为联合输入包；full vs state-only不能识别其中单项贡献。', '',
              f"训练与落盘耗时 {report['elapsed_seconds']:.1f} 秒。"]
    (P/f'{TAG}_report_zh.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(verification, ensure_ascii=False, indent=2))
    selected = summary[(summary.seed == 'ensemble')&summary.baseline.isin(['uniform_rule', 'legacy_empirical', 'horizon_only', 'state_only', 'hgb_augmented'])]
    print(selected.to_csv(index=False))


if __name__ == '__main__':
    main()
