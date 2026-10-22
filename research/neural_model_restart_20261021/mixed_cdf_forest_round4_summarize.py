# -*- coding: utf-8 -*-
"""Summarize completed mixed-CDF model experiments without any additional fit."""
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parent
TAG = 'mixed_cdf_forest_round4'
SEEDS = [11, 22, 33, 44, 55]
VARIANTS = ['per_h_CDF', 'pooled_MSE', 'pooled_CDF', 'pooled_zero_CDF']
MARKETS = ['CEA', 'GUANGZHOU', 'SHANGHAI', 'HUBEI', 'EU']


def ratio(rows, model, seed, base, base_seed, metric, market=None, horizon=None):
    lookup = {(r['market'], r['horizon'], r['model'], r['seed']): r for r in rows}
    selected = [r for r in rows if r['model'] == model and r['seed'] == seed
                and (market is None or r['market'] == market)
                and (horizon is None or r['horizon'] == horizon)]
    pairs = [(r, lookup.get((r['market'], r['horizon'], base, base_seed))) for r in selected]
    pairs = [(a, b) for a, b in pairs if b and a[metric] is not None and b[metric] is not None]
    if not pairs:
        return None
    rr = np.array([a[metric]/b[metric] for a, b in pairs])
    return dict(cells=len(rr), wins=int((rr < 1).sum()),
                geometric_gain_pct=float(100*(1-np.exp(np.log(rr).mean()))), median_ratio=float(np.median(rr)))


def main():
    report = json.loads((P/f'{TAG}.json').read_text(encoding='utf-8'))
    assert report['forest_fits_completed'] == 175
    assert report['manifest']['frozen_sha256_before'] == report['frozen_sha256_after']
    rows, component_rows, verified = report['cells'], [], {}
    predictions = np.load(P/f'{TAG}_predictions.npz')
    borrowing_origin, borrowing_cells = [], []
    pairs = [('pooled_CDF', 'pooled_MSE'), ('pooled_zero_CDF', 'pooled_CDF'),
             ('pooled_zero_CDF', 'pooled_MSE'), ('pooled_CDF', 'per_h_CDF')]
    for a, b in pairs:
        for seed in [*map(str, SEEDS), 'ensemble']:
            for metric in ['MQS', 'CRPS', 'WIS', 'MAE']:
                result = ratio(rows, a, seed, b, seed, metric)
                component_rows.append(dict(model=a, baseline=b, seed=seed, metric=metric, **result))
    pd.DataFrame(component_rows).to_csv(P/f'{TAG}_component_summary.csv', index=False, encoding='utf-8-sig')
    for market in MARKETS:
        verified[market] = {}
        for variant in VARIANTS:
            weights = np.load(P/f'{TAG}_{market}_{variant}_weights.npz')
            groups = ['pooled'] if variant != 'per_h_CDF' else ['h1', 'h5', 'h10', 'h20']
            errors, borrowed = [], {}
            for group in groups:
                w, support = weights[group+'_ensemble_weights'], weights[group+'_support']
                assert np.isfinite(w).all() and np.all(w >= 0)
                error = float(np.max(np.abs(w.sum(1, dtype=np.float64)-1.)))
                assert error < 2e-6 and np.all(np.diff(support) >= 0)
                assert np.all(weights[group+'_fit_mature'] < report['manifest']['markets'][market]['audit']['cut_index'])
                errors.append(error)
                if group == 'pooled':
                    source_h = weights[group+'_fit_horizon']
                    details = report['markets'][market]['fits'][variant][group]
                    borrowed = details['source_horizon_probability_by_query_h']
                    for masses in borrowed.values():
                        assert abs(sum(masses.values())-1) < 2e-6
                    assert set(source_h.tolist()) == {1, 5, 10, 20}
                    positions = weights[group+'_test_positions']
                    query_h = predictions[market+'_horizon'][positions]
                    origin = predictions[market+'_origin'][positions]
                    for qh in [1, 5, 10, 20]:
                        take = np.flatnonzero(query_h == qh)
                        mass = w[take][:, source_h != qh].sum(1, dtype=np.float64)
                        assert mass.min() >= -2e-6 and mass.max() <= 1+2e-6
                        borrowing_cells.append(dict(market=market, model=variant, horizon=qh,
                            n=len(take), mean=float(mass.mean()), median=float(np.median(mass)),
                            min=float(mass.min()), max=float(mass.max())))
                        borrowing_origin.extend(dict(market=market, model=variant, origin=int(t),
                            horizon=qh, other_horizon_probability_mass=float(v))
                            for t, v in zip(origin[take], mass))
            verified[market][variant] = dict(max_weight_mass_error=max(errors),
                individual_horizon_maturity_checked=True, source_horizon_probability_by_query_h=borrowed)
    pd.DataFrame(borrowing_origin).to_csv(P/f'{TAG}_borrowing_origin.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(borrowing_cells).to_csv(P/f'{TAG}_borrowing_cells.csv', index=False, encoding='utf-8-sig')
    (P/f'{TAG}_verification.json').write_text(json.dumps(verified, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 第四轮：CDF分裂、零质量与期限共享森林模型实验', '',
        '四个固定候选均实际训练，五市场×五seed(11/22/33/44/55)，每forest80树、leaf25、n_jobs2。',
        'per_h_CDF为每h单独depth6；pooled_MSE、pooled_CDF、pooled_zero_CDF为全部h共享depth8，保留context中的h列26。',
        '175个最终forest fits。所有参数、fit-only网格、scaler和分割在首个fit前保存manifest；没有inner超参搜索或测试后设置调整。', '',
        'CDF训练响应为33个fit-only经验分位去重所得区间上的sqrt(Δ)*I(z<=中点)，zero组追加sqrt(.5)*I(y==0)。',
        'Multi-output squared-error只学习分区，是有限grid近似的integrated-CRPS impurity；最终分布仍为leaf内历史真实scalar z的概率，不使用CDF坐标均值拼概率。',
        '最终完整weighted-ECDF CRPS为精确离散分布评分。所有ensemble先平均五seed完整概率权重，再算Q/CRPS。', '',
        '## 对强基线的五seed ensemble结果', '',
        '改善=1−20格score比值的几何均值，正值表示候选优于基线。', '',
        '| 候选 | 基线 | MQS改善% | 胜格 | 完整CRPS改善% | 胜格 | WIS改善% |',
        '|---|---|---:|---:|---:|---:|---:|']
    for variant in VARIANTS:
        for base in ['qrf_original', 'qrf_rule_atom', 'lstm_qr', 'tilt_full_5seed']:
            q = ratio(rows, variant, 'ensemble', base, 'fixed', 'MQS')
            c = ratio(rows, variant, 'ensemble', base, 'fixed', 'CRPS')
            w = ratio(rows, variant, 'ensemble', base, 'fixed', 'WIS')
            lines.append(f"| {variant} | {base} | {q['geometric_gain_pct']:+.3f} | {q['wins']}/20 | "+
                (f"{c['geometric_gain_pct']:+.3f} | {c['wins']}/20 | " if c else '— | — | ')+f"{w['geometric_gain_pct']:+.3f} |")
    lines += ['', '## 组件对照', '',
        'pooled_CDF vs pooled_MSE隔离CDF分裂；pooled_zero_CDF vs pooled_CDF隔离额外zero-Brier坐标。',
        'pooled_CDF vs per_h_CDF包含期限借样/深度分配改变，不能把共享收益统称zero模块。', '',
        '| 候选 | 对照 | MQS改善% | 胜格 | 完整CRPS改善% | 胜格 |', '|---|---|---:|---:|---:|---:|']
    for a, b in pairs:
        q, c = [ratio(rows, a, 'ensemble', b, 'ensemble', metric) for metric in ['MQS', 'CRPS']]
        lines.append(f"| {a} | {b} | {q['geometric_gain_pct']:+.3f} | {q['wins']}/20 | {c['geometric_gain_pct']:+.3f} | {c['wins']}/20 |")
    lines += ['', '## 逐seed对rule-atom QRF强基线', '',
        '| 候选 | seed | MQS改善% | 完整CRPS改善% |', '|---|---:|---:|---:|']
    for variant in VARIANTS:
        for seed in map(str, SEEDS):
            q, c = [ratio(rows, variant, seed, 'qrf_rule_atom', 'fixed', metric) for metric in ['MQS', 'CRPS']]
            lines.append(f"| {variant} | {seed} | {q['geometric_gain_pct']:+.3f} | {c['geometric_gain_pct']:+.3f} |")
    lines += ['', '## 各市场/期限的ensemble MQS', '',
        '| 市场 | h | per_h_CDF | pooled_MSE | pooled_CDF | pooled_zero_CDF | QRF | rule-QRF | LSTMQR |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for market in MARKETS:
        for h in [1, 5, 10, 20]:
            dd = {r['model']: r for r in rows if r['market'] == market and r['horizon'] == h and r['seed'] in ['ensemble', 'fixed']}
            names = VARIANTS+['qrf_original', 'qrf_rule_atom', 'lstm_qr']
            lines.append(f"| {market} | {h} | "+' | '.join(f"{dd[n]['MQS']:.7f}" for n in names)+' |')
    lines += ['', '## 实际跨期限借样质量', '',
        '对每query计算sum_i w_i I(source_h_i != query_h)，下表为每market/h的均值与中位数。逐原点质量另存CSV。', '',
        '| 市场 | 模型 | h | 跨h质量均值 | 跨h质量中位数 |', '|---|---|---:|---:|---:|']
    for r in borrowing_cells:
        lines.append(f"| {r['market']} | {r['model']} | {r['horizon']} | {r['mean']:.4f} | {r['median']:.4f} |")
    lines += ['', '## 解释与保存', '',
        '- pooled支持中的每个样本均按其自身t_i+h_i检查成熟时间，并严格早于fitcut；source-h只提供已知标准化响应，query用自身sigma_t*sqrt(h)反变换。',
        '- 共享不是未来标签输入，也不是测试期追加样本；保存了source horizon、mature索引及每query-h的来源h概率矩阵。',
        '- QRF强基线曾在inner选leaf10/25/50，本组固定25；per_h_CDF与旧QRF对比不是完全匹配的分裂损失消融。三组pooled是匹配设置。',
        '- LSTMQR为缓存五seed分位数平均，仅比较MQS/WIS/覆盖率，不从15tau编造完整CRPS。',
        '- 每seed和ensemble Q/CRPS/p全保留；每市场/模型独立NPZ保留ensemble完整weights、support及训练来源；seed权重哈希已保存。',
        '- 原点/h/y/scale与四套缓存逐值完全相等；所有读取冻结源/结果SHA前后一致。',
        '- 方向只落在碳价zero-aware proper-score分裂与多期限借样结构；不称新forest分布族、首创CDF特征或复现DRF/MMD。', '',
        f"训练/评分/落盘耗时 {report['elapsed_seconds']:.1f} 秒。", '',
        '运行时sanity：', '```json', json.dumps(report['sanity'], ensure_ascii=False, indent=2), '```']
    (P/f'{TAG}_report_zh.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    important = [r for r in report['summary'] if r['seed'] == 'ensemble' and r['baseline'] in ['qrf_original', 'qrf_rule_atom', 'lstm_qr', 'pooled_MSE', 'pooled_CDF', 'per_h_CDF'] and r['metric'] in ['MQS', 'CRPS']]
    print(json.dumps(important, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
