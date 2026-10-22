# -*- coding: utf-8 -*-
"""Read and verify saved QRF mixtures; produce Chinese baseline comparison."""
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parent
TAG = 'qrf_round4'
SEEDS = [11, 22, 33, 44, 55]
MARKETS = ['CEA', 'GUANGZHOU', 'SHANGHAI', 'HUBEI', 'EU']


def main():
    report = json.loads((P/f'{TAG}.json').read_text(encoding='utf-8'))
    assert set(report['markets']) == set(MARKETS)
    assert report['frozen_sha256_before'] == report['frozen_sha256_after']
    pred = np.load(P/f'{TAG}_predictions.npz')
    saved_checks = {}
    for market in MARKETS:
        key = market+'_'
        wfile = np.load(P/f'{TAG}_{market}_weights.npz')
        maxerr, max_p_error, max_quantile_average_difference = 0., 0., 0.
        max_weight_mass_error = 0.
        maximum_CRPS_convexity_error = 0.
        for h in [1, 5, 10, 20]:
            support = wfile[f'support_h{h}']
            rows = wfile[f'test_positions_h{h}']
            assert np.all(np.diff(support) >= 0)
            assert np.all(wfile[f'fit_origin_h{h}']+h < report['markets'][market]['audit']['cut_index'])
            assert np.sum(np.abs(support) <= 1e-12) == report['markets'][market]['selection'][str(h)]['final_zero_labels']
            ww = [wfile[f'seed{s}_weights_h{h}'] for s in SEEDS]
            mixture = wfile[f'ensemble_weights_h{h}']
            error = float(np.max(np.abs(np.mean(ww, 0)-mixture)))
            maxerr = max(maxerr, error)
            assert np.allclose(np.mean(ww, 0), mixture, atol=3e-8, rtol=1e-6)
            for seed, w in [(str(s), w) for s, w in zip(SEEDS, ww)]+[('ensemble', mixture)]:
                assert np.all(w >= 0) and np.isfinite(w).all()
                mass_error = float(np.max(np.abs(w.sum(1, dtype=np.float64)-1.)))
                max_weight_mass_error = max(max_weight_mass_error, mass_error)
                assert mass_error < 2e-6
                p = 1-w[:, np.abs(support) <= 1e-12].sum(1)
                stored_p = pred[key+f'qrf_{seed}_p'][rows]
                max_p_error = max(max_p_error, float(np.max(np.abs(p-stored_p))))
                assert np.allclose(p, stored_p, atol=2e-6, rtol=0)
                q = pred[key+f'qrf_{seed}_Q'][rows]
                assert np.isfinite(q).all() and np.all(np.diff(q, axis=1) >= -1e-8)
            qmean = np.mean([pred[key+f'qrf_{s}_Q'][rows] for s in SEEDS], 0)
            max_quantile_average_difference = max(max_quantile_average_difference,
                float(np.max(np.abs(qmean-pred[key+'qrf_ensemble_Q'][rows]))))
            cmean = np.mean([pred[key+f'qrf_{s}_CRPS'][rows] for s in SEEDS], 0)
            delta = pred[key+'qrf_ensemble_CRPS'][rows]-cmean
            maximum_CRPS_convexity_error = max(maximum_CRPS_convexity_error, float(delta.max()))
            assert delta.max() < 3e-7
        saved_checks[market] = dict(max_saved_mixture_error=maxerr, max_zero_mass_probability_error=max_p_error,
            max_weight_mass_error_float64=max_weight_mass_error,
            max_ensemble_quantile_vs_seed_mean_difference=max_quantile_average_difference,
            max_CRPS_minus_mean_seed_CRPS=maximum_CRPS_convexity_error,
            fit_labels_strictly_before_cut=True, original_zero_labels_preserved=True)
    (P/f'{TAG}_verification.json').write_text(json.dumps(saved_checks, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = pd.DataFrame(report['summary'])
    cells = pd.DataFrame(report['cells'])
    lines = ['# 第四轮标准QRF条件分布强基线', '',
             '每市场/期限训练RandomForestRegressor，80树、depth6、seed11/22/33/44/55、n_jobs=2。',
             'minleaf从10/25/50由seed11内层purged validation normalized MQS选择；随后全部precut labels重拟合。',
             '36维context、原点、期限、原始收益及scale均与冻结五seed tilt_full完全相同；scaler只在inner tr估计，最终在precut重估。', '',
             '这是标准QRF叶分布实现：树以均方误差学习mean分裂；query所在叶内每个原始fit样本均获1/n_leaf概率，再平均80棵树。',
             'bootstrap用于学习树分区；叶分布对全部原始fit行各计一次，不把bootstrap重复次数当响应质量。保留所有零收益标签。',
             '五seed概率矩阵在相同排序support上平均，然后计算inverse-ECDF分位数和完整weighted-ECDF CRPS；没有平均树点预测充当分布，也没有追加测试标签。',
             '不声称复现DRF的MMD splitting。', '',
             '## 五seed ensemble对照', '',
             '改善=1−20个market×h得分比值的几何均值，正值表示表中模型优于基线。', '',
             '| 模型 | 基线 | MQS改善% | 胜格 | 完整CRPS改善% | 胜格 |', '|---|---|---:|---:|---:|---:|']
    selected = summary[((summary.model == 'qrf')&(summary.seed == 'ensemble'))|(summary.model == 'tilt_full_5seed')]
    for (model, base), group in selected.groupby(['model', 'baseline'], sort=False):
        dd = {r['metric']: r for r in group.to_dict('records')}
        q, c = dd['MQS'], dd.get('CRPS')
        lines.append(f"| {model} | {base} | {q['geometric_gain_pct']:+.3f} | {q['wins']}/20 | "+
            (f"{c['geometric_gain_pct']:+.3f} | {c['wins']}/20 |" if c else '— | — |'))
    lines += ['', '## QRF逐seed对冻结五seed tilt_full', '',
              '| QRF seed | MQS改善% | 胜格 | 完整CRPS改善% | 胜格 |', '|---|---:|---:|---:|---:|']
    for seed in map(str, SEEDS):
        ss = summary[(summary.model == 'qrf')&(summary.seed == seed)&(summary.baseline == 'tilt_full_5seed')]
        dd = {r['metric']: r for r in ss.to_dict('records')}
        q, c = dd['MQS'], dd['CRPS']
        lines.append(f"| {seed} | {q['geometric_gain_pct']:+.3f} | {q['wins']}/20 | {c['geometric_gain_pct']:+.3f} | {c['wins']}/20 |")
    lines += ['', '## 各市场/期限的分数与inner选择', '',
              '| 市场 | h | leaf | QRF MQS | tilt MQS | QRF CRPS | tilt CRPS | QRF WIS | tilt WIS | QRF cov80 | tilt cov80 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for market in MARKETS:
        for h in [1, 5, 10, 20]:
            ss = cells[(cells.market == market)&(cells.horizon == h)]
            q = ss[(ss.model == 'qrf')&(ss.seed == 'ensemble')].iloc[0]
            t = ss[ss.model == 'tilt_full_5seed'].iloc[0]
            leaf = report['markets'][market]['selection'][str(h)]['chosen_minleaf']
            lines.append(f"| {market} | {h} | {leaf} | {q.MQS:.7f} | {t.MQS:.7f} | {q.CRPS:.7f} | {t.CRPS:.7f} | {q.WIS:.7f} | {t.WIS:.7f} | {q.coverage80:.3f} | {t.coverage80:.3f} |")
    lines += ['', '## 审计', '',
              '- 原点、h、y、scale与冻结followup逐值完全相等；主候选取full_ensemble五seed，未混用原两seed。',
              '- 每seed/ensemble的Q、完整CRPS、非零概率及逐市场同support完整weights均保存。',
              '- 稀疏leaf权重与逐树暴力公式一致；单叶uniform测试、零标签质量测试、权重非负/归一/有限、分位数有序均通过。',
              '- 所有预测完整CRPS用独立CDF区间积分核对；保存后再核对五seed权重均值、零质量、标签cut和CRPS混合凸性。',
              '- 冻结T/输入依赖源及旧结果SHA256前后一致；只新增qrf_round4前缀文件。',
              '- QRF与tilt共享输入、目标及precut支持信息，RF分裂损失及分布参数化不同；不是DRF/MMD splitting。',
              '- HGB为已缓存预测，lookback250与本次128不同，仅比较raw MQS/WIS/覆盖率，不能把信息差归结构。',
              '- legacy empirical的MQS为插值样本分位，完整CRPS为ECDF；uniform_static统一使用inverse ECDF表示。', '',
              f"训练及落盘耗时 {report['elapsed_seconds']:.1f} 秒。", '',
              '运行时sanity：', '```json', json.dumps(report['sanity'], ensure_ascii=False, indent=2), '```']
    (P/f'{TAG}_report_zh.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(selected.to_csv(index=False))
    print(json.dumps(saved_checks, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
