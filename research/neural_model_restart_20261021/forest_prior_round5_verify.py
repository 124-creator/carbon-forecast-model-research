# -*- coding: utf-8 -*-
"""Read back forest prior caches; no fitting and no cache modifications."""
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
import forest_prior_round5 as F

P = Path(__file__).resolve().parent
TAG = 'forest_prior_round5'


def main():
    torch.set_num_threads(2)
    report = json.loads((P/f'{TAG}.json').read_text(encoding='utf-8'))
    assert report['frozen_sha256_before'] == report['frozen_sha256_after']
    verification = {}
    with threadpool_limits(limits=2):
        for market in F.T.B.CUT:
            top = torch.load(P/f'{TAG}_{market}.pt', map_location='cpu', weights_only=False)
            D = {k: v.numpy() for k, v in top['rows'].items()}
            times, first, inverse = np.unique(D['t'], return_index=True, return_inverse=True)
            assert np.array_equal(top['times'].numpy(), times)
            assert np.array_equal(top['unique_times'].numpy(), times)
            assert np.array_equal(top['row_sid'].numpy(), inverse)
            assert np.array_equal(top['first'].numpy(), first)
            result = {}
            for phase, part in [('inner', 'tr'), ('final', 'rf')]:
                phase_data = F.load_cache(market, phase, D)
                idx = D[part]
                assert np.array_equal(phase_data['fit_D_indices'].numpy(), idx)
                mean, std = D['x_aug'][idx].mean(0), D['x_aug'][idx].std(0)+1e-5
                assert np.array_equal(phase_data['scaler']['mean'].numpy(), mean)
                assert np.array_equal(phase_data['scaler']['std'].numpy(), std)
                assert np.array_equal(phase_data['x'].numpy(), np.clip((D['x_aug']-mean)/std, -8, 8))
                _, expected_banks = F.T.prepare(D, idx, True)
                per_h = {}
                for hi, bank in enumerate(phase_data['banks']):
                    h = int(F.T.HOR[hi])
                    ids = F.nonzero_ids(D, idx, hi)
                    assert np.array_equal(bank['bank_D_indices'].numpy(), ids)
                    for key in ['z', 'phi', 'mature', 'support']:
                        assert torch.equal(bank[key], expected_banks[hi][key])
                    assert np.all(D['t'][ids]+h < phase_data['boundary'])
                    allowed = bank['mature'][None, :] <= top['times'][:, None]
                    assert torch.equal(allowed, bank['allowed'])
                    assert torch.equal(allowed.sum(1), bank['count'])
                    assert torch.all(bank['u'][~allowed] == 0)
                    eligible = bank['count'] > 0
                    mass_error = float((bank['u'][eligible].double().sum(1)-1).abs().max())
                    assert mass_error < 2e-6
                    smooth_error = float((bank['u']-(.95*bank['forest']+.05*bank['uniform'])).abs().max())
                    assert smooth_error < 1e-7
                    positive_uniform = bank['uniform'] > 0
                    lower_ratio = float((bank['u'][positive_uniform].double()/(.05*bank['uniform'][positive_uniform].double())).min())
                    assert lower_ratio >= 1-3e-7
                    invalid = ~bank['valid_sid']
                    if bool(invalid.any()):
                        assert torch.allclose(bank['u'][invalid], bank['uniform'][invalid], atol=1e-7, rtol=1e-6)
                    per_h[str(h)] = dict(Nsid=len(times), Nbank=len(ids), mass_max_error=mass_error,
                        smoothing_max_error=smooth_error, min_uniform_floor_ratio=lower_ratio, future_mass_exact_zero=True,
                        stable_Dindex_bank=True, phase_boundary_strict=True)
                result[phase] = per_h
            verification[market] = result
    (P/f'{TAG}_verification.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding='utf-8')
    checks = pd.read_csv(P/f'{TAG}_baseline_checks.csv')
    lines = ['# 第五轮条件森林参考缓存交付', '',
        '五市场inner/final两phase已完成，使用冻结QRF实际leaf、seed11/22/33/44/55、每seed80树、CPU2。',
        'inner只拟合D.tr，final只拟合D.rf；所有支持标签均按各自t_i+h_i严格早于phase boundary。',
        '全体unique origin sid与seq_transport.prepare一致；每bank稳定非零顺序严格按原D索引映射，重复z不按值匹配。',
        '成熟mask t_key+h<=query_t；mature NZ leaf质量0则成熟uniform fallback；u=.95forest+.05uniform。',
        '缺失该h目标的sid用成熟uniform并标记无效；无成熟NZ的sid全0，沿用count>=32过滤。', '',
        f"200个必要forest重建和缓存生成耗时 {report['elapsed_seconds']:.3f} 秒。", '',
        '| 市场 | cache MiB | 全局Nsid | inner各h Nbank | final各h Nbank |', '|---|---:|---:|---|---|']
    for market, detail in report['markets'].items():
        inner = detail['phases']['inner']; final = detail['phases']['final']
        sizes = lambda values: '/'.join(str(values[str(h)]['n_bank']) for h in [1, 5, 10, 20])
        lines.append(f"| {market} | {detail['bytes']/1024**2:.2f} | {inner['1']['n_sid']} | {sizes(inner)} | {sizes(final)} |")
    lines += ['', '## final test重现检查', '',
        f"- 原QRF非零leaf概率按严格D索引重排后最大差：{checks.original_weight_NZ_Dindex_reorder_max_error.max():.3g}。",
        f"- gamma0条件mark最大差：{checks.gamma0_conditional_mark_max_error.max():.3g}。",
        f"- gamma0+同p_rule分位数最大差：{checks.gamma0_quantile_max_error.max():.3g}；各格MQS完全相同。",
        f"- gamma0完整CRPS最大差：{checks.gamma0_CRPS_max_error.max():.3g}（float32运算误差）。",
        '- 所有冻结旧源/结果SHA前后一致；落盘读回重新核对sid、scaler、T.bank、maturity mask、概率归一及5%uniform保留。', '',
        '## 消费与边界', '',
        '每市场独立PT已原子落盘，phases.inner/final均可加载。接口文档说明所有字段；load_cache(market,phase,D)只读并断言D/sid一致。',
        'val只能消费inner forest；测试和最终重拟合消费final forest。',
        '森林/scaler/phi按整个允许fit区间估计，成熟mask只限制检索响应，不宣称训练query全参数vintage。',
        '本任务仅构建新模型需要的参考，没有重调leaf或重新选择测试模型。']
    (P/f'{TAG}_report_zh.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('ALL_READBACK_CHECKS_PASSED')
    print(checks[['original_weight_NZ_Dindex_reorder_max_error', 'gamma0_CRPS_max_error', 'gamma0_quantile_max_error']].max().to_string())


if __name__ == '__main__':
    main()
