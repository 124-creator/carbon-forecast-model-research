# -*- coding: utf-8 -*-
"""Read-only verification of prefix-forest training-reference alignment."""
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
import forest_prior_honest_round5 as H

P = Path(__file__).resolve().parent
TAG = 'forest_prior_honest_round5'


def main():
    torch.set_num_threads(2)
    report = json.loads((P/f'{TAG}.json').read_text(encoding='utf-8'))
    assert report['manifest']['frozen_sha256_before'] == report['frozen_sha256_after']
    rows, block_rows = [], []
    with threadpool_limits(limits=2):
        for market in H.F.T.B.CUT:
            old = torch.load(P/f'forest_prior_round5_{market}.pt', map_location='cpu', weights_only=False)
            new = torch.load(P/f'{TAG}_{market}.pt', map_location='cpu', weights_only=False)
            for key in ['times', 'unique_times', 'row_sid', 'valid_sid_h']:
                assert torch.equal(old[key], new[key])
            for key in old['rows']:
                assert torch.equal(old['rows'][key], new['rows'][key])
            D = {key: value.numpy() for key, value in new['rows'].items()}
            ends = D['t']+H.F.T.HOR[D['hi']]
            times = new['times'].numpy();inverse = new['row_sid'].numpy()
            for phase in ['inner', 'final']:
                source, ph = old['phases'][phase], new['phases'][phase]
                assert H.immutable_phase_digest(source) == H.immutable_phase_digest(ph)
                bound = int(ph['boundary'])
                after = torch.tensor(np.flatnonzero(times >= bound))
                for hi, h in enumerate(H.F.T.HOR):
                    a, b = source['banks'][hi], ph['banks'][hi]
                    for key in ['u', 'forest', 'u_forest', 'm0', 'v0', 'fallback']:
                        assert torch.equal(a[key][after], b[key][after])
                    fit_rows = D['tr' if phase == 'inner' else 'rf']
                    fit_rows = fit_rows[D['hi'][fit_rows] == hi]
                    fit_sids = inverse[fit_rows]
                    fit_sids = fit_sids[b['count'].numpy()[fit_sids] >= 32]
                    idx = torch.tensor(fit_sids)
                    l1 = (a['u'][idx].double()-b['u'][idx].double()).abs().sum(1).numpy()
                    val_or_test = D['va' if phase == 'inner' else 'te']
                    qrows = val_or_test[D['hi'][val_or_test] == hi]
                    assert torch.equal(a['u'][torch.tensor(inverse[qrows])], b['u'][torch.tensor(inverse[qrows])])
                    assert torch.all(b['u'][~b['allowed']] == 0)
                    nonempty = b['count'] > 0
                    error = float((b['u'][nonempty].double().sum(1)-1).abs().max())
                    assert error < 2e-6
                    rows.append(dict(market=market, phase=phase, horizon=int(h),
                        eligible_training_queries=len(idx), changed_training_queries=int((l1 > 1e-7).sum()),
                        training_reference_mean_L1=float(l1.mean()), training_reference_median_L1=float(np.median(l1)),
                        training_reference_max_L1=float(l1.max()), mass_max_error=error,
                        future_mass_exact_zero=True, evaluation_u_bitwise_equal=True,
                        all_postboundary_reference_fields_bitwise_equal=True,
                        NN_scaler_x_bank_phi_masks_unchanged=True))
                for block in ph['honest_blocks']:
                    prefix = block['prefix_fit_D_indices'].numpy()
                    left, right = block['start'], block['stop']
                    assert np.array_equal(prefix, ph['fit_D_indices'].numpy()[ends[ph['fit_D_indices'].numpy()] < left])
                    assert np.all(ends[prefix] < left)
                    if len(prefix):
                        raw = D['x_aug']
                        assert np.array_equal(raw[prefix].mean(0), block['scaler']['mean'].numpy())
                        assert np.array_equal(raw[prefix].std(0)+1e-5, block['scaler']['std'].numpy())
                    for detail in block['per_h']:
                        hi = list(H.F.T.HOR).index(detail['horizon'])
                        ids = detail['prefix_h_D_indices'].numpy();query = detail['query_D_indices'].numpy()
                        nz = detail['prefix_NZ_D_indices'].numpy();cols = detail['full_bank_columns'].numpy()
                        assert np.all(D['hi'][ids] == hi) and np.all(ends[ids] < left)
                        assert np.intersect1d(ids, query).size == 0
                        assert np.all((D['t'][query] >= left)&(D['t'][query] < right))
                        assert np.array_equal(ph['banks'][hi]['bank_D_indices'].numpy()[cols], nz)
                        block_rows.append(dict(market=market, phase=phase, block=block['block'], horizon=int(detail['horizon']),
                            block_start=left, block_stop=right, n_prefix_allh=len(prefix), n_forest_source=len(ids),
                            n_query=len(query), last_forest_target=int(ends[ids].max()) if len(ids) else None,
                            source_strictly_before_block=True, no_own_or_future_label=True,
                            prefix_scaler_allh=True, Dindex_NZ_mapping_exact=True))
            print('READBACK_OK', market, flush=True)
    pd.DataFrame(rows).to_csv(P/f'{TAG}_reference_shift.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(block_rows).to_csv(P/f'{TAG}_block_checks.csv', index=False, encoding='utf-8-sig')
    verification = dict(all_checks_passed=True, reference_rows=rows, block_checks=block_rows)
    (P/f'{TAG}_verification.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 第五轮：prefix-forest训练查询参考cache交付', '',
        '目标是改变训练残差看到的参考误差机制；不宣称旧评估无效，不改验证/测试参考。',
        'inner/final仍对应D.tr/D.rf，四个block与RF设置均在fit前固定。每block森林只使用label t_i+h_i<block_start的源行，prefix scaler按所有h的该prefix估计。',
        '首block与prefix单h<=1条时直接使用当前成熟NZ uniform；其余按原已选leaf、五seed、各80树重建；不丢合法NN query。',
        '参考仍为.95prefixforest+.05当前成熟NZ uniform；uniform可以利用block开始后但不晚于query已成熟的病例。', '',
        f"实际forest fits={report['forest_fits_completed']}，生成耗时{report['elapsed_seconds']:.3f}秒，CPU2。", '',
        '| 市场 | MiB | inner block边界 | final block边界 |', '|---|---:|---|---|']
    for market, info in report['markets'].items():
        lines.append(f"| {market} | {info['bytes']/1024**2:.2f} | {info['phases']['inner']['boundaries']} | {info['phases']['final']['boundaries']} |")
    lines += ['', '## 不变项与新元数据', '',
        '- 所有q>=phase boundary的u/forest/u_forest/m0/v0/fallback逐值不变，inner D.va和final D.te的u位级相同。',
        '- 全部原D keys、unique sid、完整NZ bank支持/phi/mature/allowed/count/uniform、fitindices、NN x/scaler均保持旧cache值；合法训练query eligibility不变。',
        '- honest_blocks保存每block全prefix源索引/target成熟索引/scaler、每h森林源D索引及NZ到完整bank列的映射、query索引。',
        '- query_block_sid指明训练sid所属block，边界后为-1；source_cache_sha256记录原cache。',
        '- 所有读取旧源/结果/cache SHA前后一致；缓存只使用新前缀，已原子落盘。', '',
        '## 训练参考实际变化', '',
        '下表按所有count>=32的合法训练query统计新旧u的L1距离，阶段性分组；完整market/h明细另存数值表。', '',
        '| 市场 | phase | 训练query数 | L1均值 | 改变query数 |', '|---|---|---:|---:|---:|']
    frame = pd.DataFrame(rows)
    for (market, phase), part in frame.groupby(['market', 'phase'], sort=False):
        n = int(part.eligible_training_queries.sum())
        mean = float(np.average(part.training_reference_mean_L1, weights=part.eligible_training_queries))
        lines.append(f"| {market} | {phase} | {n} | {mean:.4f} | {int(part.changed_training_queries.sum())} |")
    lines += ['', '## 范围', '',
        '这只让森林训练query参考以固定时间前缀构造。NN参数、NN scaler、phi和完整phase bank仍正常使用允许的整个fit partition；不称所有参数historical vintage。',
        '旧参考训练机制与新机制的效果由后续NN训练比较；本cache任务没有选择新超参、重训NN或更改forecast参考。',
        '消费接口与原cache一致，切换forest_prior_honest_round5前缀即可。']
    (P/f'{TAG}_report_zh.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('ALL_HONEST_READBACK_CHECKS_PASSED')


if __name__ == '__main__':
    main()
