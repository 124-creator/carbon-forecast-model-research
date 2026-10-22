# -*- coding: utf-8 -*-
"""Round-6 comparison table: our reference vs the published-style QR baselines.

Panel A: pinball-based (MQS/WIS over the same 15 taus, same rows, same cuts).
Panel B: price-level metrics of the median forecast (MAPE/RMSE/R2), which is what
         most carbon-price papers report, so the table is directly comparable.
Nothing is selected on the test window; every arm is the frozen five-seed average.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import empirical_tilt_round3 as T

TAUS = np.array(T.B.TAUS)
HOR = list(T.HOR)
MARKETS = list(T.B.CUT)
RNG = np.random.default_rng(20261021)
B = 2000

ARMS = ['zist_rule', 'qrf_original', 'qrnn', 'qrnn_anchor', 'lstm_qr', 'gru_qr', 'tcn_qr', 'transformer_qr', 'uniform_static', 'gaussian']
SRC = {'zist_rule': ('qrf', 'qrf_rule_atom_Q'), 'qrf_original': ('qrf', 'qrf_original_Q'),
       'uniform_static': ('dp', 'uniform_static_Q'), 'gaussian': ('dp', 'gaussian_Q')}


def pin(y, q):
    e = y[:, None] - q
    return float(2 * np.maximum(TAUS * e, (TAUS - 1) * e).mean())


def wis(y, q):
    """Weighted interval score over the 7 central intervals the 15 taus provide."""
    m = q[:, 7]
    parts = [0.5 * np.abs(y - m)]; weights = [0.5]
    for lo, hi in [(0, 14), (1, 13), (2, 12), (3, 11), (4, 10), (5, 9), (6, 8)]:
        alpha = 1.0 - (TAUS[hi] - TAUS[lo])
        width = q[:, hi] - q[:, lo]
        pen = 2.0 / alpha * np.maximum(0.0, q[:, lo] - y) + 2.0 / alpha * np.maximum(0.0, y - q[:, hi])
        parts.append(width + pen); weights.append(alpha / 2)
    return float((np.array(weights)[:, None] * np.stack(parts)).sum(0).mean() / (len(weights) - 0.5))


def main():
    panel = pd.read_csv(T.B.PANEL, parse_dates=['date'])
    qrf = np.load(HERE / 'qrf_round4_probability_control_predictions.npz')
    dp = np.load(HERE / 'deep_seq_round4_predictions.npz')
    qn_path = HERE / 'qrnn_round6_predictions.npz'
    qn = np.load(qn_path) if qn_path.exists() else None
    arms = [a for a in ARMS if a not in ('qrnn', 'qrnn_anchor') or qn is not None]
    cells = []
    for market in MARKETS:
        d = panel[panel.market == market].sort_values('date').reset_index(drop=True)
        price = d.price.to_numpy(float)
        origin = dp[market + '_origin']; hz = dp[market + '_horizon']; y = dp[market + '_target']; scale = dp[market + '_scale']
        assert np.array_equal(origin, qrf[market + '_origin'])
        p0 = price[origin]; p1 = p0 * np.exp(y)
        for h in HOR:
            k = hz == h
            row = dict(market=market, horizon=h, n=int(k.sum()))
            for a in arms:
                if a in SRC:
                    q = (qrf if SRC[a][0] == 'qrf' else dp)[market + '_' + SRC[a][1]]
                elif a in ('qrnn', 'qrnn_anchor'):
                    q = qn[market + '_' + a + '_Q']
                else:
                    q = dp[market + '_' + a + '_Q']
                q = q[k]; yy = y[k]
                med = q[:, 7] * scale[k]
                pred = p0[k] * np.exp(med)
                row[f'{a}_MQS'] = pin(yy, q)
                row[f'{a}_WIS'] = wis(yy, q)
                row[f'{a}_cov80'] = float(((yy >= q[:, 2]) & (yy <= q[:, 12])).mean())
                row[f'{a}_cov95'] = float(((yy >= q[:, 0]) & (yy <= q[:, 14])).mean())
                row[f'{a}_MAE'] = float(np.abs(yy - med).mean())
                row[f'{a}_MAPE'] = float((np.abs(pred - p1[k]) / p1[k]).mean() * 100)
                row[f'{a}_RMSE'] = float(np.sqrt(((pred - p1[k]) ** 2).mean()))
                ss = float(((p1[k] - p1[k].mean()) ** 2).sum())
                row[f'{a}_R2'] = float(1 - ((pred - p1[k]) ** 2).sum() / ss)
            cells.append(row)
    df = pd.DataFrame(cells)

    def gain(a, b, metric='MQS'):
        r = df[f'{a}_{metric}'] / df[f'{b}_{metric}']
        return 100 * (1 - np.exp(np.log(r).mean())), int((r < 1).sum())

    def boot(a, b, metric='MQS'):
        lr = np.log(df[f'{a}_{metric}'] / df[f'{b}_{metric}']).to_numpy()
        idx = RNG.integers(0, len(lr), size=(B, len(lr)))
        g = 100 * (1 - np.exp(lr[idx].mean(1)))
        lo, hi = np.percentile(g, [2.5, 97.5])
        return dict(gain=float(100 * (1 - np.exp(lr.mean()))), lo=float(lo), hi=float(hi), p_pos=float((g > 0).mean()))

    summary = {}
    for a in arms:
        g, w = gain(a, 'zist_rule'); gw, _ = gain(a, 'zist_rule', 'WIS')
        summary[a] = dict(MQS_gain_vs_zist=round(g, 4), MQS_wins=f'{w}/20', WIS_gain_vs_zist=round(gw, 4),
                          cov80=round(float(df[f'{a}_cov80'].mean()), 3), cov95=round(float(df[f'{a}_cov95'].mean()), 3),
                          MAPE_level=round(float(df[f'{a}_MAPE'].mean()), 3), RMSE_level=round(float(df[f'{a}_RMSE'].mean()), 3),
                          R2_level=round(float(df[f'{a}_R2'].mean()), 4))
    ci = {a: boot(a, 'zist_rule') for a in arms if a != 'zist_rule'}
    by_market = {a: {m: round(gain(a, 'zist_rule')[0], 3) for m in []} for a in []}
    pm = {}
    for a in arms:
        if a == 'zist_rule':
            continue
        pm[a] = {m: round(float(100 * (1 - np.exp(np.log(df[df.market == m][f'{a}_MQS'] / df[df.market == m][f'zist_rule_MQS']).mean()))), 3) for m in MARKETS}
        pm[a]['_by_horizon'] = {int(h): round(float(100 * (1 - np.exp(np.log(df[df.horizon == h][f'{a}_MQS'] / df[df.horizon == h][f'zist_rule_MQS']).mean()))), 3) for h in HOR}
    out = dict(arms=arms, summary=summary, bootstrap_vs_zist=ci, by_market_and_horizon=pm, cells=len(df),
               note='MQS/WIS from the same 15 taus and the same rows; MAPE/RMSE/R2 are price-level metrics of the median forecast. Positive gain = arm better than zist_rule.')
    (HERE / 'qr_baseline_table_round6.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    df.to_csv(HERE / 'qr_baseline_table_round6_cells.csv', index=False)

    L = ['# 第六轮：与文献常用分位数基准的同协议对比', '',
         f'20 个 market×horizon 单元，同一批行、同一截断、同一 15 个 τ。所有对照臂都是 5 seed 平均，未在测试窗上做任何选择。增益定义为 `100·(1−exp(mean(log(MQS_arm/MQS_zist))))`，正数表示该臂比我们的参考好。', '',
         '## Panel A：pinball 指标（MQS / WIS，15 τ）', '',
         '| 模型 | MQS 相对 zist_rule | 胜格 | WIS 相对 | 80% 覆盖 | 95% 覆盖 | 95% 区间 CI (MQS) |', '|---|---:|---:|---:|---:|---:|---|']
    for a in arms:
        s = summary[a]; c = ci.get(a)
        tag = '（本模型）' if a == 'zist_rule' else ''
        citxt = '—' if not c else f'[{c["lo"]:+.3f}, {c["hi"]:+.3f}]，P(+)= {c["p_pos"]:.3f}'
        L.append(f'| {a}{tag} | {s["MQS_gain_vs_zist"]:+.4f}% | {s["MQS_wins"]} | {s["WIS_gain_vs_zist"]:+.4f}% | {s["cov80"]:.3f} | {s["cov95"]:.3f} | {citxt} |')
    L += ['', '## Panel B：价格水平指标（中位数预测，直接对标文献报告的 MAPE/RMSE/R²）', '',
          '| 模型 | MAPE (%) | RMSE (CNY/t) | R² |', '|---|---:|---:|---:|']
    for a in arms:
        s = summary[a]
        L.append(f'| {a} | {s["MAPE_level"]:.3f} | {s["RMSE_level"]:.3f} | {s["R2_level"]:.4f} |')
    L += ['', '## 逐市场 / 逐周期（MQS 相对 zist_rule）', '',
          '| 模型 | ' + ' | '.join(MARKETS) + ' | ' + ' | '.join(f'h={h}' for h in HOR) + ' |',
          '|---|' + '---:|' * (len(MARKETS) + len(HOR))]
    for a in pm:
        L.append('| ' + a + ' | ' + ' | '.join(f'{pm[a][m]:+.2f}' for m in MARKETS) + ' | ' + ' | '.join(f'{pm[a]["_by_horizon"][int(h)]:+.2f}' for h in HOR) + ' |')
    L += ['', '说明：Panel B 的 R² 在价格水平上几乎必然接近 1（持续性所致），因此它不能区分模型；这正是文献里 R²=0.88–0.99 的来源。Panel A 才是可比的判别口径。']
    (HERE / 'QR_BASELINE_TABLE_ZH.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
