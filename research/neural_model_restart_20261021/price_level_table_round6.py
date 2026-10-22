# -*- coding: utf-8 -*-
"""Price-level metrics per market and per horizon, in the format the literature uses.

The round-6 comparison pooled four horizons into one number, which is why the level
metrics looked low.  Most carbon-price papers report one market and one step ahead,
so this script prints exactly that cell, plus h=5 for reference.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import empirical_tilt_round3 as T

MARKETS = list(T.B.CUT)
HOR = list(T.HOR)
ARMS = ['zist_rule', 'qrf_original', 'qrnn', 'lstm_qr', 'gru_qr', 'tcn_qr', 'transformer_qr', 'gaussian']
SRC = {'zist_rule': ('qrf', 'qrf_rule_atom_Q'), 'qrf_original': ('qrf', 'qrf_original_Q'), 'gaussian': ('dp', 'gaussian_Q')}


def main():
    panel = pd.read_csv(T.B.PANEL, parse_dates=['date'])
    qrf = np.load(HERE / 'qrf_round4_probability_control_predictions.npz')
    dp = np.load(HERE / 'deep_seq_round4_predictions.npz')
    qn = np.load(HERE / 'qrnn_round6_predictions.npz')
    rows = []
    for market in MARKETS:
        d = panel[panel.market == market].sort_values('date').reset_index(drop=True)
        price = d.price.to_numpy(float)
        origin = dp[market + '_origin']; hz = dp[market + '_horizon']; y = dp[market + '_target']; scale = dp[market + '_scale']
        p0 = price[origin]; p1 = p0 * np.exp(y)
        for h in HOR:
            k = hz == h
            for a in ARMS:
                if a in SRC:
                    q = (qrf if SRC[a][0] == 'qrf' else dp)[market + '_' + SRC[a][1]]
                elif a == 'qrnn':
                    q = qn[market + '_qrnn_Q']
                else:
                    q = dp[market + '_' + a + '_Q']
                med = q[k][:, 7] * scale[k]
                pred = p0[k] * np.exp(med)
                ss = float(((p1[k] - p1[k].mean()) ** 2).sum())
                rows.append(dict(market=market, horizon=h, arm=a, n=int(k.sum()),
                                 MAPE=float((np.abs(pred - p1[k]) / p1[k]).mean() * 100),
                                 RMSE=float(np.sqrt(((pred - p1[k]) ** 2).mean())),
                                 MAE=float(np.abs(pred - p1[k]).mean()),
                                 R2=float(1 - ((pred - p1[k]) ** 2).sum() / ss),
                                 mean_abs_move=float(np.abs(p1[k] - p0[k]).mean()),
                                 level_sd=float(p1[k].std())))
    df = pd.DataFrame(rows)
    # persistence ceiling: predict p[t+h] = p[t]
    ceil = []
    for market in MARKETS:
        d = panel[panel.market == market].sort_values('date').reset_index(drop=True)
        price = d.price.to_numpy(float)
        origin = dp[market + '_origin']; hz = dp[market + '_horizon']; y = dp[market + '_target']
        p0 = price[origin]; p1 = p0 * np.exp(y)
        for h in HOR:
            k = hz == h
            ss = float(((p1[k] - p1[k].mean()) ** 2).sum())
            ceil.append(dict(market=market, horizon=h, persistence_R2=float(1 - ((p0[k] - p1[k]) ** 2).sum() / ss)))
    df = df.merge(pd.DataFrame(ceil), on=['market', 'horizon'])
    df.to_csv(HERE / 'price_level_table_round6_cells.csv', index=False)
    L = ['# 价格水平指标：按文献口径（单市场、单步）', '',
         '同一批测试行、同一中位数预测。`mean_abs_move` 与 `level_sd` 说明该窗口的价格波动幅度，'
         'R² 的高低主要由它们决定，而不是由模型决定。', '']
    for h in [1, 5]:
        L += [f'## h = {h}', '', '| 市场 | ' + ' | '.join(ARMS) + ' | 平均绝对变动 | 价格水平 sd |',
              '|---|' + '---:|' * (len(ARMS) + 2)]
        for m in MARKETS:
            s = df[(df.market == m) & (df.horizon == h)]
            L.append(f'| {m} | ' + ' | '.join(f'{float(s[s.arm == a].R2.iloc[0]):.4f}' for a in ARMS)
                     + f' | {float(s.mean_abs_move.iloc[0]):.3f} | {float(s.level_sd.iloc[0]):.2f} |')
        L += ['', '| 市场 | ' + ' | '.join(ARMS) + ' |', '|---|' + '---:|' * len(ARMS)]
        for m in MARKETS:
            s = df[(df.market == m) & (df.horizon == h)]
            L.append(f'| {m} (MAPE %) | ' + ' | '.join(f'{float(s[s.arm == a].MAPE.iloc[0]):.3f}' for a in ARMS) + ' |')
        L.append('')
    (HERE / 'PRICE_LEVEL_TABLE_ZH.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
