"""Causal online recalibration of the reference dispersion.

At test origin t we may use every residual whose target has already realised
(s + h <= t).  This script rebuilds the dispersion multiplier from that trailing
window only, so nothing after t enters the forecast.  A look-ahead variant is run
as a positive control: if the look-ahead version reproduces the ~+2% figure while
the causal one does not, the effect is a regime shift that cannot be tracked with
the available sample.
"""
import sys, json
sys.path.insert(0, r'research/neural_model_restart_20261021')
import numpy as np, pandas as pd
import forest_prior_round5 as FP

T = FP.T
P = FP.P
TAUS = np.array(T.B.TAUS)
WINDOWS = [40, 80, 150]
CLIP = (0.4, 2.5)


def pin(y, q):
    e = y[:, None] - q
    return float(2 * np.maximum(TAUS * e, (TAUS - 1) * e).mean())


def main():
    panel = pd.read_csv(T.B.PANEL, parse_dates=['date'])
    z = np.load(P / 'market_info_round5_predictions.npz')
    base = np.load(P / 'qrf_round4_probability_control_predictions.npz')
    out = {}
    for mode in ['causal', 'lookahead', 'fixed_va']:
        for W in WINDOWS:
            cells = {}
            for market in T.B.CUT:
                d = panel[panel.market == market].sort_values('date').reset_index(drop=True)
                dates = d['date'].values
                Q = z[market + '_base_Q']; y = z[market + '_target']
                h = z[market + '_horizon']; o = z[market + '_origin']
                rule = base[market + '_qrf_rule_atom_Q']
                med = Q[:, 7:8]
                spread2 = np.mean((Q - med) ** 2, axis=1)
                outQ = np.array(Q, copy=True)
                for hh in T.HOR:
                    hh = int(hh)
                    idx = np.flatnonzero(h == hh)
                    idx = idx[np.argsort(dates[o[idx]])]
                    r = y[idx] - med[idx, 0]
                    ratio = np.ones(len(idx))
                    for j, row in enumerate(idx):
                        lo = 0 if mode == 'lookahead' else max(0, j - hh - W)
                        hi = min(len(idx), j + W) if mode == 'lookahead' else max(0, j - hh)
                        if hi - lo < 15:
                            ratio[j] = 1.0
                            continue
                        num = np.mean(r[lo:hi] ** 2)
                        den = np.mean(spread2[idx[lo:hi]])
                        ratio[j] = np.clip(np.sqrt(num / max(den, 1e-12)), *CLIP)
                    if mode == 'fixed_va':
                        ratio[:] = np.sqrt(np.mean(r[: max(20, len(r) // 3)] ** 2) / max(np.mean(spread2[idx[: max(20, len(idx) // 3)]]), 1e-12))
                        ratio[:] = np.clip(ratio, *CLIP)
                    outQ[idx] = med[idx] + ratio[:, None] * (Q[idx] - med[idx])
                for hh in T.HOR:
                    k = h == hh
                    cells[(market, int(hh))] = (pin(y[k], outQ[k]) / pin(y[k], rule[k]),
                                                pin(y[k], outQ[k]) / pin(y[k], Q[k]))
            r0 = np.array([v[0] for v in cells.values()])
            r1 = np.array([v[1] for v in cells.values()])
            g0 = 100 * (1 - np.exp(np.log(r0).mean()))
            g1 = 100 * (1 - np.exp(np.log(r1).mean()))
            by_h = {int(hh): round(float(100 * (1 - np.exp(np.mean([np.log(cells[(m, hh)][0]) for m in T.B.CUT])))), 2) for hh in T.HOR}
            out[f'{mode}_{W}'] = dict(vs_rule=g0, vs_base=g1, wins=int((r1 < 1).sum()), by_horizon=by_h)
            print('%-9s W=%3d  vs rule %+.3f%%  vs base %+.3f%%  wins %2d/20  h %s'
                  % (mode, W, g0, g1, int((r1 < 1).sum()), by_h), flush=True)
    (P / 'online_dispersion_round5.json').write_text(json.dumps(out, indent=2), encoding='utf-8')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
