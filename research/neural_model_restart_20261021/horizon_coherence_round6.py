"""Multi-horizon coherence: project the conditional spread profile onto a power law.

For a fixed origin t the four horizon laws are fitted independently, so their
spreads need not follow any term structure.  Enforcing spread(h) = sigma(t) * h**alpha
borrows strength across horizons: sigma(t) is the cross-horizon level at that date
(so the local volatility level is untouched) and only the *profile* is smoothed.
No parameter is estimated on the test window; alpha is either fixed a priori or
estimated from pre-cut laws.
"""
import sys, json
sys.path.insert(0, r'research/neural_model_restart_20261021')
import numpy as np, pandas as pd
import forest_prior_round5 as FP

T = FP.T
P = FP.P
TAUS = np.array(T.B.TAUS)
ALPHAS = [0.40, 0.50, 0.60, 0.70]
MODES = ['spread_only', 'spread_median']


def pin(y, q):
    e = y[:, None] - q
    return float(2 * np.maximum(TAUS * e, (TAUS - 1) * e).mean())


def project(Q, h, alpha, mode):
    """Q rows share one origin; rescale each row's spread to sigma*h**alpha."""
    med = Q[:, 7].copy()
    spread = Q.std(axis=1)
    hh = h.astype(float)
    ok = spread > 1e-12
    if ok.sum() < 2:
        return Q
    lw = np.log(spread[ok]) - alpha * np.log(hh[ok])
    sigma = float(np.exp(lw.mean()))
    f = np.ones(len(Q))
    f[ok] = sigma * hh[ok] ** alpha / spread[ok]
    out = med[:, None] + f[:, None] * (Q - med[:, None])
    if mode == 'spread_only':
        out = out - (out[:, 7] - med)[:, None]
    return out


def main():
    z = np.load(P / 'market_info_round5_predictions.npz')
    base = np.load(P / 'qrf_round4_probability_control_predictions.npz')
    out = {}
    for mode in MODES:
        for alpha in ALPHAS:
            cells = {}
            for market in T.B.CUT:
                Q = z[market + '_base_Q']; y = z[market + '_target']
                h = z[market + '_horizon']; o = z[market + '_origin']
                rule = base[market + '_qrf_rule_atom_Q']
                Qp = np.array(Q, copy=True)
                for t in np.unique(o):
                    idx = np.flatnonzero(o == t)
                    if len(idx) < 2:
                        continue
                    Qp[idx] = project(Q[idx], h[idx], alpha, mode)
                for hh in T.HOR:
                    k = h == hh
                    cells[(market, int(hh))] = (pin(y[k], Qp[k]) / pin(y[k], rule[k]),
                                                pin(y[k], Qp[k]) / pin(y[k], Q[k]))
            r0 = np.array([v[0] for v in cells.values()])
            r1 = np.array([v[1] for v in cells.values()])
            g0 = 100 * (1 - np.exp(np.log(r0).mean()))
            g1 = 100 * (1 - np.exp(np.log(r1).mean()))
            by_h = {int(hh): round(float(100 * (1 - np.exp(np.mean([np.log(cells[(m, hh)][0]) for m in T.B.CUT])))), 2) for hh in T.HOR}
            by_m = {m: round(float(100 * (1 - np.exp(np.mean([np.log(cells[(m, hh)][0]) for hh in T.HOR])))), 2) for m in T.B.CUT}
            out[f'{mode}_{alpha}'] = dict(vs_rule=g0, vs_base=g1, wins=int((r1 < 1).sum()),
                                         by_horizon=by_h, by_market=by_m)
            print('%-13s a=%.2f  vs rule %+.3f%%  vs base %+.3f%%  wins %2d/20  h %s  m %s'
                  % (mode, alpha, g0, g1, int((r1 < 1).sum()), by_h, by_m), flush=True)
    (P / 'horizon_coherence_round6.json').write_text(json.dumps(out, indent=2), encoding='utf-8')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
