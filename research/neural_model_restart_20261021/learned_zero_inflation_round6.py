"""Learned zero-inflation: replace the hand-set rule p by a classifier.

The zero atom is the single largest contributor to the reference (removing it costs
~2.8%).  Its probability is currently the fixed rule p = .8*local + .2*p_static.
Here P(target == 0) is predicted by a classifier fitted on pre-cut rows only, and
the reference quantiles are rebuilt from the cached support weights with that p.
No forest is refitted; nothing is tuned on the test window.
"""
import sys, json
sys.path.insert(0, r'research/neural_model_restart_20261021')
import numpy as np, pandas as pd, torch
import forest_prior_round5 as FP
import forest_transport_round5 as F
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

T = FP.T
P = FP.P
R = F.R
TAUS = np.array(T.B.TAUS)


def pin(y, q):
    e = y[:, None] - q
    return float(2 * np.maximum(TAUS * e, (TAUS - 1) * e).mean())


def crps_from_q(y, q):
    """average pinball x2 is not CRPS; use the exact law instead (handled by caller)."""
    return pin(y, q)


def rebuild(u, z, scale, p):
    w = torch.tensor(u, dtype=torch.float64)
    w = w / w.sum(1, keepdim=True).clamp_min(1e-30)
    op = dict(w=w, p=torch.tensor(p, dtype=torch.float64),
              a=torch.ones(len(w), dtype=torch.float64), b=torch.zeros(len(w), dtype=torch.float64))
    q, c, pe, _, _ = R.mixture(torch.zeros(len(w), dtype=torch.float64), z, [op])
    return q.numpy() * scale[:, None], c.numpy() * scale, pe.numpy()


def main():
    panel = pd.read_csv(T.B.PANEL, parse_dates=['date'])
    z = np.load(P / 'market_info_round5_predictions.npz')
    base = np.load(P / 'qrf_round4_probability_control_predictions.npz')
    out = {}
    results = {}
    for market in T.B.CUT:
        d = panel[panel.market == market].sort_values('date').reset_index(drop=True)
        D = T.add_context(d, T.E.build(d)); D['market'] = market
        cache = FP.load_cache(market, 'final', D)
        X, _ = T.prepare(D, D['rf'], True)
        xa = D['x_aug']
        sid_of = cache['sid'].numpy()
        te = D['te']
        Q_rule = z[market + '_base_Q']; y_te = z[market + '_target']; h_te = z[market + '_horizon']
        rule_q = base[market + '_qrf_rule_atom_Q']
        nb = {'rule': np.array(Q_rule, copy=True), 'logit': np.array(Q_rule, copy=True),
              'gbm': np.array(Q_rule, copy=True)}
        cr = {k: np.zeros(len(te)) for k in nb}
        for hi, h in enumerate(T.HOR):
            h = int(h)
            bank = cache['banks'][hi]
            tr = D['rf'][D['hi'][D['rf']] == hi]
            va = D['va'][D['hi'][D['va']] == hi]
            iz = np.flatnonzero(D['hi'][te] == hi)
            rows = np.concatenate([tr, va, iz])
            truth = (D['y'][rows] == 0).astype(int)
            sc = StandardScaler().fit(xa[tr])
            A = sc.transform(xa[rows])
            nfit = len(tr) + len(va)

            def fit_predict(make):
                if len(np.unique(truth[:nfit])) < 2:
                    rate = float(np.mean(truth[:nfit]))
                    return np.full(len(iz), rate)
                m = make().fit(A[:nfit], truth[:nfit])
                return m.predict_proba(A[nfit:])[:, 1]

            p_zero_l = fit_predict(lambda: LogisticRegression(C=1.0, max_iter=2000))
            p_zero_g = fit_predict(lambda: HistGradientBoostingClassifier(max_iter=200, random_state=11))
            p_l = np.clip(1.0 - p_zero_l, 0.02, 0.999)
            p_g = np.clip(1.0 - p_zero_g, 0.02, 0.999)
            for key, p in [('rule', None), ('logit', p_l), ('gbm', p_g)]:
                if p is None:
                    continue
                sid = sid_of[te[iz]]
                u = bank['u'].numpy()[sid]
                q, c, _ = rebuild(u, bank['z'].double(), D['scale'][te[iz]], p)
                nb[key][iz] = q
                cr[key][iz] = c
        for key in nb:
            for hh in T.HOR:
                k = h_te == hh
                results.setdefault(key, {})[(market, int(hh))] = (
                    pin(y_te[k], nb[key][k]) / pin(y_te[k], rule_q[k]),
                    np.mean(cr[key][k]) / np.mean(np.abs(y_te[k]) + 1e-12))
        # sanity: rule arm must reproduce the frozen base Q
        dmax = float(np.abs(nb['rule'] - Q_rule).max())
        print(market, 'rule-arm maxabs vs frozen base', round(dmax, 5), flush=True)
    for key in results:
        r = np.array([v[0] for v in results[key].values()])
        g = 100 * (1 - np.exp(np.log(r).mean()))
        by_h = {int(hh): round(float(100 * (1 - np.exp(np.mean([np.log(results[key][(m, hh)][0]) for m in T.B.CUT])))), 2) for hh in T.HOR}
        by_m = {m: round(float(100 * (1 - np.exp(np.mean([np.log(results[key][(m, hh)][0]) for hh in T.HOR])))), 2) for m in T.B.CUT}
        out[key] = dict(vs_rule=g, wins=int((r < 1).sum()), by_horizon=by_h, by_market=by_m)
        print('%-6s vs rule %+.4f%%  wins %2d/20  h %s  m %s' % (key, g, int((r < 1).sum()), by_h, by_m), flush=True)
    (P / 'learned_zero_inflation_round6.json').write_text(json.dumps(out, indent=2), encoding='utf-8')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
