"""Diagnostic with generator-informed features, not a novel predictor."""
import json
from pathlib import Path
import numpy as np


def run():
    rows = []
    poles = np.exp(-np.log(2) / np.array([1., 8., 32.]))
    gain = np.sqrt(1 - poles ** 2)
    for seed in range(400, 420):
        rng = np.random.default_rng(seed)
        r = np.zeros(5400)
        states = np.zeros((5400, 3))
        state = np.zeros(3)
        for t in range(len(r)):
            states[t] = state
            r[t] = .7 * np.prod(np.tanh(state)) + rng.normal()
            state = poles * state + gain * r[t]
        short = np.zeros_like(states)
        for lag in range(1, 61):
            short[lag:] += r[:-lag, None] * gain * poles ** (lag - 1)
        train = np.arange(400, 880)
        test = np.arange(1400, 5400)
        record = {'seed': seed}
        for name, feature in [('linear_modes', states),
                              ('known_full_interaction', np.prod(np.tanh(states), axis=1)[:, None]),
                              ('known_short_interaction', np.prod(np.tanh(short), axis=1)[:, None])]:
            # Fixed tiny ridge, no evaluation-dependent tuning or model search.
            x = feature[train]
            beta = np.linalg.solve(x.T @ x + 1e-6 * np.eye(x.shape[1]), x.T @ r[train])
            pred = feature[test] @ beta
            record[name] = float(np.mean((r[test] - pred) ** 2) / np.mean(r[test] ** 2))
            if name != 'linear_modes':record[name + '_coefficient'] = float(beta[0])
        rows.append(record)
    keys = ['linear_modes', 'known_full_interaction', 'known_short_interaction']
    return {'design': '20 fixed seeds; 480 training labels; fixed ridge; independent later 4000 observations; one-step.',
            'scope': 'Features and poles chosen with generator knowledge. Diagnostic upper control, not fair deployable baseline or innovation.',
            'means': {k: float(np.mean([r[k] for r in rows])) for k in keys}, 'rows': rows}


if __name__ == '__main__':
    result = run()
    Path(__file__).with_name('LEARNABILITY_RESULTS.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))
