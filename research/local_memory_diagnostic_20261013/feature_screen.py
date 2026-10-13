"""Fixed diagnostic feature screen; no market data, no new algorithm claim."""
import json
from pathlib import Path
import numpy as np


def run():
    poles = np.exp(-np.log(2) / np.array([1., 8., 32.]))
    gain = np.sqrt(1 - poles ** 2)
    rows = []
    for seed in range(400, 420):
        rng = np.random.default_rng(seed)
        r = np.zeros(5400)
        state = np.zeros(3)
        x = np.zeros((5400, 3))
        for t in range(len(r)):
            x[t] = state
            r[t] = .7 * np.prod(np.tanh(state)) + rng.normal()
            state = poles * state + gain * r[t]
        features = {'linear': x,
                    'raw_cubic': np.column_stack([x, np.prod(x, axis=1)]),
                    'bounded_cubic': np.column_stack([x, np.prod(np.tanh(x), axis=1)]),
                    'candidate_axis': np.column_stack([x, x[:, 0] * np.tanh(x[:, 1] * x[:, 2]), x[:, 1] * np.tanh(x[:, 0] * x[:, 2]), x[:, 2] * np.tanh(x[:, 0] * x[:, 1])])}
        train = np.arange(400, 880)
        tune = np.arange(900, 1300)
        test = np.arange(1400, 5400)
        for name, f in features.items():
            scales = np.maximum(np.sqrt(np.mean(f[train] ** 2, axis=0)), 1e-8)
            f = f / scales
            candidates = []
            for alpha in [.001, .01, .1, 1.]:
                b = np.linalg.solve(f[train].T @ f[train] / len(train) + alpha * np.eye(f.shape[1]), f[train].T @ r[train] / len(train))
                candidates.append((float(np.mean((f[tune] @ b - r[tune]) ** 2)), alpha, b))
            _, alpha, b = min(candidates, key=lambda a: (a[0], a[1]))
            rows.append({'seed': seed, 'family': name, 'alpha': alpha, 'ratio': float(np.mean((f[test] @ b - r[test]) ** 2) / np.mean(r[test] ** 2))})
    return {'scope': 'Generator-informed poles and feature axes; diagnostic, not innovation or market validation. Reused development seeds.',
            'design': 'Fixed four feature families, four penalties, 480 training, 400 tuning, 4000 evaluation; 20 seeds; one-step.',
            'means': {name: float(np.mean([r['ratio'] for r in rows if r['family'] == name])) for name in features}, 'rows': rows}


if __name__ == '__main__':
    result = run()
    Path(__file__).with_name('FEATURE_SCREEN_RESULTS.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))
