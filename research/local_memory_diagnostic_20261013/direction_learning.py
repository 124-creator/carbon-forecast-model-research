"""Bounded direction-learning diagnostic; not a market experiment."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares


def predict(theta, x):
    matrix = theta[:9].reshape(3, 3)
    return x @ theta[9:12] + theta[12] * np.prod(np.tanh(x @ matrix.T), axis=1)


def run():
    poles = np.exp(-np.log(2) / np.array([1., 8., 32.]))
    gain = np.sqrt(1 - poles ** 2)
    rows = []
    for seed in range(400, 410):
        rng = np.random.default_rng(seed)
        r = np.zeros(5400)
        x = np.zeros((5400, 3))
        state = np.zeros(3)
        for t in range(len(r)):
            x[t] = state
            r[t] = .7 * np.prod(np.tanh(state)) + rng.normal()
            state = poles * state + gain * r[t]
        tr, va, te = np.arange(400, 880), np.arange(900, 1300), np.arange(1400, 5400)
        for family in ['random_directions', 'known_axis_initialization']:
            for model_seed in [7, 17, 29]:
                initial = np.zeros(13)
                initial[:9] = (np.eye(3) if family == 'known_axis_initialization' else np.random.default_rng(model_seed).normal(size=(3, 3)) / np.sqrt(3)).ravel()
                initial[12] = .1
                choices = []
                for penalty in [.001, .01, .1]:
                    def residual(theta):
                        return np.r_[(predict(theta, x[tr]) - r[tr]) / np.sqrt(len(tr)), np.sqrt(penalty) * theta]
                    fit = least_squares(residual, initial, bounds=(-3., 3.), max_nfev=100, ftol=1e-6, xtol=1e-6, gtol=1e-6)
                    choices.append((float(np.mean((predict(fit.x, x[va]) - r[va]) ** 2)), penalty, fit))
                _, penalty, fit = min(choices, key=lambda v: (v[0], v[1]))
                rows.append({'seed': seed, 'model_seed': model_seed, 'family': family, 'penalty': penalty, 'solver_success': bool(fit.success), 'nfev': int(fit.nfev), 'ratio': float(np.mean((predict(fit.x, x[te]) - r[te]) ** 2) / np.mean(r[te] ** 2))})
    return {'scope': 'Known generator poles and interaction family; developmental diagnostic, not new architecture or independent validation. Known-axis starts have extra structural knowledge.',
            'budget': '10 data seeds, 3 model seeds, 3 penalties, 100 function evaluations per fit, same data splits as preceding diagnostics, one-step.',
            'means': {f: float(np.mean([v['ratio'] for v in rows if v['family'] == f])) for f in ['random_directions', 'known_axis_initialization']},
            'non_success_selected': sum(not v['solver_success'] for v in rows), 'rows': rows}


if __name__ == '__main__':
    result = run()
    Path(__file__).with_name('DIRECTION_RESULTS.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))
