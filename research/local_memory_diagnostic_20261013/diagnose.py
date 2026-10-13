"""Fresh controlled diagnostic; no market data or submitted code execution."""
import json
from pathlib import Path
import numpy as np


def run():
    rows = []
    poles = np.exp(-np.log(2) / np.array([1., 8., 32.]))
    gain = np.sqrt(1 - poles ** 2)
    for seed in range(400, 420):
        rng = np.random.default_rng(seed)
        returns = np.zeros(5400)
        states = np.zeros((5400, 3))
        state = np.zeros(3)
        means = np.zeros(5400)
        for t in range(5400):
            states[t] = state
            means[t] = .7 * np.prod(np.tanh(state))
            returns[t] = means[t] + rng.normal()
            state = poles * state + gain * returns[t]
        # Truncate the exact causal convolution to 60 past increments.
        approx = np.zeros_like(states)
        for lag in range(1, 61):
            approx[lag:] += returns[:-lag, None] * gain * poles ** (lag - 1)
        truncated_mean = .7 * np.prod(np.tanh(approx), axis=1)
        ix = np.arange(1400, 5400)
        denominator = np.mean(returns[ix] ** 2)
        full_mse = np.mean((returns[ix] - means[ix]) ** 2)
        short_mse = np.mean((returns[ix] - truncated_mean[ix]) ** 2)
        rows.append({'seed': seed, 'rw_mse': float(denominator),
                     'full_oracle_ratio': float(full_mse / denominator),
                     'truncated_oracle_ratio': float(short_mse / denominator),
                     'conditional_mean_discrepancy': float(np.mean((means[ix] - truncated_mean[ix]) ** 2)),
                     'signal_variance': float(np.mean(means[ix] ** 2))})
    signal = np.mean([r['signal_variance'] for r in rows])
    discrepancy = np.mean([r['conditional_mean_discrepancy'] for r in rows])
    return {'scope': 'Known generating parameters, diagnostic only; no deployable learner, no market scoring.',
            'design': 'Stage G formula independently implemented; 20 fixed seeds, burn 1400, evaluation 4000; one-step only.',
            'full_oracle_ratio': float(np.mean([r['full_oracle_ratio'] for r in rows])),
            'truncated_oracle_ratio': float(np.mean([r['truncated_oracle_ratio'] for r in rows])),
            'mean_discrepancy_relative_signal': float(discrepancy / signal),
            'warning': 'Truncated plug-in is not the optimal conditional predictor given 60 increments; this is not an information-theoretic lower bound.',
            'rows': rows}


if __name__ == '__main__':
    result = run()
    output = Path(__file__).with_name('RESULTS.json')
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))
