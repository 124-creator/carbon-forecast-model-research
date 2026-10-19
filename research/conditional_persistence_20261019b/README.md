# Conditional persistence screen and GAS-Norm carbon benchmark

This is a new, separately frozen follow-up to `distribution_update_learning_20261019`.
The previous study is not changed. The objective is model / learning-method research,
not causal economic inference. No success or publication readiness is assumed.

The first experiment identifies a narrow structural degree of freedom: a conditional
reset persistence changes multi-step tails while keeping the same one-step marginal
and cumulative variance. The Markov model and scoring rules are established methods.
The proposed restriction, exact lattice propagation, and path-assisted estimation
must earn a useful result against a fitted constant-persistence control.

The second experiment independently implements the published GAS-Norm Student-t
filter, then adapts it to the same carbon return distribution task as EWMA and global
normalization with a matched GRU16 / three-Gaussian-mixture output. It is explicitly
an implementation of the paper's filter plus a task adaptation, not reproduction of
the original paper's complete experiments or a claim to have invented GAS-Norm.

## Frozen rules

- [Protocol](provenance/PROTOCOL.json): synthetic resource gate, existing-data benchmark,
  seeds, fit budget, maturity rules, and disclosure limits.
- [Exact propagation](source/persistence.py) and [screen runner](source/run_synthetic.py).
- [Independent model audit](audits/INDEPENDENT_PERSISTENCE_AUDIT.json).
- [GAS filter](source/gas_filter.py): includes the explicit paper/code differences.

New Shenzhen prices and all reserved 2024+ prices remain unopened until the specific
conditions in the protocol are satisfied. A synthetic pass is not evidence of
market gains. A failure stops the persistence candidate before new-market fitting.

## Reproduction

Python 3.12, NumPy, SciPy, pandas, PyTorch, and Matplotlib are used. Private inputs,
model checkpoints, and per-origin market forecasts are stored outside public Git.

```bash
python research/conditional_persistence_20261019b/source/run_synthetic.py \
  --local-dir /tmp/carbon-persistence-screen-runs \
  --public-dir research/conditional_persistence_20261019b/results/synthetic \
  --workers 3
```

The complete report and verified execution receipts will be added after the runs.
