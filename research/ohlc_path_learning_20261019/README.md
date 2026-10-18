# Reported EUA price–range learning: completed development experiment

**Decision: stop this candidate.** The joint summary-state model has mean cell MSE/RW **1.020229**, with paired 95% CI **[0.987905, 1.094544]**. It does not meet the predefined price-forecasting gates. All models, ablations, seeds, negative results, and data limitations are retained.

This study follows [the completed auction-clearing study](../auction_clearing_innovation_20261018/REPORT_ZH.md). It is a bounded empirical test, not a claim of a new general forecasting theory or a successful publication-ready model. The two author datasets concern EUA, not two independent markets.

## Read the evidence

- [完整中文实验报告](REPORT_ZH.md): results, interpretation, literature, and direction verdict.
- [Frozen protocol](PROTOCOL_ZH.md) and [pre-fit file hashes](FROZEN_PROTOCOL.json).
- [Exact method](METHOD_ZH.md), [recent carbon literature](LITERATURE_ZH.md), and [direct mathematical neighbors](DIRECT_NEIGHBORS_ZH.md).
- [Data provenance and semantic audit](provenance/DATA_AUDIT_ZH.md).
- [Independent audit](AUDIT_ZH.md), [checkpoint replay](audits/REPLAY_AUDIT.json), and [scorer repair log](audits/SCORER_AUDIT_ZH.md).
- [Final scores and gates](results/summary.json), [all cell scores](results/mse_by_cell.csv), [paired comparisons](results/comparisons.csv), and [fit receipts](results/TRAINING_RECEIPTS.json).
- [Publication route and fee evidence](PUBLICATION_ROUTE_ZH.md).

![All-model results and paired comparisons](figures/performance_overview.png)

## What was actually run

Two separately analyzed author sources: 1,805 undated reconstructed Quantum records and 772 dated Ding records admitted before 2024. Six expanding chronological evaluation folds give 1,151 common origins and 3,453 correlated horizon targets. Seven neural families use three final seeds and identical selection budgets, alongside five classical/reference families.

The runner completed 84 neural selection fits and 126 final model constructions. Of the final models, 63 received gradient updates and 63 selected epoch zero; hence **147 neural fits actually took gradient steps**, totaling 16,986 optimizer steps. There were **114 classical regressor fits**. The 16 synthetic preflight calls are recorded separately.

All 126 checkpoints were replayed at three fixed test origins. Independent scoring reconstructed labels, causal scales, common origins, and paired confidence intervals from 156 local forecast files. No test-driven model rescue followed the negative result.

## Reproduce from source files

Python 3.12 is the recorded runtime. Use an isolated environment and the pinned requirements:

```bash
python -m venv .venv
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
.venv/bin/python provenance/prepare_eua_sources.py --output-dir /tmp/carbon-path-local
.venv/bin/python run.py --data-dir /tmp/carbon-path-local --out /tmp/carbon-path-run --workers 3
.venv/bin/python score_independent.py --data-dir /tmp/carbon-path-local --run-root /tmp/carbon-path-run
.venv/bin/python audits/replay_checkpoints.py
.venv/bin/python reporting/make_figures.py
```

Run these commands from this experiment directory. The optional CEA reconstruction is an audit appendix and is not required for the model experiment. Acquisition scripts pin public author commits and byte hashes; source availability and permitted use remain prerequisites. The offline option accepts only matching locally obtained source files.

For a fresh run use a new directory outside a Git worktree. The runner refuses to resume under changed source/code hashes and validates existing checkpoints and forecasts. CPU implementation details can affect final low-order floating-point bits on another platform; the audit publishes explicit numerical tolerances rather than promising identical cross-platform binary serialization.

The public release contains code, source URLs/hashes, protocols, aggregate metrics, fit histories, and scientific figures. Raw/processed price sequences, per-origin forecasts, and model weights are kept out of the public repository. Running the pinned acquisition and experiment code reconstructs the local evidence.

## Important interpretation boundaries

Quantum has no authenticated daily dates and 1,025 reported Price values outside its reported high–low range. Ding has two such observations. Prices are not repaired or reinterpreted as certified traded closes. Only high≥low and positive prices are imposed; prediction horizons count source records.

The local protocol snapshot was committed before any market fitting; it is uploaded after completion in accordance with the task sequence. This is not external independent preregistration. Scorer-only numerical/infrastructure repairs are separately logged; model, features, split, predictions, hyperparameter budget, bootstrap design, and success thresholds were unchanged after the freeze.
