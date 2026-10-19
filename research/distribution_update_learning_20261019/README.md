# Learning distribution updates with budgeted path controls

**Completed model-method development experiment, 2026-10-19.** The fixed candidate does not pass the cross-cell advancement rule. This directory is a reproducible research record, not a claim of an established novel or state-of-the-art carbon-price model.

## Main result

The primary method changes a previously fitted probabilistic forecast through a three-dimensional history-dependent CDF mixture gate. It estimates the exact quadratic CRPS improvement using a predictably capped reflected path control and an explicit sign-imbalance bias budget.

Across four distinct markets, 22 market/year/horizon cells, two ordinary neural backbones and three seeds:

| Quantity | Result |
|---|---:|
| Primary normalized CRPS / frozen base | 0.9995610784 |
| Same-joint-budget raw update / frozen base | 1.0000196388 |
| Primary relative change vs matched raw update | −0.045856% |
| Paired within-fixed-cell block interval | [−0.058995%, −0.034982%] |
| Cells improved / tied / worse | **1 / 21 / 0** |
| Neural gradient fits / statistical optimizer fits | 132 / 44 |
| Selected epoch-zero proposals | 46 of 132 |
| Fits whose primary gate differs from matched raw | 6 of 132, all in Guangzhou 2021 H20 |

The interval resamples within fixed development cells; it does not establish replication across independent markets. All six changed fits belong to a single cell. The online BOA-CDF comparator also has a better overall point score than the primary method. The executable decision is **`STOP_FIXED_CANDIDATE_NO_CONFIRMATION_OPENED`**. Newly reserved prices remain unopened.

## Read the research

- [Complete Chinese report](REPORT_ZH.md): motivation, recent literature, experiment design, full results and paper-readiness decision.
- [Precise method and proofs](methods/METHOD_ZH.md), with [English method note](methods/METHOD_EN.md).
- [Original-source literature screening](methods/LITERATURE_SCREEN_ZH.md) and [online-baseline implementation boundaries](methods/ONLINE_BASELINES_ZH.md).
- [Earlier candidate mathematics and counterexamples](methods/PRIOR_CANDIDATE_SCREEN_ZH.md), retained as a historical screening record.
- [Frozen protocol](provenance/PROTOCOL.json) and [documented pre-run cap refinement](provenance/DESIGN_ADDENDUM_PRE_RUN.json).
- [Machine-readable market summary](results/market/SUMMARY.json), [all cell metrics](results/market/CELL_METRICS.csv), and [general synthetic study](results/synthetic/GENERAL_SYNTHETIC.json).
- [Exact synthetic audit](audits/INDEPENDENT_EXACT_SYNTHETIC_ZH.md), [independent score/operator audit](audits/INDEPENDENT_CORE_CHECK_ZH.md), and the independent time/data and checkpoint-replay audits under `audits/`.

The proposal is a learning rule. The MLP, GRU, Gaussian mixtures, CRPS, control variates, forecast aggregation and empirical Bernstein inequality are existing components. Originality of the narrow combined operator is **not established** by the present source search.

## Reproduce

Run from this directory after checking out this branch. The data-acquisition script uses only the Python standard library and the repository's hash-verified older development CSVs. It downloads the fixed author archive, reads price values only after the date guard, and emits development data outside the repository. It does not open prices from 2024 onward.

```bash
python -m venv /tmp/carbon-update-venv
/tmp/carbon-update-venv/bin/pip install -r requirements.txt
/tmp/carbon-update-venv/bin/python data_sources/acquire_development.py --download --output-dir /tmp/carbon-update-data
```

CPU execution uses three workers and one numerical thread per worker. The recorded environment is Python 3.12.14 with PyTorch 2.14.1+cpu; use an appropriate CPU wheel for your platform. Choose fresh local output paths to preserve any preceding run.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /tmp/carbon-update-venv/bin/python source/run_synthetic.py --output results/synthetic
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/carbon-update-venv/bin/python source/run_market.py --data-dir /tmp/carbon-update-data/development --local-dir /tmp/carbon-update-run --public-dir results/market --workers 3
MPLCONFIGDIR=/tmp/carbon-update-mpl OPENBLAS_NUM_THREADS=1 /tmp/carbon-update-venv/bin/python source/analyze_results.py --local-dir /tmp/carbon-update-run --root .
MPLCONFIGDIR=/tmp/carbon-update-mpl /tmp/carbon-update-venv/bin/python source/plot_cell_gains.py --root .
```

The independent audit scripts expose `--help` for their exact artifact paths. They do not retrain the 132 networks when replaying checkpoints. Audit tolerances and the floating-point-zero correction are recorded explicitly.

Public files contain code, acquisition instructions, hashes, aggregate scores, validation-loss histories, method notes and audits. Newly acquired raw prices, row-level targets/forecasts and neural checkpoints remain outside this directory. The source archive's CC BY statement does not by itself clear third-party Wind/Investing redistribution rights. This restriction affects the public artifact bundle, not the completeness of the locally executed experiment.

## Interpretation boundaries

- Primary scores concern origin-normalized cumulative **log increments**; this is a prospectively changed probability-forecasting objective. Price CRPS and price-mean MSE are secondary. Lognormal means are computed analytically and need not equal the current price.
- Horizons are 5 and 20 **author quote observations**, not necessarily exchange trading days. No calendar forward/backward filling is used.
- EUA_LONG and EU_REPORTED overlap exactly throughout 2018–2022, so EU_REPORTED is excluded as an independent market. Shanghai 2021 is excluded prospectively because it has only 47 observations and a long gap.
- The exact guarantee requires an independent calibration sample and a stated conditional sign-given-magnitude bound. The market moving-block bootstrap is a fixed approximate adaptation; arbitrary nonstationary future safety is not claimed.
- `GARCH_Normal` is a Gaussian approximation using the forecast cumulative variance, not a full simulated multi-step GARCH predictive density. `quantile_BOAG_online` is a 19-atom unsmoothed task adaptation, not the whole profoc workflow.
- The frozen `fit_garch` helper has a documented unused-index limitation: `next_variance[0]` is not rescaled to return-variance units. This study only reads origins 39 and later; every used value passed independent reconstruction. Correct that entry before reusing it at origin zero.
- The protocol was committed locally before market fitting and uploaded after completion. The analytic cap refinement was recorded before the general synthetic and market runs. This is not external preregistration. A later date-guard-only repair and its lack of numerical effect are documented.

The previous OHLC experiment remains separate at `../ohlc_path_learning_20261019/`; no earlier research files are replaced by this study.
