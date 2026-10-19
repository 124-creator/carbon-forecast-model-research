# Conditional persistence: completed model experiments and GAS-Norm benchmark

**Decision: `STOP_FIXED_CANDIDATE`. This round does not establish a publishable positive model contribution.**

This separately frozen follow-up studies whether history-conditioned volatility-state
persistence improves multi-horizon carbon-return distributions beyond a fitted constant
persistence and an equal-capacity scale control. It is model / learning-method research.
The complete previous study is preserved.

## Main results

Relative CRPS is candidate / control minus one, computed per market-year / horizon cell
after averaging three seed scores, then averaged equally across cells. Lower is better.

| Prespecified contrast | Relative CRPS | Nominal 95% paired block interval | Strictly better cells |
|---|---:|---:|---:|
| GRU conditional / constant persistence, development | +0.183892% | [+0.004735%, +0.469536%] | 9 / 22 |
| GRU conditional / equal-capacity scale control | +0.258964% | [+0.098944%, +0.502603%] | 6 / 22 |
| GRU path auxiliary / endpoint-only loss | +0.352520% | [+0.029586%, +0.936724%] | 10 / 22 |
| MLP conditional / constant persistence | −0.011817% | [−0.084860%, +0.092557%] | 12 / 22 |
| GRU conditional / constant, additional Shenzhen years | +0.006976% | [−0.034884%, +0.055109%] | 3 / 4 |

Only one of six continuation conditions passes. No reserved 2024+ market prices were
opened, and no model, seed, loss weight or test year was changed to rescue the candidate.
The synthetic identification screen passed before the market experiment, but the real
market mechanism controls did not support the proposed advantage.

The candidate is descriptively better than this round's GAS-Norm + GRU-GMM adaptation
(−3.074706% CRPS on the same 22 development cells). Its simpler mechanism controls are
better still. Cross-family ranking therefore does not establish an effect of conditional
persistence. GAS-Norm is an independently implemented paper-method adaptation, **not** a
reproduction of the original paper's full benchmark or a new normalization invention.

## Completed execution

| Scope | Actual work |
|---|---|
| Synthetic mechanism screen | 720 low-dimensional model fits; 1,080 scalar optimizations; 20 seeds × 3 sizes × 3 scenarios × 4 arms |
| Candidate and ablations | 312 neural fits; 39 statistical fits; 13 market-year panels; GRU / MLP; seeds 17, 41, 89 |
| GAS / EWMA / Global benchmark | 198 neural fits; 22 GAS statistical fits; 154 fixed forecasts |
| Total formal market neural training | 510 fits; 20,400 completed epochs; 387,360 optimizer updates |
| Saved market distributions | 338 candidate artifacts and 352 GAS / fixed-control artifacts |

All neural fits completed 40 epochs, even when validation selected epoch 0.
Synthetic and implementation-smoke computations are counted separately from market fits.
Three GAS fits reached the fixed evaluation budget with finite valid parameters; their
actual optimizer status is retained, without retries or hidden budget extensions.

## Read the deliverables

- [Complete Chinese research report](REPORT_ZH.md)
- [Fourteen closest papers, source depth, adaptation boundaries, and direction decisions](literature/CLOSEST_PAPERS_ZH.md)
- [Model, mathematics, and limitations](methods/METHOD_ZH.md)
- [Exact reproduction commands and private input mapping](REPRODUCE_ZH.md)
- [Candidate results and continuation gate](results/market/persistence/SUMMARY.json)
- [GAS benchmark results](results/gas_benchmark/SUMMARY.json)
- [Contextual cross-family results and synchronized-date sensitivity](results/cross_benchmark/SUMMARY.json)
- [Synthetic screen](results/synthetic/SUMMARY.json)
- [Independent checkpoint replay](audits/INDEPENDENT_CHECKPOINT_REPLAY_ZH.md)
- [Independent candidate aggregation replay](audits/PERSISTENCE_AGGREGATE_REPLAY.json)
- [Independent GAS aggregation replay](audits/GAS_AGGREGATION_REPLAY.json)
- [Complete score replay with price-CRPS numerical qualification](audits/MARKET_SCORE_REPLAY_QUALIFIED.json)
- [Stable-expression, 80/120-digit and defining-integral price audit](audits/LOGNORMAL_CRPS_CANCELLATION_AUDIT.json)
- [Stable price-CRPS numerical companion: 934 cell rows](audits/STABLE_PRICE_CRPS_CELLS.csv)
- [Stable price recomputation and unchanged-input receipt](audits/STABLE_PRICE_CRPS_RECOMPUTATION.json)

![Matched comparisons](figures/03_matched_comparisons.png)

The intervals condition on these fitted models and annual panels. They are not
population-of-markets intervals, are not multiplicity-adjusted, and do not establish
external validation. The synchronized-date analysis is a secondary robustness check
specified during training and before reading candidate comparisons.

## Protocol and provenance

- [Initial screen and benchmark protocol](provenance/PROTOCOL.json)
- [Frozen market follow-up protocol](provenance/MARKET_PROTOCOL.json)
- [GAS implementation choices fixed before fitting](provenance/GAS_IMPLEMENTATION_PRE_FIT.json)
- [Secondary comparison specification](provenance/SECONDARY_COMPARISON_SPEC.json)
- [Environment](provenance/ENVIRONMENT.json) and [pinned dependencies](requirements.txt)

Protocols and implementation snapshots were committed locally before their relevant
fits or comparisons. They were uploaded after completion; this is an internal freeze,
**not externally timestamped preregistration**. Shenzhen 2020/2021 were first used for
model training/scoring in this study after the synthetic gate. The source workbook had
been mechanically examined in prior work, so it is not described as a never-seen market.

Public Git contains code, aggregate results, fit receipts, mathematical / implementation
audits and figures. Raw input workbooks, prepared arrays, selected checkpoints and
per-origin forecasts are retained in a separate private run archive. Read the report's
data-quality and numerical limitations before interpreting secondary price metrics.

The full independent score reconstruction covers 421,600 origin/horizon distributions
and 4,187,068 metric values. Normalized CRPS, NLL, analytic price mean/MSE, coverage and
width pass the original numerical tolerances. Original stored price CRPS has cancellation
error above that tolerance for 331 Shenzhen h20 scores (validation and test combined).
These values are explicitly qualified; they are not silently accepted by widening the
tolerance. A separate stable-price companion preserves corrected arithmetic alongside
the unchanged original forecasts. No development-panel primary decision changes.
