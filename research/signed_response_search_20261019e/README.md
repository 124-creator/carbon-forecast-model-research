# Signed conditional responses for carbon-price forecasting — stage e

**Research outcome:** the model learns and improves upon a random walk, but the incremental conditional-duration mechanism is not supported by matched controls. A single max80→max320 sensitivity does not change that conclusion. This is a completed research experiment, not a positive-results paper or a general SOTA claim.

Read [the full Chinese report](REPORT_ZH.md), [reproduction instructions](REPRODUCE_ZH.md), and [continuation handoff](HANDOFF_CONTINUATION_ZH.md). Machine-readable conclusions are in [FINAL_DECISION.json](FINAL_DECISION.json).

GitHub delivery: [Draft PR26](https://github.com/124-creator/carbon-forecast-model-research/pull/26), stacked on the stage-d branch. The complete results content was pushed as [d999f26](https://github.com/124-creator/carbon-forecast-model-research/commit/d999f26db91d8b581b14aa42292d7db4782afaf4) and read back with an exact local/remote tree match. [Delivery receipt](provenance/GITHUB_DELIVERY_RECEIPT.json) records the immutable content identity; later metadata-only commits can advance the live PR head.

## Completed experiments

| Scope | Fits | Positive optimizer updates | Market–model prediction sets |
|---|---:|---:|---:|
| Original frozen evaluation | 244 | 11,885 | 24 |
| Post-exposure max320 sensitivity | 170 | 32,683 | 10 |
| Total | 414 | 44,568 | 34 |

The two markets contain 196 and 231 distinct evaluation origins. Fit counts and seeds do not create additional independent market samples. Original CEA scoring was new within the project evidence; every later CEA sensitivity is exposed development.

## Results at a glance

Q is the equal mean across H1/H5/H20 of model MSE divided by random-walk MSE; lower is better.

| Model | Tianjin original development | CEA original evaluation |
|---|---:|---:|
| Primary conditional duration | 0.920781 | 0.878516 |
| Matched conditional amplitude | 0.915517 | 0.876827 |
| Learned static duration | 0.903971 | 0.876992 |
| Price Ridge | 0.902011 | 0.851062 |
| SWEWT–GRU adaptation | 0.990924 | 0.808418 |

The primary model's original RW improvement has block-bootstrap interval support. Several matched controls have better point estimates; CEA rankings against Ridge and neural references have intervals crossing zero. All 12 arms, both markets, both prespecified block lengths, and all supplemental arms are retained in the report.

## Evidence index

- `source/`: four frozen model/runner files, result export, figures, and a replication launcher.
- `methods/`: exact mathematical definitions and learning attribution.
- `literature/`: eleven direct neighbors with read-access limits, and one bounded next-candidate proposal.
- `provenance/`: dataset identities, extraction, exposure history, exact execution/export counts, and local/remote freeze mappings.
- `results/core/`: complete original numerical selection/refit records and summaries.
- `results/neural_references/`: complete GRU selection/refit records and summaries.
- `results/budget320/`: complete separate development sensitivity.
- `audits/`: fixed statistical plan, independent reconstruction, paired intervals, budget checks, and export scope.
- `figures/`: standalone PNG and PDF scientific figures made only from aggregate results.

Self-trained numerical parameter states are retained. Third-party raw rows, per-origin NPZ arrays, and binary checkpoints are excluded from the public tree; fixed-source reconstruction scripts and hashes are provided. Stage d remains byte-for-byte unchanged.

Local design freeze and subsequent GitHub content mirroring have separate identities and timestamps. Neither this repository nor the later upload is presented as an external preregistration before fitting.
