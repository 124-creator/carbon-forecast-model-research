# Literature-aligned carbon forecasting model search

Date: 2026-10-19. **All five planned model batches and independent result audits are complete.**

**Decision:** useful low-dimensional forecast structures were found, but the added learned gate and volume mechanism do not yet have sufficient incremental evidence for the requested innovation paper. No positive paper claim is drafted from these results.

## Start here

- [完整中文实验报告](REPORT_ZH.md): published practice, information scope, model mathematics, actual results, limitations and paper-stage decision.
- [复现命令](REPRODUCE_ZH.md): sources, runtime, all five runners and independent audits.
- [续接上下文](HANDOFF_CONTINUATION_ZH.md): user intent, closed prior attempts, decisions, run paths and exposure state.
- [机器可读裁决](FINAL_DECISION.json) and [execution totals](provenance/EXECUTION_TOTALS.json).
- [Literature audit](literature/PUBLISHED_PRACTICE_AUDIT_ZH.md): seven substantive method texts plus one separately identified author-code review.

## What was actually run

| Batch | Evaluation scope | Neural selection/refit constructions | With positive gradient steps | Main result |
|---|---|---:|---:|---|
| V1 | Four existing development series, 2022 and 2023 | 240 | 198 | Sequential gate 1.001207 MSE/RW; terminal gate 0.975430 |
| V2 | Guangzhou/Hubei with aligned volume, same two years | 120 | 84 | Volume allocation 0.974008; gains concentrated in Hubei 2023 |
| V3 | ScienceDB Beijing/Tianjin; Chongqing separate stress test | 120 | 108 | Primary volume model 0.810168 BJ/TJ mean; prespecified secondary terminal gate 0.790512 |
| SWEWT supplement | Same V3 data and targets | 30 | 21 | Price/full-input adaptations 0.809787/0.801339 BJ/TJ mean |
| V4 | Official Tianjin 2023–2026 tail, 196 common origins | 50 | 29 | Terminal gate and constant control both 0.959915; price Ridge 0.901624 |

The 560 neural tasks comprise 224 selection tasks and 336 final constructions. Of the final constructions, 216 have positive gradient steps and 120 select epoch 0. There are 528 classical estimator fits in these five batches, plus 84 in the separate information-scope diagnostic. A task or seed is not an extra independent market sample.

## Principal finding

The terminal gate was discovered as a useful control after V1 results, kept as a predeclared secondary model in V3, and fixed as the primary candidate before V4 scores. Those identities have not been rewritten.

In V4 it improves over RW by 4.0085 RW percentage points, with the prespecified family-adjusted interval above zero. But all six structural families select their own initial checkpoints; terminal gate and constant allocation predictions are exactly equal. Price Ridge, chosen using the earlier validation period, is better than both. Thus this later-period result supports a fixed predictive prior, not a new learned-gate effect.

The author-cleaned ScienceDB benchmark is used as published. The newer official Tianjin series retains all 1,077 reported daily-total records and their rounded mean prices. All horizons are subsequent observed records, not fixed calendar-day horizons.

## Files and provenance

- `source/`: the eight frozen model/helper sources, read-only aggregate exporter and figure generator.
- `methods/`: designs, equations, nearest-prior boundaries and post-result learning diagnostics.
- `audits/`: numerical checks, fixed analysis plans and independent paired block uncertainty calculations.
- `results/`: every family summary, horizon/native metrics, selections, full compressed fit traces, and checkpoint/prediction hashes.
- `data/sciencedb_819_v1/`: the CC BY 4.0 original dataset, parsed tables and attribution.
- `provenance/`: download/parse scripts, source identities, exposure and local-to-remote freeze records.
- `figures/`: static PNG and PDF research figures, each generated from the complete result tables.

Raw files without an established redistribution license, per-origin price/prediction arrays from those sources and binary checkpoints remain in the recorded local run directories. Public diagnostics may contain explicitly authorized numeric parameters trained by this project. The distinction and licensed-data exception are checked by the export-scope audit.

Local code commits preceded their own fits. GitHub content mirrors were uploaded later and are **not external preregistration**. Prior studies outside this new directory are unchanged.

