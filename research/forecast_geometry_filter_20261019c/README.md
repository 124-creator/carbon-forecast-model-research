# Forecast geometry filter — completed synthetic qualification

**Decision: STOP_FIXED_GEOMETRY_CANDIDATE.** The interrupted CRPS+KL numerical failure has been repaired without changing the scientific protocol, tolerance, or solver budgets. The frozen four-scenario experiment and independent replays are complete.

The future-CDF candidate's expected CRPS is **+0.082077%** relative to the validation-selected competitor (positive is worse; paired 20-seed nominal 95% interval **[+0.022619%, +0.142870%]**). It does not outperform the future-variance or tempered-Bayes controls. The original 1% total-CRPS gate also has a disclosed feasibility flaw: on these fixed histories, even the true oracle can improve only **0.559361%**. That gate remains unchanged, and the stopping decision has the separate adverse-comparison evidence above.

This is a completed finite synthetic method qualification, not a successful carbon-price model, market validation, or novelty certification. Actual neural and emission/transition parameter fits: **0**.

## Read first

- [中文完整报告](REPORT_ZH.md)
- [Reproduction instructions](REPRODUCE_ZH.md) and [pinned dependencies](requirements.txt)
- [Continuation / context record](HANDOFF_CONTINUATION_ZH.md)
- [Machine-readable final decision](results/FINAL_DECISION.json)
- [Original formal summary](results/synthetic/SUMMARY.json)
- [Final execution audit](audits/FINAL_EXECUTION_AUDIT.json)

## Evidence and implementation

- [Mathematical derivation](methods/GEOMETRY_DERIVATION_ZH.md)
- [Unchanged protocol](provenance/SYNTHETIC_PROTOCOL.json) and [pre-resume source freeze](provenance/SOURCE_FREEZE.json)
- [Production core](source/geometry_filter.py) and [runner](source/run_synthetic.py)
- [All 5,600 validation configurations](results/synthetic/TRIAL_VALIDATION.csv)
- [All 1,040 selected/baseline/oracle result rows](results/synthetic/SELECTED_RESULTS.csv)
- [All 130 paired comparisons](results/synthetic/PAIRED_COMPARISONS.csv)
- [Full trace hashes](results/synthetic/COMPLETE.json)
- [Original failed attempt](attempts/01_kl_dual_stagnation/FAILURE_RECEIPT.json)
- [Solver repair](audits/SOLVER_REPAIR_REPORT_ZH.md)
- [Independent test replay](audits/INDEPENDENT_TRACE_REPLAY.json)
- [Full validation-prefix replay](audits/INDEPENDENT_PREFIX_REPLAY_ZH.md)
- [Post-result oracle-headroom diagnostic](results/ORACLE_HEADROOM_POSTHOC.json)
- [Closest-literature review](literature/RECOVERY_LITERATURE_AUDIT_ZH.md)

## Scope

There are 20 independent seeds, 80 scenario–seed sequences, 5,600 formal validation filter configurations, and 1,040 evaluated result streams. Verification replays are not extra scientific samples. Four complete generated-data NPZ traces are supplied separately in the synthetic reproduction archive and can also be regenerated from this repository. No raw market data, new market fit, or 2024+ confirmation evaluation is part of this study.

Historical local freeze commits and later GitHub delivery commits have different parent objects and timestamps; matching file trees establish content equivalence. The local freezes are not external preregistration.

