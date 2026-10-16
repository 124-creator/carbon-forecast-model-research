# Carbon forecasting model research

Latest local research result (2026-10-17): `research/recurrent_response_20261014/horizon_scale_competition_20261016/REPORT_ZH.md`. A separately declared 59-fit supplement completes the interrupted 192-fit development grid without rewriting the original incomplete record. The horizon-scale candidate does not beat full GRU/LSTM controls and is not promoted. Normal/optimized independent scoring and representative fresh-fit checks are included; per-origin forecasts and parameters remain local. This entry supersedes earlier no-training statements below for current model-development status, not earlier data/source limitations.

Current data-first entry: `research/pro_data_packet_20261013/START_HERE_ZH.md`. Three licensed author workbooks and three offline-ready CSVs (4535 quotes total) are now in `data/raw` and `data/parsed/figshare_28682348`. Guangzhou's 1492 date/price pairs match held official GDEA endPrice values exactly; worksheet-name contradiction remains disclosed. Hubei/EU source limits and 13 scoped bibliographic records are explicit. No model was trained or scored in this acquisition phase. Current file-hash authority is `metadata/PRO_PACKET_PUBLICATION_MANIFEST.json`; entry statements below refer to earlier stages.

Current evidence: `research/stageC_20261013_review/INDEPENDENT_REVIEW_ZH.md`. A submitted failed experiment is archived with explicit publication exclusions; 81 tuning selections and 16 aggregate check rows were independently recomputed, not retrained locally. A newly acquired 4861-row author EUA snapshot remains local-only pending redistribution and instrument-definition checks. The earlier initialization statements below are historical; no successful new model or novelty is certified.

Latest follow-up: `reports/20261012_followup/ANALYSIS_AND_DATA_DECISION_ZH.md`, with two additional derived CSVs (ascending author prices and origin/maturity metadata), source-definition evidence and date-axis feasibility counts. No additional market observations or forecast scores were created. A local-only GDEA candidate exists but is excluded from publication. See `metadata/followup_manifest.json` for the supplement file hashes.

Current reader entry: `START_HERE_PRO_ZH.md`. The expanded package now contains 5 original workbooks and 6 freshly parsed CSV series, including one author-level carbon-price series and five potential external-variable series. It does not contain official exchange daily-price panels or new market forecasting results. See `metadata/published_data_manifest.json` for exact source grades, hashes, attributions and limitations. Earlier no-data statements below describe initialization, not the current expanded package.

Private independent research project. This is not a continuation of the submitted auction-association paper. The preceding manuscript is frozen at 124-creator/-123 release emec-submitted-v10.1-20261012; subsequent editorial requests are handled separately.

## Objective
Develop a defensible forecasting-method contribution and improve real out-of-time predictive performance. Prefer subscription/non-OA journal publication. No promised acceptance, invented improvement or fixed impressive R-squared target.

## Initial candidate, not a novelty claim
A small data-efficient forecaster with a random-walk anchor and learned multi-horizon residual corrections, with horizon-conditioned adaptation and carefully bounded forgetting using only matured historical forecast errors. Test whether adaptation adds stable incremental predictive information under shifts rather than simply increasing capacity. Exact architecture is not frozen; market and data feasibility are unresolved. Regime gating, residual forecasting, mixture-of-experts, conformal prediction and Transformer combinations already have close neighbours.

Alternative: reliable cross-market transfer with explicit market-specific heads and target-specific normalization. Use only if market/product identities and concurrent information availability are authenticated. This too needs a prior-art check.

## Literature status
No new literature retrieval has been completed in this task. Preliminary local draft references and claims of search retrieval were premature and have been withdrawn before publication. A direct-neighbour review remains necessary; no novelty is certified.

## Evaluation contract to finalize before fits
Authenticate original tables anew; old project training/model outputs remain untrusted. Preserve product, price definition, observed-date/calendar horizon and quote staleness semantics. Freeze data hashes and training/validation/test boundaries before fitting; no full-series decomposition, future backfill, test-set tuning or retroactively selected windows. Use train-only scalers and only matured labels/errors for adaptation.
Primary comparisons: target-matched random walk, linear/ridge, a tree baseline, a compact neural baseline and a relevant strong recent architecture; equal declared tuning budgets. MAE and RMSE with baseline-relative improvement by horizon/year, not cross-paper headline metrics. Probabilistic scoring only if part of the fixed question. Include ablations, repeated seeds, uncertainty and resource cost. No market observations or results presently uploaded.

## Bounded next phase
Complete a finite data and direct-neighbour feasibility review; select one main candidate and one target. Then run a small local pilot with fixed tuning budget and explicit stopping conditions. Stop or simplify if gain is absent or restricted to one period. Do not invent percentage-gain targets or continue until significant. No training has started.

## Privacy and upstream rights
Private repository does not waive third-party permissions. Do not commit credentials, email correspondence, signed URLs, old models, or restricted raw datasets. Non-OA publication does not imply secret code or exemption from reproducibility requirements.

## Executed evidence (2026-10-12)
- `reports/20261012/gdea_data_audit.json`: fresh parse of three existing official-response snapshots; 2839 records, byte receipts checked, yearly aggregate data diagnostics. Daily prices are not uploaded; close/status semantics and redistribution permission remain unresolved.
- `reports/20261012/synthetic_objective_probe.json`: constructed mathematical counterexample, not market predictions, training or performance improvement.
- `reports/20261012/RESEARCH_DECISION_ZH.md`: bounded method hypothesis, data barriers, comparisons, stopping rules and Pro handoff. This supersedes the initial broad candidate for the next feasibility review, not a certified novelty claim.
- `tools/research_probe.py`, `tests/test_probe.py`: executable stdlib audit and maturity/objective checks; run `python -B -m unittest discover -s tests -v`.

Local audit invocation: `python -B tools/research_probe.py --local-root LOCAL_SNAPSHOT_DIRECTORY --receipt LOCAL_SOURCE_RECEIPT_JSON --out reports/20261012`. Obtain originals lawfully; the executable compares exact bytes against the supplied source receipts. No model training has occurred. No old experimental results are used.
