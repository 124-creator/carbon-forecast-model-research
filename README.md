# Carbon forecasting model research

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
