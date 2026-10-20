# Actual synthetic results

All four seed/condition runs completed in 3.203 seconds (NumPy analytic training; 90 epochs each independent and joint MLP), within the 600-second ceiling. 252 seed/condition/model/market/horizon MSE cells independently checked from saved predictions. No market access, no commit/push.

| Model | shared mean MSE | no_shared mean MSE |
|---|---:|---:|
| RW |8.721105|10.667326|
| Original independent |8.890766|10.764351|
| Joint direct ridge linear |5.809822|7.721459|
| Joint reduced-rank 2 |5.823931|8.070606|
| Joint MLP |6.086068|8.159632|
| State/covariance candidate |5.816260|7.715657|
| Latent oracle AUDIT ONLY |4.970113|5.726128|

Joint linear wins every market/horizon cell versus RW in both conditions. Shared MLP loses 16/18 cells versus linear; candidate loses 13/18; reduced-rank loses 15/18. No_shared MLP loses 16/18 to linear and 1/18 to RW; reduced-rank loses 16/18 to linear and 2/18 to RW. Candidate loses 5/18 to linear in no_shared. Full cells in metrics.csv; loss/tie counts comparisons.json. These are two seeds, not statistical significance.

Shared seed17 and seed43 observed joint linear minus true-state oracle MSE gaps: .415572 and 1.263845. Thus observed histories contain usable predictable structure; previous neural losses do NOT demonstrate that shared state is unobservable. Remaining gap mixes finite-sample estimation, restricted regressors, and partial observation. No_shared improvement also indicates better representation/estimation, so shared information cannot be isolated from these comparisons. A matched own-increment linear control would be required to isolate cross-market information benefit; not included in this frozen grid.

Candidate fails the shared-setting positive gate. Do not continue gradient reweighting or claim innovation. Constant-noise complete-observation Kalman covariance becomes deterministic: uncertainty feature is effectively a state reparameterization, not new private multistep mean information. Existing factor/Kalman and Deep Factors work is directly relevant. Primary references verified by web search: https://www.sciencedirect.com/science/article/pii/S030440761100039X ; https://proceedings.mlr.press/v97/wang19k.html . PCA/AR/moment-noise implementation is a conventional heuristic, not a full dynamic-factor MLE and not claimed mathematically equivalent to Deep Factors' nonlinear architecture.

Future-perturbed genuine refit: shared seed17 joint linear fitted anew after adding 10000 to prices strictly after t=1440. Fitted coefficients, scaler, and t=1440 input exactly unchanged. Validation and train targets predate perturbation; later test targets differ and were not used in selection. Last train label=1079 < val origin=1080; last val label=1439 < test origin=1440.

Source: run.py plus original_baseline_source.py snapshot, SHA256 in verification.json; original generator file unchanged. Per-run observations, checkpoints, full training/validation histories and predictions retained. The original independent learner's exact traces are in independent_training.json. Covariance candidate estimates parameters only from observed training increments; latent values used exclusively in oracle audit.
