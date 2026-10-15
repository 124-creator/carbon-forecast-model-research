# Separated predictive decomposition: structural revision before a fresh trial

Status: complete neural-model hypothesis, not established novelty. This is one structural revision of ../predictive_decomposition_model; no third within-route redesign or rescue scan is authorized by this protocol.

## Why the previous implementation was stopped
The unrestricted four-input GRU can absorb the conservative-band basis by a linear input-weight change. The actual nonzero-feedback full-sequence numerical check is recorded in ../predictive_decomposition_model/COORDINATE_EQUIVALENCE.json. Its 61 completed fits were stopped, not ranked on market scores. Their forecasts, numeric parameters, training logs and source snapshots are preserved; no old prediction or weight is reused here. This redesign was chosen after seeing training progress, but before computing/viewing the aborted run's aggregate evaluation score. It is not an untouched research history.

Conservation is a representation identity, not sufficient model innovation. ES-dRNN already has recurrent feedback into smoothing coefficients, and Leddam already has learned decomposition, as confirmed in the actual pinned author-source audit in the parent directory. The present architecture still needs a direct-neighbor novelty review if it proves useful; there is no first-in-field claim.

## Full model and the structural restriction
Retain the three ordered filter states, zero initialization, stick-breaking rates initialized at [1/20,1/5,1/2], and four telescoping bands b whose sum equals the observed standardized price change. No band is discarded. Do not assume high frequency is noise or give components unobserved economic-factor names.

Unlike the aborted unrestricted-input design, each band j has its own 8-dimensional hidden state:

    h_j,t = shared_GRUCell(b_j,t, h_j,t-1), j=1..4
    H_t = concatenate(h_1,t, ..., h_4,t)
    v_t = W H_(t-1) + bias
    rates_t = ordered_stick_breaking(v_t)
    s_j,t = (1-rate_j,t)s_j,t-1 + rate_j,t*x_t
    forecast_(1:20) = Linear(H_last)

The filter update precedes the GRU update; rate_t uses only H_(t-1). The scalar-input GRUCell parameters are shared across bands. There is no unrestricted recurrent mixing between band states. In the coupled model, cross-band interactions enter the next decomposition rates through W, then affect subsequent band-specific states. A final linear readout combines the hidden states. Arbitrary band-basis changes can no longer simply be absorbed by a free four-input matrix, although this does not prove originality, component identification or a new universal function class.

Controls: plain = original 16-unit one-input GRU; fixed = separated shared-cell model with fixed rates; global = same model with three trainable global logits; coupled = same model with feedback W and bias. Fixed/global/coupled have identical initial shared-cell weights/readout and predictions per seed. Only original-price prediction loss trains the model and rates, with no future decomposition labels or reconstruction auxiliary loss. Shared 8-unit cell gives 32 hidden coordinates across four bands plus three filter states; plain has 16 hidden coordinates. Parameter count, recurrent width, effective cell/sample occurrences and runtime are not equal and must be disclosed. This is a mechanism screen, not a capacity-matched final superiority claim.

## Data, fitting and exact trial budget
Fresh 4 variants x seeds 17/29 x 12 refreshes = 96 from-scratch fits, plus four actual first-refresh seed-17 future-perturbed refits. No reuse from the aborted implementation. Same protocol as its frozen parent: exactly 950 audited Guangzhou records through 2021-12-31 using the strict streaming reader; all 60 observed changes as input; 20 original-price net-change labels; latest-at-most-480 complete-H origin pool; last 60 validation origins and 20-origin purge; common scale estimated only from training inputs. No future calendars, exogenous values, full-series decomposition, future-filled prices or date changes.

AdamW lr .003, weight decay .001, full batch, norm clipping 1, max 200 epochs, patience 20, initialization eligible. All-20 unweighted standardized MSE for both training and validation. Refresh end=459+40k, k=0..11. Successful-fit soft runtime check 1200 seconds between fits, with causal refits accounted separately; incomplete support is reported as incomplete, never a partial-score win. No implicit resume/overwrite. Per-fit outputs persist.

## Fixed decision gates and interpretation
451 same-year complete-H origins; equal six-cell original-price MSE/RW-MSE score, 2020/2021 x horizons 1/5/20, average seed losses rather than ensemble forecasts. Coupled mechanism gate: strictly beat plain, fixed and global overall, with neither year worse than global. Escalation additionally requires score below historical MLE .9497932301207621; that old score is a conservative screen, not a freshly matched comparison. Historical eight-member ensemble .924480 is an external, higher-cost development reference only.

Keep every seed and cell. Independently replay scores from the original permitted prefix in normal and -O modes, audit purge and source snapshots, and verify full representative causal refits. If this precise model fails, stop it without new rate/width/seed choices. Passing does not prove that the added feedback rather than capacity explains gains; later matched-capacity and near-neighbor comparisons, multiple markets and a genuinely unscored frozen set remain necessary. 2020/21 are development and 2022 is already scored; no 2023 data is accessed.
