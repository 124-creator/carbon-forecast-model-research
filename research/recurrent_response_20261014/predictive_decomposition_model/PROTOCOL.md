# State-coupled conservative decomposition: one complete model hypothesis

Status before implementation: candidate architecture, no established novelty or predictive result. Do not name an existing component as an original algorithm.

## Why this candidate, and what is already known
ES-dRNN author code already updates smoothing coefficients from recurrent-network outputs; adaptive recurrent smoothing is not our first contribution.[1] Leddam author code learns a shared convolution and forms a main/residual split; learned decomposition is not our first contribution.[2] These are actually fetched pinned source files, statically read in the ranges recorded in AUTHOR_SOURCE_RECEIPT.json. No whole-paper or author-experiment reproduction is claimed. LEM search identifies a multiscale recurrent neighbor, but its currently returned official repository is empty at the commit endpoint; no core-source review is claimed.

We retain a single experimental architecture question, not a first-in-field claim: can a small shared GRU learn better when its previous predictive state controls an ordered, exactly conserving ladder decomposition of observed price changes? This differs from our old response-time heads: the decomposition and its feedback act inside every encoder step, not only at the final output. However, this placement difference alone does not establish sufficient model innovation. The exact combination may already have direct neighbors beyond the two reviewed implementations.

## Complete state equations
For standardized observed changes x_t, three scalar filter states start at zero at the beginning of each 60-change input. Given the preceding GRU state h_(t-1), fixed/global/state-conditioned logits generate ordered rates by stick breaking:

    a1 = sigmoid(v1)
    a2 = a1 + (1-a1) sigmoid(v2)
    a3 = a2 + (1-a2) sigmoid(v3)
    s_j,t = (1-a_j,t) s_j,t-1 + a_j,t x_t
    b_t = [s1,t, s2,t-s1,t, s3,t-s2,t, x_t-s3,t]
    h_t = GRUCell(b_t, h_(t-1))
    predicted_change_(1:20) = Linear(h_last)

The components sum to x_t by construction; there are no future component labels, no separately optimized component forecasts, no reconstruction auxiliary loss, no discarded band and no assumption that a high-frequency component is unpredictable. With changing rates these are adaptive filter states, not certified permanent/transitory market factors. Rate order, not a strict spectral ordering theorem, is imposed. Each update is a convex filter update, but no global neural-state stability or statistical consistency theorem is claimed.

Fixed initial rates: [1/20, 1/5, 1/2], converted to logits once. No rate/window scan. Global variant trains three logits. Coupled variant trains v_t = W h_(t-1)+b with W initialized at zero; its initial encoder, readout and forecasts must match the fixed/global variants for each seed. The subsequent predictive-loss gradient must reach W. Plain reference is exactly the existing 16-unit GRU, with no decomposition.

## Fixed experiment and resource bounds
Four variants: plain, fixed, global, coupled. Seeds 17 and 29; twelve refreshes at 459+40k, k=0..11: 96 from-scratch fits. Four additional first-refresh, seed-17 future-price-perturbed refits. Do not reuse saved old forecasts or checkpoints. Do not add seeds, widths, windows, rates or rescue hyperparameters.

Use only the previously audited 950 Guangzhou records through 2021-12-31. Streaming prefix reader must not request the next row. Original records, duplicates, weekends and prices are unchanged. Input: all 60 observed price changes. Targets: all next 20 original-price net changes. At each fit use the same latest-at-most-480 complete-H origin pool, last 60 origins as validation, and 20-origin purge. Only training inputs estimate the common innovation standard deviation. All-20 unweighted standardized forecast MSE is both training and selection objective; no horizon rescaling. AdamW .003, weight decay .001, full batch, gradient clipping 1, max 200 epochs, patience 20, initialization eligible for selection. Same step budget does not mean equal FLOPs or equal active parameter count; disclose both parameter counts and runtime.

Save each completed fit, predictions and numeric parameters locally, with no implicit overwrite/resume. Fixed successful-fit runtime bound 1200 seconds, plus separately recorded representative causal refits; if the bound is reached report incomplete rather than score a subset as complete. Test normal and -O modes. Each representative perturbation changes p[end:] and must actually refit scaler/model and compare parameters and a pre-cutoff forecast. Only a complete fit grid gets the main score.

## Decision conditions
Main score: equal average of the six original-price MSE/random-walk-MSE cells for 2020/2021 x h=1/5/20, averaging seed losses, not averaging predictions. Expected support 451 same-year complete-H origins.

Mechanism gate: coupled must strictly beat plain, fixed and global in aggregate and must not worsen either year relative to global. Direction-escalation gate additionally requires beating historical MLE 0.9497932301207621. That historical reference is conservative screening only, not a freshly matched fair comparison. Historical eight-member feedback ensemble 0.924480 is a strong external development reference, not equal cost or a retrained competitor here. Report seed scores and all six cells, including reversals.

Passing either gate is not novelty or publication qualification. If the mechanism fails, stop this precise four-variant implementation; do not reject all decomposition or complete neural architectures. If it passes, a next stage must still establish direct-neighbor differences, matched capacity, equal-information strong baselines, broader seed/market stability and a genuinely frozen unseen evaluation. 2020/2021 are development; 2022 was previously scored and is not unseen. No 2023 data is accessed here.

Sources:
[1] https://github.com/slaweks17/ES-dRNN/blob/61439479934fd8164b0bb12a25680c70fca69feb/es_drnn.py — ES-dRNN pinned author source; static lines 800-919
[2] https://github.com/Levi-Ackman/Leddam/blob/5a145f78f4425be73c9b0b0f0ef527730f35afbf/layers/Leddam.py — Leddam pinned learned-decomposition source; static lines 1-115
