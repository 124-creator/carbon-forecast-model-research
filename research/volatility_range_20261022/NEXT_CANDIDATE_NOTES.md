# Next candidate notes — not executed

Frozen on 2026-10-22 while the first logistic-gate candidate is running, independently of its eventual score.

The expanding-window risk models show severe 2025 h=22 MSE deterioration in some assets; attribution must be checked asset by asset. A plausible hypothesis is historical high-variance level mismatch, not an established explanation.

A separate future candidate can use origin-observed EWMA97 risk scale s_t. Train on y_s/s_s, make own/range/gap predictors dimensionless relative to s_s, and forecast f_t=s_t*exp(g_t). QLIKE log(f_t)+y_t/f_t equals log(s_t)+g_t+(y_t/s_t)*exp(-g_t): normalization changes the parameterization, not the QLIKE comparison loss. The s_s values must be causal at each training origin; do not normalize historical examples by the current-origin s_t. Targets in the delivered prediction files remain original y; zeros remain zeros.

Before implementation, check scaled HAR, volatility normalization, and multiplicative component literature for prior art. This is not yet an innovation claim. Specify scale floors only if needed and without changing targets, training standardization, fixed hyperparameters, static and dynamic gate ablations, common origins, and comparisons with EWMA97. Do not start another run until the current gate results and per-asset diagnostics have been reviewed.
