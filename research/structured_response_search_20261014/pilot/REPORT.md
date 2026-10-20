# Structured direct-response pilot

No innovation claim: exponential/damped response dictionaries are established. Fixed linear coefficients are absorbable into an ordinary linear output; fixed nonlinear coefficients are a constrained MLP output.

One candidate: 6 observable block statistics -> 4 tanh units -> trend + two amplitudes, with 2 global bounded rates. Candidate 45 parameters; fixed dictionary 43; MLP 78; linear 70; linear dictionary 21. Shared information, same iterations, not equal FLOPs or parameter counts.

Positive controls are deliberately aligned with the candidate family: unknown generating coefficients, observed history only, no latent-state oracle. Independent episodes in chronological order are a capability gate, not a contiguous market replay. Null future observations are independent noise, not cumulative random-walk changes.

All 60 runs retained. Empirical gates pass; 48/60 optimizers hit maxiter, including every structured candidate. No convergence claim and NO market access. No historical MLE/linear scores copied. No future data/2022 access. Parent should verify source and report; no commit/push performed.

|process|seed|reg|model|test MSE / zero|optimizer success|
|---|---:|---:|---|---:|---|
|no_signal|17|0.001|structured_rate|1.004890|False|
|no_signal|17|0.001|fixed_dictionary|1.005032|False|
|no_signal|17|0.001|linear_dictionary|1.001101|False|
|no_signal|17|0.001|mlp|1.014710|False|
|no_signal|17|0.001|linear|1.004707|True|
|no_signal|17|0.01|structured_rate|1.004358|False|
|no_signal|17|0.01|fixed_dictionary|1.004432|False|
|no_signal|17|0.01|linear_dictionary|1.000831|False|
|no_signal|17|0.01|mlp|1.011073|False|
|no_signal|17|0.01|linear|1.004273|True|
|no_signal|43|0.001|structured_rate|1.003616|False|
|no_signal|43|0.001|fixed_dictionary|1.003792|False|
|no_signal|43|0.001|linear_dictionary|1.005600|False|
|no_signal|43|0.001|mlp|1.023620|False|
|no_signal|43|0.001|linear|1.018193|True|
|no_signal|43|0.01|structured_rate|1.006101|False|
|no_signal|43|0.01|fixed_dictionary|1.004609|False|
|no_signal|43|0.01|linear_dictionary|1.005256|False|
|no_signal|43|0.01|mlp|1.023040|False|
|no_signal|43|0.01|linear|1.017680|True|
|persistent_transient|17|0.001|structured_rate|0.091083|False|
|persistent_transient|17|0.001|fixed_dictionary|0.091698|False|
|persistent_transient|17|0.001|linear_dictionary|0.155365|False|
|persistent_transient|17|0.001|mlp|0.093399|False|
|persistent_transient|17|0.001|linear|0.154719|True|
|persistent_transient|17|0.01|structured_rate|0.090776|False|
|persistent_transient|17|0.01|fixed_dictionary|0.090086|False|
|persistent_transient|17|0.01|linear_dictionary|0.154966|False|
|persistent_transient|17|0.01|mlp|0.091256|False|
|persistent_transient|17|0.01|linear|0.154329|True|
|persistent_transient|43|0.001|structured_rate|0.108857|False|
|persistent_transient|43|0.001|fixed_dictionary|0.108710|False|
|persistent_transient|43|0.001|linear_dictionary|0.153179|False|
|persistent_transient|43|0.001|mlp|0.109027|False|
|persistent_transient|43|0.001|linear|0.153274|True|
|persistent_transient|43|0.01|structured_rate|0.108655|False|
|persistent_transient|43|0.01|fixed_dictionary|0.108279|False|
|persistent_transient|43|0.01|linear_dictionary|0.153590|False|
|persistent_transient|43|0.01|mlp|0.112410|False|
|persistent_transient|43|0.01|linear|0.153087|True|
|asymmetric|17|0.001|structured_rate|0.087563|False|
|asymmetric|17|0.001|fixed_dictionary|0.087759|False|
|asymmetric|17|0.001|linear_dictionary|0.154278|False|
|asymmetric|17|0.001|mlp|0.089649|False|
|asymmetric|17|0.001|linear|0.154056|True|
|asymmetric|17|0.01|structured_rate|0.089360|False|
|asymmetric|17|0.01|fixed_dictionary|0.087654|False|
|asymmetric|17|0.01|linear_dictionary|0.154115|False|
|asymmetric|17|0.01|mlp|0.089312|False|
|asymmetric|17|0.01|linear|0.153873|True|
|asymmetric|43|0.001|structured_rate|0.106932|False|
|asymmetric|43|0.001|fixed_dictionary|0.107457|False|
|asymmetric|43|0.001|linear_dictionary|0.151356|False|
|asymmetric|43|0.001|mlp|0.109742|False|
|asymmetric|43|0.001|linear|0.150872|True|
|asymmetric|43|0.01|structured_rate|0.106724|False|
|asymmetric|43|0.01|fixed_dictionary|0.106610|False|
|asymmetric|43|0.01|linear_dictionary|0.151188|False|
|asymmetric|43|0.01|mlp|0.109689|False|
|asymmetric|43|0.01|linear|0.150683|True|

Outcome: nonlinear maps learn the designed positive controls better than linear outputs, but learned rates do not show a meaningful or consistent advantage over fixed dictionary or unstructured MLP. This supports a trainable baseline, not a new mechanism or market improvement.

Verification: {"verified_fits": 60, "prediction_replay_max_abs_error": 0.0, "fixed_linear_absorption_error": 1.4210854715202004e-14, "empirical_gate_passed": true, "primary_optimizer_converged": false, "market_allowed": false, "reason": "All primary structured fits hit iteration limit. Objective decrease is real but convergence is not certified; conservatively withhold market access."}

Runtime receipt: {"protocol_sha256": "c86ab7f311d59e26f050b27527eef7bc46884f7b31e732ece40c4d32e8bd4434", "fit_count": 60, "elapsed_seconds": 2.969000000040978, "gate_passed": true, "market_run": false, "market_reason": "Separate market identity/source audit required even if synthetic gate passes", "optimizer_success_count": 12}