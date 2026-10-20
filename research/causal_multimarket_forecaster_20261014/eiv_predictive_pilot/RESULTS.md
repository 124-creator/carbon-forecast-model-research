# Identification failed; ordinary low-rank diagnostics only

{
  "identification_gate": "FAIL",
  "novelty_gate": "BLOCKED",
  "candidate": "not estimated: R not identified",
  "no_correction_ablation": "RRR alias",
  "oracle": "not run: no identified measurement error decomposition",
  "aggregates": {
    "shared": {
      "independentlinear": 6.1118133735824065,
      "jointlinear_matched": 5.809822002285895,
      "jointlinear": 5.809822002285897,
      "RRR": 5.782263806969355,
      "PLS": 5.796288269028478,
      "factor": 6.320381605549654
    },
    "no_shared": {
      "independentlinear": 6.96308082748239,
      "jointlinear_matched": 7.767021727711834,
      "jointlinear": 7.72145902182454,
      "RRR": 7.706911487111664,
      "PLS": 8.63107996290891,
      "factor": 9.201807558517652
    }
  },
  "metric_rows": 216,
  "all_metrics_reloaded": true,
  "future_checks": [
    {
      "seed": 17,
      "condition": "shared",
      "model": "independentlinear",
      "future_cut": 1440,
      "exact_refit_first_forecast_equal": true
    },
    {
      "seed": 17,
      "condition": "shared",
      "model": "jointlinear_matched",
      "future_cut": 1440,
      "exact_refit_first_forecast_equal": true
    },
    {
      "seed": 17,
      "condition": "shared",
      "model": "jointlinear",
      "future_cut": 1440,
      "exact_refit_first_forecast_equal": true
    },
    {
      "seed": 17,
      "condition": "shared",
      "model": "RRR",
      "future_cut": 1440,
      "exact_refit_first_forecast_equal": true
    },
    {
      "seed": 17,
      "condition": "shared",
      "model": "PLS",
      "future_cut": 1440,
      "exact_refit_first_forecast_equal": true
    },
    {
      "seed": 17,
      "condition": "shared",
      "model": "factor",
      "future_cut": 1440,
      "exact_refit_first_forecast_equal": true
    },
    {
      "seed": 17,
      "condition": "no_shared",
      "model": "independentlinear",
      "future_cut": 1440,
      "exact_refit_first_forecast_equal": true
    },
    {
      "seed": 17,
      "condition": "no_shared",
      "model": "jointlinear_matched",
      "future_cut": 1440,
      "exact_refit_first_forecast_equal": true
    },
    {
      "seed": 17,
      "condition": "no_shared",
      "model": "jointlinear",
      "future_cut": 1440,
      "exact_refit_first_forecast_equal": true
    },
    {
      "seed": 17,
      "condition": "no_shared",
      "model": "RRR",
      "future_cut": 1440,
      "exact_refit_first_forecast_equal": true
    },
    {
      "seed": 17,
      "condition": "no_shared",
      "model": "PLS",
      "future_cut": 1440,
      "exact_refit_first_forecast_equal": true
    },
    {
      "seed": 17,
      "condition": "no_shared",
      "model": "factor",
      "future_cut": 1440,
      "exact_refit_first_forecast_equal": true
    },
    {
      "seed": 43,
      "condition": "shared",
      "model": "independentlinear",
      "future_cut": 1440,
      "exact_refit_first_forecast_equal": true
    },
    {
      "seed": 43,
      "condition": "shared",
      "model": "jointlinear_matched",
      "future_cut": 1440,
      "exact_refit_first_forecast_equal": true
    },
    {
      "seed": 43,
      "condition": "shared",
      "model": "jointlinear",
      "future_cut": 1440,
      "exact_refit_first_forecast_equal": true
    },
    {
      "seed": 43,
      "condition": "shared",
      "model": "RRR",
      "future_cut": 1440,
      "exact_refit_first_forecast_equal": true
    },
    {
      "seed": 43,
      "condition": "shared",
      "model": "PLS",
      "future_cut": 1440,
      "exact_refit_first_forecast_equal": true
    },
    {
      "seed": 43,
      "condition": "shared",
      "model": "factor",
      "future_cut": 1440,
      "exact_refit_first_forecast_equal": true
    },
    {
      "seed": 43,
      "condition": "no_shared",
      "model": "independentlinear",
      "future_cut": 1440,
      "exact_refit_first_forecast_equal": true
    },
    {
      "seed": 43,
      "condition": "no_shared",
      "model": "jointlinear_matched",
      "future_cut": 1440,
      "exact_refit_first_forecast_equal": true
    },
    {
      "seed": 43,
      "condition": "no_shared",
      "model": "jointlinear",
      "future_cut": 1440,
      "exact_refit_first_forecast_equal": true
    },
    {
      "seed": 43,
      "condition": "no_shared",
      "model": "RRR",
      "future_cut": 1440,
      "exact_refit_first_forecast_equal": true
    },
    {
      "seed": 43,
      "condition": "no_shared",
      "model": "PLS",
      "future_cut": 1440,
      "exact_refit_first_forecast_equal": true
    },
    {
      "seed": 43,
      "condition": "no_shared",
      "model": "factor",
      "future_cut": 1440,
      "exact_refit_first_forecast_equal": true
    }
  ],
  "splits": [
    {
      "seed": 17,
      "condition": "shared",
      "train_last_label": 1079,
      "val_first_origin": 1080,
      "val_last_label": 1439,
      "test_first_origin": 1440
    },
    {
      "seed": 17,
      "condition": "no_shared",
      "train_last_label": 1079,
      "val_first_origin": 1080,
      "val_last_label": 1439,
      "test_first_origin": 1440
    },
    {
      "seed": 43,
      "condition": "shared",
      "train_last_label": 1079,
      "val_first_origin": 1080,
      "val_last_label": 1439,
      "test_first_origin": 1440
    },
    {
      "seed": 43,
      "condition": "no_shared",
      "train_last_label": 1079,
      "val_first_origin": 1080,
      "val_last_label": 1439,
      "test_first_origin": 1440
    }
  ],
  "R0_equivalence_max_abs": 2.220446049250313e-16,
  "OLS_max_abs": 5.828670879282072e-16,
  "nonPSD_rejected": true,
  "seconds": 1.625,
  "numpy": "2.4.6",
  "source": "E:\\\u78b3\u4ef7\u683c\\\u72ec\u7acb\u6570\u636e\u5f52\u6863\\carbon-forecast-model-research\\research\\causal_multimarket_forecaster_20261014\\joint_history\\matched_information\\run.py",
  "generator_source": "E:\\\u78b3\u4ef7\u683c\\11\\\u5b9e\u9a8c\u5341\u4e00\u5341\u4e00\\\u7530\\\u7b2c\u4e09\u6b21-tian-\u78b3\u4ef7\u683c\u5b9e\u9a8c\\research\\causal_multimarket_forecaster_20261014\\baseline\\benchmark.py",
  "source_hashes": {
    "E:\\\u78b3\u4ef7\u683c\\\u72ec\u7acb\u6570\u636e\u5f52\u6863\\carbon-forecast-model-research\\research\\causal_multimarket_forecaster_20261014\\joint_history\\matched_information\\run.py": "98774bc8776beb447be82a73cb5c3988e1a0cfdeee30691f6bcb174651dcb66c",
    "E:\\\u78b3\u4ef7\u683c\\\u72ec\u7acb\u6570\u636e\u5f52\u6863\\carbon-forecast-model-research\\research\\causal_multimarket_forecaster_20261014\\joint_history\\run.py": "a22b3159121d755f62649552bb0bd79b93e1e77cd9d9cda2f95ad427f093c5a9",
    "E:\\\u78b3\u4ef7\u683c\\11\\\u5b9e\u9a8c\u5341\u4e00\u5341\u4e00\\\u7530\\\u7b2c\u4e09\u6b21-tian-\u78b3\u4ef7\u683c\u5b9e\u9a8c\\research\\causal_multimarket_forecaster_20261014\\baseline\\benchmark.py": "e1e2fbf11b89b37368642b19146d891e0027cd6343d79106dec73ad058bc4dff",
    "E:\\\u78b3\u4ef7\u683c\\\u72ec\u7acb\u6570\u636e\u5f52\u6863\\carbon-forecast-model-research\\research\\causal_multimarket_forecaster_20261014\\eiv_predictive_pilot\\DESIGN.md": "89dac5a680070131d82d5f5b567ac62ccf7c679856ac9f8d70fe37b9f544412d",
    "E:\\\u78b3\u4ef7\u683c\\\u72ec\u7acb\u6570\u636e\u5f52\u6863\\carbon-forecast-model-research\\research\\causal_multimarket_forecaster_20261014\\eiv_predictive_pilot\\NEXT_TEST_SPEC.md": "4af351371eefc37b471ba1c891c99dc1b18d741797b77e00f3b23c3d614311f1",
    "E:\\\u78b3\u4ef7\u683c\\\u72ec\u7acb\u6570\u636e\u5f52\u6863\\carbon-forecast-model-research\\research\\causal_multimarket_forecaster_20261014\\eiv_predictive_pilot\\PROTOCOL.md": "b02f3adaaf2cef9826694a2b24bc12f70df3292e090291eb19f22b76ffbad3ff"
  }
}

No innovation evidence. Price transients cannot be relabeled measurement error. RRR is the exact no-correction degeneration, not a successfully identified EIV candidate. PLS/PCA are ordinary comparators. Search budgets differ: ridge 6, RRR/PCA 18, PLS 3 per fit; independent ridge 3 separate market fits. No market access, commits or push. All configurations and failed/null outcomes retained.
