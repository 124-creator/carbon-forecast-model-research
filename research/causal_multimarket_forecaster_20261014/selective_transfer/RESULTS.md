# Conventional selective-transfer baseline results

NOT RELEASED as innovation; exact generalized-ridge equivalence. No market access or threshold rescue.

```json
{
  "rows": 180,
  "all_metrics_reload_verified": true,
  "source_hashes": {
    "E:\\\u78b3\u4ef7\u683c\\\u72ec\u7acb\u6570\u636e\u5f52\u6863\\carbon-forecast-model-research\\research\\causal_multimarket_forecaster_20261014\\joint_history\\run.py": "a22b3159121d755f62649552bb0bd79b93e1e77cd9d9cda2f95ad427f093c5a9",
    "E:\\\u78b3\u4ef7\u683c\\11\\\u5b9e\u9a8c\u5341\u4e00\u5341\u4e00\\\u7530\\\u7b2c\u4e09\u6b21-tian-\u78b3\u4ef7\u683c\u5b9e\u9a8c\\research\\causal_multimarket_forecaster_20261014\\baseline\\benchmark.py": "e1e2fbf11b89b37368642b19146d891e0027cd6343d79106dec73ad058bc4dff",
    "E:\\\u78b3\u4ef7\u683c\\\u72ec\u7acb\u6570\u636e\u5f52\u6863\\carbon-forecast-model-research\\research\\causal_multimarket_forecaster_20261014\\selective_transfer\\PROTOCOL.md": "ff2b523e4541805cd4dbee1e3f3f0349e2eae448ee339334aa073b9e62e38db3",
    "E:\\\u78b3\u4ef7\u683c\\\u72ec\u7acb\u6570\u636e\u5f52\u6863\\carbon-forecast-model-research\\research\\causal_multimarket_forecaster_20261014\\selective_transfer\\LITERATURE.md": "54e34fd64d4ead84fa0955aff2230a8fb4c945cbb44bf2a328b044ab9048751c",
    "E:\\\u78b3\u4ef7\u683c\\\u72ec\u7acb\u6570\u636e\u5f52\u6863\\carbon-forecast-model-research\\research\\causal_multimarket_forecaster_20261014\\selective_transfer\\run.py": "61fd4addc9eed7e6e54d0a7e7d8ec9c941588ec9da8944ebd1134495b6d16e20"
  },
  "future_refits": [
    {
      "seed": 17,
      "condition": "shared",
      "all_refit_selections_and_first_origin_exactly_invariant": true
    },
    {
      "seed": 17,
      "condition": "no_shared",
      "all_refit_selections_and_first_origin_exactly_invariant": true
    },
    {
      "seed": 43,
      "condition": "shared",
      "all_refit_selections_and_first_origin_exactly_invariant": true
    },
    {
      "seed": 43,
      "condition": "no_shared",
      "all_refit_selections_and_first_origin_exactly_invariant": true
    }
  ],
  "by_seed": {
    "17_shared": {
      "independent_linear": 5.030336225441147,
      "joint_linear": 4.721438509967386,
      "group_sparse": 4.7537522840177715,
      "scalar_validation_mix": 4.731201246773694,
      "horizon_shrinkage_baseline": 4.734127638893608
    },
    "17_no_shared": {
      "independent_linear": 6.140343142737821,
      "joint_linear": 7.357668659145563,
      "group_sparse": 7.160215006007148,
      "scalar_validation_mix": 6.786223446347677,
      "horizon_shrinkage_baseline": 6.768876415090038
    },
    "43_shared": {
      "independent_linear": 7.1902997973491845,
      "joint_linear": 6.898205494604404,
      "group_sparse": 6.876525791799054,
      "scalar_validation_mix": 6.8970751496071925,
      "horizon_shrinkage_baseline": 6.897528257757632
    },
    "43_no_shared": {
      "independent_linear": 7.781519519090164,
      "joint_linear": 8.145412959157095,
      "group_sparse": 8.073012440377804,
      "scalar_validation_mix": 7.76735705873331,
      "horizon_shrinkage_baseline": 7.766713811112186
    }
  },
  "no_shared_excess_mse": {
    "17_no_shared": {
      "independent_linear": 0.0,
      "joint_linear": 1.2173255164077421,
      "group_sparse": 1.0198718632693264,
      "scalar_validation_mix": 0.6458803036098555,
      "horizon_shrinkage_baseline": 0.6285332723522163
    },
    "43_no_shared": {
      "independent_linear": 0.0,
      "joint_linear": 0.36389344006693136,
      "group_sparse": 0.2914929212876398,
      "scalar_validation_mix": -0.014162460356853757,
      "horizon_shrinkage_baseline": -0.014805707977978422
    }
  },
  "empirical_gate": false,
  "innovation_gate": false,
  "reason": "Exactly constrained generalized ridge on forecast corrections; ordinary shrinkage baseline only",
  "wall_seconds": 2.5630000000237487,
  "train_last_label": 1079,
  "val_first_origin": 1080,
  "val_last_label": 1439,
  "test_first_origin": 1440
}
```

All no_shared market/horizon harm is reconstructible from the complete 180-row metrics.csv; aggregate excess shown above. Optimizer convergence for every sparse trial is in training.json; cap failure is not convergence. Same max-horizon maturity purge. Frozen validation-only weights, not online dynamic transfer.
