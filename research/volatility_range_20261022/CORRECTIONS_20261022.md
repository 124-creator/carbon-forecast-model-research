# 2026-10-22 corrected risk experiment corrections

This note records corrections requested by the supervising agent to the earlier exploratory probe. The independent corrected implementation and its manifest are the computational evidence for the new results; old probe scores must not be mixed with them.

- Earlier HAR features omitted the latest information day; corrected rolling features include origin t.
- The earlier MA22 slice contained 23 observations; corrected MA22 contains exactly r²[t-21],...,r²[t].
- Earlier model-specific masks could change score samples; all nine corrected models are scored on exactly the same asset/horizon/year origins.
- Earlier log-HAR lacked smearing; corrected HAR uses training-residual Duan smearing of log(y+epsilon), subtracts epsilon, and explicitly flags any nonpositive forecast floor. Targets remain original, including zeros.
- The earlier flat/zero-volume proportion was misreported: independent_validation.json verifies directly against raw data that KEUA is 67/79 = 84.81%, KRBN 0/4 and KCCA 0/1.
- The earlier tick-alignment algorithm was invalid and cannot support a claim about executable tick precision.
- These corrections do not prove prices are unpredictable or that any residual-model architecture will succeed. The interaction experiment is a mechanism probe and is not a claim of model innovation.
- 2024 is viewed validation, 2025 viewed development, 2026 viewed diagnostic data; none is a clean untouched test. Source files end in June 2026 for KRBN/KCCA and March 2026 for KEUA.

Implementation conventions are explicit in manifest.json: cross-year targets are removed from both training and scoring; gap² and Parkinson aggregates are logged; flat means high==low; ridge uses mean loss + 0.01/2 times squared standardized slopes; MSE_pct_MA22 is 100*MSE_model/MSE_MA22 (100 means equality).
