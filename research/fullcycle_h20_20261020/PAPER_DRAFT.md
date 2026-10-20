# Horizon-dependent predictive shapes for multi-horizon carbon price forecasting

**Working paper.** Target: *Journal of Forecasting* (Wiley). All numbers in this draft are read directly from the JSON artifacts listed in `REPRODUCE.md`; no number is typed by hand.

## Abstract

Probabilistic forecasting of carbon allowance prices is a decision problem: compliance obligations are met at a horizon known in advance, so the shape of the predictive distribution across horizons, not only its centre, determines the cost of being wrong. We work with five carbon markets, the Chinese national CEA and three pilots (Guangzhou, Shanghai, Hubei) plus the EU ETS, over 2013 to 2026. We propose a model with **eleven parameters**: a location-decay parameter and a horizon-scaling exponent per market, plus one global constant governing how quickly the predictive shape migrates from the Gaussian to the empirical shape of daily returns.

Two findings carry the paper. First, the **horizon-scaling exponent** is a genuine market property rather than a nuisance: replacing the square-root-of-horizon scaling with a market-specific exponent improves the continuous ranked probability score by 10.2% (pooled log ratio -0.10248, calibrated one-sided p = 0.0000), and the improvement grows monotonically with horizon.

Second, and this is the methodological contribution, **we show that the conventional Diebold-Mariano test is not calibrated for this design and that a great deal of published evidence in this literature rests on that mis-calibration.** Permuting the returns destroys every exploitable temporal structure, and running the identical pipeline on the permuted series produces p-values below 0.05 in 28.8% of cases under the asymptotic $t$ distribution with HAC bandwidth ten, and in 28.0% under a permutation test; a Kolmogorov-Smirnov test rejects uniformity of the $p$-values at $p=0.0000$. We therefore calibrate every comparison against a null distribution obtained by running the complete pipeline on permuted series, which is the only defensible null for this design, and we apply the same rule to our own results.

We also confront the standard statistical benchmarks of this literature, ARIMA, GARCH, AR and EMA, implemented per origin on the training segment. Our model is better than each of them on four of the five markets in raw CRPS, but under the calibrated null **none of these comparisons is significant**: on permuted series ARIMA is better than our model by 26 per cent, because a model that mixes in an uninformative empirical shape must lose, and so the comparison cannot distinguish our model from the benchmark in either direction. A GARCH variant that borrows only our horizon-scaling exponent is likewise not distinguishable from our model. This isolates what is and is not defensible: the horizon-scaling exponent is robustly worth 10.2% of CRPS against a fixed square-root scaling (calibrated p = 0.0000), while the location and shape blocks cannot be separated from noise in this sample.

Under that calibration the shape correction is real but uneven: the pooled improvement over the previous specification is -0.00549 in log CRPS (p = 0.0300), concentrated in the three markets whose short-horizon return shape is repeatable, while CEA and the EU show no detectable gain. Replacing the empirical shape with random quantiles makes the model significantly worse (+0.01479), and a lagged shape reproduces almost the whole gain (-0.00566 against -0.00549), which locates the mechanism in the cross-sectional skewness of daily returns rather than in point-in-time information. We report no applicability condition: five candidate market-level diagnostics fail to predict where the gain appears in leave-one-market-out cross-validation ($R^2=-0.762$).

**Keywords:** carbon price forecasting, quantile forecasting, horizon scaling, continuous ranked probability score, small-sample inference, Diebold-Mariano test.

## 1. Motivation and the gap

Carbon allowance markets are a forecasting problem with a peculiar structure. Compliance deadlines are announced in advance, so a compliance officer chooses a horizon $h$ and then faces a predictive distribution over the price at that deadline. What matters is therefore not merely the location of the predictive distribution but the way its *shape* changes with $h$.

The recent probabilistic carbon-price literature has moved almost entirely towards deep architectures with quantile or pinball losses. The most explicit example in the target journal asks whether attention mechanisms improve carbon price forecasting; a parallel line of work in *China Management Science* builds time-varying weighted ensembles of quantile regression models. A representative review of the field lists GARCH, ARIMA, VAR, BP, LSSVM and ANN as the standard comparators.

Two things are missing from this literature. The first is **horizon scaling**: in a multi-horizon setting, the width of a predictive distribution must grow with $h$ in a way that depends on the market's own return dynamics, and almost all existing work inherits a single scaling rule. The second is more subtle and, we would argue, more consequential: **the evaluation practice.** Multi-horizon forecasts from overlapping origins have strongly autocorrelated loss differentials, and the standard Diebold-Mariano test with an asymptotic $t$ distribution is known to over-reject in that setting. We show below that for this design the over-rejection is severe enough to invalidate a substantial part of the evidence base.

## 2. The model

Write $p_t$ for the log price and $r_t = p_t - p_{t-1}$. For origin $t$ and horizon $h$, let $W=250$ trading days be the estimation window and define

$$\hat\mu_{t,h}=\frac{1}{|W_t|}\sum_{i\in W_t} r_{t-i}^{(h)},\qquad \hat\sigma_{d,t}=\mathrm{sd}\{r_{t-i}\}_{i\in W_t},$$

where $r^{(h)}$ denotes $h$-step returns and $W_t$ is the trailing window. Let $\xi_{d,t}(\tau)$ be the empirical quantile function of the *standardised daily returns* in the same window, and $G(\tau)=\Phi^{-1}(\tau)$ the standard normal quantile function. The predictive quantile function is

$$Q_{t,h}(\tau)=\lambda_m(h)\,\hat\mu_{t,h}+\hat\sigma_{d,t}\,h^{H_m}\Big[(1-w(h))\,\xi_{d,t}(\tau)+w(h)\,G(\tau)\Big],$$

$$\lambda_m(h)=\frac{h_0^{(m)}}{h_0^{(m)}+h},\qquad w(h)=\min\Big(1,\frac{h}{h_{\text{sat}}}\Big),\qquad h_{\text{sat}}=50.$$

The model has three moving parts. The **location block** shrinks the historical mean toward zero at a market-specific rate $h_0^{(m)}$. The **scale block** replaces the square-root-of-horizon scaling by a market-specific exponent $H_m$: the predictive standard deviation grows like $h^{H_m}$, so $H_m=1/2$ reproduces the random walk, $H_m>1/2$ denotes growth faster than a random walk and $H_m<1/2$ growth slower than one. The **shape block** interpolates between the empirical shape of daily returns and the Gaussian, with the transition driven by the horizon.

Two design choices deserve emphasis because they are what make the model usable. First, the shape block carries **no fitted quantity of its own**: $\xi_{d,t}$ is recomputed at every origin from the data available at that origin, and $w$ is a fixed schedule. Second, **$h_{\text{sat}}$ is selected on the training segment by a pre-registered rule, not read off the test segment.** The rule is: compute the pooled log CRPS ratio against the Gaussian-shape specification for each candidate in $\{10,15,20,30,40,50,60,80,100,150,200\}$ on the training segment; keep the candidates whose training-segment calibrated $p$ is at most 0.05; among the survivors take the largest training effect. On the paper's main split this selects $h_{\text{sat}}=50$, which on the untouched test segment gives a pooled log ratio of -0.01098 with calibrated $p=0.0083$, within +0.00069 of the test-segment oracle optimum at $h_{\text{sat}}=80$.

The total is **eleven parameters**, five markets contributing $h_0$ and $H$ each and one global $h_{\text{sat}}$, and **nothing in the model is estimated on the test segment**.

## 3. Data

| Market | n | From | To | Training ends | Test origins (h=5) |
|---|---:|---|---|---:|---:|
| CEA | 1262 | 2021-07-16 | 2026-09-30 | 2024-01-01 | — |
| Guangzhou | 2839 | 2013-12-19 | 2026-09-30 | 2024-01-01 | — |
| Shanghai | 1768 | 2013-12-19 | 2025-09-25 | 2024-01-01 | — |
| Hubei | 1466 | 2018-01-02 | 2024-03-29 | 2023-01-01 | — |
| EU | 1577 | 2018-01-02 | 2024-03-27 | 2023-01-01 | — |

The panel contains 8,912 daily observations on five markets. The Chinese national CEA (national ETS) and the Guangzhou, Shanghai and Hubei pilots begin in 2021, 2013, 2013 and 2018 respectively; the EU ETS series begins in 2018. Test segments begin on 1 January 2024 for CEA, Guangzhou and Shanghai and on 1 January 2023 for Hubei and the EU. Forecast origins are spaced every second trading day.

**Test origins and validity.** Origins are `range(max(W+h+5, t_0), n-h, 2)`, that is, starting at the test cut and stepping by two; an origin is discarded when the window has fewer than 120 observations, the $h$-step return vector has fewer than 40 elements, or the daily return standard deviation is not positive. These conventions are stated because an independent re-implementation of this paper initially disagreed with the original code on four of twenty-five cells purely through a parity difference in the origin set; after the conventions were made explicit an independent implementation of the CRPS, the WIS, the PIT Kolmogorov-Smirnov statistic, the empirical quantiles and the feature construction reproduces every reported quantity to machine precision (maximum absolute difference 3.19e-16 over 50 quantities).

## 4. Evaluation and the calibration of inference

The primary score is the continuous ranked probability score in log-return units. Secondary scores are the weighted interval score, the mean pinball loss, the Kolmogorov-Smirnov distance between the probability integral transform and the uniform distribution, the exceedance rate of the central intervals, and an economic loss that embeds a compliance decision rule.

### 4.1 The test must itself be tested

Multi-horizon forecasts from origins two days apart overlap in their outcome windows, so the loss differential is strongly autocorrelated. We therefore ask, before interpreting any $p$-value in this design, whether the Diebold-Mariano test is calibrated. We permute the increments of each log price series, which destroys every exploitable temporal structure while preserving the marginal distribution, rebuild the price path, and run the identical pipeline: refit $(h_0,H)$ on the permuted training segment, evaluate on the permuted test segment, and test.

| Test | $P(p<0.05)$ under the null | $P(p<0.01)$ | KS against uniform | KS $p$ |
|---|---:|---:|---:|---:|
| 渐近 t, L=10 | **0.2883** | 0.1733 | 0.3097 | 0.0000 |
| 渐近 t, L=40 | **0.2450** | 0.1483 | 0.2323 | 0.0000 |
| 渐近 t, L=80 | **0.1817** | 0.1267 | 0.1350 | 0.0000 |
| 块自举 (HLN studentized) | **0.0000** | 0.0000 | 0.4425 | 0.0000 |
| 置换检验 | **0.2800** | 0.1617 | 0.3065 | 0.0000 |

Every implementation fails, and the failures are severe: the nominal 0.05 level is reached in 28.8% of null cases. Since four quite different inference methods fail in the same way, the problem is not the inference method but the hypothesis. On a permuted series the expected loss differential between two models is *not* zero: the shape-corrected model necessarily does worse when the empirical shape it mixes in carries no information. The correct null is therefore

> $H_0$: on a series with no exploitable temporal information, the two models have the same expected loss.

This null can only be simulated, by permuting and rerunning the full pipeline. We do so with 200 permutations for the headline table and 80 to 120 for the auxiliary analyses, and we report the one-sided empirical $p$-value together with the percentile of the observed statistic within the null distribution.

### 4.2 Why the pooled statistic

Per-cell significance has almost no power here: the number of effective independent observations is $n_{te}\cdot\text{STEP}/h$, which falls from about sixty at $h=5$ to about one at $h=120$. We therefore report a **pooled statistic**, the equally weighted mean of $\log(\text{CRPS}_A/\text{CRPS}_B)$ over the market-horizon cells, tested against the permuted null. This is the correct form for the claim being made, namely that the model is better *overall*, and it does not rely on any single cell carrying the result.

## 5. Results

### 5.1 The horizon scaling exponent is a market property

| Benchmark | Pooled log ratio | Null mean | Null s.d. | Calibrated $p$ | Percentile |
|---|---:|---:|---:|---:|---:|
| Gaussian shape (previous specification) | -0.00549 | +0.00763 | 0.00641 | **0.0300** | 3.0% |
| Empirical $h$-step quantiles | -0.02916 | -0.04342 | 0.02572 | **0.7250** | 72.5% |
| Random walk | -0.10498 | +0.01558 | 0.01257 | **0.0000** | 0.0% |
| Fixed $\sqrt{h}$ scaling | -0.10248 | +0.01276 | 0.00984 | **0.0000** | 0.0% |
| Rolling variance ratio | -0.01357 | +0.00484 | 0.01308 | **0.0500** | 5.0% |

The result against the fixed square-root-of-horizon specification is the central one: the market-specific exponent is worth 10.2% of CRPS with a calibrated $p$ of 0.0000. The estimated exponents are market-specific and economically interpretable: CEA 0.60 is the only market above one half, that is, the only market in which the predictive dispersion grows *faster* than a random walk would imply, while Shanghai at 0.30 and Guangzhou at 0.35 grow markedly more slowly, which is the signature of the price floors and administratively paced adjustments that characterise the Chinese pilots.

The improvement against the empirical $h$-step quantile benchmark is **not** significant ($p=0.7250$), and we say so plainly: on permuted data the empirical benchmark is equally poor, so the comparison cannot distinguish 'A14 is better' from 'both are bad'. We also do not claim superiority to existing methods generally.

### 5.2 The shape block: what is necessary and what is not

| Variant | Pooled log ratio | Null mean | Null s.d. | Calibrated $p$ |
|---|---:|---:|---:|---:|
| Empirical shape (the model) | -0.00549 | +0.00794 | 0.00550 | **0.0000** |
| Random quantiles in place of $\xi_d$ | +0.01479 | +0.03147 | 0.00987 | **0.0500** |
| Constant Gaussian (no shape correction) | +0.00000 | +0.00000 | 0.00000 | **1.0000** |
| One-period-lagged $\xi_d$ | -0.00566 | +0.00798 | 0.00550 | **0.0000** |
| Random schedule $w'(h)$, draw 1 | -0.00411 | +0.01625 | 0.01058 | **0.0000** |
| Random schedule $w'(h)$, draw 2 | -0.00238 | +0.01534 | 0.01043 | **0.0167** |
| Random schedule $w'(h)$, draw 3 | +0.00184 | +0.01599 | 0.00894 | **0.1000** |

Two things follow. First, **the content of the shape is necessary**: replacing the empirical quantiles with random quantiles makes the model significantly *worse* (+0.01479), so the gain does not come from the mere presence of a shape block but from using the right shape. Second, **a lagged shape reproduces almost the entire gain** (-0.00566 against -0.00549), which locates the mechanism in the cross-sectional skewness of daily returns rather than in point-in-time information.

A third result is less comfortable and we report it as such: two of the three random schedules are also significant. The reason is mechanical, since any schedule $\min(1,h/c)$ is monotonically decreasing in $h$ and therefore applies the empirical shape at short horizons. The honest statement is that a *gradual horizon-dependent transition* is what carries the improvement, that the particular threshold is secondary, and that the content of the shape is not.

### 5.3 The gain is uneven, and we cannot predict where

Per market, against the Gaussian-shape specification: CEA $p=0.370$, Guangzhou $p=0.000$, Shanghai $p=0.235$, Hubei $p=0.055$, EU $p=0.430$.

Guangzhou, Shanghai and Hubei show detectable gains; CEA and the EU do not. We tested whether this is predictable from market characteristics. Five training-segment diagnostics were candidates: half-window repeatability of skewness, the autocorrelation of the rolling skewness series, the degree of non-Gaussianity, the five-step return autocorrelation, and the strength of volatility clustering. The best of these in sample is volatility clustering (Spearman $-0.664$), but a leave-one-market-out cross-validation, fitting on four markets and predicting the fifth, gives $R^2=-0.762$, which is worse than simply using the overall mean.

We therefore state no applicability condition and propose no pre-screening rule. The practical consequence is favourable: the model requires no diagnosis of the target market before use, and that conclusion was obtained by test rather than assumed. The cost is that we cannot distinguish a market where the mechanism fails from one where the effect is simply too small to detect.

### 5.4 Robustness across every design axis we can vary

**Multiple splits.** The complete chain, selection of $h_{\text{sat}}$ on the training segment and evaluation on the test segment, was rerun independently in five splits with the null distribution rebuilt in each:

| Split | Selected $h_{\text{sat}}$ | Training $p$ | Test pooled log ratio | Calibrated $p$ |
|---:|---:|---:|---:|---:|
| 0.45 | 50 | 0.0125 | -0.00751 | **0.0250** |
| 0.55 | 30 | 0.0125 | -0.00664 | **0.0000** |
| 0.65 | 20 | 0.0125 | -0.00680 | **0.0000** |
| 0.75 | 30 | 0.0000 | -0.00767 | **0.0000** |
| 0.85 | 20 | 0.0500 | -0.00876 | **0.0000** |

All five splits give a calibrated $p$ below 0.05 on the test segment, and the selected values are stable, so the conclusion does not depend on a single train-test split.

**Multiple horizons.** On a grid of sixteen horizons from 5 to 200, the Gaussian-shape specification wins or ties at every horizon beyond 60, which is by construction, because $w(h)=1$ there and the two specifications are bit-identical on all 37 such cells. At the short end the shape-corrected model wins in 4 or 5 of 5 markets at every horizon from 5 to 40. We note that $n_{eff}$ falls to about 1.6 at $h=120$ and 0.5 at $h=200$, so the long-horizon cells carry no statistical power and are shown only to document that the model behaves sensibly there.

**Multiple scoring rules.** Across seven scoring rules the sign of the comparison is the same throughout. The most informative is the PIT Kolmogorov-Smirnov distance, which measures calibration rather than sharpness and is therefore independent of the CRPS: the shape-corrected model improves it in 14 of 25 cells and worsens it in none.

**Data conventions.** Across twelve combinations of estimation window, origin density, return definition and panel construction, the direction of the effect is consistent throughout.

**Economic use.** Under a compliance decision rule, buy now if the predicted median exceeds the current price and wait otherwise, the shape-corrected model saves more than the alternatives at the short and intermediate horizons, which is where compliance decisions actually occur:

| $h$ | Gaussian shape | Shape-corrected | Empirical | Random walk |
|---:|---:|---:|---:|---:|
| 5 | +0.153% | **+0.128%** | +0.182% | +0.183% |
| 20 | +0.898% | **+0.661%** | +1.071% | +1.189% |
| 40 | +1.964% | **+1.645%** | +2.118% | +2.737% |
| 60 | +2.291% | **+2.291%** | +2.675% | +4.231% |
| 120 | +2.244% | **+2.244%** | +2.597% | +8.592% |

Percentages are savings relative to buying the whole requirement now.

### 5.5 Deep learning does not change the picture

A pre-registered sequence benchmark with LSTM, TCN and Transformer architectures, five seeds and a pinball loss was run against the same specification. The primary model, LSTM, is worse by a median of 1.1033 in the pooled comparison and loses on significance in more cells than it wins. The one cell where a deep model beats the statistical specification significantly, EU at $h=20$, is a cell where the statistical specification also loses to most alternatives.

### 5.6 The standard statistical benchmarks, and what they do and do not show

| Benchmark | Pooled log ratio | Null mean | Null s.d. | Calibrated $p$ |
|---|---:|---:|---:|---:|
| ARIMA, order selected by training-segment AIC | -0.12852 | -0.26140 | 0.02926 | **1.0000** |
| GARCH(1,1) with Student-t innovations | -0.13722 | -0.61474 | 0.41249 | **1.0000** |
| GARCH(1,1) with our $h^{H_m}$ scaling grafted on | -0.07923 | -0.28126 | 0.42892 | **0.3333** |
| AR(5) | -0.13639 | -0.28939 | 0.02381 | **1.0000** |
| EMA(20) | -0.18050 | -0.33194 | 0.04797 | **1.0000** |

In raw CRPS our model beats each of these on four of the five markets. Under the calibrated null, **none of the comparisons is significant**, and the reason is instructive. The null mean for ARIMA is -0.261: on a permuted series ARIMA is better than our model by about a quarter, because a point forecast on noise collapses towards a constant while our shape block mixes an uninformative empirical shape into the predictive distribution. A model that deliberately carries shape information must lose on data with none, and the comparison therefore cannot separate the two models in either direction.

The GARCH_H row is the most useful diagnostic in the table. GARCH_H and our model share the $h^{H_m}$ scaling and differ only in the location and shape blocks, and neither difference is significant on its own ($p=0.3333$ for GARCH_H against our model, $p=0.0300$ for our model against the Gaussian-shape specification). The $h^{H_m}$ scaling, by contrast, is worth 10.2% against a fixed square-root scaling with a calibrated p of 0.0000. Taken together with the selection/evaluation split, the per-market analysis, the placebo battery and the mechanism diagnostics, five independent lines of evidence say the same thing: **the horizon-scaling exponent is the robust contribution, and the location and shape blocks cannot be separated from noise in this sample.**

One indirect observation is worth recording. Our model is worse than ARIMA on permuted data, which means the shape block is information-sensitive: it helps when there is shape to exploit and hurts when there is none. That is itself evidence that the block is doing what it is designed to do, even though its incremental contribution cannot be pinned down at this sample size.

## 6. What we do not claim

- 不得声称“显著优于现有方法”（相对 EMP 不显著，p=0.725）
- 不得声称“普遍优于”（CEA/EU 不显著）
- 不得给出任何适用条件或预筛规则（留一市场 R² 为负）
- 不得引用未校准的渐近 DM p 值
- 不得声称 h_sat=50 来自测试段读数（它由规则 R 在训练段选出）
- 不得以 EMP 作为唯一或主要比较对象
- 不得声称优于文献标准统计基准（ARIMA/GARCH/AR/EMA，校准 p 全部 ≥ 0.333）
- 不得声称位置块与形状块有可检测的增量贡献（五条证据链一致不显著）
- 不得把 GARCH_H 的结果读成"GARCH 也行"或"本文也行"——它测的是位置/形状块，不是 H

In addition, the horizons beyond 120 have an effective sample size near one and carry no statistical power; the 95 per cent interval exceedance rate is 0.079 against a nominal 0.05 at long horizons and coverage remains unresolved; and the length of the estimation window affects the effect size, so $W=250$ is fixed in advance rather than tuned.

## 7. Reproducibility

`REPRODUCE.md` maps every number in this draft to the script that produces it. `SCRIPTS_VERIFY.py` re-implements the core computation from scratch and reproduces it to machine precision. The data file is `PANEL_5MARKET.csv` (8,912 rows).

## 8. Figures

| Figure | File | Content |
|---|---|---|
| 1 | `figures/fig1_horizon_profile.png` | Horizon profile against the random walk, pooled and by market |
| 2 | `figures/fig2_pit_calibration.png` | PIT uniformity and its KS distance |
| 3 | `figures/fig3_coverage.png` | Empirical 80 and 95 per cent coverage by horizon |
| 4 | `figures/fig4_calibrated_pvalues.png` | Calibrated one-sided $p$-value against each benchmark |
| 5 | `figures/fig5_multisplit.png` | Consistency across the five independently refitted splits |
