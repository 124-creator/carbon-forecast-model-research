# Stage M: duration-dependent shared forecast responses

Status: one research candidate, not an established novel model. Source main ba282791e1900aeed8c120535798e8532a4f5670. This protocol is committed before this stage's market fitting. Prior Stage L source/report are supplied in the conversation archive; its old weights and predictions are NOT training inputs. Stage L was not uploaded at that time. Current GitHub write actions are available again.

## Research question and prior art
Stage L's four shared-transition heads all beat its direct head on repeatedly used development data. That does not identify Markov dynamics: low-dimensional horizon sharing, basis smoothing, output initialization and parameterization remain competing explanations. This experiment explicitly adds learned rank-4/rank-6 and fixed-basis heads instead of promoting Stage L's result into novelty.

One new candidate RENEWAL allows a forecast-response state's departure rate to depend on its elapsed residence time. This is a specific neural duration parameterization, not the invention of semi-Markov models, hazard functions, GRUs, matrix powers, or reduced-rank forecasting. A hypothesis rather than an observed carbon-market fact: duration dependence could express response shapes not efficiently captured by the previous four-state memoryless readout. The latent states/ages below are computational response coordinates, NOT observed regimes, inferred actual market ages or calibrated regime probabilities.

Actual pre-protocol source scope: arXiv:2509.18414v1 HTML sections3.2-3.4 read (router plus duration hazard already exists for observed activity sequences); arXiv:2305.18803v2 HTML local-operator and forecast-objective passages read (Koopa matrix-power forecasting already exists); arXiv:1905.10437v4 HTML obtained, abstract/architecture overview read (N-BEATS basis expansion exists; detailed core-method review pending). ACM DOI10.1145/3773365.3773428 search-return abstract/intro/conclusion describes semi-Markov durations for commodity forecasting; complete equations and experimental protocol not audited. No author code executed or comprehensive novelty exclusion claimed. Carbon-price 2026 publisher abstracts were searched, not treated as full texts or comparable result tables.

## Data and information boundary
Read only the existing approved Guangzhou author CSV prefix: header plus950 records,29204bytes, ending2021-12-31, SHA256=1366ca4814dcdb8aa43296eb64768d1336d6f0fc9088648d6d7ccdbe43318070. Never request the following byte or hash/read the full original file. No new download, no2022-2024, noHubei/EU. Source_excel_row and Excel formatting are not features. Attribution follows the existing Yingying Fan2025 Figshare28682348.v1 CC BY4.0 record, not a new official-market/vintage certification.

Use origins459..929; score451 whose entire20-step target remains in the origin year (224 in2020,227 in2021). These are repeatedly exposed development years, not independent validation. Every refit T=459+40k uses origins j with j+20<T, last<=480 windows; last60 are internal validation with20-origin purge. Training support299-400; scale s_T is the training-input standard deviation only. Every model sees the same60 original price increments and the same as-issued causal reference. No posterior state truth, future state label, future date, unapproved holdout or old checkpoint.

## Shared reference and encoder
Use the inspected Stage L31-member scalar Gaussian reference bank, as-is: RW plus rho{0,.25,.5,.75,.9,.97} x permanent variance fraction{.02,.1,.25,.5,.8}. Choose the smallest profiled past innovation likelihood using previous240 observations at60,80,...; exclude current observation from model selection, allow it in the state update; never rewrite earlier b[j,h]. Additional reference windows120/480 are ordinary baselines.

Shared feature f=tanh(U(GRU(x)-GRU(0))+Vq+c)-tanh(c),dimension8, GRU(1,16),q=[tanh(mu/s_T),rho*(1-v),1-v]. Same common parameters are initialized identically across matched seeds. Output target is raw-price net change1..20. Reference b plus normalized correction is forecast; coefficient1 correction-amplitude penalty inherited from exposed prior development, not selected anew.

## One candidate, complete response equations
The common four-state transition is P=.98 rowSoftmax(sum_m alpha_m(f)L_m)+.02J/4,with3 logit bases. Bases are initialized exactly as supplied Stage L. Initial probability pi=softmax(A_pi f+b_pi), payoff v=3tanh(A_v f+b_v) centered across states, drift d=tanh(A_d f+b_d). Payoff/drift output parameters start at0.

For state s define e_s=1-P_ss and Q_sr=P_sr/e_s (r!=s),Q_ss=0. Candidate duration parameter beta_s(f)=2tanh(A_beta f+b_beta),A_beta=b_beta=0 initially. Let g(a)=log(1+a)/log(20),a=0..19. Exit hazard lambda_s(a)=sigmoid(logit(e_s)+beta_s(f)g(a)).

Forecast-age mass starts z_0(s,0)=pi_s,all other ages0. Age0 is a response-clock initialization at EACH forecast origin, not an estimate of the real regime age. For h=0..19:
 z_(h+1)(r,0)=sum_(s!=r,a) z_h(s,a)lambda_s(a)Q_sr;
 z_(h+1)(s,a+1)=z_h(s,a)(1-lambda_s(a)).
Retain ages0..20 without truncating probability inside this20-step forecast. Define m_h(s)=sum_a z_h(s,a). Forecast correction is (m_h-pi)v+(h/20)d, then p_hat=p_t+b_h+s_T*correction_h. All network and transition parameters trained jointly from original-price forecast loss; no likelihood of fabricated regime labels.

At beta=0, marginalized response exactly reduces to the original MARKOV decoder(pi P^h-pi)v+(h/20)d. RENEWAL does not generally obey a4-state semigroup, but IS an ordinary Markov chain after state-age augmentation; this equivalence is disclosed, not hidden as a novel theorem. Mass conservation/bounded response do not identify true states or guarantee predictive improvement.

## Matched controls
1.MARKOV: exact original Stage L memoryless decoder (1228parameters), newly trained.
2.CLOCK: identical beta and all common parameters, but replace g(age) by g(h) at forecast step h. Thus identical parameter count and age-mass computation, but exit depends on total forecast horizon rather than residence. It tests duration versus ordinary horizon modulation.
3.RANK4 and4.RANK6: common encoder, learned20xr basis B, coefficients3tanh(Af+b), plus the same bounded linear drift. Initial B is piecewise-linear interpolation basis at r fixed geometrically spaced horizon knots (unique rounded geomspace1..20, with a fixed zero at horizon0); B then learned freely. These are generic reduced-rank heads, not N-BEATS reproductions. RANK6approximately matches candidate parameter count; no dummy capacity.
5.FIXED6: same asRANK6 but freeze B at those interpolation weights. Tests simple smooth shared shape, with fewer actual trainable parameters disclosed.
6.DIRECT: previous ordinary FULL direct20-output head.
7.GRU: previous ordinary GRU baseline without reference.
Together with RENEWAL there are8modes. No second candidate will be selected after seeing scores. The better baseline is retained as baseline only.

## Fixed training and finite budget
8modes x4seeds(17,29,41,53) x12refits=384formal neural fits. AdamW lr.003,weight_decay.001,full training batch,clip norm1,max160epochs,patience20; checkpoint uses unpenalized20-horizon validation MSE,epoch0 allowed. Forecast loss=MSE+mean(correction^2),coefficient1; ordinaryGRU=MSE. Same steps do NOT mean sameFLOPs; age propagation increases state dimension/computation. Log all epochs, selected epochs, parameter counts, allocated/active-capacity differences, seconds, and failures. Float64,MKLDNNoff,deterministicTorch,single-threadBLAS,<=3independent workers. No hyperparameter,window,rank,seed or horizon rescue scan.

Fresh ordinary baselines: RW; referenceW120/240/480; driftW120/240/480; RidgeL5/20 xalpha.01/.1/1/10; HistGradientBoosting leaves7/15 xlr.05/.1,80rounds,no random validation. Trees/Ridge forecast onlyh1/5/20,othersNaN. Reference/GRU fixed and matured-OLS combinations generated anew. Full fast-online eight-network comparison only if the primary mechanism gate passes; historical scores are not substitutes for a new paired comparison.

## Pre-fit checks, analysis and stop
Before market fits: exact beta0/Markov equality, mass/nonnegativity, finite-difference hazard gradient, step1 equality, age-versus-clock divergence after renewal, state-age lifted-matrix equivalence, label maturation, train-only scales, prefix perturbation, output-neutral initialization, common initialization, rank bounds, fixed/learned basis initialization equivalence, affine price transform, deterministic replay of a fixed input. Algebra fixtures are not carbon prediction results. Do not create favorable synthetic-score evidence.

Primary R=equal mean of six year2020/2021 xh1/5/20 original-priceMSE/RW cells, averaging seed LOSSES. Separately score averaged predictions. Full tables include annual/horizonRMSE,MAE and all seeds. Use2000paired within-year moving-block resamples at lengths20/40/60; same samples for all models/horizons. These are descriptive, selected repeated-development intervals with no cross-round correction.

Continue this candidate only if mean R improves>=1% against MARKOV,CLOCK,RANK6 and FIXED6, with>=3/4paired seed wins against MARKOV and CLOCK; candidate ensemble must beat the same-four-network noncandidate best, and mean R must beat RW by>=2% and the fresh noncandidate best by>=1%; no year worse thanRW by>5%. These are project screening criteria, not journal thresholds. Report rank4 too. If fail,stop before new historical data or newmarkets; no renamed control as success. If finite/range tests fail,fix with explicit timing record before scoring or report incomplete.

## Record and delivery
Save complete trained numeric weights,predictions,response-mass diagnostics,all epoch traces,scores,environment,prefix receipt,code hashes and independent scoring. Audit all saved forecasts from saved numeric state dictionaries(allow_pickle=False),representative full future-price-perturbed refits,and include source-level/algorithmic equivalences. Push all own source and permitted numerical materials to this new branch,without merging main/changing visibility. Do not redistribute third-party papers. Record actual remote status and cross-environment differences only after execution; no claim of remote reproduction from upload alone.
