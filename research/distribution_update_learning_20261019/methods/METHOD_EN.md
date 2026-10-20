# A Provisional Research Method for Conditional Distribution Updates with Bounded Reflection Controls

**Version: 2026-10-19; original protocol 1.0 and pre-run addendum 1.1.** This document specifies a proposed calibration operator and the conditions under which its algebra supports a risk statement. It does not establish originality, empirical superiority, or readiness for publication. The complete mathematical and implementation specification is in `METHOD_ZH.md`; empirical conclusions belong to `REPORT_ZH.md`.

## 1. Forecasting task and fixed experts

At quote origin o, let P_o>0 be the observed price, H_o the available history, s_o a historical daily return scale, and u_o=s_o√h. The response and its observed future increments are

\[
Y=\frac{\log P_{o+h}-\log P_o}{u_o}
=\sum_{j=1}^h r_j,\qquad
r_j=\frac{\log P_{o+j}-\log P_{o+j-1}}{u_o}.
\]

Market horizons are h=5 and 20 actual quote observations. Missing calendar dates are not fabricated. Only H_o-measurable features X are prediction inputs; the future path becomes calibration information after its endpoint matures.

Two ordinary Gaussian scale mixtures are fitted before calibration:

\[
F_0(y)=\sum_{m=1}^3\pi_m\Phi(y/\sigma_m),\qquad
F_1(y\mid X)=\sum_{m=1}^3\pi_m\Phi(y/[f_\theta(X)\sigma_m]),
\]

where σ_m=s_*ρ_m, ρ=(0.5,1,2), and fθ=0.5+sigmoid(gθ). The output head is initialized to zero, so epoch zero has f=1 and exactly reproduces F0. An MLP or GRU is trained with ordinary CRPS, including epoch zero in validation selection. Experts are then frozen. The proposed control modifies the subsequent calibration update, not neural training gradients.

For a predictive draw Y from Fα=(1−α)F0+αF1, the positive price P_o exp(u_oY) has a finite lognormal mixture distribution. Its predictive moments satisfy, for every fixed origin and finite real q,

\[
E_{F_\alpha}[P^q]=P_o^q\sum_m\pi_m\left[
(1-\alpha)e^{q^2u_o^2\sigma_m^2/2}
+\alpha e^{q^2u_o^2f_\theta(X)^2\sigma_m^2/2}\right]<\infty.
\]

The price point forecast is this expression at q=1. These are model moments at a fixed origin, not unconditional moment claims about the true market. The primary risk is normalized log-increment CRPS; price CRPS, price-mean MSE, and interval measures remain separate outcomes. A nonlinear price transformation does not preserve CRPS rankings.

## 2. Exact quadratic CRPS risk

Use the established loss convention [Gneiting and Raftery](https://doi.org/10.1198/016214506000001437):

\[
S(F,y)=\int(F(z)-\mathbf1\{y\le z\})^2dz
=E|Z-y|-\tfrac12E|Z-Z'|.
\]

For d_x=F1−F0, define

\[
a(x)=\int_{\mathbb R} d_x(z)^2dz,\qquad
b(x,y)=\int d_x(z)\{\mathbf1(y\le z)-F_0(z\mid x)\}dz.
\]

Then, exactly,

\[
S(F_\alpha,y)-S(F_0,y)=a(x)\alpha^2-2b(x,y)\alpha,
\qquad b=\tfrac12\{a+S(F_0,y)-S(F_1,y)\}.
\]

Writing A=Ea and B=Eb gives ΔR(α)=Aα²−2Bα. Ordinary empirical CRPS mixing uses clip(Σb/Σa,0,1). The proposed operator changes how finite calibration data justify the update, while preserving the same valid CDF family.

All mixture scores and a are analytic finite sums. In particular, normal absolute distances use Ψ(z,s)=2sφ(z/s)+z[2Φ(z/s)−1]; convolution adds variances. No simulated forecast mean substitutes for an analytic quantity.

## 3. Reflection control and a pathwise budget

With the origin and experts fixed, let S_(j−1)=Σ_(l<j)r_l and define

\[
G_j(s;x)=E_{Z\sim N(0,(h-j)/h)}[b(x,s+Z)],\qquad
M_j=\tfrac12\{G_j(S_{j-1}+r_j;x)-G_j(S_{j-1}-r_j;x)\}.
\]

G_j uses a predetermined reference continuation. It need not equal the true conditional continuation. Holding the past and reference fixed makes M_j odd in the current increment, regardless of reference accuracy. This algebra does not itself imply a zero mean under an asymmetric true process. Future magnitudes may depend on earlier signs; the construction does not condition on the entire future magnitude path and randomize all its signs.

For the common-scale experts,

\[
W(x)=W_1(F_0,F_1)=|f(x)-1|\sqrt{2/\pi}\sum_m\pi_m\sigma_m.
\]

Since b'(y)=F0(y)−F1(y), common symmetry gives osc_y b≤W/2. Averaging cannot increase oscillation, so |M_j|≤W/4. The primary total budget is K=W/4:

\[
K_j=\max\{K-\sum_{l<j}|\widetilde M_l|,0\},\quad
\widetilde M_j=\operatorname{clip}(M_j,-K_j,K_j),\quad
C=\sum_j\widetilde M_j.
\]

The predictable symmetric clipping preserves oddness and ensures Σ|M̃_j|≤K, hence |C|≤K, on every path. The range statement fixes x; it must not be mistaken for a global range when X also varies. K=W/4 is not claimed to be statistically optimal.

Protocol 1.0 used K=W. Addendum 1.1 adopted W/4 on the analytic range argument after the separate four-path audit, but before general-operator synthetic runs or market fitting. Both records and the full-cap comparison are retained. This is a local design record, not external preregistration. Without clipping, the available bound is hW/4. With no prefix, or h=1, the present symmetric construction gives C=0 exactly.

## 4. Explicit conditional sign-bias assumption

The needed true-process assumption is

\[
\left|E[\operatorname{sgn}(r_j)\mid |r_j|,H_{j-1}]\right|\le\delta
\quad\text{almost surely},\qquad\operatorname{sgn}(0)=0.
\]

Oddness conditional on the current magnitude and past, followed by the pathwise budget, yields

\[
|E[C\mid X]|\le\delta K(X),\qquad
|E[v_k(X)C]|\le\delta E[v_k(X)K(X)]
\]

for any nonnegative historical basis v_k. A zero mean, a conditional martingale difference, or a marginal sign balance does not imply this assumption. At δ=0, the assumption and integrability imply a martingale difference; δ>0 permits nonzero drift. An ordinary per-step reflection-total-variation bound with the same δ is insufficient. For example, increments equal to +1 with probability δ and otherwise zero, together with a control spending K at the first nonzero event, have EC=K[1−(1−δ)^h]; their conditional sign imbalance at nonzero magnitude is one. They lie outside the stated assumption.

The market primary δ=0.05 and sensitivities 0, 0.025, 0.10 are assumed budgets, not certified market bounds. A wrong reference affects efficiency; a wrong δ can invalidate the risk argument.

## 5. Simultaneous calibration and a three-parameter update

Fix Λ={0,0.5,1}. In the scalar case, simultaneous lower bounds for μλ=E[b−λC], an upper bound κ+≥EK, and U_A≥A give

\[
L=\max_{\lambda\in\Lambda}
\{\widehat\mu_\lambda-u_\lambda-|\lambda|\delta\kappa_+\},\qquad
\widehat\alpha=\operatorname{clip}(L/U_A,0,1).
\]

On their joint event, ΔR(α̂)≤U_Aα̂²−2Lα̂≤0. Zero is the fallback if U_A=0. Choosing λ requires simultaneous protection; its multiplicity cost is not removed by including λ=0.

The main gate uses three nonnegative basis functions summing to one:

\[
v_k(X)\propto\exp[-(z(X)-c_k)^2/(2\tau^2)],\quad
c=(-1,0,1),\quad\tau=0.75,
\]

where z is historical log EWMA scale standardized using training data. Set α_w(X)=wᵀv(X), w∈[0,1]³. Now A=E[avvᵀ], B=E[bv], and ΔR(w)=wᵀAw−2Bᵀw.

For each coordinate, form a jointly protected lower bound

\[
L_k=\max_{\lambda\in\Lambda}
\{\widehat\mu_{\lambda,k}-u_{\lambda,k}
-|\lambda|\delta\kappa_{+,k}\}.
\]

If symmetric elementwise errors satisfy |A_jk−Â_jk|≤e_jk, the standard Gershgorin-type envelope

\[
U=\widehat A+\operatorname{diag}(\sum_k e_{jk})
\]

satisfies A≼U and U≽0. Solve the convex box problem

\[
\widehat w\in\arg\min_{w\in[0,1]^3}
\{w^\top Uw-2L^\top w\}.
\]

Because w≥0 and zero is feasible, the joint event implies ΔR(ŵ)≤0. The bound is uniform over the continuous box, requiring no additional grid over w. It concerns population-average risk, not every history separately. Implementation enumerates the 27 faces, checks feasible stationary solutions, and retains zero as a fallback.

## 6. A finite-sample statement and its market limitation

**Conditional iid proposition.** Condition on independent earlier fitting data. Suppose calibration trajectories are iid, evaluation has the same law, the sign-bias assumption holds, risks are finite, and calibration statistics have known support bounds. If all preceding statistical bounds fail with total probability at most ε, then the calibrated update has risk no greater than F0 with probability at least 1−ε over calibration data. Dependence inside one trajectory is allowed.

The proof combines B≥L, A≼U, and zero feasibility. A valid one-sided mean radius from [Maurer and Pontil, Theorem 4](https://arxiv.org/pdf/0907.3740) is

\[
u(Z;\eta)=\sqrt{2s_n^2\log(2/\eta)/n}
+\frac{7R_Z\log(2/\eta)}{3(n-1)},
\]

where R_Z is the full support length, not an observed sample range. For three λ and three gate coordinates, allocating ε across nine linear lower bounds, three budget upper bounds, and two sides of six matrix entries gives 24 one-sided events. A known envelope W≤M implies a≤M and |b|≤M without clipping the true response. The implemented iid scalar synthetic case has fixed experts and known a,K; it does not implement this entire random-state matrix theorem.

**Market calibration is approximate.** It uses 399 circular block replicates of the 252 calibration origins, block length 2h, with common indices for all 18 statistics: nine corrected linear terms, six unique matrix entries, and three budget terms. Studentized replicate deviations form the simultaneous maximum max(z_B*,|z_A*|,−z_K*). Its nonnegative nominal 95% quantile supplies the respective lower, two-sided, and upper radii before the bias penalty and quadratic program. Coordinates with se≤1e−14 contribute zero standardized deviations; the original se is retained in the radii. [Circular block resampling](https://statistics.stanford.edu/technical-reports/circular-block-resampling-procedure-stationary-data) is established; this particular calibration is marked `nominal_only`. It has no finite-sample guarantee proved here under overlap, dependence, unknown δ, or future distribution change.

## 7. Maturity, information budgets, and scope of comparisons

For first test-year origin T, the 252 calibration origins end at T−h−1; the 126 validation origins end h+1 indices before calibration starts; training ends h+1 indices before validation starts. Thus the preceding role's last target index is strictly earlier than the next role's first origin. Calibration is 252 quote origins, not a fixed calendar year. Test origin and target must both lie within the declared year. All features use available history. Gate coefficients are frozen after calibration, although α_w(X) changes with observed history.

Matched controls use the same F0 and F1: empirical scalar and gate mixing, raw calibration, and raw calibration with the same joint-statistic budget. Frozen BOA and deployment-updated BOA are separate information budgets. Online feedback uses only matured, previously issued forecasts and their issued weights. Delayed adaptation does not automatically inherit the no-delay BOA regret theorem.

The quantile BOAG adapter uses 19 midpoint levels (j−0.5)/19, rearrangement, and the exact CRPS of the resulting equal-weight atomic distribution. It is not a complete profoc reproduction; tail discretization and rearrangement limit direct theoretical transfer. GARCH_Normal uses expected cumulative conditional variance in a Gaussian approximation, not the complete multistep GARCH predictive density. The fixed comparisons do not exhaust GAS, skew-t, or neural volatility models.

## 8. Direct neighbors and the provisional contribution boundary

| Original work | Established overlap and remaining distinction |
|---|---|
| [Berrisch–Ziel, CRPS Learning](https://arxiv.org/pdf/2102.00968v3); [Wintenberger, BOA](https://arxiv.org/pdf/1404.1356v5) | Distribution aggregation and second-order online learning already exist. Here mature paths modify calibration evidence for a fixed pair of experts. |
| [Hesterberg–Nelson, probability and quantile control variates](https://users.iems.northwestern.edu/~nelsonb/Publications/HesterbergNelson.pdf) | Probability control variates are established. The proposed target is the relative-CRPS linear coefficient. |
| [Sabate-Vidales et al., unbiased deep solvers](https://arxiv.org/pdf/1810.05094) | Predictable stochastic controls are established. Our reference may be wrong; validity instead requires an explicit true-process sign assumption. |
| [Pasupathy et al., estimated control means](https://doi.org/10.1080/0740817X.2011.610430) | Bias, control-mean error, and coefficient shrinkage are established. Our specific envelope uses conditional sign imbalance and pathwise absolute budget. |
| [Learn then Test](https://arxiv.org/pdf/2110.01052) | Simultaneous risk-based selection is established. The continuous gate bound here follows from exact quadratic geometry and a matrix envelope. |
| [Almeida et al., Appendix D](https://research.feedzai.com/oshuzuxa/2025/10/high_probability_risk_control_under_covariate_shift.pdf) | Control variates have already been used in risk calibration, including range and sample-splitting costs. Our control information is a path assumption rather than known-mean importance weights. |
| [Semi-supervised PPI risk control](https://arxiv.org/pdf/2412.11174); [active anytime risk control](https://arxiv.org/pdf/2406.10490) | Auxiliary predictions, residual corrections, and protected risk calibration are established. This proposal uses neither extra unlabeled observations nor known random labeling probabilities. |
| [Chen–Dobriban–Lee, group-theoretic data augmentation](https://jmlr.org/papers/volume21/20-163/20-163.pdf) | Symmetry averaging and variance reduction are established. The present operator is a sequential odd control with a predictable budget, not whole-path orbit averaging. |

The research question is whether this specific analytic reflection and budget improve finite-calibration update decisions after paying bias, range, and selection costs. The formulas alone establish neither that this is sufficiently new nor that it is useful. The method description must remain separate from the experimental decision about continuing its development.
