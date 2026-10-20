# Synthetic mechanism pilot results

No novelty candidate: conventional shared/private and covariance-penalty controls only. NumPy analytic-gradient tanh MLP training, not Transformer training. No market files were read.

| seed | method | task 1 MSE | task 2 | task 3 | null task 4 | first-three mean |
|---|---|---|---|---|---|---|
| 17 | independent | 0.311902 | 0.446739 | 0.683509 | 1.103749 | 0.480717 |
| 17 | fully_shared | 0.823749 | 1.072514 | 0.622399 | 1.365344 | 0.839554 |
| 17 | shared_private | 0.398438 | 0.490886 | 0.411217 | 1.169646 | 0.433514 |
| 17 | sp_orthogonal | 0.351911 | 0.449089 | 0.389692 | 1.205217 | 0.396898 |
| 29 | independent | 0.391799 | 0.267617 | 0.667935 | 0.864116 | 0.442450 |
| 29 | fully_shared | 0.754782 | 0.888505 | 0.707296 | 1.221676 | 0.783528 |
| 29 | shared_private | 0.394266 | 0.561581 | 0.678461 | 1.308565 | 0.544769 |
| 29 | sp_orthogonal | 0.406351 | 0.558250 | 0.669511 | 1.312659 | 0.544704 |

## Interpretation
- Seed 17: preregistered SP positive gate=True; null-task excess MSE versus independent=0.065897.
- Seed 29: preregistered SP positive gate=False; null-task excess MSE versus independent=0.444449.

The two-seed positive gate fails. Negative transfer remains. This is evidence against treating ordinary decomposition/orthogonality as a reliable solution, not evidence against all cross-task learning. No successful innovation is claimed.

## Mathematical implementation
Let h_b(x)=tanh(xW_b+b_b), and q_bj(x)=v_bj h_bj(x). Independent f_t=sum_j q_tj+c_t. Fully shared f_t=(1/2)sum_bj q_bj+c_t. SP f_t=(1/2)sum_b sum_{j<6}q_bj+sum_{j>=6}q_tj+c_t. Orthogonal control adds 0.1/4 sum_t Cov_train(s(x_t),p_t(x_t))^2. It is a soft scalar-output covariance penalty, not exact hidden-space orthogonality, PCGrad, or a novel decoupling objective. Task identity is only a routing index. True generating common/private functions are never learner features.

## Fairness and limitations
- Exact aggregate stored/trainable parameter count 340, common initialization, training arrays, optimizer steps and validation checkpoints. Effective functional capacity/per-task accessibility and FLOPs differ; fully shared has redundant bank decompositions. These are constrained mechanism controls, not a strongest multi-head benchmark.
- A single equally weighted validation mean chooses each joint checkpoint, including independent banks; independent taskwise checkpoint selection is not implemented and could strengthen that comparator.
- Shared and private target responses are nonlinear functions of observed AR(1) covariates. This tests temporally driven heterogeneous responses, not autoregressive target-state dynamics or Transformer representation learning.
- Fourth task has no common target component but can still have accidental finite-sample correlations. Independent is the no-sharing control; matched training with task 4 excluded is not included, so its causal contamination of tasks 1–3 is not isolated.
- Branch removal is inference ablation without refitting. Covariance on/off is retrained; saved traces include gradient norms and 4x4 task-gradient cosine matrices at 13 checkpoints.
- Two seeds are a pilot, not significance or robustness certification. Fixed 1200 steps certify objective descent, not stationarity/convergence.

## Verification
- 8/8 fits, 13 checkpoints each; exact saved-prediction reproduction and 24 split-metric checks. Max metric discrepancy 0.
- Coordinate finite differences: maximum absolute gradient error 3.81e-10.
- Training execution 9.421 seconds. Train-only scaling and held-out label invariance passed.
- Sources checked: Liu, Qiu & Huang (ACL 2017, P17-1001), shared/private separation; Yu et al. (2020, arXiv:2001.06782), gradient surgery. Neither is renamed as an innovation.

Reproduce: `python run_pilot.py` then `python verify_pilot.py` from this directory. Parent should execute verifier and inspect source/report before acceptance. No commit/push performed.