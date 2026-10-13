"""Verification of a rejected formulation, NOT new-candidate training.
No market files, archive contents, or pretrained weights are read.
"""
from pathlib import Path
import json, time, hashlib, sys
import numpy as np
import scipy
from scipy.optimize._numdiff import approx_derivative

ROOT=Path(__file__).resolve().parent
P=json.loads((ROOT/'PROTOCOL.json').read_text(encoding='utf-8'))
t0=time.perf_counter()
rng=np.random.default_rng(P['test_seed'])
n,d,m,H=P['verification_shapes']; lam=P['lambda']
X=rng.normal(size=(n,d)); Y=rng.normal(size=(n,H)); theta=rng.normal(size=(d,m))*.2
weights=np.array([1.,.5,2.,3.])

def feature(t): return np.tanh(X@t.reshape(d,m))
def profile(t):
 A=feature(t); W=np.linalg.solve(A.T@A+lam*np.eye(m),A.T@Y)
 R=A@W-Y
 return float(np.sum(R*R*weights)+lam*np.sum(W*W*weights)),W,A,R
f,W,A,R=profile(theta)
# Same-capacity one-hidden-layer MLP: identical forward map at same parameters.
mlp=np.tanh(X@theta)@W
prediction_error=float(np.max(np.abs(mlp-A@W)))
# Fixed theta reduces exactly to a fixed-feature ridge kernel.
K=A@A.T
pred_dual=K@np.linalg.solve(K+lam*np.eye(n),Y)
primal_dual=float(np.max(np.abs(pred_dual-A@W)))
# Any arbitrary output head satisfies the profiled-objective decomposition.
V=rng.normal(size=(m,H)); D=V-W
arbitrary=float(np.sum((A@V-Y)**2*weights)+lam*np.sum(V*V*weights))
extra=float(sum(weights[h]*(D[:,h]@(A.T@A+lam*np.eye(m))@D[:,h]) for h in range(H)))
profile_identity=abs(arbitrary-f-extra)
# Envelope gradient (with lambda>0, unique W) versus finite differences.
grad=2*X.T@(((R*weights)@W.T)*(1-A*A))
# Complex-step analytic continuation avoids subtractive cancellation.
fd=np.empty((d,m))
for j in range(d):
 for k in range(m):
  z=theta.astype(complex); z[j,k]+=1e-20j
  AZ=feature(z); WZ=np.linalg.solve(AZ.T@AZ+lam*np.eye(m),AZ.T@Y)
  RZ=AZ@WZ-Y
  fd[j,k]=np.imag(np.sum(RZ*RZ*weights)+lam*np.sum(WZ*WZ*weights))/1e-20
grad_error=float(np.max(np.abs(grad-fd)))
# Linear learned bottleneck has no new function class versus ordinary reduced rank.
B=rng.normal(size=(m,H)); linear_error=float(np.max(np.abs((X@theta)@B-X@(theta@B))))
# Non-orthogonal feature-coordinate changes preserve predictions iff penalty transforms.
T=np.eye(m)+.07*rng.normal(size=(m,m)); AT=A@T
WT=np.linalg.solve(AT.T@AT+lam*T.T@T,AT.T@Y)
coordinate_error=float(np.max(np.abs(A@W-AT@WT)))
# Plain isotropic ridge after rescaling changes estimator, not architecture.
Wwrong=np.linalg.solve(AT.T@AT+lam*np.eye(m),AT.T@Y)
wrong_penalty_difference=float(np.max(np.abs(A@W-AT@Wwrong)))
# Alternative problem's mechanism: E[f(X)] != f(E[X]) for noisy quadratic map.
x=.4; noise_variance=.09
true_second_mean=x**4+noise_variance
iterated_mean=x**4
jensen_gap=true_second_mean-iterated_mean
checks={'mlp_prediction_equivalence':prediction_error,'fixed_kernel_primal_dual':primal_dual,
 'profile_objective_decomposition':profile_identity,'envelope_gradient_complex_step':grad_error,
 'linear_factorization_equivalence':linear_error,'transformed_penalty_coordinate_equivalence':coordinate_error}
if any(v>P['absolute_tolerance'] for v in checks.values()): raise RuntimeError(checks)
result={'configuration':P,'versions':{'numpy':np.__version__,'scipy':scipy.__version__},
 'checks':checks,'isotropic_penalty_coordinate_change_prediction_difference':wrong_penalty_difference,
 'alternative_quadratic_noise_example':{'x':x,'variance':noise_variance,'true_second_conditional_mean':true_second_mean,'iterated_first_mean':iterated_mean,'gap':jensen_gap},
 'decision':'REJECT: same-capacity MLP function class and profiled objective; variable projection is an existing optimizer',
 'candidate_training_runs':0,'market_files_opened':0,'benchmark_scores':None,
 'runtime_seconds':time.perf_counter()-t0,
 'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(ROOT/('EQUIVALENCE_RESULTS_OPTIMIZED.json' if sys.flags.optimize else 'EQUIVALENCE_RESULTS.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
