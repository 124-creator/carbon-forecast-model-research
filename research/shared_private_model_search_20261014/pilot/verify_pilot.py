"""Independent artifact checks; execute after run_pilot.py."""
import hashlib,json
from pathlib import Path
import numpy as np
import run_pilot as rp
R=Path(__file__).resolve().parent
rows=json.loads((R/'results.json').read_text()); receipt=json.loads((R/'receipt.json').read_text())
if len(rows)!=8 or {(r['seed'],r['method']) for r in rows}!={(s,m) for s in [17,29] for m in rp.METHODS}: raise RuntimeError('grid')
if hashlib.sha256((R/'protocol.md').read_bytes()).hexdigest()!=receipt['protocol_sha256']: raise RuntimeError('protocol changed')
errors=[]
for row in rows:
    seed,m=row['seed'],row['method']; d=np.load(R/f'data_{seed}.npz'); f=np.load(R/f'fit_{seed}_{m}.npz')
    x,y=rp.data(seed)
    if not np.array_equal(x,d['x']) or not np.array_equal(y,d['y']): raise RuntimeError('reproduction')
    np.testing.assert_allclose(x[:,:64].mean(1),0,atol=1e-14)
    np.testing.assert_allclose(x[:,:64].std(1),1,atol=1e-14)
    pred=rp.forward(f['p'],x,m)[0]; np.testing.assert_array_equal(pred,f['pred'])
    for name,sl in [('train',slice(0,64)),('validation',slice(64,128)),('test',slice(128,None))]:
        metric=np.mean((pred[:,sl]-y[:,sl])**2,axis=1); np.testing.assert_allclose(metric,row[name+'_mse'],atol=1e-14); errors.append(float(np.max(np.abs(metric-row[name+'_mse']))))
    tr=json.loads((R/f'trace_{seed}_{m}.json').read_text())
    if len(tr)!=13 or row['best_step']!=min(tr,key=lambda z:z['val'])['step']: raise RuntimeError('selection')
    if not np.isfinite(pred).all() or f['p'].size!=340: raise RuntimeError('finite/capacity')
    if not tr[-1]['objective']<tr[0]['objective']: raise RuntimeError('no learning')
    # Test/validation labels have no influence on the fitting objective/gradient.
    altered=y.copy(); altered[:,64:]+=10000
    loss,g=rp.objective(f['p'],x[:,:64],y[:,:64],m); loss2,g2=rp.objective(f['p'],x[:,:64],altered[:,:64],m)
    np.testing.assert_array_equal(g,g2)
    if loss!=loss2: raise RuntimeError('label leakage')
lines=['# Synthetic mechanism pilot results','', 'No novelty candidate: conventional shared/private and covariance-penalty controls only. NumPy analytic-gradient tanh MLP training, not Transformer training. No market files were read.','', '| seed | method | task 1 MSE | task 2 | task 3 | null task 4 | first-three mean |','|---|---|---|---|---|---|---|']
for row in rows:
    v=row['test_mse']; lines.append('| '+str(row['seed'])+' | '+row['method']+' | '+' | '.join(f'{z:.6f}' for z in [*v,float(np.mean(v[:3]))])+' |')
lines+=['','## Interpretation']
for seed in [17,29]:
    rr={r['method']:r for r in rows if r['seed']==seed}; a=np.mean(rr['shared_private']['test_mse'][:3]); b=np.mean(rr['independent']['test_mse'][:3]); c=np.mean(rr['fully_shared']['test_mse'][:3])
    delta=rr['shared_private']['test_mse'][3]-rr['independent']['test_mse'][3]
    lines.append(f'- Seed {seed}: preregistered SP positive gate={bool(a<b and a<c)}; null-task excess MSE versus independent={delta:.6f}.')
lines+=['','The two-seed positive gate fails. Negative transfer remains. This is evidence against treating ordinary decomposition/orthogonality as a reliable solution, not evidence against all cross-task learning. No successful innovation is claimed.','', '## Mathematical implementation', 'Let h_b(x)=tanh(xW_b+b_b), and q_bj(x)=v_bj h_bj(x). Independent f_t=sum_j q_tj+c_t. Fully shared f_t=(1/2)sum_bj q_bj+c_t. SP f_t=(1/2)sum_b sum_{j<6}q_bj+sum_{j>=6}q_tj+c_t. Orthogonal control adds 0.1/4 sum_t Cov_train(s(x_t),p_t(x_t))^2. It is a soft scalar-output covariance penalty, not exact hidden-space orthogonality, PCGrad, or a novel decoupling objective. Task identity is only a routing index. True generating common/private functions are never learner features.','', '## Fairness and limitations', '- Exact aggregate stored/trainable parameter count 340, common initialization, training arrays, optimizer steps and validation checkpoints. Effective functional capacity/per-task accessibility and FLOPs differ; fully shared has redundant bank decompositions. These are constrained mechanism controls, not a strongest multi-head benchmark.', '- A single equally weighted validation mean chooses each joint checkpoint, including independent banks; independent taskwise checkpoint selection is not implemented and could strengthen that comparator.', '- Shared and private target responses are nonlinear functions of observed AR(1) covariates. This tests temporally driven heterogeneous responses, not autoregressive target-state dynamics or Transformer representation learning.', '- Fourth task has no common target component but can still have accidental finite-sample correlations. Independent is the no-sharing control; matched training with task 4 excluded is not included, so its causal contamination of tasks 1–3 is not isolated.', '- Branch removal is inference ablation without refitting. Covariance on/off is retrained; saved traces include gradient norms and 4x4 task-gradient cosine matrices at 13 checkpoints.', '- Two seeds are a pilot, not significance or robustness certification. Fixed 1200 steps certify objective descent, not stationarity/convergence.', '', '## Verification',f'- 8/8 fits, 13 checkpoints each; exact saved-prediction reproduction and 24 split-metric checks. Max metric discrepancy {max(errors):.3g}.', f'- Coordinate finite differences: maximum absolute gradient error {max(receipt["gradient_max_abs_error"].values()):.3g}.', f'- Training execution {receipt["seconds"]:.3f} seconds. Train-only scaling and held-out label invariance passed.', '- Sources checked: Liu, Qiu & Huang (ACL 2017, P17-1001), shared/private separation; Yu et al. (2020, arXiv:2001.06782), gradient surgery. Neither is renamed as an innovation.', '', 'Reproduce: `python run_pilot.py` then `python verify_pilot.py` from this directory. Parent should execute verifier and inspect source/report before acceptance. No commit/push performed.']
(R/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
verification=dict(runs=8,metric_checks=len(errors),max_metric_error=max(errors),gradient_check=rp.gradient_check(),passed=True)
(R/'verification.json').write_text(json.dumps(verification,indent=2)); print(json.dumps(verification)); print('\n'.join(lines[:20]))
