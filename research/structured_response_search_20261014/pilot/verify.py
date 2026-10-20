"""Independent artifact verification; deliberately stricter market permission."""
import json, hashlib
from pathlib import Path
import numpy as np
from run import dictionary, generate, factory
R=Path(__file__).resolve().parent
rows=json.loads((R/'results.json').read_text()); receipt=json.loads((R/'receipt.json').read_text())
if len(rows)!=60 or len({(r['process'],r['seed'],r['reg'],r['model']) for r in rows})!=60: raise RuntimeError('missing fits')
if hashlib.sha256((R/'protocol.json').read_bytes()).hexdigest()!=receipt['protocol_sha256']: raise RuntimeError('protocol changed')
max_error=0.
for r in rows:
    name=f"{r['process']}_{r['seed']}_{r['reg']}_{r['model']}"
    d=np.load(R/f"data_{r['process']}_{r['seed']}.npz"); a=np.load(R/(name+'.npz'))
    past,target=generate(r['seed'],r['process'])
    if not np.array_equal(past,d['past']) or not np.array_equal(target,d['targets']): raise RuntimeError('generator replay mismatch')
    mse=float(np.mean((a['prediction'][800:]-target[800:])**2))
    if abs(mse-r['test_mse'])>1e-12: raise RuntimeError('metric mismatch')
    _,calc=factory(r['model'],d['features'][:600],target[:600],r['reg'])
    pred=calc(a['parameters'],data=d['features'],gradient=False)
    max_error=max(max_error,float(np.max(np.abs(pred-a['prediction']))))
    # Learner API accepts measurements/targets, never generator parameters.
    if r['gradient_check']>1e-6 or not np.isfinite(a['parameters']).all(): raise RuntimeError('invalid learning')
rng=np.random.default_rng(4); X=rng.normal(size=(20,6)); W=rng.normal(size=(6,3)); D=dictionary()[0]
equiv=float(np.max(np.abs((X@W)@D-X@(W@D))))
primary=[r for r in rows if r['model']=='structured_rate' and r['reg']==.001]
converged=all(r['success'] for r in primary)
out=dict(verified_fits=len(rows),prediction_replay_max_abs_error=max_error,fixed_linear_absorption_error=equiv,empirical_gate_passed=receipt['gate_passed'],primary_optimizer_converged=converged,market_allowed=receipt['gate_passed'] and converged,reason='All primary structured fits hit iteration limit. Objective decrease is real but convergence is not certified; conservatively withhold market access.')
(R/'verification.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
lines=['# Structured direct-response pilot','', 'No innovation claim: exponential/damped response dictionaries are established. Fixed linear coefficients are absorbable into an ordinary linear output; fixed nonlinear coefficients are a constrained MLP output.', '', 'One candidate: 6 observable block statistics -> 4 tanh units -> trend + two amplitudes, with 2 global bounded rates. Candidate 45 parameters; fixed dictionary 43; MLP 78; linear 70; linear dictionary 21. Shared information, same iterations, not equal FLOPs or parameter counts.', '', 'Positive controls are deliberately aligned with the candidate family: unknown generating coefficients, observed history only, no latent-state oracle. Independent episodes in chronological order are a capability gate, not a contiguous market replay. Null future observations are independent noise, not cumulative random-walk changes.', '', 'All 60 runs retained. Empirical gates pass; 48/60 optimizers hit maxiter, including every structured candidate. No convergence claim and NO market access. No historical MLE/linear scores copied. No future data/2022 access. Parent should verify source and report; no commit/push performed.', '', '|process|seed|reg|model|test MSE / zero|optimizer success|','|---|---:|---:|---|---:|---|']
for r in rows: lines.append(f"|{r['process']}|{r['seed']}|{r['reg']}|{r['model']}|{r['test_mse']/r['zero_test_mse']:.6f}|{r['success']}|")
lines+=['', 'Outcome: nonlinear maps learn the designed positive controls better than linear outputs, but learned rates do not show a meaningful or consistent advantage over fixed dictionary or unstructured MLP. This supports a trainable baseline, not a new mechanism or market improvement.', '', 'Verification: '+json.dumps(out), '', 'Runtime receipt: '+json.dumps(receipt)]
(R/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
