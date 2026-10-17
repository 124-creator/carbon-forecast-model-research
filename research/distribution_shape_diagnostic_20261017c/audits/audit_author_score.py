"""Independent fixed-source two-point counterexample; no market/model fitting."""
import argparse
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--vendor-dir',required=True)
    ap.add_argument('--output',required=True)
    args=ap.parse_args()
    root=Path(args.vendor_dir).resolve()
    manifest=json.loads((root/'SOURCE_MANIFEST.json').read_text())
    item=next(f for f in manifest['files'] if f['path']=='discrete_distribution.py')
    path=root/item['path'];raw=path.read_bytes()
    sha256=hashlib.sha256(raw).hexdigest()
    git_blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    if sha256 != item['sha256'] or git_blob != item['git_blob_sha']:
        raise AssertionError('Author bytes differ from the fixed manifest')
    spec=importlib.util.spec_from_file_location('audited_author_distribution',path)
    author=importlib.util.module_from_spec(spec);spec.loader.exec_module(author)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    grid=[.05,.1,.25,.5,.75,.9,.95]
    rows=[];max_error=0.
    for p in grid:
        probability=torch.tensor(p,requires_grad=True)
        values=torch.tensor([[-1.,1.],[-1.,1.]])
        probs=torch.stack([probability,1-probability]).expand(2,2)
        truth=torch.tensor([[-1.],[1.]])
        distribution=author.DiscreteDistribution(values=values,probs=probs,validate_args=False)
        raw_scores=distribution.rps(truth)
        expected=raw_scores.mean()
        derivative=torch.autograd.grad(expected,probability)[0]
        # Independent pairwise law score, with truth P(-1)=P(1)=1/2.
        a=np.array([-1.,1.]);w=np.array([p,1-p])
        true_crps=[]
        for y in (-1.,1.):
            true_crps.append(float(sum(w[i]*abs(a[i]-y) for i in range(2))-.5*sum(w[i]*w[j]*abs(a[i]-a[j]) for i in range(2) for j in range(2))))
        errors=[abs(float(raw_scores[0])),abs(float(raw_scores[1])-p),abs(float(expected)-p/2),abs(float(derivative)-.5),abs(float(np.mean(true_crps))-(p*p+(1-p)**2))]
        max_error=max(max_error,*errors)
        if max(errors)>1e-14:
            raise AssertionError((p,errors))
        rows.append(dict(probability_at_minus_one=p,author_rps_at_minus_one=float(raw_scores[0].detach()),
            author_rps_at_plus_one=float(raw_scores[1].detach()),expected_author_rps=float(expected.detach()),
            expected_author_rps_derivative=float(derivative),expected_exact_crps=float(np.mean(true_crps))))
    true_row=next(r for r in rows if r['probability_at_minus_one']==.5)
    false_row=next(r for r in rows if r['probability_at_minus_one']==.1)
    assert false_row['expected_author_rps']<true_row['expected_author_rps']
    assert false_row['expected_exact_crps']>true_row['expected_exact_crps']
    receipt={'status':'PASS','completed_utc':datetime.now(timezone.utc).isoformat(),
        'claim':'The exact fixed DiscreteDistribution.rps function is not a proper score on the general class of finite laws. This is a function/version counterexample, not an assertion about every DeepNPTS version or the paper benchmark.',
        'repository':manifest['repository'],'commit':manifest['commit'],'license':manifest['license'],
        'upstream_path':item['upstream_path'],'function':'DiscreteDistribution.rps',
        'sha256':sha256,'git_blob_sha':git_blob,'source_bytes':len(raw),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'truth':{'atoms':[-1,1],'probabilities':[.5,.5]},
        'forecast':{'atoms':[-1,1],'probabilities':'[p,1-p], 0<p<1'},
        'identities':{'author_expected_loss':'p/2','author_expected_derivative':'1/2','exact_crps_expected_loss':'p^2+(1-p)^2'},
        'grid_cases':rows,'checks':37,'maximum_abs_error':max_error,
        'runtime':{'numpy':np.__version__,'torch':torch.__version__},
        'market_reads':0,'confirmation_reads':0,'market_fits':0,'optimizer_steps':0,'network_requests':0,
        'limits':['The fixed-support counterexample is not the same parameter family as the study weighted-zero-mean/unit-variance31-atom adapter.',
                  'It therefore does not predict the task-adapted arms numerical ranking; both actual training arms and independent proper scoring are still required.',
                  'Replacing this score by the established exact CRPS is not new forecasting methodology.',
                  'The original source and the frozen running study have not been edited.']}
    Path(args.output).write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:receipt[k] for k in ('status','checks','maximum_abs_error','commit','sha256','git_blob_sha')},ensure_ascii=False))


if __name__=='__main__':
    main()
