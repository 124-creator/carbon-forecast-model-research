"""Design-only checks; no market data or forecasting training."""
import hashlib
import json
import math
import random
from pathlib import Path
ROOT = Path(__file__).resolve().parent

def cumulative(h, m, u, rho):
    return h*m + (rho**h-1)*u

def run():
    rng=random.Random(20261014)
    loss_errors=[]
    for _ in range(1000):
        e,ep,lam=(rng.uniform(-4,4),rng.uniform(-4,4),rng.random()*3)
        lhs=0.5*(e*e+ep*ep)+lam*(ep-e)**2
        common=(e+ep)/math.sqrt(2); difference=(ep-e)/math.sqrt(2)
        rhs=0.5*common**2+(0.5+2*lam)*difference**2
        loss_errors.append(abs(lhs-rhs))
    assert max(loss_errors)<1e-12
    rows=[]
    for h in [1,2,4,8,16,32]:
        m,u,rho=0.1,2.0,0.8
        assert math.isclose(cumulative(h,m,u,rho),sum(m+(rho-1)*rho**(j-1)*u for j in range(1,h+1)),abs_tol=1e-12)
        positive=cumulative(h,0,1,0.9)
        negative=cumulative(h,0,-1,0.5)
        rows.append({'h':h,'trend_response_unit':float(h),'positive_deviation_response':positive,'negative_deviation_response':negative,'asymmetry_sum':positive+negative})
    assert abs(rows[-1]['asymmetry_sum'])>1e-3
    assert abs(rows[-1]['positive_deviation_response'])>abs(rows[0]['positive_deviation_response'])
    # Reanchoring controls: each scenario uses its own last price.
    last,last_prime=100,101
    y,y_prime=102,101.5
    correct=(y_prime-last_prime)-(y-last)
    wrong=(y_prime-last)-(y-last)
    assert correct==-1.5 and wrong==-0.5
    search=json.loads((ROOT/'search_actual.json').read_text(encoding='utf-8'))
    assert search['query_count']==8 and len(search['queries'])==8
    receipts=[]
    for name in ['primary_receipts.json','additional_receipts.json']:
        receipts.extend(json.loads((ROOT/name).read_text(encoding='utf-8')))
    verified=[]
    for receipt in receipts:
        if receipt.get('status')==200 and receipt.get('file'):
            path=ROOT/receipt['file']
            assert path.is_file()
            actual=hashlib.sha256(path.read_bytes()).hexdigest()
            assert actual==receipt['sha256'],str(path)
            verified.append(receipt['name'])
    result={'status':'passed','scope':'algebra and retrieved-artifact integrity only; no model trained','max_pair_loss_identity_error':max(loss_errors),'identity_draws':len(loss_errors),'response_rows':rows,'anchoring_control':{'correct':correct,'incorrect_shared_anchor':wrong},'query_count':len(search['queries']),'successful_searches':sum(q.get('result',{}).get('success',False) for q in search['queries']),'source_hashes_checked':len(verified),'sources_checked':verified}
    (ROOT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    run()
