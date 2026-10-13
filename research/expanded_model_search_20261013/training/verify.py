"""Independent saved-output audit; optional matched gate control, no fitting.
The Kalman-gate control was added AFTER seeing primary scores: secondary only.
"""
import json
import hashlib
from pathlib import Path
import numpy as np
import prototype as p

ROOT=Path(__file__).resolve().parent

def require(value,message):
    if not value: raise RuntimeError(message)

def main():
    rows=json.loads((ROOT/'ALL_RESULTS.json').read_text(encoding='utf-8'))
    require(len(rows)==36,'must retain 36 cells')
    controls=[]; checked=0; replay_checks=0
    for row in rows:
        mode,seed=row['mode'],row['seed']
        a=np.load(ROOT/f'pred_{mode}_{seed}.npz')
        x,release,y,pred=a['x'],a['release'],a['y'],a['pred']
        trace=json.loads((ROOT/f'trace_{mode}_{seed}.json').read_text(encoding='utf-8'))
        expected,_=p.generate(mode,seed)
        for actual,wanted in zip([x,release,y],expected):
            require(np.array_equal(actual,wanted),'synthetic reproducibility mismatch')
        require(np.all(np.isfinite(pred[300:1199])),'nonfinite deploy forecast')
        mse=np.mean((pred[600:1199]-y[600:1199,None])**2,axis=0)
        for j,m in enumerate(p.METHODS):
            require(np.isclose(mse[j],row['mse'][m],atol=1e-12,rtol=1e-12),'MSE mismatch')
        gated=np.full(len(y),np.nan); gate_trace=[]
        for audit,model in zip(trace['audit'],trace['estimates']):
            t=audit['t']; tr=np.array(audit['train']); val=np.array(audit['val'])
            require(np.all(release[tr]<=val[0]),'internal validation release leak')
            require(np.all(tr<val[0]),'internal validation target leak')
            require(np.all(release[val]<=t),'unreleased validation source')
            require(np.all(val+1<=t),'immature validation label')
            checked+=1
            v,r,rho,beta=model['kalman']; r=np.array(r)
            vp=np.array([beta*p.filter_at(x,release,int(s),v,r,rho) for s in val])
            g=p.gate(vp,y[val]); end=min(t+20,1199)
            gated[t:end]=g*pred[t:end,3]; gate_trace.append([t,g])
            # Fitted model predicts identically after replacing all unpublished
            # measurements with large values; predictions cannot use the future.
            m=model['joint']; m['r']=np.array(m['r'])
            xp=x.copy(); xp[release>t]=1e9
            z1=p.static_predict(x,release,t,m,model['rho'])
            z2=p.static_predict(xp,release,t,m,model['rho'])
            require(z1==z2,'candidate release leak')
            k1=p.filter_at(x,release,t,v,r,rho)
            k2=p.filter_at(xp,release,t,v,r,rho)
            require(k1==k2,'Kalman release leak'); replay_checks+=2
        kmse=float(np.mean((gated[600:1199]-y[600:1199])**2))
        controls.append(dict(mode=mode,seed=seed,mse=kmse,ratio=kmse/row['mse']['rw'],gate_trace=gate_trace))
        np.savez_compressed(ROOT/f'secondary_kalman_gate_{mode}_{seed}.npz',pred=gated)
    summary={mode:dict(mean_ratio=float(np.mean([r['ratio'] for r in controls if r['mode']==mode])),
                       pooled_mse=float(np.mean([r['mse'] for r in controls if r['mode']==mode])))
             for mode in ['signal','null','reliability_shift']}
    p.save_json('SECONDARY_MATCHED_GATE.json',dict(posthoc=True,not_primary=True,results=controls,summary=summary))
    receipt=json.loads((ROOT/'RECEIPT.json').read_text(encoding='utf-8'))
    require(receipt['protocol_sha256']==hashlib.sha256((ROOT/'PROTOCOL_ZH.md').read_bytes()).hexdigest(),'protocol modified since training')
    manifest={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in ROOT.iterdir() if f.is_file() and f.name not in ['VERIFICATION.json','MANIFEST.json','VERIFY.log','VERIFY_OPTIMIZED.log']}
    p.save_json('MANIFEST.json',manifest)
    p.save_json('VERIFICATION.json',dict(cells=len(rows),audited_refits=checked,unpublished_perturbation_checks=replay_checks,
                  synthetic_only=True,primary_mse_recomputed=True,protocol_hash_matched=True,secondary_control=summary))
    print(json.dumps(dict(cells=len(rows),audited_refits=checked,secondary=summary),indent=2))

if __name__=='__main__': main()
