"""Read-only saved-artifact audit. Does not fit or change pilot outputs."""
import json, hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
from run import generate, features, factory, dictionary
R=Path(__file__).resolve().parent
rows=json.loads((R/'results.json').read_text())
receipt=json.loads((R/'receipt.json').read_text())
def require(ok,msg):
    if not ok: raise RuntimeError(msg)
require(len(rows)==60 and len({(r['process'],r['seed'],r['reg'],r['model']) for r in rows})==60,'fit inventory')
require(hashlib.sha256((R/'protocol.json').read_bytes()).hexdigest()==receipt['protocol_sha256'],'protocol hash')
groups=defaultdict(list); maxpred=maxmetric=maxgrad=0.; details=[]
for r in rows:
    name=f"{r['process']}_{r['seed']}_{r['reg']}_{r['model']}"
    with np.load(R/f"data_{r['process']}_{r['seed']}.npz",allow_pickle=False) as d, np.load(R/(name+'.npz'),allow_pickle=False) as a:
        past,y=generate(r['seed'],r['process'])
        require(np.array_equal(past,d['past']) and np.array_equal(y,d['targets']),name+' generator')
        f=features(past); mu=f[:600].mean(0); sd=f[:600].std(0); x=(f-mu)/sd
        require(np.array_equal(x,d['features']) and np.array_equal(mu,d['mean']) and np.array_equal(sd,d['std']),name+' scaling')
        _,calc=factory(r['model'],x[:600],y[:600],r['reg'])
        pred=calc(a['parameters'],data=x,gradient=False)
        pe=float(np.max(np.abs(pred-a['prediction']))); maxpred=max(maxpred,pe); require(pe<=1e-12,name+' prediction')
        checks={'train_mse':np.mean((pred[:600]-y[:600])**2),'validation_mse':np.mean((pred[600:800]-y[600:800])**2),'test_mse':np.mean((pred[800:]-y[800:])**2),'zero_test_mse':np.mean(y[800:]**2),'initial':calc(a['initial_parameters'])[0],'final':calc(a['parameters'])[0]}
        for k,v in checks.items():
            e=abs(float(v)-r[k]); maxmetric=max(maxmetric,e); require(e<=1e-12,name+' '+k)
        _,g=calc(a['parameters']); gn=float(np.linalg.norm(g)); gi=float(np.max(np.abs(g)))
        require(abs(gn-r['gradient_norm'])<=1e-12,name+' gradient norm'); maxgrad=max(maxgrad,gi)
        obj=a['accepted_objectives']; require(len(obj)==r['accepted_steps']==r['iterations'],name+' steps')
        tail=(float(obj[-11]-obj[-1])/max(1.,abs(float(obj[-11])))) if len(obj)>10 else None
        detail=dict(name=name,model=r['model'],process=r['process'],reg=r['reg'],seed=r['seed'],iterations=r['iterations'],message=r['message'],success=r['success'],gradient_inf=gi,gradient_l2=gn,stationary_gtol=gi<=1e-6,last10_objective_drop_scaled=tail,test_ratio=r['test_mse']/r['zero_test_mse'],validation_zero_ratio=float(np.mean((pred[600:800]-y[600:800])**2)/np.mean(y[600:800]**2)))
        if r['model']=='structured_rate': detail['learned_half_lives']=(np.log(2)/dictionary(a['parameters'][-2:])[2]).tolist()
        details.append(detail); groups[r['model']].append(detail)
summary={k:dict(count=len(v),maxiter=sum('ITERATIONS REACHED LIMIT' in z['message'] for z in v),success=sum(z['success'] for z in v),stationary_gtol=sum(z['stationary_gtol'] for z in v),gradient_inf_range=[min(z['gradient_inf'] for z in v),max(z['gradient_inf'] for z in v)],last10_drop_range=[min(z['last10_objective_drop_scaled'] for z in v),max(z['last10_objective_drop_scaled'] for z in v)]) for k,v in groups.items()}
comparisons=[]
for process in ['no_signal','persistent_transient','asymmetric']:
    for reg in [.001,.01]:
        subset=[r for r in rows if r['process']==process and r['reg']==reg]
        means={kind:float(np.mean([r['test_mse'] for r in subset if r['model']==kind])) for kind in groups}
        comparisons.append(dict(process=process,reg=reg,mean_seed_test_mse=means,learned_vs_fixed_relative=means['structured_rate']/means['fixed_dictionary']-1,learned_vs_mlp_relative=means['structured_rate']/means['mlp']-1))
print(json.dumps(dict(verified_fits=len(rows),max_prediction_error=maxpred,max_metric_error=maxmetric,summary=summary,comparisons=comparisons,details=details,validation_trajectory_available=False),indent=2))
