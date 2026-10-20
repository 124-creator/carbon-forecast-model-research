from pathlib import Path
import time,json,hashlib,copy,sys
import numpy as np
import torch
from core import *
ROOT=Path(__file__).parent
START=time.monotonic(); LIMIT=1200
TRUE_K=np.interp(np.arange(13),np.arange(0,13,2),[1,.95,.85,.45,.2,.05,0])
def save(name,obj): (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def generate(family,seed):
    r=np.random.default_rng(seed);n=2640;eta=r.normal(size=n);u=np.zeros(n);noise=np.zeros(n);state=0.
    if family=='G1':u=r.normal(0,.35,n);noise=r.normal(0,.1,n)
    if family=='G2':
        noise=r.normal(0,.1,n)
        for j in range(n):
            state=.8*state+r.normal(0,.4)
            if r.random()<.05:
                shock=r.normal();p=1/(1+np.exp(-state))
                if r.random()<p:u[j]+=shock
                else:eta[j]+=shock
    if family=='G5':
        u=r.normal(0,.6,n);noise=r.normal(0,.2,n)
        for j in range(1,n,37):u[j]=-u[j-1]
    y=eta.cumsum()+np.convolve(u,TRUE_K)[:n]+noise
    return y[240:], {'kernel':TRUE_K.tolist(),'variances':[1,.1225,.01] if family=='G1' else [1,.36,.04]}
def dataset(y,scale,cap,window=60):
    origins=np.arange(window-1,len(y)-1)
    ds=np.array([np.diff(y[t-window+1:t+1])/scale for t in origins]);d=torch.tensor(ds);f=features(d,cap)
    targets=np.zeros((len(origins),3));mask=np.zeros_like(targets,dtype=bool)
    for i,h in enumerate([1,5,10]):
        ok=origins+h<len(y);targets[ok,i]=(y[origins[ok]+h]-y[origins[ok]])/scale;mask[ok,i]=True
    return origins,d,f,torch.tensor(targets),torch.tensor(mask)
def masks(orig,base):
    out={}
    for name,lo,hi in [('train',59,1200),('val',1210,1800),('test',1810,2400)]:
        out[name]=base & torch.tensor((orig[:,None]>=lo)&(orig[:,None]+np.array([1,5,10])<hi))
    return out
def loss(pred,target,mask):
    return torch.stack([((pred[:,j]-target[:,j])[mask[:,j]]**2).mean() for j in range(3)]).mean()
def metrics(p,t,m):
    return [float(((p[:,j]-t[:,j])[m[:,j]]**2).mean()) for j in range(3)]
def fit(model,d,f,t,ms,tag):
    opt=torch.optim.Adam(model.parameters(),lr=.001);trace=[];best=float('inf');checkpoint=None;predbest=None
    initial=float(loss(model(d,f),t,ms['train'])); trainrows=ms['train'].any(1);rows=ms['val'].any(1)
    for epoch in range(500):
        if time.monotonic()-START>LIMIT:break
        opt.zero_grad();p=model(d[trainrows],f[trainrows]);l=loss(p,t[trainrows],ms['train'][trainrows]);l.backward()
        grad=float(torch.sqrt(sum((x.grad*x.grad).sum() for x in model.parameters())));opt.step()
        record={'epoch':epoch+1,'train':float(l),'gradient':grad}
        if epoch%10==0 or epoch==499:
            with torch.no_grad():v=float(loss(model(d[rows],f[rows]),t[rows],ms['val'][rows]))
            record['validation']=v
            if np.isfinite(v) and v<best:best=v;checkpoint=copy.deepcopy(model.state_dict())
        trace.append(record)
    if checkpoint is None:raise RuntimeError('no validated checkpoint')
    laststate=copy.deepcopy(model.state_dict());torch.save({'last':laststate,'selected':checkpoint},ROOT/(tag+'.pt'))
    opt.zero_grad();lastloss=loss(model(d[trainrows],f[trainrows]),t[trainrows],ms['train'][trainrows]);lastloss.backward()
    terminalgrad=float(torch.sqrt(sum((x.grad*x.grad).sum() for x in model.parameters())))
    vals=[x['validation'] for x in trace if 'validation' in x][-5:];stable=len(vals)==5 and (max(vals)-min(vals))/max(np.mean(vals),1e-12)<=.02 and float(lastloss)<initial
    model.load_state_dict(checkpoint)
    with torch.no_grad():pred=model(d,f)
    save(tag+'_trace.json',trace)
    return pred,{'initial_train':initial,'last_train':float(lastloss),'last_gradient':terminalgrad,'stable_learning':bool(stable),'epochs':len(trace),'selected_validation':best,'parameters':sum(x.numel() for x in model.parameters()),'optimizer_sufficient':False}
def fit_mle(d,f,t,ms,raw,tag):
    # Exact stationary joint likelihood of ALL training differences, not horizon likelihood.
    route=Route('constant');lv=nn.Parameter(torch.tensor([0.,-2.,-4.]));opt=torch.optim.Adam([route.v,lv],lr=.03)
    train=torch.tensor(raw);n=len(train);best=float('inf');selected=None;trace=[]
    for epoch in range(100):
        if time.monotonic()-START>LIMIT:break
        opt.zero_grad();k=route.kernel();v=lv.exp();C=covariance(k,v,n);chol=torch.linalg.cholesky(C)
        nll=(2*chol.diag().log().sum()+train@torch.cholesky_solve(train[:,None],chol).squeeze(1))/n
        nll.backward();g=float(torch.sqrt((route.v.grad**2).sum()+(lv.grad**2).sum()));opt.step()
        with torch.no_grad():p=gaussian_predict(d,route.kernel(),lv.exp());val=float(loss(p,t,ms['val']))
        trace.append({'epoch':epoch+1,'joint_nll':float(nll),'gradient':g,'validation':val})
        if val<best:best=val;selected=(route.v.detach().clone(),lv.detach().clone())
    last=(route.v.detach().clone(),lv.detach().clone());torch.save({'last':last,'selected':selected},ROOT/(tag+'.pt'))
    opt.zero_grad();C=covariance(route.kernel(),lv.exp(),n);ch=torch.linalg.cholesky(C);l=(2*ch.diag().log().sum()+train@torch.cholesky_solve(train[:,None],ch).squeeze(1))/n;l.backward()
    terminalgrad=float(torch.sqrt((route.v.grad**2).sum()+(lv.grad**2).sum()));vals=[x['validation'] for x in trace][-5:]
    stable=len(vals)==5 and (max(vals)-min(vals))/np.mean(vals)<=.02 and float(l)<trace[0]['joint_nll']
    route.v.data.copy_(selected[0]);lv.data.copy_(selected[1]);save(tag+'_trace.json',trace)
    with torch.no_grad():p=gaussian_predict(d,route.kernel(),lv.exp())
    return p,{'epochs':len(trace),'parameters':8,'last_gradient':terminalgrad,'stable_learning':bool(stable),'optimizer_sufficient':False,'likelihood':'exact joint stationary Gaussian training differences','kernel':route.kernel().detach().tolist(),'variances':lv.exp().detach().tolist()}
def diagnostics(model,d,f):
    out={};worstinf=0;worstspec=0;worstcond=0;maxe=0;reconstruction=0
    with torch.no_grad():
        for i in range(0,len(d),64):
            p,e,T,q=model(d[i:i+64],f[i:i+64],True);inv=torch.linalg.solve_triangular(T,torch.eye(d.shape[1]).expand(len(T),-1,-1),upper=False)
            inf=inv.abs().sum(2).amax(1);index=int(inf.argmax());worstinf=max(worstinf,float(inf.max()));worstspec=max(worstspec,float(torch.linalg.matrix_norm(inv[index],ord=2)));worstcond=max(worstcond,float(torch.linalg.cond(T[index])))
            maxe=max(maxe,float(e.abs().max()));reconstruction=max(reconstruction,float((T@e[:,:,None]-d[i:i+64,:,None]).abs().max()))
    out.update(inverse_infinity_upper_bound=worstinf,inverse_spectral_sample_lower_bound=worstspec,condition_at_batch_infinity_max=worstcond,max_abs_innovation=maxe,reconstruction_max=reconstruction)
    # Every inverse infinity norm <=10 certifies all spectral norms? sqrt(||A||1||A||inf) is needed.
    with torch.no_grad():
        ub=0
        for i in range(0,len(d),64):
            _,_,T,_=model(d[i:i+64],f[i:i+64],True);inv=torch.linalg.solve_triangular(T,torch.eye(d.shape[1]).expand(len(T),-1,-1),upper=False)
            ub=max(ub,float(torch.sqrt(inv.abs().sum(1).amax(1)*inv.abs().sum(2).amax(1)).max()))
        delta=torch.zeros_like(d[:8]);delta[:,::3]=1e-6;changed=d[:8]+delta;dp=model(changed,features(changed,float(f[:,:,0].max())))-model(d[:8],f[:8]);out['perturb_forecast_gain']=float(dp.norm()/delta.norm())
    out['all_origin_spectral_upper_bound']=ub;out['inverse_gate']=ub<=10
    return out

def run():
    save('RUN_START.json',{'protocol_sha256':hashlib.sha256((ROOT/'PROTOCOL_ZH.md').read_bytes()).hexdigest(),'python':sys.executable,'torch':torch.__version__,'budget_seconds':LIMIT,'seeds':[7,17],'market_read':False})
    results=[]
    for family in ['G0','G1','G2','G5']:
      for seed in [7,17]:
        y,truth=generate(family,seed);scale=float(np.std(np.diff(y[:1200])));cap=float(np.quantile(np.abs(np.diff(y[:1200]))/scale,.995));orig,d,f,t,m=dataset(y,scale,cap);ms=masks(orig,m)
        np.savez_compressed(ROOT/f'{family}_{seed}_synthetic.npz',y=y,origins=orig,targets=t.numpy(),**{name:mask.numpy() for name,mask in ms.items()})
        predictions={};p=torch.zeros_like(t);predictions['rw']=p.numpy();results.append({'family':family,'seed':seed,'method':'rw','test_h':metrics(p,t,ms['test']),'val_h':metrics(p,t,ms['val'])})
        for mode in ['full','constant','age','evidence','small','mle']:
            if time.monotonic()-START>LIMIT:save('RESULTS.json',results);save('INCOMPLETE.json',{'next':[family,seed,mode]});return
            torch.manual_seed(seed);tag=f'{family}_{seed}_{mode}';tick=time.monotonic()
            if mode=='mle':p,receipt=fit_mle(d,f,t,ms,np.diff(y[:1200])/scale,tag)
            else:
                model=Small() if mode=='small' else Route(mode);p,receipt=fit(model,d,f,t,ms,tag)
                if mode!='small':receipt['diagnostics']=diagnostics(model,d,f)
            receipt.update(family=family,seed=seed,method=mode,test_h=metrics(p,t,ms['test']),val_h=metrics(p,t,ms['val']),runtime_seconds=time.monotonic()-tick)
            predictions[mode]=p.numpy();results.append(receipt);save('RESULTS.json',results);print(tag,receipt['test_h'],receipt['runtime_seconds'],flush=True)
        if family in ['G1','G5']:
            v=torch.tensor(truth['variances'])/scale**2;k=torch.tensor(TRUE_K);po=gaussian_predict(d,k,v);predictions['linear_oracle']=po.numpy();results.append({'family':family,'seed':seed,'method':'linear_oracle_DIAGNOSTIC','test_h':metrics(po,t,ms['test'])})
            o120,d120,f120,t120,m120=dataset(y,scale,cap,120);ms120=masks(o120,m120);p120=gaussian_predict(d120,k,v)
            results.append({'family':family,'seed':seed,'method':'linear_oracle120_DIAGNOSTIC','test_h':metrics(p120,t120,ms120['test'])})
            np.savez_compressed(ROOT/f'{family}_{seed}_oracle120.npz',origins=o120,pred=p120.numpy(),targets=t120.numpy(),mask=ms120['test'].numpy())
        np.savez_compressed(ROOT/f'{family}_{seed}_predictions.npz',**predictions)
        save('RESULTS.json',results)
    save('COMPLETE.json',{'rows':len(results),'runtime_seconds':time.monotonic()-START})
if __name__=='__main__':run()
