import copy,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent));import run as base
from prefix_loader import read_prefix
MODES=['zero','level']

class Model(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.GRU(2,16,batch_first=True);self.head=nn.Linear(16,20)
    def forward(self,x):return self.head(self.encoder(x)[0][:,-1])

def inputs(p,orig,sd,center,scale,mode):
    d=np.array([np.diff(p[t-60:t+1]) for t in orig])/sd
    v=np.array([p[t-59:t+1] for t in orig]);v=(v-center)/scale if mode=='level' else np.zeros_like(v)
    return torch.tensor(np.stack([d,v],axis=-1),dtype=torch.float32)

def fit(p,end,mode,seed):
    ids,_,y,tr,va,sd=base.arrays(p,end)
    center=float(np.mean(p[ids[tr]]));scale=max(float(np.std(p[ids[tr]])),1e-6)
    x=inputs(p,ids,sd,center,scale,mode);target=torch.tensor(y/sd,dtype=torch.float32)
    torch.manual_seed(seed);m=Model();opt=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=.001)
    with torch.no_grad():best=float(((m(x[va])-target[va])**2).mean())
    state=copy.deepcopy(m.state_dict());selected=0;wait=0;hist=[]
    for step in range(200):
        opt.zero_grad();loss=((m(x[tr])-target[tr])**2).mean();loss.backward();gn=float(nn.utils.clip_grad_norm_(m.parameters(),1));opt.step()
        with torch.no_grad():vl=float(((m(x[va])-target[va])**2).mean())
        hist.append([float(loss.detach()),vl,gn])
        if not np.isfinite(hist[-1]).all():raise ValueError('nonfinite')
        if vl<best-1e-6:best=vl;selected=step+1;state=copy.deepcopy(m.state_dict());wait=0
        else:wait+=1
        if wait>=20:break
    m.load_state_dict(state)
    meta=dict(mode=mode,seed=seed,end=end,epochs=len(hist),selected=selected,n=len(ids),history=hist,max_training_target=int(ids[tr[-1]]+20),validation_first=int(ids[va[0]]),max_validation_target=int(ids[va[-1]]+20),parameters=sum(v.numel() for v in m.parameters()),scale_max_source=int(ids[tr[-1]]))
    return m,(sd,center,scale),meta

def predict(m,scales,p,orig,mode):
    sd,center,scale=scales
    with torch.no_grad():return m(inputs(p,orig,sd,center,scale,mode)).numpy().astype(float)*sd

def fixtures():
    torch.manual_seed(17);a=Model();torch.manual_seed(17);b=Model()
    if not all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()):raise ValueError('same initialization')
    p=np.arange(100,dtype=float);x=inputs(p,[60,61],1.,30.,4.,'zero')
    if x.shape!=(2,60,2) or torch.count_nonzero(x[:,:,1]):raise ValueError('zero input')
    y=inputs(p+100,[60,61],1.,130.,4.,'level');z=inputs(p,[60,61],1.,30.,4.,'level')
    if not torch.equal(y,z):raise ValueError('translation equivariance')
    return dict(passed=True,same_seed_initial_parameters=True,zero_channel=True,translation_equivariance=True)

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);fx=fixtures()
    rows,audit=read_prefix(R.parents[2]/'data/parsed/figshare_28682348/guangzhou_reported.csv',950,'2021-12-31')
    p=np.array([float(v['author_reported_price']) for v in rows]);dates=np.array([v['date'] for v in rows]);orig=np.array([t for t in range(459,930) if dates[t][:4]==dates[t+20][:4]]);y=np.array([p[t+1:t+21]-p[t] for t in orig]);years=np.array([v[:4] for v in dates[orig]])
    saved=dict(origins=orig,truth=y,years=years);logs=[];causal=[];local=R/'local_only';local.mkdir(exist_ok=True);elapsed=0.
    for mode in MODES:
        for seed in [17,29]:
            saved[f'{mode}_{seed}']=np.full(y.shape,np.nan)
            for block in range(12):
                if elapsed>600:raise RuntimeError('budget incomplete')
                end=459+40*block;ix=np.where((orig-459)//40==block)[0];start=time.perf_counter();m,sc,row=fit(p,end,mode,seed);seconds=time.perf_counter()-start;elapsed+=seconds
                saved[f'{mode}_{seed}'][ix]=predict(m,sc,p,orig[ix],mode);logs.append(dict(block=block,seconds=seconds,**row))
                if seed==17 and block==0:
                    q=p.copy();q[end:]+=10000;n,ns,nrow=fit(q,end,mode,seed)
                    if sc!=ns or row!=nrow or not all(torch.equal(v,n.state_dict()[k]) for k,v in m.state_dict().items()) or not np.array_equal(predict(m,sc,p,[458],mode),predict(n,ns,q,[458],mode)):raise ValueError('future refit')
                    causal.append(dict(mode=mode,seed=seed,end=end,passed=True,scope='representative fresh training and both fitted scales'))
                np.savez_compressed(local/'PREDICTIONS.npz',**saved);(R/'FIT_LOGS.json').write_text(json.dumps(logs));print(mode,seed,block,row['selected'],round(seconds,2),flush=True)
    scores={};seed_scores={}
    for mode in MODES:
        cells=[];seed_scores[mode]={}
        for yr in ['2020','2021']:
            mask=years==yr
            for h in [1,5,20]:
                ratio=np.mean([np.mean((saved[f'{mode}_{s}'][mask,h-1]-y[mask,h-1])**2) for s in [17,29]])/np.mean(y[mask,h-1]**2);cells.append(dict(year=yr,h=h,ratio=float(ratio)))
        scores[mode]=dict(score=float(np.mean([v['ratio'] for v in cells])),cells=cells)
        for seed in [17,29]:seed_scores[mode][str(seed)]=float(np.mean([np.mean((saved[f'{mode}_{seed}'][years==yr,h-1]-y[years==yr,h-1])**2)/np.mean(y[years==yr,h-1]**2) for yr in ['2020','2021'] for h in [1,5,20]]))
    gate=scores['level']['score']<scores['zero']['score'] and all(np.mean([v['ratio'] for v in scores['level']['cells'] if v['year']==yr])<np.mean([v['ratio'] for v in scores['zero']['cells'] if v['year']==yr]) for yr in ['2020','2021'])
    result=dict(scores=scores,seed_scores=seed_scores,fits=len(logs),origins=len(orig),fit_seconds=elapsed,fixtures=fx,causal=causal,audit=audit,development_gate=gate,novelty='ordinary input diagnostic; source acquisition failed; no novelty claim',parameter_counts={m:next(v['parameters'] for v in logs if v['mode']==m) for m in MODES},initialization='fresh random initialization for every block; no checkpoint loading')
    (R/'RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
