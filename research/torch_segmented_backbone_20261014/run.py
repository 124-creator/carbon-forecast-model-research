import json,time,sys,subprocess,copy
from pathlib import Path
import numpy as np
import torch
from backbone import Backbone,mapping_check
OUT=Path(__file__).parent
def generate(seed,regime):
 r=np.random.RandomState(seed+2000);z=r.uniform(-1,1,384);x=np.arange(60)[None,:]*z[:,None]/60+r.normal(0,.015,(384,60));mean=np.abs(z)[:,None]*np.array([.2,.5,1.])[None,:]
 if regime=='null':mean=np.zeros((384,3))
 return x,mean+r.normal(0,.1,(384,3)),mean
def main():
 start=time.monotonic();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 check=mapping_check();(OUT/'MAPPING.json').write_text(json.dumps(check,indent=2));rows=[]
 for regime in ['nonlinear','null']:
  for seed in [7,17]:
   x,y,mean=generate(seed,regime);sx=max(np.std(x[:192]-x[:192,-1:]),.05);sy=np.maximum(np.std(y[:192],axis=0),.1);xx=torch.from_numpy(x/sx);yy=torch.from_numpy(y/sy)
   m=Backbone(seed);opt=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=.001);rng=np.random.RandomState(seed);best=float('inf');state=None;patience=0;logs=[];updates=0
   for ep in range(200):
    norms=[]
    for ids in np.array_split(rng.permutation(192),6):
     opt.zero_grad();loss=((m(xx[ids])-yy[ids])**2).mean();loss.backward();norms.append(float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.)));opt.step();updates+=1
    with torch.no_grad():train=float(((m(xx[:192])-yy[:192])**2).mean());valid=float(((m(xx[192:256])-yy[192:256])**2).mean())
    logs.append(dict(epoch=ep+1,train=train,valid=valid,gradient_norms=norms))
    if valid<best-1e-5:best=valid;state=copy.deepcopy(m.state_dict());patience=0;selected=ep+1
    else:patience+=1
    if patience>=25:break
   m.load_state_dict(state)
   with torch.no_grad():pred=m(xx).numpy()*sy
   mse=float(np.mean((pred[256:]-y[256:])**2));zero=float(np.mean(y[256:]**2));intercept=float(np.mean((y[:192].mean(0)-y[256:])**2))
   gate=mse<.5*zero if regime=='nonlinear' else mse<=1.1*zero
   row=dict(regime=regime,seed=seed,mse=mse,zero_mse=zero,intercept_mse=intercept,gate=gate,selected_epoch=selected,epochs=len(logs),updates=updates,history=logs)
   rows.append(row);np.savez(OUT/f'{regime}_{seed}.npz',prediction=pred,target=y,mean=mean);(OUT/'RESULTS.json').write_text(json.dumps(rows,indent=2));print({k:v for k,v in row.items() if k!='history'},flush=True)
 (OUT/'GATES.json').write_text(json.dumps(dict(pass_all=all(r['gate'] for r in rows),seconds=time.monotonic()-start),indent=2))
 (OUT/'requirements.lock.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
 (OUT/'ENVIRONMENT.json').write_text(json.dumps(dict(executable=sys.executable,python=sys.version,torch=torch.__version__,numpy=np.__version__,cuda_available=torch.cuda.is_available(),threads=torch.get_num_threads(),market_read=False),indent=2))
if __name__=='__main__':main()
