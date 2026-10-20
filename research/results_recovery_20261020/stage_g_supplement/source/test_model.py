"""Synthetic numerical checks only; no market model scores."""
import json, sys
import numpy as np
import torch
from anchored_response import Response, solve, rates, DT, Config, fit

def numpy_basis(x,s):
 sig=lambda a:1/(1+np.exp(-a))
 def rr(a):
  low=.98*sig(a[...,0]);high=low+(.98-low)*sig(a[...,1]);return np.stack((low,high),-1)
 raw=np.asarray(s['state']['raw']);r=rr(raw);z=np.column_stack((np.log1p(abs(x[:,-1,0])),x[:,-1,1]*x[:,-1,2]))/s['state']['context_scale']
 if s['kind']=='price_only':z=z[:,:1]
 q=2*np.tanh(z@np.asarray(s['state']['gate']).T)
 forward=rr(raw[None]+q) if s['kind'] in ('duration','price_only') else np.broadcast_to(r,(len(x),2))
 amp=np.exp(q) if s['kind']=='amplitude' else np.ones_like(q)
 state=np.zeros((len(x),2))
 for t in range(64):state=state*r+(1-r)*x[:,t,0,None]
 out=np.zeros((len(x),3,3));out[:,:,0]=[1,5,20]
 for i,h in enumerate((1,5,20)):
  accum=np.zeros_like(forward)
  for step in range(1,h+1):accum+=forward**step
  out[:,i,1:]=accum*state*amp
 return out

def main():
 torch.set_num_threads(1);rng=np.random.default_rng(19)
 xx=rng.normal(size=(80,64,3));xx[:,:,2]=1;x=torch.tensor(xx,dtype=DT)
 m=Response();m.gate.data[:]=torch.tensor([[.2,-.1],[.1,.3]],dtype=DT)
 y=torch.tensor(rng.normal(size=(80,3)),dtype=DT);v=y.square().mean(0);a=.1
 beta,obj,_=solve(m.basis(x),y,v,a);loss=obj+.1*((m.raw-rates(torch.zeros(2,dtype=DT))).square().sum()+m.gate.square().sum())
 loss.backward();analytic=torch.cat([p.grad.flatten() for p in m.parameters()]).numpy();numeric=[]
 def value():
  return float((solve(m.basis(x),y,v,a)[1]+.1*((m.raw-rates(torch.zeros(2,dtype=DT))).square().sum()+m.gate.square().sum())).detach())
 for p in m.parameters():
  for k in range(p.numel()):
   orig=float(p.flatten()[k].detach())
   with torch.no_grad():p.flatten()[k]=orig+1e-5
   hi=value()
   with torch.no_grad():p.flatten()[k]=orig-1e-5
   lo=value()
   with torch.no_grad():p.flatten()[k]=orig
   numeric.append((hi-lo)/2e-5)
 gap=float(np.max(abs(analytic-numeric)));assert gap<1e-7
 basis_gap=float(np.max(abs(numpy_basis(xx,m.export())-m.basis(x).detach().numpy())));assert basis_gap<1e-12
 phi=m.basis(x).detach().numpy();aa=(phi/np.sqrt(v.numpy())[None,:,None]).reshape(-1,3);bb=(y.numpy()/np.sqrt(v.numpy())).reshape(-1);ss=np.maximum(np.sqrt((aa*aa).mean(0)),1e-8)
 independent=np.linalg.lstsq(np.vstack((aa/ss/np.sqrt(len(aa)),np.sqrt(a)*np.eye(3))),np.r_[bb/np.sqrt(len(aa)),np.zeros(3)],rcond=None)[0]/ss
 beta_gap=float(np.max(abs(independent-beta.detach().numpy())));assert beta_gap<1e-10
 sm,srec=fit('static',x,y,v,Config(.1,.01),epochs=10)
 wm,wrec=fit('duration',x,y,v,Config(.1,.01,.1,160),start=sm.raw.detach().tolist(),scaled=True,epochs=0)
 anchor_gap=float((sm(x)-wm(x)).detach().abs().max());assert anchor_gap<1e-12
 restored=Response.restore(wm.export());replay_gap=float((wm(x)-restored(x)).detach().abs().max());assert replay_gap==0
 print(json.dumps({'status':'PASS_SYNTHETIC_ONLY','gradient_gap':gap,'basis_gap':basis_gap,'ridge_svd_beta_gap':beta_gap,'warm_zero_equals_fitted_static_gap':anchor_gap,'checkpoint_replay_gap':replay_gap,'synthetic_gradient_steps':10,'market_fits':0}))
if __name__=='__main__':main()
