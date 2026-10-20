"""Apache-2.0 independent compact price-change SegRNN subset.
Attribution: lss-1138/SegRNN, 8e869ecfdf1daab3a0ba14d1d620796c1a5d2c4f.
Modified: sparse 1/5/20 queries, no RevIN/channel ID/full-file loader.
NumPy reference is verification-only; optimization uses nn.GRU and AdamW.
"""
import numpy as np
import torch
from torch import nn
D=16
MAP={'emb':('emb.weight',True),'eb':('emb.bias',False),'wi':('gru.weight_ih_l0',True),'wh':('gru.weight_hh_l0',True),'bi':('gru.bias_ih_l0',False),'bh':('gru.bias_hh_l0',False),'pos':('pos',False),'out':('out.weight',True),'ob':('out.bias',False)}
def initialize(seed):
 r=np.random.RandomState(seed)
 def w(a,b):return r.normal(size=(a,b))/np.sqrt(a)
 return dict(emb=w(5,D),eb=np.zeros(D),wi=w(D,3*D),wh=w(D,3*D),bi=np.zeros(3*D),bh=np.zeros(3*D),pos=r.normal(size=(2,D)),out=w(D,2),ob=np.zeros(2))
class Backbone(nn.Module):
 def __init__(self,seed=7):
  super().__init__();self.emb=nn.Linear(5,D);self.gru=nn.GRU(D,D,batch_first=True);self.pos=nn.Parameter(torch.empty(2,D));self.out=nn.Linear(D,2)
  self.double(); self.load_numpy(initialize(seed))
 def load_numpy(self,p):
  state=self.state_dict()
  for k,(name,tr) in MAP.items():state[name]=torch.from_numpy(p[k].T.copy() if tr else p[k].copy())
  self.load_state_dict(state)
 def forward(self,x):
  v=(x-x[:,-1:]).reshape(-1,12,5);_,h=self.gru(torch.relu(self.emb(v)));ys=[]
  for j in range(2):
   _,hy=self.gru(self.pos[j].reshape(1,1,D).expand(len(x),1,D),h);ys.append(self.out(hy[0]))
  return torch.stack([ys[0][:,0],ys[0][:,1],ys[1][:,1]],1)
def reference(p,x):
 def step(x,h):
  a=x@p['wi']+p['bi'];b=h@p['wh']+p['bh'];r=1/(1+np.exp(-(a[:,:D]+b[:,:D])));z=1/(1+np.exp(-(a[:,D:2*D]+b[:,D:2*D])))
  return (1-z)*np.tanh(a[:,2*D:]+r*b[:,2*D:])+z*h
 v=(x-x[:,-1:]).reshape(-1,12,5);e=np.maximum(v@p['emb']+p['eb'],0);h=np.zeros((len(x),D))
 for t in range(12):h=step(e[:,t],h)
 ys=[step(np.tile(p['pos'][j],(len(x),1)),h)@p['out']+p['ob'] for j in range(2)]
 return np.stack([ys[0][:,0],ys[0][:,1],ys[1][:,1]],1)
def mapping_check():
 p=initialize(7);x=np.random.RandomState(12).normal(size=(3,60));m=Backbone(7);y=m(torch.from_numpy(x));loss=(y*y).mean();loss.backward();errors={};eps=1e-5
 for k,(name,tr) in MAP.items():
  analytic=dict(m.named_parameters())[name].grad.numpy();analytic=analytic.T if tr else analytic
  numeric=np.zeros_like(p[k])
  for ix in np.ndindex(p[k].shape):
   old=p[k][ix];p[k][ix]=old+eps;a=np.mean(reference(p,x)**2);p[k][ix]=old-eps;b=np.mean(reference(p,x)**2);p[k][ix]=old;numeric[ix]=(a-b)/(2*eps)
  errors[k]=float(np.max(np.abs(numeric-analytic)))
 return dict(forward_max_error=float(np.max(np.abs(y.detach().numpy()-reference(p,x)))),gradient_max_error=max(errors.values()),parameter_gradient_errors=errors,gate_order='r,z,n; transpose wi/wh; separate input and hidden biases; reset after hidden affine')
