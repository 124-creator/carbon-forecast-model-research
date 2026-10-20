"""Independent NHITS-style coordinate residual/interpolation model; no package edits."""
import math
import torch
from torch import nn
from torch.nn import functional as F

def bridge_matrix(n):
 i=torch.arange(1,n,dtype=torch.float64)[:,None]
 r=torch.arange(1,n,dtype=torch.float64)[None,:]
 q=(2/n)**.5*torch.sin(math.pi*i*r/n)
 d=torch.zeros(n,n,dtype=torch.float64); d[:,0]=1
 b=torch.zeros(n+1,n-1,dtype=torch.float64); b[1:n]=q
 d[:,1:]=b[1:]-b[:-1]
 e=torch.linalg.inv(d)
 return e,d

def ordinary(n,kind):
 if kind=='haar':
  rows=[torch.ones(n,dtype=torch.float64)/math.sqrt(n)]
  span=n
  while span>=2:
   for start in range(0,n,span):
    row=torch.zeros(n,dtype=torch.float64); row[start:start+span//2]=1/math.sqrt(span); row[start+span//2:start+span]=-1/math.sqrt(span); rows.append(row)
   span//=2
  e=torch.stack(rows)
 else:
  i=torch.arange(1,n+1,dtype=torch.float64)[:,None]; r=torch.arange(1,n+1,dtype=torch.float64)[None,:]
  e=(math.sqrt(2/(n+1))*torch.sin(math.pi*i*r/(n+1))).T
 return e,e.T

class Block(nn.Module):
 def __init__(self,L,H,pool,freq,maxpool=False):
  super().__init__(); self.L=L; self.H=H; self.pool=pool; self.maxpool=maxpool; self.knots=max(H//freq,1)
  from neuralforecast.models.nhits import NHITSBlock, _IdentityBasis
  basis=_IdentityBasis(L,H,'linear')
  self.official=NHITSBlock(input_size=L,h=H,n_theta=L+self.knots,mlp_units=[[32,32]],basis=basis,futr_input_size=0,hist_input_size=0,stat_input_size=0,n_pool_kernel_size=pool,pooling_mode='MaxPool1d' if maxpool else 'AvgPool1d',dropout_prob=0.,activation='ReLU')
 def forward(self,x):
  back,forecast=self.official(x,None,None,None)
  return back,forecast.squeeze(-1)

class Forecaster(nn.Module):
 def __init__(self,kind):
  super().__init__(); self.kind=kind
  transform='haar' if kind=='haar' else 'dst'
  e,d=bridge_matrix(8) if kind in ['bridge','no_bridge','no_route'] else ordinary(8,transform)
  self.register_buffer('E',e.float())
  for n in (8,4):
   _,d=bridge_matrix(n) if kind in ['bridge','no_bridge'] else ordinary(n,transform)
   self.register_buffer('D'+str(n),d.float())
  if kind.startswith('raw'):
   self.blocks=nn.ModuleList([Block(64,20,p,p,kind=='rawMax') for p in [4,2,1]])
  else:
   self.blocks=nn.ModuleList([Block(8,3,p,p) for p in [4,4,2,2,1,1,1,1]])
 def encode(self,x): return x.reshape(-1,8,8)@self.E.T
 def decode(self,c):
  return torch.cat([c[:,0]@self.D8.T,c[:,1]@self.D8.T,c[:,2,:4]@self.D4.T],dim=-1)
 def forward(self,x):
  if self.kind.startswith('raw'):
   residual=x; out=torch.zeros(x.shape[0],20,device=x.device,dtype=x.dtype)
   for b in self.blocks:
    back,f=b(residual); residual=residual-back; out=out+f
   return out.cumsum(-1)
  c=self.encode(x)
  if self.kind=='no_bridge': c=torch.cat([c[:,:,:1],torch.zeros_like(c[:,:,1:])],dim=-1)
  output=torch.stack([block(c[:,:,r])[1] for r,block in enumerate(self.blocks)],dim=-1)
  return self.decode(output).cumsum(-1)

