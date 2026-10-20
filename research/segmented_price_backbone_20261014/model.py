"""Apache-2.0; independent SegRNN GRU/PMF mathematical subset.
Attribution lss-1138/SegRNN commit 8e869ecfdf1daab3a0ba14d1d620796c1a5d2c4f.
Modified: strictly prices only, compact active queries, overlapping-lag GRU control.
"""
import autograd.numpy as np
import numpy as onp
from autograd import grad
from autograd.misc.flatten import flatten
D=16
H=onp.array([1,5,20])
def initialize(seed):
 r=onp.random.RandomState(seed)
 def w(a,b): return r.normal(size=(a,b))/onp.sqrt(a)
 return dict(emb=w(5,D),eb=onp.zeros(D),wi=w(D,3*D),wh=w(D,3*D),bi=onp.zeros(3*D),bh=onp.zeros(3*D),pos=r.normal(size=(2,D)),out=w(D,2),ob=onp.zeros(2))
def step(p,x,h):
 a=np.dot(x,p['wi'])+p['bi']; b=np.dot(h,p['wh'])+p['bh']
 r=1/(1+np.exp(-(a[:,:D]+b[:,:D]))); z=1/(1+np.exp(-(a[:,D:2*D]+b[:,D:2*D])))
 return (1-z)*np.tanh(a[:,2*D:]+r*b[:,2*D:])+z*h
def predict(p,x,mode):
 v=x-x[:,-1:]; n=len(x)
 if mode=='segmented': v=np.reshape(v,(n,12,5))
 elif mode=='unsegmented': v=np.stack([v[:,i:i+5] for i in range(56)],axis=1)
 else: raise ValueError(mode)
 e=np.maximum(np.dot(v,p['emb'])+p['eb'],0); h=np.zeros((n,D))
 for t in range(e.shape[1]): h=step(p,e[:,t],h)
 ys=[np.dot(step(p,np.tile(p['pos'][j],(n,1)),h),p['out'])+p['ob'] for j in range(2)]
 return np.stack([ys[0][:,0],ys[0][:,1],ys[1][:,1]],axis=1)
def loss(p,x,y,mode):
 mask=onp.isfinite(y); safe=onp.where(mask,y,0.)
 errors=(predict(p,x,mode)-safe)**2*mask
 return np.mean(np.sum(errors,axis=0)/np.sum(mask,axis=0))
def train(x,y,mode,seed,deadline,epochs=60):
 import time
 p=initialize(seed); f,u=flatten(p); m=onp.zeros_like(f); v=m.copy(); rng=onp.random.RandomState(seed)
 g=grad(lambda ff,xx,yy:loss(u(ff),xx,yy,mode)); initial=float(loss(p,x,y,mode)); logs=[]; k=0
 first_norm=None
 for ep in range(epochs):
  for ids in onp.array_split(rng.permutation(len(x)),8):
   if time.monotonic()>deadline: raise TimeoutError('15 minute global budget')
   df=g(f,x[ids],y[ids]); norm=float(onp.linalg.norm(df))
   if first_norm is None: first_norm=norm
   df=df/max(1,norm)+1e-4*f; k+=1; m=.9*m+.1*df; v=.999*v+.001*df**2
   f-=.001*(m/(1-.9**k))/(onp.sqrt(v/(1-.999**k))+1e-8)
  logs.append(float(loss(u(f),x,y,mode)))
 return u(f),dict(initial=initial,history=logs,first_gradient_norm=first_norm,updates=k,parameters=len(f))
