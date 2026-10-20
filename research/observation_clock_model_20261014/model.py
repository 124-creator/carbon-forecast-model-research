"""Apache-2.0. SegRNN architecture attribution: lss-1138/SegRNN,
commit 8e869ecfdf1daab3a0ba14d1d620796c1a5d2c4f.
MODIFIED: independent NumPy/autograd rewrite of univariate GRU/PMF core;
no torch, upstream loader, runner, weights or RevIN executed.
"""
import autograd.numpy as np
import numpy as onp
from autograd import grad
from autograd.misc.flatten import flatten

H=onp.array([1,5,20]); D=16

def initialize(seed):
    r=onp.random.RandomState(seed)
    def w(a,b): return r.normal(size=(a,b))/onp.sqrt(a)
    return dict(emb=w(5,D), eb=onp.zeros(D), wi=w(D,3*D), wh=w(D,3*D),
                bi=onp.zeros(3*D), bh=onp.zeros(3*D), pos=r.normal(size=(4,D)),
                out=w(D,5), ob=onp.zeros(5), gap=w(5,D),
                clock=w(1,D), q=w(D,9), qb=onp.zeros(9))

def step(p,x,h):
    a=np.dot(x,p['wi'])+p['bi']; b=np.dot(h,p['wh'])+p['bh']
    r=1/(1+np.exp(-(a[:,:D]+b[:,:D])))
    z=1/(1+np.exp(-(a[:,D:2*D]+b[:,D:2*D])))
    n=np.tanh(a[:,2*D:]+r*b[:,2*D:])
    return (1-z)*n+z*h

def predict(p,x,g,mode,representatives=None,mask=None):
    """Returns standardized increments, q and conditional increments; no future gaps API."""
    n=x.shape[0]; v=np.reshape(x-x[:,-1:],(n,12,5))
    e=np.dot(v,p['emb'])+p['eb']
    if mode!='original': e=e+np.dot(np.reshape(g,(n,12,5)),p['gap'])
    e=np.maximum(e,0); hidden=np.zeros((n,D))
    for t in range(12): hidden=step(p,e[:,t],hidden)
    logits=np.reshape(np.dot(hidden,p['q'])+p['qb'],(n,3,3))
    logits=logits-np.max(logits,axis=2,keepdims=True)
    q=np.exp(logits)/np.sum(np.exp(logits),axis=2,keepdims=True)
    queries=[]; conditional=[]
    for j,h in enumerate(H):
        query=np.tile(p['pos'][(h-1)//5],(n,1))
        for k in range(3):
            z=query
            if mode=='marginal':
                z=z+representatives[j,k]*p['clock'][0]
            hy=step(p,z,hidden)
            if mask is not None: hy=hy*mask
            y=np.dot(hy,p['out'])+p['ob']
            queries.append(y[:,(h-1)%5])
        conditional.append(np.stack(queries[-3:],axis=1))
    mu=np.stack(conditional,axis=1)
    return np.sum(q*mu,axis=2),q,mu

def loss(p,x,g,y,k,mode,reps,mask=None):
    pred,q,mu=predict(p,x,g,mode,reps,mask)
    if mode in ('marginal','auxiliary','no_condition'):
        chosen=np.sum(mu*np.eye(3)[k],axis=2)
        return .5*np.mean((chosen-y)**2)-np.mean(np.log(q+1e-12)*np.eye(3)[k])*3
    return .5*np.mean((pred-y)**2)

def train(x,g,y,k,mode,reps,seed,epochs=40,deadline=None):
    import time
    p=initialize(seed); flat,unflatten=flatten(p); m=np.zeros_like(flat); v=np.zeros_like(flat)
    derivative=grad(lambda f,xx,gg,yy,kk,mask:loss(unflatten(f),xx,gg,yy,kk,mode,reps,mask))
    r=onp.random.RandomState(seed); log=[]; stepnum=0; start=time.monotonic()
    for ep in range(epochs):
        for ids in onp.array_split(r.permutation(len(x)),max(1,len(x)//32)):
            if deadline is not None and time.monotonic()>deadline: raise TimeoutError('fixed global budget exhausted')
            mask=(r.rand(len(ids),D)>.1)/.9
            df=derivative(flat,x[ids],g[ids],y[ids],k[ids],mask)
            norm=onp.linalg.norm(df); df=df/max(1.,norm)+1e-4*flat
            stepnum+=1; m=.9*m+.1*df; v=.999*v+.001*df**2
            flat=flat-.001*(m/(1-.9**stepnum))/(np.sqrt(v/(1-.999**stepnum))+1e-8)
        log.append(float(loss(unflatten(flat),x,g,y,k,mode,reps)))
    return unflatten(flat),log,time.monotonic()-start
