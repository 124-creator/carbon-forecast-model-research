"""Ordinary fixed-node piecewise-linear regression splines, not novel."""
import numpy as np
from scipy.linalg import solve

def features(x,state):
 z=(x-state['mu'])/state['sd']; cols=[np.ones(len(x)),*z.T]
 if state['kind']!='linear':
  for j in range(z.shape[1]):
   for k in state['knots'][j]:cols.append(np.maximum(z[:,j]-k,0))
 if state['kind']=='interaction':
  # One pair, rank <=2 coefficient matrix: fixed two-dimensional marginals,
  # four freely estimated tensor coefficients (no nonconvex factorization).
  a=np.column_stack([z[:,0],np.maximum(z[:,0],0)])-state['am']
  b=np.column_stack([z[:,1],np.maximum(z[:,1],0)])-state['bm']
  t=np.einsum('ni,nj->nij',a,b).reshape(len(x),4)
  main=np.column_stack(cols)
  cols.extend((t-main@state['projection']).T)
 return np.column_stack(cols)

def fit(x,y,kind,alpha):
 mu=x.mean(0);sd=x.std(0);sd=np.maximum(sd,1e-8);z=(x-mu)/sd
 state=dict(mu=mu,sd=sd,kind=kind,knots=np.quantile(z,[.1,.3,.5,.7,.9],axis=0).T)
 if kind=='interaction':
  state['am']=np.column_stack([z[:,0],np.maximum(z[:,0],0)]).mean(0)
  state['bm']=np.column_stack([z[:,1],np.maximum(z[:,1],0)]).mean(0)
  state['kind']='additive';main=features(x,state);state['kind']=kind
  a=np.column_stack([z[:,0],np.maximum(z[:,0],0)])-state['am'];b=np.column_stack([z[:,1],np.maximum(z[:,1],0)])-state['bm']
  t=np.einsum('ni,nj->nij',a,b).reshape(len(x),4)
  state['projection']=np.linalg.lstsq(main,t,rcond=1e-10)[0]
 A=features(x,state);center=A[:,1:].mean(0);scale=np.maximum(A[:,1:].std(0),1e-8)
 A[:,1:]=(A[:,1:]-center)/scale
 penalty=np.ones(A.shape[1]);penalty[0]=0
 if kind=='interaction':penalty[-4:]=4
 G=A.T@A+len(x)*alpha*np.diag(penalty);rhs=A.T@y
 coef=solve(G,rhs,assume_a='pos')
 state.update(center=center,scale=scale,coef=coef,alpha=alpha,gradient_norm=float(np.linalg.norm((G@coef-rhs)/len(x))),parameters=int(coef.size),effective_df=float(np.trace(solve(G,A.T@A,assume_a='pos'))),training_mse=float(np.mean((A@coef-y)**2)))
 return state

def predict(state,x):
 A=features(x,state);A[:,1:]=(A[:,1:]-state['center'])/state['scale'];return A@state['coef']
