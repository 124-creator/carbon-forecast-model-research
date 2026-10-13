import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np

def fit(X,Y,method,lag=20):
    n,d=X.shape
    G=X.T@X
    inv=np.linalg.pinv(G,rcond=1e-10)
    B=inv@X.T@Y
    E=Y-X@B
    leverage=np.einsum('ij,jk,ik->i',X,inv,X)
    e=E/np.maximum(1-leverage[:,None],.1)
    N=e.T@(e*leverage[:,None])/n
    for k in range(1,min(lag,n-1)+1):
        w=np.einsum('ij,jk,ik->i',X[k:],inv,X[:-k])
        T=e[k:].T@(e[:-k]*w[:,None])/n
        N+=(1-k/(lag+1))*(T+T.T)
    N=(N+N.T)/2
    vals,U=np.linalg.eigh(N)
    N=(U*np.maximum(vals,0))@U.T
    S=B.T@G@B/n
    if method=='full':
        out=B
    elif method=='rw':
        out=np.zeros_like(B)
    else:
        noise=N if method=='corrected' else np.diag(np.diag(N)) if method=='diagonal' else np.zeros_like(N)
        vals,V=np.linalg.eigh(S-noise)
        v=V[:,-1:]
        # Exact unweighted LS on the selected output subspace.
        out=B@v@v.T
    return out,{'objective':float(np.mean((Y-X@out)**2)), 'noise_norm':float(np.linalg.norm(N)), 'signal_norm':float(np.linalg.norm(S)), 'rank':int(np.linalg.matrix_rank(out))}
