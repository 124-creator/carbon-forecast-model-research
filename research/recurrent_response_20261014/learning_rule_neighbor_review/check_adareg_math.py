import json
import numpy as np

def covariance_update(w,left,lower=.001,upper=1000.):
    p=w.shape[1];q=w.shape[0]
    gram=w.T@(left@left.T)@w
    u,s,_=np.linalg.svd(gram)
    with np.errstate(divide='ignore'):v=np.clip(p/s,lower,upper)
    right=(u*np.sqrt(v))@u.T
    gram=w@(right@right.T)@w.T
    u,s,_=np.linalg.svd(gram)
    with np.errstate(divide='ignore'):v=np.clip(q/s,lower,upper)
    left=(u*np.sqrt(v))@u.T
    return left,right

rng=np.random.default_rng(17);w=rng.normal(size=(20,16));left,right=covariance_update(w,np.eye(20))
if not np.isfinite(left).all() or not np.isfinite(right).all():raise ValueError('nonfinite')
for v in [left,right]:
    ev=np.linalg.eigvalsh(v@v.T)
    if min(ev)<.001-1e-8 or max(ev)>1000+1e-8:raise ValueError('bounds')
def penalty(a):return np.sum((left@a@right)**2)
g=2*left.T@left@w@right@right.T
checks=[]
for i,j in [(0,0),(5,7),(19,15)]:
    delta=np.zeros_like(w);delta[i,j]=1e-5
    fd=(penalty(w+delta)-penalty(w-delta))/(2e-5)
    if not np.isclose(fd,g[i,j],rtol=1e-5,atol=1e-6):raise ValueError('gradient')
    checks.append(dict(row=i,column=j,difference=float(abs(fd-g[i,j]))))
lz,rz=covariance_update(np.zeros((20,16)),np.eye(20))
if not np.isfinite(lz).all() or not np.isfinite(rz).all():raise ValueError('zero weight')
print(json.dumps(dict(passed=True,scope='independent NumPy transcription of sequential covariance update and frozen-covariance penalty; not author demo execution or forecasting',shape=list(w.shape),gradient_checks=checks,zero_weight_finite=True)))
