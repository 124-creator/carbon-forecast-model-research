import json
import numpy as np

def decompose(inputs):
    means=inputs.mean(axis=-1,keepdims=True)
    centered=inputs-means
    std=np.sqrt(np.var(centered,axis=-1,keepdims=True)+1e-7)
    return centered/std,means,std

rng=np.random.default_rng(17)
x=rng.normal(size=(2,60,1));y=x+np.linspace(0,3,60)[None,:,None]
a,am,av=decompose(x);b,bm,bv=decompose(y)
if np.any(a!=0) or np.any(b!=0) or not np.array_equal(av,bv):raise ValueError('single channel counterexample')
if np.array_equal(am,bm):raise ValueError('statistics should retain varying information')
z=rng.normal(size=(2,60,3));c,cm,cv=decompose(z)
if not np.any(c!=0):raise ValueError('multichannel positive control')
print(json.dumps(dict(passed=True,scope='numpy transcription of reviewed last-axis decomposition only; not executing author network',single_channel_normalized_max=float(np.max(abs(a))),statistics_changed=not np.array_equal(am,bm),multichannel_nonzero=True)))
