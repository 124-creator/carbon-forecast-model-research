"""NumPy float64 structural probe of the upstream SegRNN encoder.
No market files, weights, pretrained modules, or external datasets are read.
PyTorch gate ordering r,z,n and reset-after recurrent candidate are reproduced.
Ridge heads are trained; the common random encoder is intentionally frozen.
"""
from pathlib import Path
import json
import numpy as np
L,S,D=48,8,16

def sigmoid(x):
    return 1/(1+np.exp(-np.clip(x,-40,40)))

class Encoder:
    def __init__(self,seed=11):
        g=np.random.default_rng(seed)
        self.W=g.normal(0,1/np.sqrt(S),(D,S))
        self.b=g.normal(0,0.1,D)
        self.Wi=g.uniform(-1/np.sqrt(D),1/np.sqrt(D),(3*D,D))
        self.Wh=g.uniform(-1/np.sqrt(D),1/np.sqrt(D),(3*D,D))
        self.bi=g.uniform(-1/np.sqrt(D),1/np.sqrt(D),3*D)
        self.bh=g.uniform(-1/np.sqrt(D),1/np.sqrt(D),3*D)
    def count(self):
        return sum(a.size for a in [self.W,self.b,self.Wi,self.Wh,self.bi,self.bh])
    def windows(self,x):
        return np.stack([np.roll(x,-t,axis=-1)[...,:S] for t in range(L)],axis=-2)
    def embedding(self,w):
        return np.maximum(w@self.W.T+self.b,0)
    def gru(self,e):
        h=np.zeros(e.shape[:-2]+(D,))
        for t in range(e.shape[-2]):
            a=e[...,t,:]@self.Wi.T+self.bi
            b=h@self.Wh.T+self.bh
            r=sigmoid(a[...,:D]+b[...,:D])
            z=sigmoid(a[...,D:2*D]+b[...,D:2*D])
            n=np.tanh(a[...,2*D:]+r*b[...,2*D:])
            h=(1-z)*n+z*h
        return h
    def bank(self,x):
        e=self.embedding(self.windows(x))
        return np.stack([self.gru(e[...,p::S,:]) for p in range(S)],axis=-2)
    def hard(self,x):
        return self.gru(self.embedding(x.reshape(*x.shape[:-1],L//S,S)))
    def slide(self,x):
        # same finite observed interval, left-pad to preserve every observation
        a=np.pad(x,[(0,0)]*(x.ndim-1)+[(S-1,0)])
        w=np.stack([a[...,t:t+S] for t in range(L)],axis=-2)
        return self.gru(self.embedding(w))
    def group(self,x):
        return np.mean(np.stack([self.hard(np.roll(x,-p,axis=-1)) for p in range(L)],axis=-2),axis=-2)

def run_checks():
    e=Encoder(); g=np.random.default_rng(99); x=g.normal(size=(12,L))
    windows=e.windows(x); dense=e.embedding(windows)
    phases=np.stack([e.embedding(windows[:,p::S]) for p in range(S)],axis=1)
    restored=phases.transpose(0,2,1,3).reshape(12,L,D)
    # polyphase paths permute windows only; no added observed samples
    raw=np.stack([windows[:,p::S] for p in range(S)],axis=1)
    raw_restored=raw.transpose(0,2,1,3).reshape(12,L,S)
    shifted=np.roll(x,1,axis=-1)
    dense_shift=e.embedding(e.windows(shifted))
    # dense starts shift one position exactly; hard starts have no such one-step action
    hard_shift=e.embedding(shifted.reshape(12,L//S,S))
    hard=e.embedding(x.reshape(12,L//S,S))
    h=e.bank(x); q=g.normal(size=D)
    a=h@q; a=np.exp(a-a.max(axis=-1,keepdims=True)); a/=a.sum(axis=-1,keepdims=True)
    # Proposed learned pooling with a LINEAR decoder is literally weighted ensemble.
    v=g.normal(size=D); b=0.3
    pool=np.sum(a[...,None]*h,axis=-2)@v+b
    ensemble=np.sum(a*(h@v+b),axis=-1)
    c=e.count()+D+1
    # Explicitly injective ReLU segment encoding: positive/negative pair.
    Wpair=np.concatenate([np.eye(S),-np.eye(S)],axis=0)
    pair=np.maximum(windows@Wpair.T,0)
    recover=pair[...,:S]-pair[...,S:]
    e0=dense[:,::S,:]
    pair_hard=np.maximum(x.reshape(12,L//S,S)@Wpair.T,0)
    pair_shift=np.maximum(shifted.reshape(12,L//S,S)@Wpair.T,0)
    return {
      'injective_relu_reconstruction_max_error':float(np.max(abs(recover-windows))),
      'injective_hard_shift_defect':float(np.linalg.norm(pair_shift-pair_hard)/np.linalg.norm(pair_hard)),
      'multiphase_gru_vs_one_sliding_gru_error':float(np.max(abs(e.bank(x).mean(axis=1)-e.gru(dense)))),
      'hard_vs_polyphase_zero_error':float(np.max(abs(e.hard(x)-e.gru(e0)))), 
      'seed':11,'L':L,'segment_length':S,'hidden':D,'dtype':'float64',
      'polyphase_vs_sliding_embedding_max_error':float(np.max(abs(restored-dense))),
      'information_sorted_window_error':float(np.max(abs(raw_restored-windows))),
      'learned_pool_vs_ensemble_max_error':float(np.max(abs(pool-ensemble))),
      'hard_embedding_shift_defect':float(np.linalg.norm(hard_shift-hard)/np.linalg.norm(hard)),
      'dense_embedding_shift_defect':float(np.max(abs(dense_shift-np.roll(dense,1,axis=1)))),
      'full_group_invariance_error':float(np.max(abs(e.group(x)-e.group(shifted)))),
      'phase_mean_not_full_group_error':float(np.max(abs(e.bank(x).mean(axis=1)-e.bank(np.roll(x,S,axis=-1)).mean(axis=1)))),
      'parameter_counts':{k:c for k in ['hard','phase_mean','sliding','full_group']},
      'recurrent_steps_per_sample':{'hard':L//S,'phase_mean':L,'sliding':L,'full_group':L*(L//S)},
      'embedding_weight_rank':int(np.linalg.matrix_rank(e.W)),
      'normalization_note':'algebra checks use already-centered synthetic arrays; forecasting probes below have identical zero endpoint, so subtract_last is exactly identity',
    }

def data(seed,n):
    g=np.random.default_rng(seed)
    # One latent example supplies ALL positions, preserving label, noise, amplitude.
    label=g.choice([-1.,1.],n)
    amp=g.uniform(.8,1.2,n)
    base=g.normal(0,.008,(n,L)); base[:,-1]=0
    xs=[]
    for p in range(S):
        x=base.copy(); start=24+p
        x[:,start]+=amp; x[:,start+1]+=amp*label
        xs.append(x)
    return np.stack(xs,axis=1),label

def ridge(x,y,lam=.01):
    a=np.column_stack([x,np.ones(len(x))]); r=np.eye(a.shape[-1])*lam; r[-1,-1]=0
    return np.linalg.solve(a.T@a+r,a.T@y)

def task_probe():
    train,y=data(203,300); test,yt=data(204,200)
    results=[]
    # Three independent initializations, shared core tensors within each comparison.
    for seed in [11,22,33]:
        e=Encoder(seed)
        functions={'hard':e.hard,'phase_mean':lambda x:e.bank(x).mean(axis=-2),'sliding':e.slide,'full_group':e.group}
        for name,f in functions.items():
            ft=f(train[:,2]); flattened=test.reshape(-1,L); fs=f(flattened).reshape(200,S,D)
            head=ridge(ft,y)
            pred=fs@head[:-1]+head[-1]
            row={'seed':seed,'model':name,'train_phase':2,'train_n':300,'test_n_per_phase':200,
                 'mse_per_phase':np.mean((pred-yt[:,None])**2,axis=0).tolist(),
                 'classification_accuracy_per_phase':np.mean(np.sign(pred)==yt[:,None],axis=0).tolist(),
                 'within_example_prediction_std':float(np.mean(np.std(pred,axis=1))),
                 'worst_phase_mse':float(np.max(np.mean((pred-yt[:,None])**2,axis=0)))}
            results.append(row)
    return results

if __name__=='__main__':
    p=Path(__file__).resolve().parent
    checks=run_checks(); print(json.dumps(checks,indent=2))
    probe=task_probe()
    out={'checks':checks,'synthetic_probe':probe,
         'scope':'Random shared encoder + trained ridge probe; NOT end-to-end SegRNN/PMF, NOT market forecasting evidence. Circular paths only algebra/control; sliding uses finite zero-left-pad.',
         'decision':'REJECT: learned phase pooling with linear head is exact learned weighted multiphase ensemble; all phase embeddings equal ordinary shared sliding linear/ReLU filter.'}
    (p/'results.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    for r in probe: print(r['seed'],r['model'],'worst MSE',round(r['worst_phase_mse'],5),'phase SD',round(r['within_example_prediction_std'],5))
