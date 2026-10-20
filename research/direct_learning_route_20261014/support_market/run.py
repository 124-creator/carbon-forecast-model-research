"""Isolated child-support adaptation; author source files unchanged."""
import sys,json,time,hashlib,contextlib,io
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
B=R.parent/'baseline'
sys.path.insert(0,str(B));from runner import training_data,model_signature
import mbtr.mbtr as author
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(R.parents[1]/'torch_segmented_backbone_20261014'))
from prefix_loader import read_prefix
OriginalTree=author.Tree

class SupportTree(OriginalTree):
    support=40
    time_boxes=0
    def _fit_node(self,x,y,node,learning_rate=1.,hessian=None):
        G=y.sum(0);H=hessian.sum(0)
        best=float(self.loss.eval_optimal_loss(G**2,H)); choice=None
        if len(x)>=2*self.support:
            for i in range(x.shape[1]-1):
                qs=np.unique(np.quantile(x[:,i],np.linspace(1/self.n_q,1-1/self.n_q,self.n_q)))
                for q in qs:
                    left=x[:,i]<q;right=~left
                    if min(int(left.sum()),int(right.sum()))<self.support:continue
                    if self.time_boxes and min(len(np.unique((x[left,-1]//20).astype(int))),len(np.unique((x[right,-1]//20).astype(int))))<self.time_boxes:continue
                    gl=y[left].sum(0);hl=hessian[left].sum(0)
                    cost=float(self.loss.eval_optimal_loss(gl**2,hl)+self.loss.eval_optimal_loss((G-gl)**2,H-hl))
                    if cost<best:best=cost;choice=(i,float(q),left,right)
        self.g.nodes[node]['loss']=best
        self.g.nodes[node]['leaf']=choice is None
        if choice is None:
            self.g.nodes[node]['response']=self.loss.eval_optimal_response(G,H)*learning_rate
        else:
            i,q,left,right=choice
            self.g.nodes[node]['variable']=i;self.g.nodes[node]['level']=q
            child=self._add_new_node(node)
            self._fit_node(x[left],y[left],child,learning_rate,hessian[left])
            self._fit_node(x[right],y[right],child+1,learning_rate,hessian[right])

def fit(x,y,kind):
    SupportTree.support=80 if kind=='strict80' else 40
    SupportTree.time_boxes=4 if kind=='time40' else 0
    author.Tree=SupportTree
    try:
        m=author.MBT(n_boosts=100,early_stopping_rounds=100,val_ratio=0,learning_rate=.05,n_q=10,min_leaf=40,loss_type='mse',lambda_weights=.1,lambda_leaves=.1,verbose=0,refit=False)
        with threadpool_limits(limits=1),contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):m.fit(x,y)
    finally:author.Tree=OriginalTree
    return m

def support_check(m,x,kind):
    minimum=80 if kind=='strict80' else 40
    nleaf=0; low=len(x);boxes=100000
    for tree in m.trees:
        masks=tree._find_regions(x,np.zeros((len(x),20)))
        for mask in masks:
            n=int(mask.sum());low=min(low,n);nleaf+=1
            b=len(np.unique((x[mask,-1]//20).astype(int)));boxes=min(boxes,b)
            if n<minimum:raise ValueError('leaf support violated')
            if kind=='time40' and b<4:raise ValueError('time support violated')
    return {'leaves':nleaf,'minimum_support':low,'minimum_boxes':boxes,'retained_trees':len(m.trees)}

rows,audit=read_prefix(R.parents[2]/'data/parsed/figshare_28682348/guangzhou_reported.csv',950,'2021-12-31')
p=np.array([float(row['author_reported_price']) for row in rows]);dates=np.array([row['date'] for row in rows]);origins=np.array([t for t in range(459,930) if dates[t][:4]==dates[t+20][:4]])
if len(origins)!=451:raise ValueError('origins')
truth=np.array([p[t+1:t+21] for t in origins]);rw=np.repeat(p[origins,None],20,1)
preds={k:np.full((451,20),np.nan) for k in ['strict40','time40','strict80']}
receipts=[];causal=[]
if (R/'FIT_RECEIPTS.json').exists():
    receipts=json.loads((R/'FIT_RECEIPTS.json').read_text())
    with np.load(R/'local_only/PREDICTIONS.npz',allow_pickle=False) as old:
        if not np.array_equal(old['origins'],origins) or not np.array_equal(old['truth'],truth):raise ValueError('resume data')
        for kind in preds:preds[kind]=old[kind].copy()
source=B/'vendor/mbtr/mbtr.py';before=hashlib.sha256(source.read_bytes()).hexdigest()
for block in range(12):
    end=459+40*block;ids=np.where((origins-459)//40==block)[0];x,y,s=training_data(p,end);x=np.column_stack([x,s]);xp=np.column_stack([np.array([np.diff(p[t-60:t+1]) for t in origins[ids]]),origins[ids]])
    for kind in preds:
        if any(row['kind']==kind and row['block']==block for row in receipts):
            if not np.isfinite(preds[kind][ids]).all():raise ValueError('resume mismatch')
            continue
        start=time.perf_counter();m=fit(x,y,kind);elapsed=time.perf_counter()-start
        preds[kind][ids]=m.predict(xp)+p[origins[ids],None]
        cap=support_check(m,x,kind);receipts.append(dict(kind=kind,block=block,fit_end=end,train_n=len(s),latest_label=int(s[-1]+20),seconds=elapsed,**cap))
        if block==0:
            q=p.copy();q[end:]+=10000
            a,b,c=training_data(q,end);n=fit(np.column_stack([a,c]),b,kind)
            support_check(n,x,kind)
            if model_signature(m)!=model_signature(n) or not np.array_equal(m.predict(x[-3:]),n.predict(x[-3:])):raise ValueError('future refit')
            causal.append(dict(kind=kind,fit_end=end,passed=True))
        local=R/'local_only';local.mkdir(exist_ok=True)
        np.savez_compressed(local/'PREDICTIONS.npz',origins=origins,truth=truth,rw=rw,years=np.array([d[:4] for d in dates[origins]]),**preds)
        (R/'FIT_RECEIPTS.json').write_text(json.dumps(receipts,indent=2));print(kind,block,round(elapsed,2),cap,flush=True)
if not causal:
    x,y,s=training_data(p,459);x=np.column_stack([x,s]);q=p.copy();q[459:]+=10000
    a,b,c=training_data(q,459)
    m=fit(x,y,'strict40');n=fit(np.column_stack([a,c]),b,'strict40')
    if model_signature(m)!=model_signature(n) or not np.array_equal(m.predict(x[-3:]),n.predict(x[-3:])):raise ValueError('resumed future refit')
    causal.append(dict(kind='strict40',fit_end=459,passed=True))
if hashlib.sha256(source.read_bytes()).hexdigest()!=before:raise ValueError('source changed')
scores={}
for kind,pred in preds.items():
    if not np.isfinite(pred).all():raise ValueError('incomplete')
    cells=[]
    for year in ['2020','2021']:
        mask=np.array([d[:4]==year for d in dates[origins]])
        ratios=np.mean((pred[mask]-truth[mask])**2,0)/np.mean((rw[mask]-truth[mask])**2,0)
        cells.extend([dict(year=year,h=h,ratio=float(ratios[h-1])) for h in [1,5,20]])
    scores[kind]=dict(score=float(np.mean([c['ratio'] for c in cells])),cells=cells)
(R/'RESULTS.json').write_text(json.dumps(dict(scores=scores,causal_refits=causal,source_hash=before,novelty='unverified; support constraints are not claimed new'),indent=2))
print(json.dumps(scores,indent=2))
