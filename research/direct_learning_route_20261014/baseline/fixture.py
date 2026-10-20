from pathlib import Path
import sys,json,time
import numpy as np
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'vendor'))
from mbtr.mbtr import Tree,MBT,leaf_stats
from mbtr.losses import MSE
start=time.monotonic()
loss=MSE(lambda_weights=.1,lambda_leaves=.1);loss.set_dimension(20)
x=np.arange(12,dtype=float).reshape(-1,1)
y=np.tile(np.array([-2.]*6+[3.]*6)[:,None],(1,20))
h=np.ones_like(y);g=y.sum(0)
np.testing.assert_allclose(loss.eval_optimal_response(g,h.sum(0)),g/(12+.1))
np.testing.assert_allclose(loss.eval_optimal_loss(g*g,h.sum(0)),-np.sum(g*g)/(12+.1))
edges=np.array([-np.inf,5.5,np.inf]);a,b=leaf_stats(y,edges,x[:,0],1);c,d=leaf_stats(h,edges,x[:,0],2)
t=Tree(n_q=2,min_leaf=8,lambda_weights=.1,lambda_leaves=.1)
t.loss.set_dimension(20)
actual=t.compute_loss(a,b,c,d,0)
expected=-np.sum(y[:6].sum(0)**2)/(6+.1)-np.sum(y[6:].sum(0)**2)/(6+.1)
np.testing.assert_allclose(actual,expected)
t.fit(x,y)
regions=t._find_regions(x,y);supports=[int(z.sum()) for z in regions]
if min(supports)>=8:raise RuntimeError('expected author parent-only stopping distinction')
m=MBT(n_boosts=2,early_stopping_rounds=2,val_ratio=0,min_leaf=8,n_q=2,refit=False)
old=np.random.permutation
def forbidden(*args): raise RuntimeError('random validation called')
np.random.permutation=forbidden
try:m.fit(x,y)
finally:np.random.permutation=old
if m.predict(x).shape!=(12,20):raise RuntimeError('output shape')
result=dict(passed=True,formula='response=G/(n+lambda); score=-sum(G_h^2)/(n+lambda); split=sum(child_scores)',split_score=float(actual),parent_score=float(loss.eval_optimal_loss(g*g,h.sum(0))),leaf_supports=supports,strict_minimum_child_support=False,author_parent_stop_threshold=8,validation_random_permutation_called=False,seconds=time.monotonic()-start,optimized=not __debug__)
(R/('FIXTURE_O.json' if not __debug__ else 'FIXTURE_NORMAL.json')).write_text(json.dumps(result,indent=2));print(result)
