from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
import sys,json,hashlib,time,contextlib,io
sys.path.insert(0,str(R/'vendor'))
from mbtr.mbtr import MBT
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from threadpoolctl import threadpool_limits

def fit_model(kind,x,y,trees=100,seed=17):
    with threadpool_limits(limits=1):
        if kind=='MBTR':
            m=MBT(n_boosts=trees,early_stopping_rounds=trees,val_ratio=0,learning_rate=.05,n_q=10,min_leaf=40,loss_type='mse',lambda_weights=.1,lambda_leaves=.1,verbose=0,refit=False)
            with contextlib.redirect_stdout(io.StringIO()):m.fit(x,y)
        else:
            m=MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=trees,learning_rate=.05,max_leaf_nodes=7,min_samples_leaf=40,l2_regularization=.1,early_stopping=False,random_state=seed),n_jobs=1).fit(x,y)
    return m

def model_signature(m):
    h=hashlib.sha256()
    if isinstance(m,MBT):
        h.update(np.asarray(m.y_0).tobytes())
        for t in m.trees:
            for node,v in t.g.nodes(data=True):
                h.update(str(node).encode())
                for k,val in sorted(v.items()):
                    h.update(k.encode());h.update(np.asarray(val).tobytes() if isinstance(val,np.ndarray) else str(val).encode())
            h.update(str(list(t.g.edges())).encode())
    else:
        for e in m.estimators_:
            h.update(np.asarray(e._baseline_prediction).tobytes())
            for stage in e._predictors:
                for tree in stage:h.update(tree.nodes.tobytes())
    return h.hexdigest()

def training_data(p,fit_end):
    s=np.arange(max(60,fit_end-20-480),fit_end-20)
    x=np.array([np.diff(p[t-60:t+1]) for t in s],dtype=np.float64)
    y=np.array([p[t+1:t+21]-p[t] for t in s],dtype=np.float64)
    return x,y,s
