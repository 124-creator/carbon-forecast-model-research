"""Synthetic algebra checks only. No market files or training."""
import json
from pathlib import Path
import numpy as np

def require(v, msg):
    if not v: raise RuntimeError(msg)

def run():
    rng=np.random.default_rng(20261013)
    X=rng.normal(size=(80,5)); C=rng.normal(size=(80,3)); y=C.sum(axis=1)
    B=np.linalg.lstsq(X,C,rcond=None)[0]
    b=np.linalg.lstsq(X,y,rcond=None)[0]
    same=float(np.max(np.abs(X@B.sum(axis=1)-X@b)))
    require(same<1e-10,'common-feature OLS sum identity')
    lam=2.; Br=np.linalg.solve(X.T@X+lam*np.eye(5),X.T@C)
    # Joint sum loss plus equal branch ridge implies lambda/K aggregate ridge.
    bj=np.linalg.solve(X.T@X+lam/3*np.eye(5),X.T@y)
    blocks=np.tile(X,(1,3)); joint=np.linalg.solve(blocks.T@blocks+lam*np.eye(15),blocks.T@y)
    joint_equiv=float(np.max(np.abs(blocks@joint-X@bj)))
    require(joint_equiv<1e-10,'joint ridge rescaling')
    E=rng.normal(size=(100,3))+np.array([1.,-2.,.5])
    mu=E.mean(0); centered=E-mu; cov=centered.T@centered/len(E)
    risk=float(np.mean(E.sum(1)**2)); identity=float(cov.sum()+mu.sum()**2)
    require(abs(risk-identity)<1e-10,'risk with bias')
    # Two deterministic zero-mean error distributions. Lower marginal MSE can hurt sum.
    A=np.array([[1.,-1.],[-1.,1.]])
    D=np.array([[.8,.8],[-.8,-.8]])
    margA=float(np.mean(np.sum(A*A,axis=1))); margD=float(np.mean(np.sum(D*D,axis=1)))
    totalA=float(np.mean(A.sum(1)**2)); totalD=float(np.mean(D.sum(1)**2))
    require(margD<margA and totalD>totalA,'counterexample')
    # Distinct features: independent component supervision != aggregate supervision.
    a=np.array([-1.,-1.,1.,1.]); z=np.array([-1.,1.,-1.,1.])
    c1=z; c2=-z; xx=np.column_stack([a,z]); target=c1+c2
    independent=np.column_stack([a,z])@np.array([a@c1/(a@a),z@c2/(z@z)])
    combined=xx@np.linalg.lstsq(xx,target,rcond=None)[0]
    require(np.mean(independent**2)>np.mean(combined**2),'distinct-feature difference')
    # SUR same-design unregularized GLS equals OLS for any positive definite task precision.
    Q=np.array([[2.,.3,.1],[.3,1.,.2],[.1,.2,1.5]])
    normal=float(np.max(np.abs(X.T@(C-X@B)@Q)))
    require(normal<1e-10,'same-design GLS normal equation')
    # covariance-only reallocation cannot change a fixed aggregate forecast.
    d=rng.normal(size=100); redistributed=E.copy(); redistributed[:,0]+=d; redistributed[:,1]-=d
    require(np.max(np.abs(redistributed.sum(1)-E.sum(1)))<1e-10,'zero-sum invariance')
    return dict(common_feature_OLS_max_difference=same,joint_ridge_scaled_lambda_max_difference=joint_equiv,risk=risk,risk_identity=identity,counterexample=dict(independent_marginal_sum=margA,smaller_marginal_sum=margD,independent_aggregate=totalA,smaller_marginal_aggregate=totalD),distinct_feature_independent_risk=float(np.mean(independent**2)),distinct_feature_joint_risk=float(np.mean(combined**2)),SUR_normal_equation_error=normal,checks_passed=7,market_files_read=False)
if __name__=='__main__':
    result=run(); print(json.dumps(result,indent=2)); Path(__file__).with_name('MATH_RESULTS.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
