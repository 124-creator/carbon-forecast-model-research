"""Exact population moment counterexample and finite scalar-estimator controls.

Neither these scalar estimators nor this covariance identity are claimed new.
No market data, candidate neural model or hyperparameter selection are used.
"""
import json
from pathlib import Path
import numpy as np
import core

def generate(seed,a,n):
 r=np.zeros(n+200)
 rng=np.random.default_rng(seed)
 for t in range(1,len(r)):r[t]=a*r[t-1]+rng.normal()
 return r[200:]

def run(out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);n=256;rows=[]
 i=np.arange(n)[:,None];j=np.arange(n)[None,:]
 for h in [1,5,20]:
  cov=((i>j)&(i<=j+h)).astype(float)
  for mode in ['IDENTITY','SYMMETRIC','CAUSAL']:
   u=core.operator_array(n,mode)[h-1];num=float(np.sum(u*cov));den=float(np.trace(u))
   rows.append({'h':h,'mode':mode,'population_numerator':num,'population_denominator':den,'population_equation_root':num/den,
                'independent_test_expected_MSE_at_population_root':h+(num/den)**2})
 if any(x['population_numerator']!=0 for x in rows if x['mode']!='SYMMETRIC'):raise AssertionError('invalid null moment')
 detail=[];n=300
 for a,label in [(0.,'null'),(.5,'AR05')]:
  for seed in range(32):
   r=generate(8500+seed,a,n+21);s=generate(9500+seed,a,n+21);x=r[:n];xx=s[:n]
   for h in [1,5,20]:
    y=np.convolve(r[1:],np.ones(h),'valid')[:n];yy=np.convolve(s[1:],np.ones(h),'valid')[:n]
    for mode in ['IDENTITY','SYMMETRIC','CAUSAL']:
     u=core.operator_array(n,mode)[h-1];b=float(x@u@y/(x@u@x))
     mse=float(np.mean((xx*b-yy)**2));rw=float(np.mean(yy**2))
     detail.append({'case':label,'seed':seed,'h':h,'mode':mode,'coefficient':b,'oracle_coefficient':float(sum(a**k for k in range(1,h+1))),
                    'relative_MSE':mse/rw,'MSE':mse,'RW_MSE':rw})
 summary={}
 for label in ['null','AR05']:
  summary[label]={mode:float(np.mean([x['relative_MSE'] for x in detail if x['case']==label and x['mode']==mode])) for mode in ['IDENTITY','SYMMETRIC','CAUSAL']}
 result={'scope':'exact moment algebra plus32 independently generated scalar fits/checks; not neural carbon performance or novel GMM theory',
         'exact_population':rows,'finite_sample_summary':summary,'gate':summary['null']['CAUSAL']<=1.05,
         'limits':'zero expected moment is not an exactly unbiased finite-sample ratio estimator; regularized neural fitting has no such unbiasedness guarantee',
         'records':len(detail)}
 (out/'MATHEMATICAL_CONTROL.json').write_text(json.dumps(result,indent=2));(out/'MATHEMATICAL_DETAIL.json').write_text(json.dumps(detail,indent=2))
 print(json.dumps(result,indent=2));return result
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();run(a.out)
