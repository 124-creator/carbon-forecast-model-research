"""Independent reconstruction from saved coefficients only. No optimizer calls."""
from pathlib import Path
import sys,json,ast,hashlib
import numpy as np
import pandas as pd
sys.dont_write_bytecode=True
P=Path(__file__).resolve().parent;OLD=P.parent/'volatility_range_20261022'
sys.path.insert(0,str(OLD));import corrected_risk_experiment as base
H=[1,5,22]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def window(x,eps=None):
 a=np.column_stack([pd.Series(x).rolling(w).mean() for w in H]);return np.log(a+eps) if eps is not None else a
def label(x,h):return np.r_[np.array([np.mean(x[t+1:t+h+1]) for t in range(len(x)-h)]),np.full(h,np.nan)]
def vector(row,k):return np.array(ast.literal_eval(row[k]),float)
def design(x,row):return np.column_stack([np.ones(len(np.atleast_2d(x))),(np.atleast_2d(x)-vector(row,'mu'))/vector(row,'sd')])
def positive(x,row):return float(row['scale_c'])*np.exp(design(x,row)@vector(row,'beta'))
def main():
 m=json.loads((P/'manifest.json').read_text(encoding='utf-8'));assert m['status']=='completed'
 before={name:sha(P/name) for name in m['outputs']};assert before==m['outputs']
 assert all(sha(p)==v for p,v in m['input_hashes'].items())
 assert sha(P/'PROTOCOL_ZH.md')==m['protocol_sha256'] and sha(P/'structure_experiment.py')==m['script_sha256']
 f=pd.read_csv(P/'predictions_common.csv',float_precision='round_trip');fd=pd.read_csv(P/'fit_diagnostics.csv',float_precision='round_trip');comp=pd.read_csv(P/'A_components.csv',float_precision='round_trip');multi=pd.read_csv(P/'A_multistart.csv',float_precision='round_trip');eps=pd.read_csv(P/'mix_epsilon.csv',float_precision='round_trip');mixobs=pd.read_csv(P/'mix_observations.csv',float_precision='round_trip')
 fits={}
 for row in fd.to_dict('records'):
  start=row['start'] if isinstance(row.get('start'),str) else ''
  fits[(row['asset'],int(row['horizon']),int(row['refit_origin_index']),row['model'],start)]=row
 selected={(r.asset,int(r.horizon),int(r.refit_origin_index)):r.selected for r in multi.itertuples()}
 data={};native_error=0.
 for asset in base.ASSETS:
  d,r,r2,epsilon,xs,_,_=base.feature_data(asset);oa=d.open*d.adjclose/d.close;gap=np.log(oa/d.adjclose.shift(1)).to_numpy();idv=np.log(d.adjclose/oa).to_numpy();years=d.date.dt.year.to_numpy()
  values={'gap':gap*gap,'id':idv*idv,'cc':r2,'cross':2*gap*idv};x={k:window(values[k],epsilon) for k in ['gap','id','cc']};x['cross']=window(gap*idv)
  y={h:{k:label(v,h) for k,v in values.items()} for h in H}
  finite=np.logical_and.reduce([np.isfinite(v).all(1) for v in x.values()]);eligible={h:np.array([s+h<len(d) and years[s]==years[s+h] for s in range(len(d))]) for h in H}
  data[asset]=dict(d=d,r2=r2,x=x,xs=xs,y=y,finite=finite,eligible=eligible,years=years)
  for h,g in f[(f.asset==asset)&(f.model=='A2_bounded')].groupby('horizon'):
   o=g.origin_index.to_numpy(int);native_error=max(native_error,float(abs(y[h]['cc'][o]-g.target).max()));np.testing.assert_allclose(y[h]['cc'][o],g.target,rtol=1e-10,atol=1e-15)
   assert eligible[h][o].all() and np.array_equal(d.date.iloc[o+h].to_numpy(),pd.to_datetime(g.target_end).to_numpy())
 cache={};max_eps_error=0.;max_c_error=0.;max_objective_error=0.;max_relative_prediction=0.;train_rows_checked=0
 def mixing(asset,t):
  nonlocal max_eps_error
  key=(asset,t)
  if key in cache:return cache[key]
  when=data[asset]['d'].date.iloc[t];series={}
  for source in ['KRBN','KCCA']:
   obj=data[source];dates=obj['d'].date;v=obj['r2'];mask=(dates<when).to_numpy()&np.isfinite(v)&(v>0);ep=float(v[mask].mean()*1e-4)
   saved=eps[(eps.target_asset==asset)&(eps.refit_date==str(when.date()))&(eps.source_asset==source)].iloc[0]
   max_eps_error=max(max_eps_error,abs(ep-saved.epsilon));assert saved.n_positive_past==mask.sum() and saved.source_date_max==str(dates[mask].max().date())
   series[source]=pd.Series(v+ep,index=dates)
  pair=pd.concat(series,axis=1).dropna();a=pair.KRBN;b=pair.KCCA;gm=np.sqrt(a*b);parts={}
  for w in H:
   parts[f'x{w}']=np.log(gm.rolling(w).mean());parts[f'k{w}']=gm.rolling(w).sum()/np.sqrt(a.rolling(w).sum()*b.rolling(w).sum())
  frame=pd.DataFrame(parts).reindex(pd.DatetimeIndex(data[asset]['d'].date));cache[key]=frame;return frame
 for asset,h,t in selected:
  obj=data[asset];x=obj['x'];ys=obj['y'][h];idx=np.arange(len(obj['d']));valid=np.logical_and.reduce([np.isfinite(v) for v in ys.values()]);train=np.flatnonzero(obj['finite']&valid&obj['eligible'][h]&(idx+h<=t))
  for name in ['gap','id','cc','cross']:
   model='A_'+name if name!='cross' else 'A3_cross';row=fits[(asset,h,t,model,'')];assert row['n_train']==len(train) and row['train_origin_max']==train[-1] and row['train_label_end_max']==train[-1]+h
   xx=x[name][train];sd=xx.std(0);sd=np.where(sd>1e-12,sd,1.)
   np.testing.assert_allclose(vector(row,'mu'),xx.mean(0),rtol=1e-12,atol=1e-14);np.testing.assert_allclose(vector(row,'sd'),sd,rtol=1e-12,atol=1e-14)
   if name!='cross':
    c=np.mean(ys[name][train]*np.exp(-design(xx,row)@vector(row,'beta')));max_c_error=max(max_c_error,abs(c-row['scale_c']))
   train_rows_checked+=1
  g=positive(x['gap'][train],fits[(asset,h,t,'A_gap','')]);i=positive(x['id'][train],fits[(asset,h,t,'A_id','')]);values={}
  for start in ['zero','uniform']:
   row=fits[(asset,h,t,'A2_rho',start)];beta=vector(row,'beta');rho=np.clip(np.tanh(design(x['cc'][train],row)@beta),-1+1e-6,1-1e-6);ff=(np.sqrt(g)-np.sqrt(i))**2+2*np.sqrt(g*i)*(1+rho)
   val=float(np.mean(np.log(ff)+ys['cc'][train]/ff)+.005*np.dot(beta[1:],beta[1:]));max_objective_error=max(max_objective_error,abs(val-row['objective']));values[start]=row['objective'];assert row['max_gradient']<=1e-4 and row['success']
  assert selected[(asset,h,t)]==('uniform' if values['uniform']<values['zero'] else 'zero')
  mx=mixing(asset,t);allx=np.column_stack([obj['xs']['QHAR_liquidity'],mx[[f'x{w}' for w in H]].to_numpy()]);bt=np.flatnonzero(np.isfinite(allx).all(1)&np.isfinite(ys['cc'])&obj['eligible'][h]&(idx+h<=t))
  for model,xx in [('DM_pair',allx),('K3_no_DM',obj['xs']['QHAR_liquidity'])]:
   row=fits[(asset,h,t,model,'')];assert row['n_train']==len(bt) and row['train_origin_max']==bt[-1] and row['train_label_end_max']==bt[-1]+h
   np.testing.assert_allclose(vector(row,'mu'),xx[bt].mean(0),rtol=1e-12,atol=1e-14);train_rows_checked+=1
 new=f[f.model.isin(m['models'])]
 for row in new.itertuples():
  asset=row.asset;h=int(row.horizon);t=int(row.origin_index);ref=int(row.refit_origin_index);obj=data[asset];x=obj['x'];key=(asset,h,ref)
  if row.model.startswith('A') or row.model.startswith('K1') or row.model.startswith('K2'):
   g=float(positive(x['gap'][t],fits[key+('A_gap','')])[0]);i=float(positive(x['id'][t],fits[key+('A_id','')])[0])
   if row.model=='A2_bounded':
    fit=fits[key+('A2_rho',selected[key])];z=float((design(x['cc'][t],fit)@vector(fit,'beta'))[0]);rho=float(np.clip(np.tanh(z),-1+1e-6,1-1e-6));pred=(np.sqrt(g)-np.sqrt(i))**2+2*np.sqrt(g*i)*(1+rho)
   elif row.model=='A3_free_cross':
    fit=fits[key+('A3_cross','')];pred=g+i+float((design(x['cross'][t],fit)@vector(fit,'beta'))[0])
   elif row.model=='K1_direct_CC':pred=float(positive(x['cc'][t],fits[key+('A_cc','')])[0])
   else:pred=g+i
  else:
   fit=fits[key+(row.model,'')];xx=obj['xs']['QHAR_liquidity'][t]
   if row.model=='DM_pair':xx=np.r_[xx,mixing(asset,ref)[[f'x{w}' for w in H]].iloc[t].to_numpy()]
   pred=float(positive(xx,fit)[0])
  max_relative_prediction=max(max_relative_prediction,abs(pred-row.forecast)/max(abs(row.forecast),1e-15))
  assert np.isclose(pred,row.forecast,rtol=1e-9,atol=1e-15)
 for row in mixobs.itertuples():
  ref=int(np.flatnonzero(data[row.asset]['d'].date==pd.Timestamp(row.refit_date))[0]);expected=mixing(row.asset,ref)[f'k{row.window}'].iloc[row.origin_index]
  assert abs(expected-row.kappa)<1e-12 and row.members=='KRBN|KCCA' and row.n_assets==2
 assert max_c_error<1e-12 and max_objective_error<1e-9
 assert (f.groupby(['asset','horizon','origin_index']).size()==24).all();assert not f.duplicated(['asset','horizon','origin_index','model']).any()
 nonpositive=f[(f.model=='A3_free_cross')&(f.forecast<=0)];assert len(nonpositive)==10 and nonpositive.qlike.isna().all()
 summary=pd.read_csv(P/'summary.csv',dtype={'period':str});pair=pd.read_csv(P/'pairwise.csv',dtype={'period':str})
 assert summary.loc[summary.QLIKE_undefined>0,'QLIKE'].isna().all() and pair.loc[pair.QLIKE_undefined>0,'delta_QLIKE'].isna().all()
 assert before=={name:sha(P/name) for name in before}
 result=dict(no_refits=True,all_saved_coefficient_predictions_rebuilt=len(new),max_relative_forecast_difference=max_relative_prediction,max_native_target_difference=native_error,max_epsilon_difference=max_eps_error,max_multiplicative_calibration_difference=max_c_error,max_rho_objective_difference=max_objective_error,training_records_checked=train_rows_checked,all_common_origins_preserved=True,all_input_and_output_hashes_unchanged=True,A3_nonpositive_breakdown=nonpositive.groupby(['asset','year','horizon']).size().reset_index(name='n').to_dict('records'))
 (P/'independent_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
 m['independent_validation']=result;m['validation_script_sha256']=sha(__file__);m['outputs']['independent_validation.json']=sha(P/'independent_validation.json')
 (P/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(result,ensure_ascii=False,indent=2));print('MANIFEST_SHA256',sha(P/'manifest.json'))
if __name__=='__main__':main()
