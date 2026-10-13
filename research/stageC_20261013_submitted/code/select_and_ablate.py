from run_market import *
import datetime

def select(allow_missing=False):
 d=load();rows=[];registry=configs()
 for c in registry:
  file=P/f"{c['id']}.npz"
  if not file.exists():
   if allow_missing:continue
   raise RuntimeError('Not all frozen configurations completed: '+c['id'])
  pred=np.load(file)['delta'];rr=score(d,pred,[2019,2020]);val=float(np.mean([v['relative_mse'] for v in rr]));rows.append(dict(**c,tuning_relative_mse=val,params_json=json.dumps(c['params'],sort_keys=True)))
 tab=pd.DataFrame(rows);tab.drop(columns=['params']).to_csv(O/'all_tuning_configurations.csv',index=False)
 selections={}
 for fam,df in tab.groupby('family'):
  ix=df.sort_values(['tuning_relative_mse','id']).iloc[0];selections[fam]=dict(id=ix['id'],family=fam,params=ix['params'],tuning_relative_mse=float(ix['tuning_relative_mse']))
 selections['_receipt']={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'all_complete':len(rows)==len(registry),'scoring_years_used':[2019,2020],'test_scores_used':False}
 (O/('selected_complete.json' if len(rows)==len(registry) else 'selected_partial_neural.json')).write_text(json.dumps(selections,indent=2))
 print(tab.sort_values('tuning_relative_mse')[['id','tuning_relative_mse']].head(15).to_string(index=False));return selections

def ablate():
 d=load();sel=select(True);c=sel['conditional'];(O/'candidate_frozen_before_check.json').write_text(json.dumps(c,indent=2));start=time.time()
 for fam in ['no_gap','fixed','no_drift']:
  file=P/f'ablation_{fam}.npz'
  if file.exists():continue
  preds=[];taus=[];logs=[]
  for seed in [11,29,47]:
   p,t,l=neural_forecasts(d,fam,**c['params'],seed=seed);preds.append(p);taus.append(t);logs.append(l)
  np.savez_compressed(file,delta=np.array(preds),tau=np.array(taus))
  (O/f'ablation_{fam}_receipt.json').write_text(json.dumps(dict(params=c['params'],logs=logs),indent=2));print('ABLATION',fam,'complete',flush=True)
 (O/'ablations_complete.json').write_text(json.dumps({'completed':True,'seconds':time.time()-start,'settings_from_tuning_only':True},indent=2))
if __name__=='__main__':
 import sys
 if '--ablate' in sys.argv:ablate()
 else:select()
