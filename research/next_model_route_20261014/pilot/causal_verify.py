"""Re-estimate from legal prefix and perturb future prices/immature labels."""
import json
import numpy as np
import run as pilot
pbase=pilot.base;dates,p=pbase.read_prefix();initial=int(np.sum(dates<'2020-01-01'));x=pilot.blocks(p);rows=[]
for end in range(initial,len(p)-20,40):
 idx=np.arange(max(60,end-20-480),end-20);y=p[idx[:,None]+pbase.H]-p[idx,None]
 for kind in pilot.P['kinds']:
  state,hs=pilot.select(x[idx],y,kind,'causal_refit')
  for t in [end,min(end+39,len(p)-21)]:
   prefix=pilot.blocks(p[:t+1]);q=p.copy();q[t+1:]+=12345+np.arange(len(p)-t-1)*1000;future=pilot.blocks(q)
   original=pilot.spline.predict(state,x[t:t+1])*hs
   for alternative in [prefix[t:t+1],future[t:t+1]]:
    if not np.array_equal(original,pilot.spline.predict(state,alternative)*hs):raise ValueError('issuance prefix/future feature leakage')
  # Corrupt every horizon-specific immature label in a full target buffer.
  target=np.full((len(p),3),np.nan)
  for k,h in enumerate(pbase.H):target[:-h,k]=p[h:]-p[:-h]
  corrupt=target.copy()
  for k,h in enumerate(pbase.H):corrupt[max(0,end-int(h)):end,k]=999999
  if not np.array_equal(target[idx],corrupt[idx]):raise ValueError('immature label accessed')
  other,oh=pilot.select(x[idx],corrupt[idx],kind,'immature_label_refit')
  error=float(np.max(np.abs(pilot.spline.predict(state,x[end:end+1])*hs-pilot.spline.predict(other,x[end:end+1])*oh)))
  if error!=0:raise ValueError('immature target influences forecast')
  rows.append(dict(end=end,kind=kind,max_training_target=int(idx[-1]+20),inner_latest_target=int(idx[len(idx)-120-20-1]+20),inner_validation_first=int(idx[len(idx)-120]),forecast_difference=error))
pilot.dump('CAUSAL_INDEPENDENT.json',dict(passed=True,checks=rows,check_count=len(rows),scope='all refresh boundaries, first/last issuance prefix replay, future prices and all h immature labels'))
print(json.dumps(dict(passed=True,checks=len(rows),max_error=max(r['forecast_difference'] for r in rows))))
