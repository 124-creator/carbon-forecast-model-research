from pathlib import Path
import json,itertools,math
D=Path(__file__).parent
def encode(x):
 k=len(x);s=list(itertools.accumulate(x));m=s[-1]/k;b=[v-(i+1)*m for i,v in enumerate(s)];w=[math.sin(math.pi*(i+1)/k) for i in range(k)]
 return {'max':max(x),'avg':m,'signed_extrema':[max(0,max(x)),min(0,min(x))],'net':s[-1],'bridge':b,'bridge_area':sum(b)/k,'kernel_bridge':sum(v*z for v,z in zip(b,w))/sum(w)}
pairs=[([2,-2],[2,2]),([2,-2,0,0],[0,0,-2,2]),([-3,-1],[-1,-1])]
out=[]
for x,y in pairs:
 a,c=encode(x),encode(y);out.append({'x':x,'y':y,'x_encoding':a,'y_encoding':c})
 if a['max']!=c['max']:raise RuntimeError('max collision expected')
for x in itertools.product([-2.,0.,2.],repeat=4):
 e=encode(x);prev=0.;r=[]
 for b in e['bridge']:
  r.append(e['avg']+b-prev);prev=b
 if max(abs(a-b) for a,b in zip(x,r))>1e-12:raise RuntimeError('reconstruction')
 if abs(e['bridge'][-1])>1e-12:raise RuntimeError('endpoint')
# Symmetric +/-1 iid enumerated exactly: positive bias of max, no forecast claim.
iid=list(itertools.product([-1.,1.],repeat=4));bias=sum(max(x) for x in iid)/len(iid)
result={'pairs':out,'iid_max_expectation_k4':bias,'iid_avg_expectation':sum(sum(x)/4 for x in iid)/len(iid),'exact_reconstruction_cases':81,'training_run':False}
(D/'synthetic_results.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
