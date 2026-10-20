import json,time,hashlib,importlib.metadata as md
from pathlib import Path
import numpy as np
import torch
from model import Forecaster, bridge_matrix, ordinary
ROOT=Path(__file__).parent
KINDS=['bridge','dst','haar','no_bridge','no_route','rawAvg','rawMax']
REGIMES=['persistent','interior_reversal','nonaligned_reversal','no_signal']

def generate(seed,regime):
 rng=np.random.default_rng(seed+9000); n=2400
 state=np.zeros(n); innovations=rng.normal(0,.18,n)
 for t in range(1,n): state[t]=.94*state[t-1]+innovations[t]
 noise=rng.normal(0,.35,n)
 if regime=='no_signal': return rng.normal(0,1,n)
 if regime=='persistent': return state+noise
 if regime=='interior_reversal':
  phase=rng.uniform(0,2*np.pi); period=8.0
  return state+1.1*np.sin(2*np.pi*np.arange(n)/period+phase)+noise
 sign=np.ones(n); current=1.; remain=0
 for t in range(n):
  if remain==0: current=-current; remain=int(rng.integers(5,18))
  sign[t]=current; remain-=1
 return state+.9*sign+noise

def prepare(x):
 n=len(x); bounds=[64,int(.6*n),int(.8*n),n]
 scale=float(np.std(x[:bounds[1]],ddof=0))
 sets=[]; origins=[]
 for start,end in zip(bounds[:-1],bounds[1:]):
  ids=np.arange(start,end-20+1)
  assert ids[-1]+20<=end and np.all(ids>=64)
  h=np.stack([x[t-64:t] for t in ids])/scale
  y=np.stack([np.cumsum(x[t:t+20]) for t in ids])
  sets.append((torch.tensor(h,dtype=torch.float32),torch.tensor(y,dtype=torch.float32)))
  origins.append(ids)
 return sets,scale,origins

def metrics(p,y):
 err=(p-y).numpy()**2
 return {'mse20':float(err.mean()),'h1':float(err[:,0].mean()),'h5':float(err[:,4].mean()),'h20':float(err[:,19].mean())}

def verify():
 out={}
 for n in (8,4):
  e,d=bridge_matrix(n); x=torch.randn(19,n,dtype=torch.float64); c=x@e.T
  out['inverse_error_n'+str(n)]=float(((c@d.T)-x).abs().max())
  assert torch.allclose(c[:,0],x.mean(-1),atol=1e-12)
  assert torch.allclose(d[:,1:].sum(0),torch.zeros(n-1,dtype=torch.float64),atol=1e-12)
  for kind in ('dst','haar'):
   a,b=ordinary(n,kind); assert torch.allclose((x@a.T)@b.T,x,atol=1e-12)
 torch.manual_seed(18); m=Forecaster('bridge').double(); x=torch.randn(2,64,dtype=torch.float64,requires_grad=True)
 loss=m(x).square().mean(); loss.backward(); analytic=float(x.grad[0,9]); eps=1e-5
 z=x.detach().clone(); z[0,9]+=eps; a=float(m(z).square().mean()); z[0,9]-=2*eps; b=float(m(z).square().mean()); numerical=(a-b)/(2*eps)
 assert abs(analytic-numerical)<1e-6
 out['input_gradient_analytic']=analytic; out['input_gradient_finite_difference']=numerical
 raw=generate(17,'persistent'); first,scale,ids=prepare(raw); mutated=raw.copy(); mutated[1440:]+=9999; second,scale2,_=prepare(mutated)
 assert scale==scale2 and torch.equal(first[0][0],second[0][0]) and torch.equal(first[0][1],second[0][1])
 out['future_perturbation_training_and_scale_unchanged']=True
 out['split_origins']=[{'first':int(i[0]),'last':int(i[-1]),'count':len(i),'last_label_exclusive':int(i[-1]+20)} for i in ids]
 out['zero_detail_endpoints']=[8,16,20]
 out['history_mapping']='x[t-64:t] => chronological 8 blocks; future x[t:t+20] => blocks8,8,4; p anchor index t-1'
 return out

def run():
 torch.set_num_threads(1); start=time.monotonic()
 result={'protocol':'PROTOCOL.md','versions':{k:md.version(k) for k in ['torch','numpy','neuralforecast']},'verification':verify(),'runs':[]}
 source=Path('E:/碳价格/11/实验十一十一/田/第三次-tian-碳价格实验/official_nhits_baseline_20261014/design')
 result['readonly_source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [source/'DESIGN_signed_bridge_nhits.md',source/'synthetic_counterexamples.py']}
 result['implementation']='Reuse installed official NHITSBlock and _IdentityBasis (mlp_units=[[32,32]], ReLU only after second hidden linear), custom routing/coordinate wrappers. NOT full official NHITS model; no installed package edits. First diagnostic used extra input ReLU; discarded for architecture mismatch, not seed or metric selection; archived separately.'
 import inspect
 from neuralforecast.models.nhits import NHITSBlock
 installed=Path(inspect.getfile(NHITSBlock))
 result['official_module']={'path':str(installed),'sha256_before':hashlib.sha256(installed.read_bytes()).hexdigest()}

 local=ROOT/'local'; local.mkdir(exist_ok=True)
 for regime in REGIMES:
  for seed in [17,43]:
   x=generate(seed,regime); sets,scale,origins=prepare(x); (xt,yt),(_,yv),(xe,ye)=sets
   np.savez_compressed(local/f'data_{regime}_{seed}.npz',increments=x,scale=scale,train_origins=origins[0],validation_origins=origins[1],test_origins=origins[2])
   zero=metrics(torch.zeros_like(ye),ye)
   for kind in KINDS:
    if time.monotonic()-start>210: raise RuntimeError('bounded execution cap exceeded; partial results saved')
    torch.manual_seed(seed); model=Forecaster(kind); opt=torch.optim.Adam(model.parameters(),lr=.001)
    g=torch.Generator().manual_seed(seed)
    for step in range(160):
     idx=torch.randint(len(xt),(64,),generator=g); opt.zero_grad(); pred=model(xt[idx])*scale; loss=(pred-yt[idx]).square().mean(); loss.backward(); opt.step()
    with torch.no_grad():
     pred=model(xe)*scale; score=metrics(pred,ye)
    params=sum(p.numel() for p in model.parameters())
    mac=sum(mod.in_features*mod.out_features for mod in model.modules() if isinstance(mod,torch.nn.Linear))
    row={'regime':regime,'seed':seed,'kind':kind,'scale_train_only':scale,'parameters':params,'linear_macs_per_forward':mac,'steps':160,'test_count':len(xe),'zero':zero,**score,'ratio_to_zero':score['mse20']/zero['mse20']}
    result['runs'].append(row)
    np.savez_compressed(local/f'parameters_{regime}_{seed}_{kind}.npz',**{k:v.detach().numpy() for k,v in model.state_dict().items()})
    (ROOT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(regime,seed,kind,round(score['mse20'],4),flush=True)
 result['elapsed_seconds']=time.monotonic()-start
 result['ablations']=[]
 for regime in REGIMES:
  for seed in [17,43]:
   rows={r['kind']:r for r in result['runs'] if r['regime']==regime and r['seed']==seed}
   result['ablations'].append({'regime':regime,'seed':seed,'bridge_information_increment_mse':rows['no_bridge']['mse20']-rows['bridge']['mse20'],'endpoint_route_increment_mse':rows['no_route']['mse20']-rows['bridge']['mse20'],'best_structural_control':min(['dst','haar'],key=lambda k:rows[k]['mse20']),'candidate_beats_both_strong':all(rows['bridge']['mse20']<rows[k]['mse20'] for k in ['dst','haar'])})
 assert len(result['runs'])==56
 result['official_module']['sha256_after']=hashlib.sha256(installed.read_bytes()).hexdigest()
 assert result['official_module']['sha256_before']==result['official_module']['sha256_after']
 result['readonly_source_unchanged']=all(hashlib.sha256((source/k).read_bytes()).hexdigest()==v for k,v in result['readonly_source_sha256'].items())
 (ROOT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf8')
 print('COMPLETE',len(result['runs']),result['elapsed_seconds'],json.dumps(result['ablations']),flush=True)
if __name__=='__main__': run()
