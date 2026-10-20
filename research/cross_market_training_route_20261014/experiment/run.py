"""Fixed historical exploratory pilot, independent implementation; no novelty claim."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import csv,hashlib,json,time,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call
R=Path(__file__).resolve().parent;ROOT=R.parents[2]
SEEDS=[17,29];KINDS=['independent','pooled','staticprivate','dynamic_noepisodic','dynamic_episodic']
torch.set_num_threads(1)
class Timeout(Exception):pass
spent=0.; deadline=None

def save(name,obj):
 p=R/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8');t.replace(p)

def load():
 counts={'guangzhou':950,'hubei':936,'eu':999};out={};receipts={}
 for m,n in counts.items():
  lines=[]
  with (ROOT/f'data/parsed/figshare_28682348/{m}_reported.csv').open('rb',buffering=0) as f:
   for _ in range(n+1):
    line=bytearray()
    while True:
     b=f.read(1)
     if not b:raise ValueError('short audited prefix')
     line+=b
     if b==b'\n':break
    lines.append(bytes(line))
  raw=b''.join(lines);rows=list(csv.DictReader(raw.decode('utf-8-sig').splitlines()));d=np.array([r['date'] for r in rows]);p=np.array([float(r['author_reported_price']) for r in rows])
  if len(rows)!=n or d[-1]!='2021-12-31' or len(set(d))!=n or np.any(d[1:]<=d[:-1]) or not np.isfinite(p).all():raise ValueError('prefix numeric/date qualification')
  out[m]=(p,d);receipts[m]={'rows':n,'last':str(d[-1]),'sha256':hashlib.sha256(raw).hexdigest(),'boundary_read':False}
 return out,receipts

def make_pack(p):
 origins=np.arange(60,len(p)-20);val=origins[-40:];train=origins[origins+20<val[0]]
 # Scale fit only on raw prices needed by purged training, NOT whole prefix or validation.
 end=int(train[-1]+20);diff=np.diff(p[:end+1]);scale=max(float(np.median(np.abs(diff-np.median(diff)))*1.4826),1e-3)
 x=np.array([np.diff(p[t-60:t+1])/scale for t in origins],dtype=np.float32)
 y=np.array([(p[t+1:t+21]-p[t])/scale for t in origins],dtype=np.float32)
 return {'x':torch.tensor(x),'y':torch.tensor(y),'origins':origins,'train_origins':train,'val_origins':val,'train':np.searchsorted(origins,train),'val':np.searchsorted(origins,val),'scale':scale,'scale_last_price_index':end,'complete':len(origins)}

class Block(nn.Module):
 def __init__(self,pool,k,private,dynamic):
  super().__init__();self.pool=pool;self.k=k;self.dynamic=dynamic
  self.hidden=nn.Sequential(nn.Linear(60//pool,16),nn.ReLU(),nn.Linear(16,16),nn.ReLU());self.back=nn.Linear(16,60);self.fore=nn.Linear(16,k)
  if private:
   self.V=nn.Parameter(torch.randn(3,2,16)*.02);self.Q=nn.Parameter(torch.randn(3,2,4)*.02) if dynamic else None;self.c=nn.Parameter(torch.randn(3,2)*.02);self.U=nn.Parameter(torch.zeros(3,k,2))
  else:self.V=None
 def forward(self,x,u,m):
  z=self.hidden(F.avg_pool1d(x[:,None,:],self.pool).squeeze(1));a=self.fore(z)
  if self.V is not None:
   v=z@self.V[m].T+self.c[m]
   if self.Q is not None:v=v+u@self.Q[m].T
   a=a+torch.tanh(v)@self.U[m].T
  return self.back(z),F.interpolate(a[:,None,:],size=20,mode='linear',align_corners=True).squeeze(1)
class Net(nn.Module):
 def __init__(self,kind):
  super().__init__();self.blocks=nn.ModuleList([Block(p,k,'private' in kind or 'dynamic' in kind,'dynamic' in kind) for p,k in [(1,20),(4,5),(12,2)]])
 def forward(self,x,m=0):
  u=torch.stack([x[:,-5:].mean(1),x[:,-20:].mean(1),x.mean(1),x[:,-20:].square().mean(1).sqrt()],1);r=x;pred=torch.zeros((len(x),20))
  for b in self.blocks:
   back,fore=b(r,u,m);r=r-back;pred=pred+fore
  return pred

def private_name(n):return n.split('.')[-1] in ['U','V','Q','c']
def fingerprint(net):
 h=hashlib.sha256()
 for n,p in net.state_dict().items():h.update(n.encode());h.update(p.numpy().tobytes())
 return h.hexdigest()

def fit(data,b,kind,seed,tag):
 global spent
 start=time.monotonic();torch.manual_seed(seed);rng=np.random.default_rng(seed);net=Net(kind);opt=torch.optim.Adam(net.parameters(),lr=.001);packs=[];ends=[]
 cutoff=data['guangzhou'][1][b-1]
 for m in ['guangzhou','hubei','eu']:
  p,d=data[m];n=b if m=='guangzhou' else int(np.sum(d<cutoff));packs.append(make_pack(p[:n]));ends.append(n)
 if b==459 and [z['complete'] for z in packs]!=[379,405,400]:raise ValueError('initial full-label pool')
 log=open(R/(tag+'_loss.jsonl'),'w',encoding='utf8');calls=0
 def loss_for(pack,idx,m,params=None):
  x,y=pack['x'][idx],pack['y'][idx];pred=net(x,m) if params is None else functional_call(net,params,(x,m));return F.huber_loss(pred,y,delta=1.)
 def batch(idx):return rng.choice(idx,size=min(64,len(idx)),replace=False)
 try:
  episodic=kind=='dynamic_episodic';iterations=19 if episodic else (120 if kind=='independent' else 40)
  for step in range(iterations):
   if spent+time.monotonic()-start>=900:raise Timeout('15min cumulative fit budget')
   opt.zero_grad();losses=[]
   for m in ([0] if kind=='independent' else range(3)):
    pack=packs[m]
    if episodic:
     tr=pack['train'];split=len(tr)//2;query=tr[split:];support=tr[pack['origins'][tr]+20<pack['origins'][query[0]]]
     pars=dict(net.named_parameters());names=[n for n in pars if private_name(n)];ls=loss_for(pack,batch(support),m);g=torch.autograd.grad(ls,[pars[n] for n in names]);calls+=1
     adapted=dict(pars)
     for n,grad in zip(names,g):adapted[n]=pars[n]-.01*grad.detach()
     loss=loss_for(pack,batch(query),m,adapted)
    else:loss=loss_for(pack,batch(pack['train']),m)
    losses.append(loss);calls+=1
   combined=torch.stack(losses).mean();combined.backward();gn=float(torch.sqrt(sum(p.grad.square().sum() for p in net.parameters() if p.grad is not None)));opt.step();log.write(json.dumps({'step':step,'gradient_calls':calls,'train_huber':float(combined.detach()),'gradient_norm':gn})+'\n');log.flush()
  if episodic:
   params=[p for n,p in net.named_parameters() if private_name(n)];adaptopt=torch.optim.SGD(params,lr=.01)
   for j in range(6):
    adaptopt.zero_grad();loss=loss_for(packs[0],batch(packs[0]['train']),0);grads=torch.autograd.grad(loss,params)
    for p,g in zip(params,grads):p.grad=g
    adaptopt.step();calls+=1;log.write(json.dumps({'adapt_step':j,'gradient_calls':calls,'train_huber':float(loss.detach())})+'\n')
  with torch.no_grad():validation=float(loss_for(packs[0],packs[0]['val'],0))
  if calls!=120:raise ValueError('gradient call budget mismatch')
  torch.save(net.state_dict(),R/(tag+'_weights_LOCAL_ONLY.pt'))
  return net,packs[0]['scale'],{'tag':tag,'kind':kind,'seed':seed,'fit_end_exclusive':b,'cutoff':str(cutoff),'source_ends':ends,'complete_pools':[p['complete'] for p in packs],'train_pools':[len(p['train']) for p in packs],'validation_only_target_huber':validation,'scale_train_only':[p['scale'] for p in packs],'scale_last_price_index':[p['scale_last_price_index'] for p in packs],'gradient_calls':calls,'parameter_count':sum(p.numel() for p in net.parameters()),'private_parameters':sum(p.numel() for n,p in net.named_parameters() if private_name(n)),'sha256':fingerprint(net),'fit_seconds':time.monotonic()-start}
 finally:log.close();spent+=time.monotonic()-start

def predict(net,scale,p,origins):
 x=torch.tensor(np.array([np.diff(p[t-60:t+1])/scale for t in origins],dtype=np.float32))
 with torch.no_grad():return p[origins,None]+scale*net(x,0).numpy()

def replay():
 z=np.load(R/'predictions_LOCAL_ONLY.npz',allow_pickle=False);rows=[];overview={}
 for kind in KINDS:
  preds=[z[f'{kind}_{s}'] for s in SEEDS];valid=np.logical_and.reduce([np.isfinite(p).all(1) for p in preds]);ratios=[]
  for year in ['2020','2021']:
   idx=(z['years']==year)&valid
   for h in range(20):
    if not idx.any():continue
    mse=float(np.mean([np.mean((p[idx,h]-z['truth'][idx,h])**2) for p in preds]));mae=float(np.mean([np.mean(abs(p[idx,h]-z['truth'][idx,h])) for p in preds]));rw=float(np.mean((z['rw'][idx,h]-z['truth'][idx,h])**2));ratio=mse/rw
    rows.append({'kind':kind,'year':year,'h':h+1,'n':int(idx.sum()),'mae':mae,'rmse':mse**.5,'mse':mse,'rw_mse':rw,'ratio':ratio});ratios.append(ratio)
  overview[kind]={'both_seed_origins':int(valid.sum()),'full451':bool(valid.all()),'year_H20_mean_RW_ratio':float(np.mean(ratios)) if ratios else None}
 save('scores.json',{'overview':overview,'year_H20':rows,'seed_aggregation':'average losses, not forecasts','MLE_stronglinear':'historical reuse only, NOT retrained; no invented scores','novelty':'hypothesis, nearest-neighbor full-text review incomplete'})
 print(json.dumps(overview),flush=True)
 return overview

def run():
 if (R/'predictions_LOCAL_ONLY.npz').exists():raise RuntimeError('fresh run required; refusing overwrite')
 data,audit=load();p,d=data['guangzhou'];origins=np.array([t for t in range(459,len(p)-20) if d[t][:4]==d[t+20][:4]])
 if len(origins)!=451:raise ValueError('451 same-year H20 contract')
 save('protocol.json',{'seeds':SEEDS,'kinds':KINDS,'configurations':1,'fit_budget_seconds':900,'gradient_calls_per_fit':120,'batch_size':64,'optimizer':'Adam lr .001; episodic first-order private SGD .01; final private SGD 6 steps','validation':'last 40 mature target origins; train endpoints strictly before first query; no selection (one fixed configuration)','source':'both HB and author EU predeclared, identity caveat','scaler':'MAD differences on purged train-label range only, floor .001','refresh':'from scratch every40, frozen in each block','L':60,'H':20,'audit':audit,'score':'451 2020-21 same-year H20; full year/H20 plus h1/5/20 original price; average seed losses','not_real_time':True,'gradient_call_matching_not_equal_FLOPs':True,'license':'Fan 2025 DOI 10.6084/m9.figshare.28682348.v1 CC BY4; historical metadata not newly verified; EU semantics unknown','original_design':'DESIGN_ZH.md copied unchanged; candidate provisional not first'})
 saved={'origins':origins,'years':np.array([x[:4] for x in d[origins]]),'truth':np.array([p[t+1:t+21] for t in origins]),'rw':np.repeat(p[origins,None],20,1)}
 for k in KINDS:
  for s in SEEDS:saved[f'{k}_{s}']=np.full((451,20),np.nan)
 receipts=[];causal=[]
 def persist():
  np.savez(R/'predictions_LOCAL_ONLY.npz',**saved);save('fit_receipts.json',receipts);save('causality.json',causal);save('budget.json',{'fit_seconds':spent,'status':'running','completed_fits':len(receipts)})
 persist()
 try:
  for block,b in enumerate(range(459,930,40)):
   ts=origins[(origins>=b)&(origins<b+40)]
   if not len(ts):continue
   for kind in KINDS:
    for seed in SEEDS:
     net,scale,receipt=fit(data,b,kind,seed,f'b{block:02d}_{kind}_{seed}');pred=predict(net,scale,p,ts);saved[f'{kind}_{seed}'][np.searchsorted(origins,ts)]=pred;receipts.append(receipt);persist()
     if block==0 and kind=='dynamic_episodic' and seed==17:
      changed={}
      for m,(mp,md) in data.items():
       cp=mp.copy();n=b if m=='guangzhou' else int(np.sum(md<d[b-1]));cp[n:]+=10000+np.arange(len(cp)-n)*17;changed[m]=(cp,md.copy())
      repeat,rs,rr=fit(changed,b,kind,seed,'future_perturbed_refit');first=predict(net,scale,p,np.array([b-1]));second=predict(repeat,rs,changed['guangzhou'][0],np.array([b-1]));check={'parameter_identical':receipt['sha256']==rr['sha256'],'issuance_prediction_identical':bool(np.array_equal(first,second)),'refit':rr,'perturbed':'every market after its permitted fit endpoint'};causal.append(check);persist()
      if not check['parameter_identical'] or not check['issuance_prediction_identical']:raise ValueError('future perturb real refit failed')
  status='complete'
 except Timeout:status='timeout'
 except Exception as e:
  persist();save('budget.json',{'fit_seconds':spent,'status':'error','error':repr(e),'completed_fits':len(receipts)});raise
 persist();save('budget.json',{'fit_seconds':spent,'status':status,'completed_fits':len(receipts),'missing_stay_nan':True});replay()
if __name__=='__main__':
 if '--replay' in sys.argv:replay()
 else:run()
