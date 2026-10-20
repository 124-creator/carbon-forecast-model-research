from pathlib import Path
import sys,time,json,copy,cProfile,pstats,io,traceback
BASE=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(BASE))
import run as r
from core import *
import numpy as np
OUT=Path(__file__).resolve().parent
START=time.monotonic(); DEADLINE=START+600
r.ROOT=OUT
r.START=START;r.LIMIT=600

def save(n,v): (OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
results=[];failures=[]
save('BUDGET.json',{'seconds':600,'scope':'profiling, training, diagnostics and saving','no_candidates_added':True,'optimization':'none; original torch triangular solve retained'})
data={}
for seed in [7,17]:
 y,_=r.generate('G2',seed);scale=float(np.std(np.diff(y[:1200])));cap=float(np.quantile(np.abs(np.diff(y[:1200]))/scale,.995));o,d,f,t,m=r.dataset(y,scale,cap);ms=r.masks(o,m);data[seed]=(y,scale,cap,o,d,f,t,ms)
# Schema only: selected and last, no optimizer state.
schemas={}
for seed,mode in [(7,'full'),(7,'constant')]:
 ck=torch.load(BASE/f'G2_{seed}_{mode}.pt',weights_only=True);schemas[f'{seed}_{mode}']={'keys':list(ck),'state_shapes':{k:list(v.shape) for k,v in ck['last'].items()},'optimizer_saved':False}
save('CHECKPOINT_SCHEMA.json',schemas)
# Real forward/backward profile; no optimizer updates and no training reuse.
y,scale,cap,o,d,f,t,ms=data[7];torch.manual_seed(7);model=Route('full');rows=ms['train'].any(1)
prof=cProfile.Profile();tick=time.monotonic();prof.enable()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as tp:
 for _ in range(3):
  model.zero_grad();r.loss(model(d[rows],f[rows]),t[rows],ms['train'][rows]).backward()
prof.disable();s=io.StringIO();pstats.Stats(prof,stream=s).sort_stats('cumtime').print_stats(35)
(OUT/'PYTHON_PROFILE.txt').write_text(s.getvalue(),encoding='utf-8')
(OUT/'TORCH_PROFILE.txt').write_text(tp.key_averages().table(sort_by='self_cpu_time_total',row_limit=30),encoding='utf-8')
save('PROFILE.json',{'three_forward_backward_seconds':time.monotonic()-tick,'python_triangular_loop':False,'implementation':'torch.linalg.solve_triangular, native batched solve; no equivalent optimization applied'})
# Existing complete full is replayed, never trained again.
for seed,mode in [(7,'full')]:
 y,scale,cap,o,d,f,t,ms=data[seed];tr=json.loads((BASE/f'G2_{seed}_{mode}_trace.json').read_text());ck=torch.load(BASE/f'G2_{seed}_{mode}.pt',weights_only=True);model=Route(mode);model.load_state_dict(ck['selected'])
 with torch.no_grad():p=model(d,f)
 results.append({'seed':seed,'method':mode,'status':'existing_complete_replayed','epochs':len(tr),'steps':len(tr),'selected_epoch':min((x for x in tr if 'validation' in x),key=lambda x:x['validation'])['epoch'],'test_h':r.metrics(p,t,ms['test']),'val_h':r.metrics(p,t,ms['val'])})
 np.savez_compressed(OUT/f'G2_{seed}_{mode}_predictions.npz',pred=p.numpy(),targets=t.numpy(),test_mask=ms['test'].numpy())
# No score-based ordering; paired ablations followed by remaining seed controls.
queue=[(7,'age'),(7,'evidence'),(17,'full'),(17,'age'),(17,'evidence'),(17,'constant'),(7,'constant')]
for seed,mode in queue:
 tag=f'G2_{seed}_{mode}'
 if time.monotonic()>=DEADLINE:
  results.append({'seed':seed,'method':mode,'status':'not_started_budget','epochs':0,'steps':0});continue
 try:
  y,scale,cap,o,d,f,t,ms=data[seed];torch.manual_seed(seed);model=Route(mode);tr=[];offset=0;best=float('inf');selected=None;resume=seed==7 and mode=='constant'
  if resume:
   old=json.loads((BASE/(tag+'_trace.json')).read_text());offset=len(old);ck=torch.load(BASE/(tag+'.pt'),weights_only=True);model.load_state_dict(ck['last']);selected=copy.deepcopy(ck['selected']);brec=min((x for x in old if 'validation' in x),key=lambda x:x['validation']);best=brec['validation'];sel_epoch=brec['epoch']
  else:sel_epoch=None
  opt=torch.optim.Adam(model.parameters(),lr=.001);trainrows=ms['train'].any(1);valrows=ms['val'].any(1);tick=time.monotonic()
  for epoch in range(offset,500):
   if time.monotonic()>=DEADLINE:break
   opt.zero_grad();p=model(d[trainrows],f[trainrows]);l=r.loss(p,t[trainrows],ms['train'][trainrows]);l.backward();grad=float(torch.sqrt(sum((x.grad*x.grad).sum() for x in model.parameters())));opt.step();rec={'epoch':epoch+1,'train':float(l),'gradient':grad,'steps_this_session':len(tr)+1}
   if epoch%10==0 or epoch==499:
    with torch.no_grad():v=float(r.loss(model(d[valrows],f[valrows]),t[valrows],ms['val'][valrows]))
    rec['validation']=v
    if np.isfinite(v) and v<best:best=v;selected=copy.deepcopy(model.state_dict());sel_epoch=epoch+1
   tr.append(rec)
  torch.save({'last':copy.deepcopy(model.state_dict()),'selected':selected,'optimizer':opt.state_dict()},OUT/(tag+'.pt'));save(tag+'_trace.json',tr)
  rec={'seed':seed,'method':mode,'status':'complete' if offset+len(tr)==500 else 'truncated','epochs':offset+len(tr),'steps':offset+len(tr),'additional_steps':len(tr),'selected_epoch':sel_epoch,'selected_validation':best,'runtime_seconds':time.monotonic()-tick,'resume':resume,'fresh_budget_equivalent':not resume,'optimizer_moments_reset':resume}
  if selected is not None:
   model.load_state_dict(selected)
   with torch.no_grad():p=model(d,f)
   rec.update(test_h=r.metrics(p,t,ms['test']),val_h=r.metrics(p,t,ms['val']))
   np.savez_compressed(OUT/(tag+'_predictions.npz'),pred=p.numpy(),targets=t.numpy(),test_mask=ms['test'].numpy())
   if time.monotonic()<DEADLINE:rec['diagnostics']=r.diagnostics(model,d,f)
   else:rec['diagnostics_status']='not_run_budget'
  results.append(rec);save('RESULTS.json',results);print(tag,rec,flush=True)
 except Exception:
  failures.append({'tag':tag,'traceback':traceback.format_exc()});save('FAILURES.json',failures)
save('RESULTS.json',results);save('FAILURES.json',failures)
save('COMPLETION.json',{'elapsed_seconds':time.monotonic()-START,'complete_500epoch_count':sum(x.get('epochs')==500 for x in results),'target_count':8,'market_read':False,'null_gate':'original G0 seed17 full/RW 1.027213406 exceeds 1.01; unchanged failed','release_allowed':False})
print('FINAL',json.dumps(results),flush=True)
