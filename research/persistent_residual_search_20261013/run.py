import sys,json,time
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent/'multi_angle_model_search_20261013/state'))
import importlib.util
spec=importlib.util.spec_from_file_location('state_reference',OUT.parent/'multi_angle_model_search_20261013/state/run.py')
old=importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
old.base.OUT=OUT
H=old.H
NAMES=['MLE','linear','constant','ewma_independent','ewma_shared','kalman','age_independent','age_shared']
def dump(n,x):
 (OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def bias(labels,m,fit,name):
 if labels.shape!=m.shape or fit<161:raise ValueError('invalid labels/fit')
 nums=[];dens=[];vals=[]
 for k,h in enumerate(H):
  idx=np.arange(max(120,fit-int(h)-480),fit-int(h))
  if len(idx)<40 or np.any(idx+h>=fit):raise ValueError('immature support')
  e=(labels[idx,k]-m[idx,k])/np.sqrt(h)
  if not np.isfinite(e).all():raise ValueError('nonfinite mature error')
  age=fit-1-idx if name.startswith('age') else fit-1-(idx+h)
  w=np.ones(len(idx)) if name=='constant' else np.exp(-age/30)
  nums.append(float(w@e));dens.append(float(w.sum()))
  if name=='kalman':
   b=0.;v=1.
   for z in e:
    v+=.01;gain=v/(v+1);b+=gain*(z-b);v*=1-gain
   vals.append(b)
  else:vals.append(nums[-1]/(20+dens[-1]))
 if name.endswith('shared'):vals=[sum(nums)/(20+sum(dens))]*3
 return np.array(vals)*np.sqrt(H)
def replay(p,m,s,orig,initial,name,labels=None):
 if labels is None:
  labels=np.full(m.shape,np.nan)
  for k,h in enumerate(H):labels[:-h,k]=p[h:]-p[:-h]
 pred=m[orig].copy();logs=[]
 for fit in range(initial,len(p),40):
  pos=np.where((orig>=fit)&(orig<fit+40))[0]
  if not len(pos):continue
  if name=='linear':pred[pos],_=old.correct(p,m,s,fit,orig[pos],dict(family='linear',seed=None,alpha=10.))
  elif name!='MLE':pred[pos]+=bias(labels,m,fit,name)
  logs.append(dict(fit=fit,h_support=[dict(h=int(h),max_origin=fit-int(h)-1,max_target=fit-1,n=min(480,fit-int(h)-120)) for h in H]))
 return pred,logs
def interval(pred,ref,y,years):
 rng=np.random.default_rng(73);draw=[]
 for _ in range(2000):
  cells=[]
  for year in ['2020','2021']:
   ix=np.where(years==year)[0];n=len(ix);starts=rng.integers(0,n,size=(n+39)//40);bs=ix[((starts[:,None]+np.arange(40))%n).ravel()[:n]]
   cells.extend((np.mean((pred[bs]-y[bs])**2-(ref[bs]-y[bs])**2,0)/np.mean(y[ix]**2,0)).tolist())
  draw.append(np.mean(cells))
 return np.quantile(draw,[.025,.975]).tolist()
def main():
 start=time.monotonic();dates,p=old.base.read_prefix();initial=int(np.sum(dates<'2020-01-01'))
 orig=np.array([t for t in range(initial,len(p)-20) if dates[t][:4]==dates[t+20][:4]])
 if len(orig)!=451:raise ValueError('origin count')
 m,s=old.history(p,initial);y=p[orig[:,None]+H]-p[orig,None];years=np.array([d[:4] for d in dates[orig]])
 saved=dict(origins=orig,dates=dates[orig],true_changes=y,issuance_mle=m,states=s);results=[];logs={}
 for name in NAMES:
  if time.monotonic()-start>900:raise TimeoutError('budget')
  pred,ll=replay(p,m,s,orig,initial,name);saved[name]=pred;logs[name]=ll
  score,rows=old.base.score(pred,y,years);results.append(dict(name=name,score=score,by_year=rows));print(name,score,flush=True)
 if abs(results[1]['score']-.942705467815)<1e-7:pass
 else:raise ValueError('linear replication mismatch')
 intervals={name:interval(saved['age_shared'],saved[name],y,years) for name in NAMES if name!='age_shared'}
 gates={r['name']:results[-1]['score']<r['score'] for r in results[:-1]}
 gates['yearwise_linear']=all(np.mean(a['relative_mse_h'])<np.mean(b['relative_mse_h']) for a,b in zip(results[-1]['by_year'],results[1]['by_year']))
 gates['block_interval']=intervals['linear'][1]<0
 np.savez_compressed(OUT/'PREDICTIONS_LOCAL_ONLY.npz',**saved)
 dump('ALL_RESULTS.json',results);dump('TRAINING_LOGS.json',logs);dump('INTERVALS.json',dict(block_length=40,replicates=2000,seed=73,candidate_minus_reference=intervals))
 dump('RUN_RECEIPT.json',dict(seconds=time.monotonic()-start,configurations=NAMES,gates=gates,escalate=all(gates.values()),development_only=True,read_2022=False))
 lines=['# 成熟发行残差：共享短期偏差检验','仅2020—2021重复开发，不是未见测试。冻结一次候选；不追加配置。','|方法|相对RW MSE|2020 h1/h5/h20|2021 h1/h5/h20|','|---|---|---|---|']
 for r in results:lines.append('|'+r['name']+'|'+str(r['score'])+'|'+str(r['by_year'][0]['relative_mse_h'])+'|'+str(r['by_year'][1]['relative_mse_h'])+'|')
 lines+=['','门检验：'+json.dumps(gates),'候选减linear时间块95%区间：'+str(intervals['linear']),'结论：'+('只能进入独立验证，不宣布创新。' if all(gates.values()) else '没有通过增量门，停止该候选，不挑配置。'),'','创新边界：这仍是带零中心收缩的指数加权多任务截距估计，不是新Kalman/DMA算法。发行年龄而不是成熟到达年龄只是具体估计差异；跨h误差相关使等权池化不能解释为独立观测。简单Kalman是到达时间随机游走偏差控制，不是严格延迟状态平滑最优基准；即便成功也须更强延迟滤波对照。sqrt(h)共享尺度是冻结假设，不经开发择优。','每h独立成熟支持保留h1最近信息，全部40点刷新。区间针对开发损失差、长度40年份内循环块，非新测试置信度；只有一个块长，不证明所有依赖尺度稳健。普通及-O验证结果见VERIFY文件；父核重评分另行完成。']
 (OUT/'REPORT_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
if __name__=='__main__':main()
