"""One frozen low-capacity conditional asymmetric residual candidate."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time
from pathlib import Path
import numpy as np
from scipy.linalg import solve
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
sys.path.insert(0,str(ROOT/'research/deep_direction_review_20261013/calibration'))
import calibrate as base
base.OUT=OUT
H=base.H

def history(p,initial):
 bank,ll,pars=base.filter_bank(p);m=np.zeros((len(p),3));states=np.zeros((len(p),2));j=0
 for t in range(120,len(p)):
  # Historical issuance parameters refresh on same 40-point lattice.
  fit=initial+40*((t-initial)//40)
  fit=max(120,fit)
  if t==120 or (t-initial)%40==0:
   start=max(1,fit-480);n=fit-start;a=ll[start:fit].sum(0)
   j=int(np.argmin(a[:,0]+n*np.log(np.maximum(a[:,1]/n,1e-14))))
  m[t]=bank[t,j]
  sd=max(float(np.std(np.diff(p[max(0,t-120):t+1]))),1e-8)
  states[t,0]=np.tanh((p[t]-np.mean(p[t-59:t+1]))/(sd*np.sqrt(60)))
  states[t,1]=np.std(np.diff(p[t-20:t+1]))/sd
 return m,states

def design(m,s,family,seed):
 g=s[:,0];v=s[:,1];one=np.ones(len(m))
 if family=='feedback':return one[:,None]
 if family=='amplitude':return np.column_stack([one,m])
 if family=='candidate':return np.column_stack([one,m,m*g,np.abs(m)*g])
 if family=='symmetric':return np.column_stack([one,m,m*g])
 if family=='no_state':return np.column_stack([one,m,np.abs(m)])
 if family=='linear':return np.column_stack([one,m,g,v])
 if family=='rff':
  x=np.column_stack([m,g,v]);rng=np.random.default_rng(seed)
  return np.column_stack([one,np.sqrt(2/8)*np.cos(x@rng.normal(size=(3,8))/np.sqrt(3)+rng.uniform(0,2*np.pi,8))])
 raise ValueError(family)

def correct(p,m,s,fit,orig,c):
 out=m[orig].copy();logs=[]
 for k,h in enumerate(H):
  idx=np.arange(max(120,fit-int(h)-480),fit-int(h))
  if len(idx)<40 or idx[-1]+h>=fit:raise ValueError('maturity/support')
  A=design(m[idx,k],s[idx],c['family'],c['seed']);T=design(m[orig,k],s[orig],c['family'],c['seed'])
  sd=np.sqrt(np.mean(A*A,0));sd[sd<1e-8]=1;A=A/sd;T=T/sd
  y=p[idx+h]-p[idx]-m[idx,k]
  coef=solve(A.T@A+c['alpha']*len(idx)*np.eye(A.shape[1]),A.T@y,assume_a='pos')
  out[:,k]+=T@coef
  logs.append(dict(fit=int(fit),h=int(h),min_origin=int(idx[0]),max_origin=int(idx[-1]),max_target=int(idx[-1]+h),n=len(idx),coefficients=(coef/sd).tolist(),dimension=A.shape[1]))
 return out,logs

def configurations():
 cs=[dict(name='MLE480',family='MLE',alpha=None,seed=None)]
 for family in ['feedback','amplitude','candidate','symmetric','no_state','linear','rff']:
  for a in [.1,1.,10.]:
   for seed in ([17,29] if family=='rff' else [None]):
    cs.append(dict(name=f'{family}_a{a}_s{seed}',family=family,alpha=a,seed=seed))
 return cs

def main():
 start=time.monotonic();cs=configurations();base.dump('CONFIGURATIONS.json',cs)
 dates,p=base.read_prefix();initial=int(np.sum(dates<'2020-01-01'))
 orig=np.array([t for t in range(initial,len(p)-20) if dates[t][:4]==dates[t+20][:4]])
 if len(orig)!=451:raise ValueError('451 origins')
 m,s=history(p,initial);true=p[orig[:,None]+H]-p[orig,None];years=np.array([d[:4] for d in dates[orig]])
 saved=dict(origins=orig,dates=dates[orig],true_changes=true,issuance_mle=m,states=s);results=[];logs=[]
 for c in cs:
  pred=m[orig].copy()
  if c['family']!='MLE':
   for fit in range(initial,len(p),40):
    if time.monotonic()-start>900:raise TimeoutError('frozen budget')
    pos=np.where((orig>=fit)&(orig<fit+40))[0]
    if len(pos):
     pred[pos],ll=correct(p,m,s,fit,orig[pos],c);logs.extend([dict(name=c['name'],**l) for l in ll])
  value,rows=base.score(pred,true,years)
  results.append(dict(configuration=c,score=value,by_year=rows));saved[c['name']]=pred
  base.dump('ALL_RESULTS.json',results);base.dump('TRAINING_LOGS.json',logs)
  print(c['name'],value,flush=True)
 np.savez_compressed(OUT/'PREDICTIONS_LOCAL_ONLY.npz',**saved)
 agg=[]
 for family in ['MLE','feedback','amplitude','candidate','symmetric','no_state','linear','rff']:
  for a in ([None] if family=='MLE' else [.1,1.,10.]):
   rr=[r for r in results if r['configuration']['family']==family and r['configuration']['alpha']==a]
   agg.append(dict(family=family,alpha=a,score=float(np.mean([r['score'] for r in rr])),year_h=np.mean([[row['relative_mse_h'] for row in r['by_year']] for r in rr],axis=0).tolist(),seeds=[r['configuration']['seed'] for r in rr]))
 base.dump('AGGREGATES.json',agg)
 best={f:min([r for r in agg if r['family']==f],key=lambda r:r['score']) for f in {r['family'] for r in agg}}
 candidate=best['candidate'];comparators=['MLE','feedback','amplitude','linear','rff']
 gates={f:candidate['score']<best[f]['score'] for f in comparators}
 gates['yearwise_MLE']=bool(np.all(np.mean(candidate['year_h'],axis=1)<np.mean(best['MLE']['year_h'],axis=1)))
 gates['state_ablation']=candidate['score']<best['no_state']['score'];gates['asymmetry_ablation']=candidate['score']<best['symmetric']['score']
 base.dump('RUN_RECEIPT.json',dict(seconds=time.monotonic()-start,configs=len(cs),fit_h_requests=len(logs),initial=initial,origins=len(orig),development_selection=True,read_2022=False,best=best,gates=gates,escalate=all(gates.values())))
 lines=['# 状态条件非对称暂态修正实际开发结果','', '所有结果是2020—2021开发选择成绩，不是独立测试。唯一候选、冻结配置，未读取2022。低容量参数并不构成算法创新；属于平滑状态转换/非对称回归校准邻域，创新待证。','', '|族（族内开发择优）|alpha|相对RW MSE|2020 h1/h5/h20|2021 h1/h5/h20|','|---|---|---|---|---|']
 for f in ['MLE','feedback','amplitude','candidate','symmetric','no_state','linear','rff']:
  r=best[f];lines.append(f"|{f}|{r['alpha']}|{r['score']:.8f}|{r['year_h'][0]}|{r['year_h'][1]}|")
 lines+=['','门检验：'+json.dumps(gates,ensure_ascii=False),'结论：'+('只可继续独立验证，不宣布创新。' if all(gates.values()) else '未通过全部强对照与消融门，不升级；停止，不追加救结果配置。'),'','每h反馈具有相同成熟支持但不同h之间支持不同。MLE只使用一步创新标签，修正采用已发行MLE预测误差，不能声称MLE拥有同样多步反馈。截距也使用零中心岭惩罚。RFF固定随机节点8个，两个种子平均损失；未择种子。没有读取旧权重、修改旧稿、提交或推送。', '验证：python research/multi_angle_model_search_20261013/state/test_state.py；python research/multi_angle_model_search_20261013/state/verify.py。']
 (OUT/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
if __name__=='__main__':main()
