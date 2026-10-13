"""Truncated symmetric rolling mean feasibility; not a new decomposition."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,hashlib
from pathlib import Path
import numpy as np
from scipy.linalg import solve
from scipy.stats import spearmanr
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
sys.path.insert(0,str(ROOT/'research/deep_direction_review_20261013/calibration'))
import calibrate as base
base.OUT=OUT

def dump(name,obj):base.dump(name,obj)

def smooth(p,s,t,w):
    return float(np.mean(p[max(0,s-w):min(s+w,t)+1]))

def features(p,w):
    # Every row recreates its OWN issuance vintage, not a final smoothed history.
    n=len(p);X=np.zeros((n,18));rev=np.full(n,np.nan)
    for s in range(n-w):rev[s]=smooth(p,s,s+w,w)-smooth(p,s,s,w)
    for t in range(8,n):
        ids=np.arange(t-8,t+1)
        trend=np.array([smooth(p,int(s),t,w) for s in ids])
        resid=p[ids]-trend
        X[t,:8]=np.diff(trend)[::-1];X[t,8:16]=np.diff(resid)[::-1]
        last=t-w-1
        if last>=19:
            r=rev[last-19:last+1]
            if not np.isfinite(r).all():raise ValueError('revision maturity failed')
            X[t,16:]=[r.mean(),np.sqrt(np.mean(r*r))]
    return X,rev

def ridge(p,X,fit,origins,alpha):
    idx=np.arange(max(60,fit-20-480),fit-20)
    if idx[-1]+20>=fit:raise ValueError('training maturity')
    A=X[idx];mu=A.mean(0);sd=A.std(0);sd[sd==0]=1;A=(A-mu)/sd
    Y=p[idx[:,None]+base.H]-p[idx,None];ym=Y.mean(0)
    B=solve(A.T@A+alpha*len(idx)*np.eye(X.shape[1]),A.T@(Y-ym),assume_a='pos')
    return ((X[origins]-mu)/sd)@B+ym,dict(fit_end_exclusive=int(fit),max_training_target=int(idx[-1]+20),min_training_origin=int(idx[0]),n_training_origins=len(idx),feature_dimensions=X.shape[1],alpha=alpha,coefficients=B.tolist(),intercept=ym.tolist(),feature_mean=mu.tolist(),feature_sd=sd.tolist())

def corr(x,y):
    return dict(pearson=float(np.corrcoef(x,y)[0,1]),spearman=float(spearmanr(x,y).statistic))

def bootstrap_diff(a,b,true,years,rng):
    values=[]
    for _ in range(500):
        cells=[]
        for y in ['2020','2021']:
            ix=np.where(years==y)[0];starts=rng.integers(0,len(ix),int(np.ceil(len(ix)/20)))
            take=ix[np.concatenate([(s+np.arange(20))%len(ix) for s in starts])[:len(ix)]]
            cells.extend((((a[take]-true[take])**2-(b[take]-true[take])**2).mean(0)/(true[take]**2).mean(0)).tolist())
        values.append(float(np.mean(cells)))
    return np.quantile(values,[.025,.5,.975]).tolist()

def main():
    begin=time.monotonic();dates,p=base.read_prefix();initial=int(np.sum(dates<'2020-01-01'))
    origins=np.array([t for t in range(initial,len(p)-20) if dates[t][:4]==dates[t+20][:4]])
    if len(origins)!=451:raise ValueError('451 protocol')
    years=np.array([d[:4] for d in dates[origins]]);true=p[origins[:,None]+base.H]-p[origins,None]
    bank=base.filter_bank(p);raw=base.raw_features(p)
    banks={};revs={}
    for w in [5,10,20]:banks[w],revs[w]=features(p,w)
    configs=[dict(name='RW',family='RW'),dict(name='MLE480',family='MLE')]
    for a in [.1,1.,10.]:
        for d in [8,16]:configs.append(dict(name=f'RAW{d}_a{a}',family='RAW',alpha=a,dim=d))
        for w in [5,10,20]:
            for f in ['DECOMP','MATURE']:configs.append(dict(name=f'{f}_w{w}_a{a}',family=f,w=w,alpha=a))
    dump('CONFIGURATIONS.json',configs)
    results=[];logs=[];saved=dict(origins=origins,dates=dates[origins],true_changes=true)
    for c in configs:
        pr=np.full(true.shape,np.nan)
        for fit in range(initial,len(p),40):
            if time.monotonic()-begin>900:raise TimeoutError('budget exceeded')
            pos=np.where((origins>=fit)&(origins<fit+40))[0]
            if not len(pos):continue
            if c['family'] in ['RW','MLE']:
                cc=dict(family=c['family'],window=480);a,log=base.predict(p,fit,origins[pos],cc,dict(bank=bank))
            else:
                X=raw[:,:c['dim']] if c['family']=='RAW' else banks[c['w']][:,:16 if c['family']=='DECOMP' else 18]
                a,log=ridge(p,X,fit,origins[pos],c['alpha'])
                if c['family']!='RAW':log.update(max_revision_source_offset=-c['w']-1,max_revision_maturity_offset=-1)
            pr[pos]=a;logs.append(dict(name=c['name'],**log))
        if not np.isfinite(pr).all():raise ValueError('forecast finite')
        score,rows=base.score(pr,true,years)
        results.append(dict(name=c['name'],configuration=c,score=score,by_year=rows));saved[c['name']]=pr
        print(c['name'],score,flush=True)
    diagnostics=[];rng=np.random.default_rng(20261013)
    byname={r['name']:r for r in results}
    for w in [5,10,20]:
        # Predeclared alpha=1 mechanism diagnostic, all other configs retained.
        a=f'DECOMP_w{w}_a1.0';b=f'MATURE_w{w}_a1.0';rev=revs[w]
        rows=[]
        for y in ['2020','2021']:
            ix=origins[years==y];valid=ix[ix+w<len(p)]
            rr=np.sqrt(np.mean(rev[valid]**2));dr=np.sqrt(np.mean(np.diff(p[np.r_[valid[0]-1,valid]])**2))
            mask=years==y;hist=banks[w][origins[mask],17]
            associations=[corr(hist,np.abs(saved[a][mask,j]-true[mask,j])) for j in range(3)]
            rows.append(dict(year=y,count=len(ix),future_revision_diagnostic_count=len(valid),revision_rms=rr,increment_rms=dr,revision_ratio=rr/dr,association_h=associations))
        ci=bootstrap_diff(saved[b],saved['MLE480'],true,years,rng)
        ci_ab=bootstrap_diff(saved[b],saved[a],true,years,rng)
        h1=all(r['revision_ratio']>=.1 for r in rows)
        h2=all(c['pearson']>.1 for r in rows for c in r['association_h'])
        h3=byname[b]['score']<=.99*byname[a]['score'] and byname[b]['score']<byname['MLE480']['score'] and ci[-1]<0
        diagnostics.append(dict(w=w,alpha=1.,by_year=rows,bootstrap_mature_minus_mle=ci,bootstrap_mature_minus_decomp=ci_ab,H1=bool(h1),H2=bool(h2),H3=bool(h3),mechanism_established=bool(h1 and h2 and h3)))
    np.savez_compressed(OUT/'PREDICTIONS_LOCAL_ONLY.npz',**saved)
    dump('ALL_RESULTS.json',results);dump('TRAINING_LOGS.json',logs);dump('DIAGNOSTICS.json',diagnostics)
    dump('RUN_RECEIPT.json',dict(seconds=time.monotonic()-begin,budget_seconds=900,configs=len(configs),origins=len(origins),initial=initial,fit_requests=len(logs),development_only=True,read_2022=False,method='truncated symmetric moving average',not_vmd_ceemdan=True))
    lines=['# 有界因果分解实际试验','', '不是VMD/CEEMDAN，也不是新分解算法。仅广州作者报告价格；无外生变量。2020—2021已用开发数据，451同年完整目标起点，1/5/20观察点预测。所有结果为原价平方误差的统一相对RW评分。','', '| 配置 | 评分 |','|---|---|']
    lines += [f"| {r['name']} | {r['score']:.8f} |" for r in results]
    lines += ['', '## 机制判断',json.dumps(diagnostics,ensure_ascii=False,indent=2),'','未建立机制时只说明当前数据条件不足，不升级注意力。分解8+残差8和RAW16同维同Ridge，MATURE只追加2项历史已成熟修订信息；RAW8是重构信息相同但维数不同的补充基线。训练只使用原价增量目标，未替换为分量损失。future_revision是事后诊断，不作当前特征。有限样本关联不代表因果关系；alpha及窗口搜索均为开发探索，置信区间未校正多重搜索。MLE采用更多近期单步成熟标签，信息支持不同。bootstrap为年份内循环20点块，500次，种子20261013。预测数组只供本地审核。']
    (OUT/'REPORT_ZH.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':main()
