"""Bounded Stage F baseline recalibration; no archive execution or old weights."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import csv,hashlib,json,time,zipfile,stat
from pathlib import Path,PurePosixPath
import numpy as np
from scipy.linalg import solve
H=np.array([1,5,20])
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
ZIP=Path('D:/默认下载/carbon_stageF_20261013_complete.zip')

def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def configurations():
    return ([dict(family='RW')]+[dict(family='DRIFT',window=w) for w in [120,240,480]]+
            [dict(family='RIDGE',lags=l,alpha=a,window=480) for l in [8,60] for a in [.001,.01,.1,1.]]+
            [dict(family='MLE',window=w) for w in [120,240,480]])

def filter_bank(p):
    pars=np.array([(0.,1.)]+[(r,q) for r in [0.,.25,.5,.75,.9,.97] for q in [.02,.1,.25,.5,.8]])
    rho,q=pars.T;r=1-q;z=np.zeros(len(pars));P=r/(1-rho*rho)
    pred=np.zeros((len(p),len(pars),3));ll=np.zeros((len(p),len(pars),2))
    for t in range(1,len(p)):
        e=p[t]-p[t-1]-(rho-1)*z;S=(rho-1)**2*P+q+r;C=rho*(rho-1)*P+r
        z=rho*z+C/S*e;P=np.maximum(rho*rho*P+r-C*C/S,0)
        pred[t]=(rho[:,None]**H-1)*z[:,None];ll[t,:,0]=np.log(S);ll[t,:,1]=e*e/S
    return pred,ll,pars

def raw_features(p):
    x=np.zeros((len(p),60))
    for t in range(60,len(p)):x[t]=np.diff(p[t-60:t+1])[::-1]
    return x

def predict(p,fit_end,origins,c,cache=None):
    # Parameters use only p[:fit_end]; causal features/filter states may use p[:t+1].
    family=c['family'];origins=np.asarray(origins,int)
    log=dict(fit_end_exclusive=int(fit_end),max_parameter_price_index=int(fit_end-1))
    if family=='RW':return np.zeros((len(origins),3)),log
    if family=='DRIFT':
        start=max(0,fit_end-c['window']-1);d=np.diff(p[start:fit_end]).mean()
        log.update(increment_count=fit_end-start-1,training_label_kind='one_step_increments')
        return np.tile(H*d,(len(origins),1)),log
    if family=='MLE':
        bank,ll,pars=filter_bank(p[:origins.max()+1]) if cache is None else cache['bank']
        start=max(1,fit_end-c['window']);n=fit_end-start;a=ll[start:fit_end].sum(0)
        j=int(np.argmin(a[:,0]+n*np.log(np.maximum(a[:,1]/n,1e-14))))
        log.update(increment_count=n,training_label_kind='one_step_Gaussian_innovations',selected_rho=float(pars[j,0]),selected_permanent_fraction=float(pars[j,1]))
        return bank[origins,j],log
    raw=raw_features(p) if cache is None else cache['raw']
    if c['lags']==8:
        widths=[1,1,1,2,5,10,20,20];q=np.zeros((8,60));s=0
        for j,w in enumerate(widths):q[j,s:s+w]=1/np.sqrt(w);s+=w
        X=raw@q.T
    else:X=raw
    idx=np.arange(max(60,fit_end-20-c['window']),fit_end-20)
    if len(idx)<40:raise ValueError('Insufficient mature support')
    if idx[-1]+20>=fit_end:raise ValueError('Target maturity violation')
    Y=p[idx[:,None]+H]-p[idx,None]
    mu=X[idx].mean(0);sd=X[idx].std(0);sd[sd==0]=1
    A=(X[idx]-mu)/sd;ym=Y.mean(0)
    B=solve(A.T@A+c['alpha']*len(idx)*np.eye(A.shape[1]),A.T@(Y-ym),assume_a='pos')
    log.update(training_label_kind='complete_1_5_20_targets',n_training_origins=len(idx),max_training_target=int(idx[-1]+20),min_training_origin=int(idx[0]))
    return ((X[origins]-mu)/sd)@B+ym,log

def score(pred,true,years):
    rows=[]
    for year in sorted(set(years)):
        m=years==year;den=np.mean(true[m]**2,0)
        if np.any(den<=0):raise ValueError('Degenerate RW denominator')
        mse=np.mean((pred[m]-true[m])**2,0)
        rows.append(dict(year=str(year),origins=int(m.sum()),mse_h=mse.tolist(),rw_mse_h=den.tolist(),relative_mse_h=(mse/den).tolist()))
    return float(np.mean([r['relative_mse_h'] for r in rows])),rows

def read_prefix():
    path=ROOT/'data/parsed/figshare_28682348/guangzhou_reported.csv'
    consumed=[]
    with path.open('rb',buffering=0) as f:
        # Exact record count from prior audit; never request the next record.
        for _ in range(951):
            line=bytearray()
            while True:
                b=f.read(1)
                if not b:raise ValueError('Premature EOF')
                line+=b
                if b==b'\n':break
            consumed.append(bytes(line))
    raw=b''.join(consumed);rows=list(csv.DictReader(raw.decode('utf-8-sig').splitlines()))
    dates=np.array([r['date'] for r in rows]);p=np.array([float(r['author_reported_price']) for r in rows])
    if len(p)!=950 or dates[-1][:4]!='2021' or np.any(dates>'2021-12-31') or not np.all(dates[1:]>dates[:-1]) or not np.isfinite(p).all():raise ValueError('Prefix audit failure')
    dump('INPUT_AUDIT.json',dict(path=str(path),rows=950,first_date=dates[0],last_date=dates[-1],prefix_sha256=hashlib.sha256(raw).hexdigest(),bytes_consumed=len(raw),boundary_record_requested=False,dependency='Prior independently audited 950-row prefix; no whole-file hash'))
    return dates,p

def main():
    begin=time.monotonic();dates,p=read_prefix();initial=int(np.sum(dates<'2020-01-01'))
    origins=np.array([t for t in range(initial,len(p)-20) if dates[t][:4]==dates[t+20][:4]],int)
    if len(origins)!=451:raise ValueError(f'Expected 451 origins, got {len(origins)}')
    true=p[origins[:,None]+H]-p[origins,None];years=np.array([d[:4] for d in dates[origins]])
    with zipfile.ZipFile(ZIP) as z:
        infos=z.infolist();names=[i.filename for i in infos]
        if len(names)!=len(set(names)) or sum(i.file_size for i in infos)>200_000_000:raise ValueError('Unsafe archive size/duplicates')
        for i in infos:
            pp=PurePosixPath(i.filename)
            if pp.is_absolute() or '..' in pp.parts or '\\' in i.filename or stat.S_ISLNK(i.external_attr>>16):raise ValueError('Unsafe archive path')
        sources={name:hashlib.sha256(z.read('carbon_stageF_20261013/'+name)).hexdigest() for name in ['src/model.py','src/market.py','src/experiment.py','prior/lowdim.py']}
        dump('SOURCE_REVIEW.json',dict(zip=str(ZIP),entries=len(infos),expanded_bytes=sum(i.file_size for i in infos),selected_source_crc_checked=True,full_archive_crc_checked=False,reason='Do not read archived price data or old weights',source_sha256=sources,extracted=False,archive_code_executed=False,old_weights_loaded=False))
    cache=dict(raw=raw_features(p),bank=filter_bank(p));results=[];logs=[];saved=dict(origins=origins,dates=dates[origins],true_changes=true)
    for c in configurations():
        name=c['family']+'_'+ '_'.join(f'{k}{v}' for k,v in c.items() if k!='family');pr=np.full(true.shape,np.nan)
        for fit_end in range(initial,len(p),40):
            pos=np.where((origins>=fit_end)&(origins<fit_end+40))[0]
            if not len(pos):continue
            a,log=predict(p,fit_end,origins[pos],c,cache);pr[pos]=a;log.update(configuration=name);logs.append(log)
        if not np.isfinite(pr).all():raise ValueError('Nonfinite forecast')
        value,rows=score(pr,true,years);results.append(dict(name=name,configuration=c,score=value,by_year=rows,rmse_h=np.sqrt(np.mean((pr-true)**2,0)).tolist()))
        saved[name]=pr;print(name,value,flush=True)
    np.savez_compressed(OUT/'PREDICTIONS_LOCAL_ONLY.npz',**saved)
    dump('ALL_RESULTS.json',results);dump('TRAINING_LOGS.json',logs)
    best={f:min([r for r in results if r['configuration']['family']==f],key=lambda r:r['score']) for f in ['RW','DRIFT','RIDGE','MLE']}
    dump('RUN_RECEIPT.json',dict(seconds=time.monotonic()-begin,compute_budget_seconds=900,configs=len(results),fit_requests=len(logs),initial_rows=initial,common_origins=len(origins),by_year={y:int(np.sum(years==y)) for y in sorted(set(years))},best_by_family=best,development_only=True,seed_ensemble=False))
    lines=['# 广州统一校准实际执行报告','',f'使用广州作者报告价格的950行许可前缀，最后日期为{dates[-1]}。2020和2021均为已经使用过的开发数据，不是未见测试集。','',f'预测步长为交易观察点1、5、20；同年完整目标共同起点451个，初始训练{initial}行。每40个观察点重新估计参数，估计边界为独占fit_end，当前预测可使用当前起点价格。无外生变量、无分解。','', '损失保留原价平方误差；每个年份、每个步长分别除以同一单元的RW均方误差，再对两个年份与三个步长等权平均。价格增量预测加上起点价格，与原价预测的平方误差完全相同。','', '| 方法（开发集事后择优） | 配置 | 等权相对RW MSE |','|---|---|---|']
    for f,r in best.items():lines.append(f"| {f} | {r['name']} | {r['score']:.8f} |")
    lines+=['','## 方法与信息差异','- RW没有训练标签。局部漂移使用截至参数边界的120/240/480个一步增量，块内漂移不再更新。','- Ridge使用最多480个共同成熟的1/5/20目标起点；8维为Stage F的分块线性投影（未使用局部波动率归一化），60维为原始倒序增量。训练标准化、无惩罚截距，alpha乘以训练起点数。8维版本明确是补充校准基线，不声称复现Stage F的LINEAR8。60维匹配Stage F LINEAR60定义。','- 低维MLE是Stage F CORE_ML的31点有限参数网格，不是连续优化MLE。采用永久随机游走加AR(1)暂态价格滤波，Gaussian创新方差整体剖面消去；窗口120/240/480。参数估计仅使用成熟一步创新，而过滤状态随起点因果更新。','- 因此MLE/漂移获得更多近期的一步标签，不能把优劣全归因于模型结构；Ridge训练标签严格完全成熟，最晚20步目标不越过参数边界。','','## 限制与审核','15个预列配置均实际运行，完整结果和训练边界见JSON；各族最佳配置是在2020—2021开发评分后选择，不能作为独立确认，也未作多重比较校正。未安装或调用torch，未执行归档代码，未加载旧权重。未读取2022价格或下一边界记录。预测数组含价格变化信息，仅限本地，GitHub同步前应排除或进一步审核。','此前两个梯度步弱原型的失败不能作为机制无效证据；本次仅校准简单基线，不检验或否定新机制。',f'实际计算用时：{time.monotonic()-begin:.3f}秒；预算900秒。没有提交或推送。']
    (OUT/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':main()
