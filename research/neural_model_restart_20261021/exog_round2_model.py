# -*- coding: utf-8 -*-
"""Bounded candidate: age-conditioned dual-task exogenous lag attention.
Only current canonical targets; validation-only selection; full pre-cut refit.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse, copy, hashlib, json, time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.linear_model import Ridge
from scipy.stats import norm
from threadpoolctl import threadpool_limits
import multiscale_round1 as M

HERE = Path(__file__).resolve().parent
HOR, TAUS = M.HOR, M.TAUS
VARIANTS = ['endogenous_only', 'mean_pool', 'shared_query', 'dual_query', 'dual_no_age']
GROUPS = [[0,1,2], [3,4,5,6], [7,8]]
L, WIDTH = 20, 12

def hashfile(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def fit_exog(D, raw, refit):
    """Scaler only sees source observations in training-origin history, never test."""
    ids = D['indices']['development' if refit else 'train']
    last_t = int(D['t'][ids].max())
    levels, masks, source, dates = [raw[k] for k in ['levels','observed_mask','feature_source_date','date']]
    scale = np.ones(9)
    for f in range(9):
        good = masks[:last_t+1,f]
        # Avoid overweighting asof repeats in median_abs estimation.
        loc = np.flatnonzero(good)
        if len(loc):
            _, first = np.unique(source[loc,f], return_index=True)
            vals = np.abs(levels[loc[first],f])
            scale[f] = max(float(np.median(vals)), 1e-8)
    transformed = np.arcsinh(levels / scale)
    change = np.full_like(transformed, np.nan)
    valid = masks[1:] & masks[:-1] & (source[1:] > source[:-1])
    change[1:] = np.where(valid, transformed[1:]-transformed[:-1], np.nan)
    finite = np.isfinite(change)
    mean, std = np.zeros(9), np.ones(9)
    for f in range(9):
        vals = change[:last_t+1,f][finite[:last_t+1,f]]
        if len(vals):
            mean[f], std[f] = float(vals.mean()), max(float(vals.std()),1e-5)
    z = np.where(finite, (change-mean)/std, 0.).astype(np.float32)
    ts = D['t'][:,None]-np.arange(L-1,-1,-1)[None]
    x, mask = z[ts], finite[ts]
    # Difference uses both original rows; conservatively age its older endpoint.
    older = source.copy()
    older[1:] = source[:-1]
    origin_days = dates[D['t']].astype('datetime64[D]')[:,None,None]
    info = older[ts].astype('datetime64[D]')
    age = (origin_days-info).astype('timedelta64[D]').astype(np.float64)
    assert ((info < origin_days) | ~mask).all()
    age = np.where(mask, np.maximum(age,0), 0.).astype(np.float32)
    audit = {'last_scaler_origin_index':last_t,'last_scaler_origin_date':str(dates[last_t]), 'asinh_median_abs':scale.tolist(),'difference_mean':mean.tolist(),'difference_std':std.tolist(), 'valid_change_fraction':float(mask.mean()), 'all_missing_windows':int((~mask.any((1,2))).sum()), 'maximum_age_days':float(age.max()), 'input_rule':'20 canonical historical slots; changes require both finite endpoints and strictly increasing source date. No carried row treated as fresh zero change; no fill. Difference age uses older source endpoint relative to query origin.'}
    return {'exo':torch.tensor(x), 'mask':torch.tensor(mask), 'age':torch.tensor(np.log1p(age)/np.log(21.), dtype=torch.float32)}, audit

def prepared(D, raw, refit):
    X, base = M.prepare(D,refit)
    E, audit = fit_exog(D,raw,refit)
    X.update(E)
    return X, base, audit

class ExogNet(nn.Module):
    def __init__(self, base, variant):
        super().__init__()
        self.variant = variant
        self.carbon = nn.Sequential(nn.Linear(3*M.K*3+9,24),nn.Tanh(),nn.Linear(24,24),nn.Tanh())
        self.enc = nn.ModuleList([nn.Sequential(nn.Linear(3*len(g),WIDTH),nn.Tanh()) for g in GROUPS])
        self.pos = nn.Parameter(torch.zeros(L,WIDTH))
        nn.init.normal_(self.pos, std=.03)
        self.query_location = nn.Linear(24,WIDTH)
        self.query_gaps = nn.Linear(24,WIDTH)
        self.raw_age = nn.Parameter(torch.full((2,3),-2.))
        self.location = nn.Sequential(nn.Linear(24+3*WIDTH,24),nn.Tanh(),nn.Linear(24,1))
        self.gaps = nn.Sequential(nn.Linear(24+3*WIDTH,24),nn.Tanh(),nn.Linear(24,14))
        for head in [self.location,self.gaps]:
            nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)
        self.register_buffer('base',torch.tensor(base))
        self.register_buffer('hlog',torch.tensor(np.log(HOR)/np.log(max(HOR)),dtype=torch.float32))
    def attend(self, tokens, valid, age, query, task, group):
        if self.variant == 'mean_pool':
            weights = valid.float()/valid.sum(1,keepdim=True).clamp_min(1)
        else:
            logits = (tokens*query[:,None]).sum(-1)/(WIDTH**.5)
            if self.variant != 'dual_no_age':
                penalty = F.softplus(self.raw_age[0 if self.variant=='shared_query' else task,group])
                logits = logits-penalty*age
            weights = logits.masked_fill(~valid,-1e4).softmax(1)*valid
            weights = weights/weights.sum(1,keepdim=True).clamp_min(1e-8)
        return (tokens*weights[:,:,None]).sum(1)
    def forward(self, seq, ctx, hi, exo, mask, age):
        carbon = self.carbon(torch.cat([seq.flatten(1),ctx,self.hlog[hi,None]],1))
        if self.variant == 'endogenous_only':
            loc = gap = torch.zeros((len(hi),3*WIDTH),device=carbon.device)
        else:
            ql = self.query_location(carbon)
            qg = ql if self.variant=='shared_query' else self.query_gaps(carbon)
            locs,gaps = [],[]
            for gi,g in enumerate(GROUPS):
                obs = mask[:,:,g]
                value = exo[:,:,g]*obs
                token_age = torch.zeros_like(age[:,:,g]) if self.variant=='dual_no_age' else age[:,:,g]*obs
                token = self.enc[gi](torch.cat([value,obs.float(),token_age],2))+self.pos[None]
                valid = obs.any(-1)
                # Feature-wise ages enter a conservative oldest-observed group age.
                a = (age[:,:,g]*obs).amax(-1)
                locs.append(self.attend(token,valid,a,ql,0,gi))
                gaps.append(self.attend(token,valid,a,qg,1,gi))
            loc,gap = torch.cat(locs,1),torch.cat(gaps,1)
        base = self.base[hi]
        median = base[:,7]+self.location(torch.cat([carbon,loc],1)).squeeze(1)
        distance = (base[:,1:]-base[:,:-1])*torch.exp(.8*torch.tanh(self.gaps(torch.cat([carbon,gap],1))))
        cum = torch.cat([torch.zeros((len(hi),1),device=carbon.device),distance.cumsum(1)],1)
        return median[:,None]+cum-cum[:,7:8]

def forward(net,X,ids):
    return net(*[X[k][ids] for k in ['seq','ctx','hi','exo','mask','age']])

def train(net,X,ids,epochs,seed,va=None):
    torch.manual_seed(seed+180)
    opt = torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.001)
    ids = torch.tensor(ids)
    def val():
        net.eval()
        with torch.no_grad():
            score = M.losses(X['z'][va],forward(net,X,va))
            return float(torch.stack([score[X['hi'][va]==i].mean() for i in range(4)]).mean())
    best = val() if va is not None else None
    state,ep_best,bad = copy.deepcopy(net.state_dict()),0,0
    ran=0
    for ep in range(epochs):
        net.train()
        for idx in ids[torch.randperm(len(ids))].split(512):
            q = forward(net,X,idx)
            loss = (M.losses(X['z'][idx],q)*X['weight'][idx]).mean()
            assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
        ran=ep+1
        if va is not None:
            value=val()
            if value<best-1e-5:
                best,ep_best,bad=value,ep+1,0;state=copy.deepcopy(net.state_dict())
            else:
                bad+=1
            if ep>=15 and bad>=12:
                break
    if va is not None:
        net.load_state_dict(state)
    net.eval()
    return {'best_epoch':ep_best,'validation_normalized_pinball':best,'epochs_run':ran}

def fit(D, Xi,bi, Xf,bf,variant,seed,epochs):
    torch.manual_seed(seed)
    net=ExogNet(bi,variant)
    log=train(net,Xi,D['indices']['train'],epochs,seed,torch.tensor(D['indices']['validation']))
    torch.manual_seed(seed)
    final=ExogNet(bf,variant)
    train(final,Xf,D['indices']['development'],log['best_epoch'],seed)
    with torch.no_grad():
        q=forward(final,Xf,torch.tensor(D['indices']['test'])).numpy()*D['scale'][D['indices']['test'],None]
        # A group with no observations returns exactly zero independent of learned logits.
        zero=final.attend(torch.ones(2,L,WIDTH),torch.zeros(2,L,dtype=torch.bool),torch.zeros(2,L),torch.ones(2,WIDTH),0,0)
        assert torch.equal(zero,torch.zeros_like(zero))
    assert np.isfinite(q).all() and (np.diff(q,axis=1)>=-1e-6).all()
    log.update(seed=seed,refit_epochs=log['best_epoch'],parameter_count=sum(p.numel() for p in final.parameters()),age_penalties=F.softplus(final.raw_age).detach().numpy().tolist(),all_missing_attention_zero_verified=True)
    return q,log

def summaries(D,X):
    x,mask,age=[X[k].numpy() for k in ['exo','mask','age']]
    denom=mask.sum(1).clip(1)
    mean=(x*mask).sum(1)/denom
    sd=np.sqrt((((x-mean[:,None])*mask)**2).sum(1)/denom)
    fraction=mask.mean(1)
    ages=(age*mask).sum(1)/denom
    return np.concatenate([X['ctx'].numpy(),D['seq'][:,:,:,0].mean(2),D['seq'][:,:,:,0].std(2),mean,sd,fraction,ages],1)

def ridge(D,Xi,Xf):
    a,b=summaries(D,Xi),summaries(D,Xf)
    tr,va,dev,te=[D['indices'][s] for s in ['train','validation','development','test']]
    out=np.empty((len(te),15));fits={}
    for hi,h in enumerate(HOR):
        it,iv,ide=[z[D['hi'][z]==hi] for z in [tr,va,dev]]
        kt=np.flatnonzero(D['hi'][te]==hi)
        best=(np.inf,None)
        for alpha in [.1,1.,10.,100.]:
            model=Ridge(alpha=alpha).fit(a[it],D['z'][it])
            rq=np.quantile(D['z'][it]-model.predict(a[it]),TAUS)
            score=M.B.score(D['z'][iv],model.predict(a[iv])[:,None]+rq).mean()
            if score<best[0]:best=(float(score),alpha)
        final=Ridge(alpha=best[1]).fit(b[ide],D['z'][ide])
        rq=np.quantile(D['z'][ide]-final.predict(b[ide]),TAUS)
        out[kt]=(final.predict(b[te[kt]])[:,None]+rq)*D['scale'][te[kt],None]
        fits[str(h)]={'alpha':best[1],'validation_MQS':best[0]}
    return out,fits

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--epochs',type=int,default=64)
    parser.add_argument('--markets',nargs='+',default=['HUBEI','GUANGZHOU'])
    args=parser.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    panel=pd.read_csv(M.B.PANEL,parse_dates=['date'])
    inputfile=HERE/'exog_round2_inputs.npz'
    with np.load(inputfile,allow_pickle=False) as f:rawall={k:f[k] for k in f.files}
    report={'status':'exploratory candidate; reused outer history, no fresh confirmatory holdout or novelty claim','config':vars(args),'seeds':[11,22],'variants':VARIANTS,'horizons':HOR,'markets':{},'inputs_sha256':{'canonical':hashfile(M.B.PANEL),'exog_npz':hashfile(inputfile),'model_source':hashfile(__file__),'multiscale_source':hashfile(M.__file__)},'selection':'chronological purged validation only, epoch0 eligible; reset same seed and refit all precut targets with newly fitted scalers','exclusions':'EU all nine archived candidate columns missing, GZ DAX missing; no CEA/SH common source passed audit. Markets excluded by input coverage before model/test evaluation.'}
    arrays={'taus':TAUS};rows=[];start=time.perf_counter()
    for market in args.markets:
        d=panel[panel.market.eq(market)].sort_values('date').reset_index(drop=True)
        raw={k:rawall[k][rawall['market']==market] for k in ['date','price','levels','observed_mask','feature_source_date']}
        assert np.array_equal(raw['date'],d.date.to_numpy(dtype='datetime64[D]'))
        assert np.array_equal(raw['price'],d.price.to_numpy(float))
        assert raw['observed_mask'].any(), 'Unsupported no-coverage market'
        D=M.extract_data(d)
        Xi,bi,ai=prepared(D,raw,False);Xf,bf,af=prepared(D,raw,True)
        te=D['indices']['test'];hi=D['hi'][te];y=D['y'][te]
        preds={'empirical_anchor':bf[hi]*D['scale'][te,None], 'gaussian_rw':norm.ppf(TAUS)[None]*D['scale'][te,None]}
        preds['ridge_exog_summary'],rf=ridge(D,Xi,Xf)
        result={'audit':D['audit'],'scalers_inner_train':ai,'scalers_refit':af,'ridge_fits':rf,'fits':{},'by_horizon':{}}
        for variant in VARIANTS:
            qs=[];logs=[]
            for seed in [11,22]:
                q,log=fit(D,Xi,bi,Xf,bf,variant,seed,min(args.epochs,64));qs.append(q);logs.append(log)
                arrays[market+'_'+variant+'_seed'+str(seed)]=q
                print(market,variant,seed,'best',log['best_epoch'],'elapsed',round(time.perf_counter()-start,1),flush=True)
            preds[variant]=np.mean(qs,0);result['fits'][variant]=logs
        origins=d.date.to_numpy(dtype='datetime64[D]')[D['t'][te]]
        targets=d.date.to_numpy(dtype='datetime64[D]')[D['t'][te]+np.array(HOR)[hi]]
        arrays.update({market+'_origin_index':D['t'][te],market+'_origin_date':origins,market+'_target_date':targets,market+'_horizon':np.array(HOR)[hi],market+'_target_log_return':y})
        for name,q in preds.items():
            arrays[market+'_'+name]=q
            qs=M.B.score(y,q);ws=M.B.wis(y,q)
            for j in range(len(y)):
                rows.append({'market':market,'origin_date':str(origins[j]),'target_date':str(targets[j]),'horizon':int(HOR[hi[j]]),'model':name,'MQS':float(qs[j]),'WIS':float(ws[j]),'target_log_return':float(y[j]),'median':float(q[j,7]),'coverage80':int(q[j,2]<=y[j]<=q[j,12]),'coverage95':int(q[j,0]<=y[j]<=q[j,14])})
        for hidx,h in enumerate(HOR):
            keep=hi==hidx;scores={}
            for name,q in preds.items():
                yh,qh=y[keep],q[keep]
                scores[name]={'MQS':float(M.B.score(yh,qh).mean()),'WIS':float(M.B.wis(yh,qh).mean()),'MAE':float(np.abs(yh-qh[:,7]).mean()),'coverage80':float(((yh>=qh[:,2])&(yh<=qh[:,12])).mean()),'coverage95':float(((yh>=qh[:,0])&(yh<=qh[:,14])).mean())}
            result['by_horizon'][str(h)]={'n':int(keep.sum()),'scores':scores}
        report['markets'][market]=result
        report['elapsed_seconds']=time.perf_counter()-start
        (HERE/'exog_round2_model_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        np.savez_compressed(HERE/'exog_round2_model_predictions.npz',**arrays)
        pd.DataFrame(rows).to_csv(HERE/'exog_round2_model_origin_scores.csv',index=False)
    report['summary']={}
    for base in [k for k in preds if k!='dual_query']:
        ratios=[cell['scores']['dual_query']['MQS']/cell['scores'][base]['MQS'] for m in report['markets'].values() for cell in m['by_horizon'].values()]
        report['summary']['dual_query_vs_'+base]={'median_MQS_ratio':float(np.median(ratios)),'mean_log_MQS_ratio':float(np.log(ratios).mean()),'wins':int((np.array(ratios)<1).sum()),'cells':len(ratios)}
    report['elapsed_seconds']=time.perf_counter()-start
    report['input_hashes_unchanged_after_run']={'canonical':hashfile(M.B.PANEL)==report['inputs_sha256']['canonical'],'exog_npz':hashfile(inputfile)==report['inputs_sha256']['exog_npz']}
    assert all(report['input_hashes_unchanged_after_run'].values())
    (HERE/'exog_round2_model_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 外生分任务滞后注意力候选小实验','','候选机制：碳价历史状态与logh生成query，energy/equity-risk/fx各组20个历史槽位，以位置与正分位间距分别选择滞后；logit减可学习age惩罚。不是首次注意力主张。','','只在输入审计确认有覆盖的HUBEI/GUANGZHOU运行；EU九列全NA。相同canonical目标、原点和h1/5/10/20，seed11/22均值；epoch0可选，最多64轮，验证早停，选轮后从头全precut refit；asinh/diff/scaler分别在inner train和最终refit重估。','','|dual_query相对基线|MQS中位比|平均log比|胜出cell|','|---|---:|---:|---:|']
    for name,s in report['summary'].items():lines.append(f"|{name}|{s['median_MQS_ratio']:.5f}|{s['mean_log_MQS_ratio']:.5f}|{s['wins']}/{s['cells']}|")
    lines+=['','|市场|h|模型|MQS|WIS|80%覆盖|95%覆盖|','|---|---:|---|---:|---:|---:|---:|']
    for m,res in report['markets'].items():
        for h,cell in res['by_horizon'].items():
            for model,s in cell['scores'].items():lines.append(f"|{m}|{h}|{model}|{s['MQS']:.6f}|{s['WIS']:.6f}|{s['coverage80']:.4f}|{s['coverage95']:.4f}|")
    lines+=['','限制：外生真实发布日/上游填补未核实；age只是归档来源日的保守年龄；GZ尾部陈旧达260天，DAX全缺失；EU不具备候选覆盖。测试历史已用于本项目探索，不能作为新盲测。单次有界比较不支持创新优先权或确定正增量；详细seed选轮、缩放、逐原点评分与15分位预测均已保存。','',f"运行秒数：{report['elapsed_seconds']:.1f}。"]
    (HERE/'exog_round2_model_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(report['summary'],ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    with threadpool_limits(limits=2):main()
