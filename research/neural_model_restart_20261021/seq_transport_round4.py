# -*- coding: utf-8 -*-
"""Conditional LSTM location/scale + moment-normalized empirical shape transport.
Only seq_transport_round4* is written. Exact CRPS and actual seed mixtures.
"""
from __future__ import annotations
import os,sys
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
sys.dont_write_bytecode=True
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
from threadpoolctl import threadpool_limits
import empirical_tilt_round3 as T
import deep_seq_round4 as S
from verify_empirical_affine_round4 import cdf_integral
P=Path(__file__).resolve().parent
VARIANTS=['seq_transport','no_sequence_transport','seq_location_scale_only','seq_reweight_only']
SEEDS=[11,22,33,44,55]
FLOOR=1e-8


def add_cache(bank,times,dtype=torch.float32):
    z=bank['z'].to(dtype);allowed=bank['mature'][None,:]<=times[:,None]
    count=allowed.sum(1);u=allowed.to(dtype)/count.clamp_min(1)[:,None]
    m0=(u*z).sum(1);v0=(u*(z-m0[:,None])**2).sum(1)
    bank.update(z=z,phi2=bank['phi'][:,:2].to(dtype),allowed=allowed,count=count,m0=m0,v0=v0)
    return bank


def operator(raw,bank,sids,local,variant):
    z=bank['z'];allowed=bank['allowed'][sids]
    beta=.35*torch.tanh(raw[:,2:4])
    if variant=='seq_location_scale_only':beta=beta*0
    logits=beta@bank['phi2'].T
    # Bounded logits in [-2.1,2.1]: exp-normalization is a stable exact softmax.
    unnormalized=torch.exp(logits)*allowed.to(logits.dtype)
    w=unnormalized/unnormalized.sum(1,keepdim=True).clamp_min(1e-30)
    m0=bank['m0'][sids];v0=bank['v0'][sids]
    mw=(w*z).sum(1);vw=(w*(z-mw[:,None])**2).sum(1)
    delta=1.5*torch.tanh(raw[:,0]);ell=1.5*torch.tanh(raw[:,1])
    mt=m0+torch.sqrt(v0.clamp_min(FLOOR))*delta
    st=torch.sqrt(v0.clamp_min(FLOOR))*torch.exp(ell)
    if variant=='seq_reweight_only':a=torch.ones_like(mt);b=torch.zeros_like(mt)
    else:a=st/torch.sqrt(vw.clamp_min(FLOOR));b=mt-a*mw
    p=.8*local+.2*torch.full_like(local,bank['p'])
    return dict(w=w,p=p,a=a,b=b,beta=beta,v0=v0,vw=vw,mt=mt,st=st,count=bank['count'][sids])


def fast_crps(y,z,o):
    w,p,a,b=o['w'],o['p'],o['a'],o['b']
    m=a[:,None]*z+b[:,None]
    cw=w.cumsum(1)-w;cwz=(w*z).cumsum(1)-w*z
    half=(w*(z*cw-cwz)).sum(1)
    return (1-p)*y.abs()+p*(w*(m-y[:,None]).abs()).sum(1)-p*p*a*half-p*(1-p)*(w*m.abs()).sum(1)


def cheap_quantiles(z,o):
    w,p,a,b=o['w'],o['p'],o['a'],o['b'];m=a[:,None]*z+b[:,None]
    below=p*(w*(m<0)).sum(1);through=1-p+p*(w*(m<=0)).sum(1)
    tau=T.TT.to(z.dtype)[None,:].expand(len(w),-1)
    target=torch.where(tau<below[:,None],tau/p[:,None].clamp_min(1e-20),(tau-(1-p[:,None]))/p[:,None].clamp_min(1e-20)).clamp(0,1)
    cw=w.cumsum(1).contiguous();cw[:,-1]=1
    ix=torch.searchsorted(cw,target.contiguous()).clamp_max(len(z)-1)
    q=a[:,None]*z[ix]+b[:,None]
    return torch.where((tau>=below[:,None])&(tau<=through[:,None]),torch.zeros_like(q),q)


def generic_distribution(y,values,weights):
    values,order=torch.sort(values,dim=1);weights=weights.gather(1,order)
    cw=weights.cumsum(1);cw0=cw-weights;aw=(weights*values).cumsum(1)-weights*values
    crps=(weights*(values-y[:,None]).abs()).sum(1)-(weights*(values*cw0-aw)).sum(1)
    cw[:,-1]=1
    ix=torch.searchsorted(cw.contiguous(),T.TT.to(values.dtype)[None,:].expand(len(y),-1).contiguous()).clamp_max(values.shape[1]-1)
    q=values.gather(1,ix)
    p_eff=(weights*(values!=0)).sum(1)
    return q,crps,p_eff,values,weights


def mixture(y,z,ops):
    vals=[];ww=[];p=torch.stack([o['p'].double() for o in ops]).mean(0)
    for o in ops:
        w=o['w'].double();w=w/w.sum(1,keepdim=True)
        vals.append(o['a'].double()[:,None]*z.double()+o['b'].double()[:,None])
        ww.append(o['p'].double()[:,None]*w/len(ops))
    vals.append(torch.zeros_like(y[:,None],dtype=torch.float64));ww.append((1-p)[:,None])
    return generic_distribution(y.double(),torch.cat(vals,1),torch.cat(ww,1))


def stats_new():return dict(query_visits=0,min_uniform_variance=None,min_weighted_variance=None,uniform_floor_visits=0,weighted_floor_visits=0,min_allowed_count=None)

def stats_add(s,o):
    v0=o['v0'].detach();vw=o['vw'].detach();n=len(v0)
    s['query_visits']+=n
    for key,value in [('min_uniform_variance',float(v0.min())),('min_weighted_variance',float(vw.min())),('min_allowed_count',int(o['count'].min()))]:s[key]=value if s[key] is None else min(s[key],value)
    s['uniform_floor_visits']+=int((v0<FLOOR).sum());s['weighted_floor_visits']+=int((vw<FLOOR).sum())


def sanity():
    dtype=torch.float64;z=torch.linspace(-2.3,3.1,40,dtype=dtype)
    bank=dict(z=z,phi=torch.tensor(T.basis(z.numpy()),dtype=dtype),mature=torch.arange(40),p=.71)
    add_cache(bank,torch.tensor([31,33,36,39]),dtype);sid=torch.arange(4);local=torch.tensor([.1,.3,.7,.95],dtype=dtype)
    zero=operator(torch.zeros(4,4,dtype=dtype),bank,sid,local,'seq_transport')
    restore=float((zero['a'][:,None]*z+zero['b'][:,None]-z).abs().max());assert restore<1e-12
    assert float((zero['w']*(~bank['allowed'])).abs().max())==0
    raw=torch.tensor([[.1,-.07,.2,-.1],[-.2,.1,-.3,.1],[.2,.2,.1,.3],[-.1,.12,-.2,.4]],dtype=dtype)
    o=operator(raw,bank,sid,local,'seq_transport');m=o['a'][:,None]*z+o['b'][:,None]
    em=(o['w']*m).sum(1);ev=(o['w']*(m-em[:,None])**2).sum(1)
    me=float((em-o['mt']).abs().max());ve=float((ev-o['st']**2).abs().max());assert me<1e-10 and ve<1e-10
    y=torch.tensor([-.7,.31,0.,2.],dtype=dtype);q,c,p,values,weights=mixture(y,z,[o])
    fast=fast_crps(y,z,o);cdf=cdf_integral(y.numpy(),values.numpy(),weights.numpy())
    ce=float(np.max(np.abs(fast.numpy()-cdf)));assert ce<1e-10
    assert float((q-cheap_quantiles(z,o)).abs().max())<1e-10
    op0=dict(o);op0['p']=torch.zeros(4,dtype=dtype);ae=float((fast_crps(y,z,op0)-y.abs()).abs().max());assert ae<1e-12
    g=raw[:1].clone().requires_grad_()
    grad=torch.autograd.gradcheck(lambda head:fast_crps(y[1:2],z,operator(head,bank,torch.tensor([0]),local[1:2],'seq_transport')),(g,),eps=1e-6,atol=1e-5,rtol=1e-4)
    o2=operator(-raw,bank,sid,local,'seq_transport');qq,mix,_,vv,ww=mixture(y,z,[o,o2]);integral=cdf_integral(y.numpy(),vv.numpy(),ww.numpy())
    ee=float(np.max(np.abs(mix.numpy()-integral)));assert ee<1e-10
    shape=raw.clone();shape[:,2:]*=-2;other=operator(shape,bank,sid,local,'seq_transport')
    assert torch.allclose(o['mt'],other['mt']) and torch.allclose(o['st'],other['st'])
    return dict(zerohead_mature_subset_restore_error=restore,conditional_mean_error=me,conditional_variance_error=ve,fast_CRPS_vs_independent_CDF_error=ce,pureatom_error=ae,gradient_check=bool(grad),different_support_mixture_CDF_error=ee,future_weight_exact_zero=True,shape_preserves_target_moments=True)


class TransportNet(nn.Module):
    def __init__(self,variant):
        super().__init__();self.variant=variant;self.encoder=nn.LSTM(2,16,batch_first=True);self.context=nn.Sequential(nn.Linear(36,16),nn.Tanh());self.shared=nn.Sequential(nn.Linear(32,16),nn.Tanh());self.head=nn.Linear(16,4);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,X,idx):
        if self.variant=='no_sequence_transport':
            out,_=self.encoder(torch.zeros_like(X['seq'][:1]));h=out[:,-1].expand(len(idx),-1)
        else:
            unique,inverse=torch.unique(X['sid'][idx],return_inverse=True)
            out,_=self.encoder(X['seq'][unique]);h=out[:,-1][inverse]
        return self.head(self.shared(torch.cat([h,self.context(X['x'][idx])],1)))


def prepare(D,part):
    X,banks=T.prepare(D,D[part],True)
    times,first,inverse=np.unique(D['t'],return_index=True,return_inverse=True)
    X['seq']=torch.tensor(D['seq'][first]);X['sid']=torch.tensor(inverse);X['origin_times']=torch.tensor(times)
    for b in banks:add_cache(b,X['origin_times'])
    return X,banks


def eligible(D,X,banks,part):
    ids=[]
    for h,b in enumerate(banks):
        rows=torch.tensor(D[part][D['hi'][D[part]]==h]);rows=rows[b['count'][X['sid'][rows]]>=32];ids.append(rows)
    out=torch.cat(ids);assert len(out)>32;return out


def run_fit(D,X,banks,variant,seed,args,part,epochs,validation=None):
    torch.manual_seed(seed);net=TransportNet(variant);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.003)
    ids=eligible(D,X,banks,part);origins=torch.unique(X['sid'][ids]);sids=X['sid'][ids];counts=torch.bincount(X['hi'][ids],minlength=4);rowweight=len(ids)/(4*counts.float())
    stats=stats_new();vstats=stats_new();encoder_gradient=False
    initial=[p.detach().clone() for p in net.encoder.parameters()]
    def evaluate():
        net.eval();loss=[]
        with torch.no_grad():
            rows=torch.tensor(validation);raw=net(X,rows)
            for h,b in enumerate(banks):
                mask=X['hi'][rows]==h;ii=rows[mask]
                o=operator(raw[mask],b,X['sid'][ii],X['local'][ii],variant);stats_add(vstats,o)
                q=cheap_quantiles(b['z'],o);e=X['z'][ii,None]-q
                loss.append(torch.maximum(T.TT*e,(T.TT-1)*e).mean())
        return float(torch.stack(loss).mean())
    best=evaluate() if validation is not None else None;best_ep=0;bad=0;hist=[[0,best]] if validation is not None else []
    torch.manual_seed(seed+731)
    for ep in range(1,epochs+1):
        net.train();order=origins[torch.randperm(len(origins))]
        for group in order.split(128):
            batch=ids[torch.isin(sids,group)];raw=net(X,batch);loss=torch.tensor(0.)
            for h,b in enumerate(banks):
                mask=X['hi'][batch]==h;ii=batch[mask]
                if not len(ii):continue
                o=operator(raw[mask],b,X['sid'][ii],X['local'][ii],variant);stats_add(stats,o)
                loss=loss+(fast_crps(X['z'][ii],b['z'],o)*rowweight[h]).sum()/len(batch)+.003*o['beta'].square().mean()*len(ii)/len(batch)
            assert torch.isfinite(loss);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
            if not encoder_gradient:encoder_gradient=any(p.grad is not None and bool(torch.any(p.grad!=0)) for p in net.encoder.parameters())
            opt.step()
        if validation is not None:
            value=evaluate();hist.append([ep,value])
            if value<best-1e-6:best,best_ep,bad=value,ep,0
            else:bad+=1
            if ep>=args.min_epochs and bad>=args.patience:break
    change=float(sum(((p.detach()-q)**2).sum() for p,q in zip(net.encoder.parameters(),initial)).sqrt())
    return net,dict(best_epoch=best_ep,validation_normalized_macro_pinball=best,epochs_run=ep if epochs else 0,validation_history=hist,eligible_queries=len(ids),encoder_nonzero_gradient=encoder_gradient,encoder_parameter_L2_change=change,training_geometry=stats,validation_geometry=vstats)


def evaluate_seed(net,D,X,banks):
    te=torch.tensor(D['te']);Q=np.empty((len(te),15));C=np.empty(len(te));PRO=np.empty(len(te));BRANCH=np.empty(len(te));AA=np.empty(len(te));BB=np.empty(len(te));ops={};diag={};net.eval()
    with torch.no_grad():
        raw=net(X,te)
        for h,b in enumerate(banks):
            positions=np.flatnonzero(D['hi'][D['te']]==h);ii=te[positions];o=operator(raw[positions],b,X['sid'][ii],X['local'][ii],net.variant);stats=stats_new();stats_add(stats,o)
            assert stats['min_allowed_count']>=32
            q,c,p,v,w=mixture(X['z'][ii],b['z'],[o]);sc=D['scale'][D['te'][positions]]
            Q[positions]=q.numpy()*sc[:,None];C[positions]=c.numpy()*sc;PRO[positions]=p.numpy();BRANCH[positions]=o['p'].numpy();AA[positions]=o['a'].numpy();BB[positions]=o['b'].numpy()
            marks=o['a'].double()[:,None]*b['z'].double()+o['b'].double()[:,None]
            final_mark_w=o['w'].double()/o['w'].double().sum(1,keepdim=True)
            collision=(marks==0)&(final_mark_w>0);cm=o['p'].double()*(final_mark_w*collision).sum(1)
            diag[str(T.HOR[h])]=dict(**stats,collision_count=int(collision.sum()),collision_mass_max=float(cm.max()),collision_mass_mean=float(cm.mean()),final_nonzero_vs_branch_max_error=float((p-o['p'].double()).abs().max()))
            ops[h]={key:o[key].detach().clone() for key in ['w','p','a','b']}
    assert np.isfinite(Q).all() and np.all(np.diff(Q,axis=1)>=-1e-8) and np.isfinite(C).all()
    return (Q,C,PRO),BRANCH,AA,BB,ops,diag


def metrics(y,q,c,p):
    return dict(MQS=float(T.B.score(y,q).mean()),CRPS=None if c is None else float(np.mean(c)),WIS=float(T.B.wis(y,q).mean()),MAE=float(np.abs(y-q[:,7]).mean()),coverage80=float(((y>=q[:,2])&(y<=q[:,12])).mean()),coverage95=float(((y>=q[:,0])&(y<=q[:,14])).mean()),Brier=None if p is None else float(np.mean((p-(y!=0))**2)))


def summary(markets):
    cells=[c['scores'] for v in markets.values() for c in v['by_horizon'].values()];out={}
    pairs=[('seq_transport',b) for b in VARIANTS[1:]+['frozen_tilt','qrf_original','rule_qrf','lstm_qr','uniform_rule']]
    for m,b in pairs:
        out[m+'_vs_'+b]={}
        for metric in ['MQS','CRPS','WIS','MAE']:
            valid=[c for c in cells if c[b][metric] is not None];rr=[c[m][metric]/c[b][metric] for c in valid]
            if rr:out[m+'_vs_'+b][metric]=dict(geometric_gain_percent=float(100*(1-np.exp(np.log(rr).mean()))),wins=int(np.sum(np.array(rr)<1)),cells=len(rr))
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--markets',nargs='+',default=list(T.B.CUT));ap.add_argument('--seeds',nargs='+',type=int,default=SEEDS);ap.add_argument('--epochs',type=int,default=64);ap.add_argument('--min-epochs',type=int,default=30);ap.add_argument('--patience',type=int,default=16);ap.add_argument('--tag',default='seq_transport_round4');args=ap.parse_args();assert args.tag.startswith('seq_transport_round4')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);check=sanity();print('geometry',json.dumps(check),flush=True)
    frozen_paths=[]
    for pat in ['empirical_tilt_round3*','qrf_round4*','deep_seq_round4*','verify_empirical_affine_round4*']:
        frozen_paths.extend(p for p in P.glob(pat) if p.is_file())
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen_paths}
    base=np.load(P/'empirical_tilt_round3_followup_predictions.npz');qrf=np.load(P/'qrf_round4_predictions.npz');rule=np.load(P/'qrf_round4_probability_control_predictions.npz');qr=np.load(P/'deep_seq_round4_predictions.npz');panel=pd.read_csv(T.B.PANEL,parse_dates=['date']);start=time.perf_counter()
    report=dict(status='exploratory same reused holdouts, no publication/novelty success claim',config=vars(args),geometry=check,frozen_sha256=hashes,mechanism='uniform matured subset mean/std + predicted bounded location/log-scale; tilted empirical shape normalized to prescribed conditional moments',probability='branch p=.8local+.2fitted nonzero fraction fixed; Brier uses final F nonzero mass, including exact transformed-zero collisions',variance_floor=FLOOR,floor_count_unit='query visits including repeated training epochs; no unit-variance claim for floor-active/degenerate subsets',batching='group128origins; deduplicate64step LSTM computation, all4h train from first epoch',ensemble='merge all seed query-specific a_s*z+b_s supports and zero, equal probability mixture; sort then exact ECDF CRPS and inverseCDF',checkpoints='per market stores20 fitted model states, common refit support/phi/maturity and scaler for reconstruction of w',markets={})
    arrays={};rows=[]
    for market in args.markets:
        data=panel[panel.market==market].sort_values('date').reset_index(drop=True);D=S.build(data);te=D['te'];hi=D['hi'][te];y=D['y'][te];sc=D['scale'][te]
        for field,value in [('origin',D['t'][te]),('horizon',T.HOR[hi]),('target',y),('scale',sc)]:
            for source in [base,qrf,rule,qr]:assert np.array_equal(value,source[market+'_'+field])
            arrays[market+'_'+field]=value
        arrays[market+'_origin_date']=base[market+'_origin_date'];Xi,bi=prepare(D,'tr');Xf,bf=prepare(D,'rf');states={}
        refs={'frozen_tilt':(base,'full_ensemble'),'qrf_original':(qrf,'qrf_ensemble'),'rule_qrf':(rule,'qrf_rule_atom'),'uniform_rule':(base,'uniform_rule')}
        pred={m:source[market+'_'+key+'_Q'] for m,(source,key) in refs.items()};cr={m:source[market+'_'+key+'_CRPS'] for m,(source,key) in refs.items()};pro={m:source[market+'_'+key+'_p'] for m,(source,key) in refs.items()}
        pred['lstm_qr']=qr[market+'_lstm_qr_Q'];cr['lstm_qr']=None;pro['lstm_qr']=None
        mres=dict(audit=D['audit'],fits={},by_horizon={},strict_alignment=True)
        for variant in VARIANTS:
            allops=[];seedcs=[];fits=[]
            for seed in args.seeds:
                _,info=run_fit(D,Xi,bi,variant,seed,args,'tr',args.epochs,D['va']);ep=info['best_epoch']
                net,refit=run_fit(D,Xf,bf,variant,seed,args,'rf',ep)
                (q,c,p),branch,aa,bb,ops,diagnostics=evaluate_seed(net,D,Xf,bf)
                expected=rule[market+'_qrf_rule_atom_p'];assert np.array_equal(branch,expected)
                states[f'{variant}_seed{seed}']={k:v.detach().clone() for k,v in net.state_dict().items()}
                key=f'{market}_{variant}_seed{seed}'
                for suffix,value in [('Q',q),('CRPS',c),('p',p),('branch_p',branch),('a',aa),('b',bb)]:arrays[key+'_'+suffix]=value
                allops.append(ops);seedcs.append(c);info.update(seed=seed,refit_epochs=ep,refit=refit,test_geometry=diagnostics,allocated_parameters=sum(p.numel() for p in net.parameters()));fits.append(info)
                for j,h in enumerate(T.HOR):
                    mask=hi==j;rows.append(dict(market=market,horizon=int(h),model=variant,seed=seed,n=int(mask.sum()),**metrics(y[mask],q[mask],c[mask],p[mask])))
                print(market,variant,seed,'best',ep,'run',info['epochs_run'],'elapsed',round(time.perf_counter()-start,1),flush=True)
            Q=np.empty((len(te),15));C=np.empty(len(te));PR=np.empty(len(te));convex=0.
            for j,b in enumerate(bf):
                positions=np.flatnonzero(hi==j);ops=[oo[j] for oo in allops]
                qq,cc,pp,_,_=mixture(torch.tensor(D['z'][te[positions]]),b['z'],ops)
                Q[positions]=qq.numpy()*sc[positions,None];C[positions]=cc.numpy()*sc[positions];PR[positions]=pp.numpy();convex=max(convex,float(np.max(C[positions]-np.mean(seedcs,0)[positions])))
            assert convex<1e-6 and np.isfinite(Q).all() and np.all(np.diff(Q,axis=1)>=-1e-8)
            pred[variant]=Q;cr[variant]=C;pro[variant]=PR;mres['fits'][variant]=fits;mres.setdefault('mixture_convexity_error',{})[variant]=convex
        meta=dict(context_mean=D['x_aug'][D['rf']].mean(0),context_std=D['x_aug'][D['rf']].std(0)+1e-5,banks=[{k:b[k] for k in ['z','phi2','mature','p']} for b in bf],state_dicts=states,variants=VARIANTS,config=vars(args),panel_sha256=hashlib.sha256(T.B.PANEL.read_bytes()).hexdigest())
        torch.save(meta,P/f'{args.tag}_{market}_checkpoints.pt')
        for j,h in enumerate(T.HOR):
            mask=hi==j;ss={m:metrics(y[mask],q[mask],None if cr[m] is None else cr[m][mask],None if pro[m] is None else pro[m][mask]) for m,q in pred.items()};mres['by_horizon'][str(h)]=dict(n=int(mask.sum()),scores=ss)
            for m in pred:rows.append(dict(market=market,horizon=int(h),model=m,seed='ensemble',n=int(mask.sum()),**ss[m]))
        for m in pred:
            arrays[market+'_'+m+'_Q']=pred[m]
            if cr[m] is not None:arrays[market+'_'+m+'_CRPS']=cr[m]
            if pro[m] is not None:arrays[market+'_'+m+'_p']=pro[m]
        report['markets'][market]=mres;report['summary']=summary(report['markets']);report['elapsed_seconds']=time.perf_counter()-start
        (P/(args.tag+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');np.savez_compressed(P/(args.tag+'_predictions.npz'),**arrays);pd.DataFrame(rows).to_csv(P/(args.tag+'_cells.csv'),index=False)
    assert all(hashlib.sha256((P/name).read_bytes()).hexdigest()==sha for name,sha in hashes.items());report['frozen_sha_unchanged']=True;report['elapsed_seconds']=time.perf_counter()-start
    (P/(args.tag+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# LSTM条件位置尺度与矩归一经验形状输运','',f'完成{sum(len(fs) for m in report["markets"].values() for fs in m["fits"].values())}fit，运行{report["elapsed_seconds"]:.2f}秒。四消融/所有市场/seed均保留，不按test选择。','',
      '每query仅用t_key+h<=query_origin的至少32个非零响应。均匀成熟子集给m0/s0；有界两系数倾斜给w，再减mw除sw并施加目标m0+s0*delta、s0*exp(ell)。LSTM16、ctx16、shared16及4zerohead同分配参数；no-seq将64步输入置0，loc-scale-only将beta置0，reweight-only不施加affine。','',
      '训练使用给定O(M)完整CRPS，max64/min30/pat16，AdamW.001/wd.003及beta²正则.003，same inner equal-horizon normalized pinball选epoch0和full-precut refit。方差floor1e-8仅数值保护，实际最小方差/floor query访问次数逐fit保存，不声称退化subset有单位方差。','',
      'p_rule是固定branch mixing概率；输运mark碰到0时，最终非零概率按完整F重算用于Brier，碰撞count/mass另存。五seed支持按query变化，合并全部a_s*z+b_s及零atom后排序，精确计算同一mixture的Q/CRPS，没有平均参数/seed分数/quantiles冒充mixture。','',
      '| seq_transport / 参照 | MQS改善 | 完整CRPS改善 |','|---|---:|---:|']
    for key,val in report['summary'].items():
        q=val['MQS'];c=val.get('CRPS');ct='不可由15q识别' if c is None else f"{c['geometric_gain_percent']:+.4f}% ({c['wins']}/20)";lines.append(f"| {key} | {q['geometric_gain_percent']:+.4f}% ({q['wins']}/20) | {ct} |")
    lines+=['','正值表示seq_transport更好。所有模型单seed与mixture Q/CRPS/最终p、branch p及a/b另存NPZ；每市场checkpoint包含全部seed state/refitbank/scaler，可重建w。canonical origin/h/target/scale严格对齐四类冻结参考，所有读取旧文件SHA保持不变。','', '这是历史复用测试上的模型探索，不证明新颖性、显著性或可发表成功；负结果不否定整个ML/DL方向。']
    (P/(args.tag+'_report_zh.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(report['summary'],indent=2),flush=True);print('elapsed',report['elapsed_seconds'],'frozen unchanged',True,flush=True)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    with threadpool_limits(limits=1):main()
