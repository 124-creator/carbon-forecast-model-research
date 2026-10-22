# -*- coding: utf-8 -*-
"""Round5: forest-reference embedded neural moment-normalized transport.
Only forest_transport_round5* outputs. Imported round4 source files stay frozen.
"""
from __future__ import annotations
import os,sys
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
sys.dont_write_bytecode=True
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
import seq_transport_round4 as R
P=Path(__file__).resolve().parent
LEARNED=['forest_transport','forest_location_scale_only','forest_reweight_only']
REFERENCE='forest_reference_only'
SEEDS=[11,22,33,44,55]
GAMMA=.05
_BASE_CHEAP=R.cheap_quantiles
_RAW_DIAGNOSTIC=None


def diagnostic_quantiles(z,o):
    q=_BASE_CHEAP(z,o)
    state=_RAW_DIAGNOSTIC;bank=o.get('_bank')
    if state is not None and bank is not None and bank.get('phase')=='inner' and torch.equal(o['_sids'],bank['diagnostic_val_sids']):
        e=bank['diagnostic_val_y'][:,None]-q.double()*bank['diagnostic_val_scale'][:,None]
        tau=torch.tensor(R.T.B.TAUS,dtype=torch.float64)
        score=float(2*torch.maximum(tau*e,(tau-1)*e).mean())
        state['current'].append(float(np.log(max(score,1e-30)/max(bank['diagnostic_anchor_MQS'],1e-30))))
        if len(state['current'])==4:
            state['history'].append([len(state['history']),float(np.mean(state['current']))]);state['current']=[]
    return q


def forest_operator(raw,bank,sids,local,variant):
    z=bank['z'];u=bank['u'][sids]
    beta=.35*torch.tanh(raw[:,2:4])
    if variant in ['forest_location_scale_only',REFERENCE]:beta=beta*0
    numerator=u*torch.exp(beta@bank['phi2'].T)
    w=numerator/numerator.sum(1,keepdim=True).clamp_min(1e-30)
    m0=bank['m0'][sids];v0=bank['v0'][sids];vu=bank['vu'][sids]
    mw=(w*z).sum(1);vw=(w*(z-mw[:,None])**2).sum(1)
    delta=1.5*torch.tanh(raw[:,0]);ell=1.5*torch.tanh(raw[:,1])
    mt=m0+torch.sqrt(v0.clamp_min(R.FLOOR))*delta
    st=torch.sqrt(v0.clamp_min(R.FLOOR))*torch.exp(ell)
    if variant in ['forest_reweight_only',REFERENCE]:a=torch.ones_like(mt);b=torch.zeros_like(mt)
    else:a=st/torch.sqrt(vw.clamp_min(R.FLOOR));b=mt-a*mw
    p=.8*local+.2*torch.full_like(local,bank['p'])
    return dict(w=w,p=p,a=a,b=b,beta=beta,v0=v0,vw=vw,vu=vu,mt=mt,st=st,count=bank['count'][sids],_bank=bank,_sids=sids)


def stats_new():
    return dict(query_visits=0,min_reference_variance=None,min_weighted_variance=None,min_uniform_variance=None,reference_floor_visits=0,weighted_floor_visits=0,min_allowed_count=None,min_reference_uniform_variance_ratio=None,min_weighted_uniform_variance_ratio=None)


def stats_add(s,o):
    v0,vw,vu=[o[k].detach() for k in ['v0','vw','vu']];s['query_visits']+=len(v0)
    values=[('min_reference_variance',float(v0.min())),('min_weighted_variance',float(vw.min())),('min_uniform_variance',float(vu.min())),('min_allowed_count',int(o['count'].min()))]
    good=vu>0
    if bool(good.any()):values += [('min_reference_uniform_variance_ratio',float((v0[good]/vu[good]).min())),('min_weighted_uniform_variance_ratio',float((vw[good]/vu[good]).min()))]
    for key,value in values:s[key]=value if s[key] is None else min(s[key],value)
    s['reference_floor_visits']+=int((v0<R.FLOOR).sum());s['weighted_floor_visits']+=int((vw<R.FLOOR).sum())


def activate():
    # Local process bindings only: no source/old result file is edited.
    R.operator=forest_operator;R.stats_new=stats_new;R.stats_add=stats_add;R.cheap_quantiles=diagnostic_quantiles


def run_inner(D,X,banks,model,seed,args):
    global _RAW_DIAGNOSTIC
    state={'history':[],'current':[]};_RAW_DIAGNOSTIC=state
    try:net,info=R.run_fit(D,X,banks,model,seed,args,'tr',args.epochs,D['va'])
    finally:_RAW_DIAGNOSTIC=None
    assert not state['current'] and len(state['history'])==info['epochs_run']+1
    best=min(state['history'],key=lambda row:row[1])
    info.update(raw_relative_validation_history=state['history'],best_raw_epoch=best[0],best_raw_log_MQS_ratio=best[1],raw_diagnostic_only=True)
    return net,info


def reference_moments(bank,u):
    bank['u']=u
    ref=u/u.sum(1,keepdim=True).clamp_min(1e-30)
    mean=(ref*bank['z']).sum(1)
    bank['m0']=mean;bank['v0']=(ref*(bank['z']-mean[:,None])**2).sum(1)


def sanity():
    dtype=torch.float64;z=torch.linspace(-2.3,3.1,40,dtype=dtype)
    bank=dict(z=z,phi=torch.tensor(R.T.basis(z.numpy()),dtype=dtype),mature=torch.arange(40),p=.71)
    R.add_cache(bank,torch.tensor([31,33,36,39]),dtype);bank['vu']=bank['v0'].clone()
    allowed=bank['allowed'];uniform=allowed.to(dtype)/allowed.sum(1,keepdim=True)
    rawu=torch.exp(torch.sin(torch.arange(160,dtype=dtype).reshape(4,40)*.37)*2)*allowed
    rawu=rawu/rawu.sum(1,keepdim=True);u=.95*rawu+.05*uniform
    reference_moments(bank,u)
    sid=torch.arange(4);local=torch.tensor([.1,.3,.7,.95],dtype=dtype);head=torch.zeros(4,4,dtype=dtype)
    zero=forest_operator(head,bank,sid,local,'forest_transport');ref=forest_operator(head,bank,sid,local,REFERENCE)
    ie=float((zero['a'][:,None]*z+zero['b'][:,None]-z).abs().max());assert ie<1e-12
    we=float((zero['w']-u/u.sum(1,keepdim=True)).abs().max());assert we<1e-12
    assert float((zero['w']*(~allowed)).abs().max())==0
    head=torch.tensor([[.1,-.07,.2,-.1],[-.2,.1,-.3,.1],[.2,.2,.1,.3],[-.1,.12,-.2,.4]],dtype=dtype)
    o=forest_operator(head,bank,sid,local,'forest_transport');v=o['a'][:,None]*z+o['b'][:,None]
    mean=(o['w']*v).sum(1);var=(o['w']*(v-mean[:,None])**2).sum(1)
    me=float((mean-o['mt']).abs().max());ve=float((var-o['st']**2).abs().max());assert me<1e-11 and ve<1e-11
    y=torch.tensor([-.7,.31,0.,2.],dtype=dtype);q,c,p,values,weights=R.mixture(y,z,[o]);closed=R.fast_crps(y,z,o)
    error=float(np.max(np.abs(closed.numpy()-R.cdf_integral(y.numpy(),values.numpy(),weights.numpy()))));assert error<1e-11
    zero_q,zero_c,_,_,_=R.mixture(y,z,[zero]);ref_q,ref_c,_,_,_=R.mixture(y,z,[ref]);assert torch.equal(zero_q,ref_q) and torch.allclose(zero_c,ref_c,atol=1e-12,rtol=0)
    headg=head[:1].clone().requires_grad_()
    grad=torch.autograd.gradcheck(lambda raw:R.fast_crps(y[1:2],z,forest_operator(raw,bank,torch.tensor([0]),local[1:2],'forest_transport')),(headg,),eps=1e-6,atol=1e-5,rtol=1e-4)
    other=forest_operator(-head,bank,sid,local,'forest_transport');_,mixed,_,vv,ww=R.mixture(y,z,[o,other]);em=float(np.max(np.abs(mixed.numpy()-R.cdf_integral(y.numpy(),vv.numpy(),ww.numpy()))));assert em<1e-11
    atom=dict(o);atom['p']=torch.zeros_like(o['p']);assert torch.allclose(R.fast_crps(y,z,atom),y.abs(),atol=1e-12,rtol=0)
    assert float((o['v0']/o['vu']).min())>=GAMMA-1e-12 and float((o['vw']/o['vu']).min())>=GAMMA*np.exp(-4.2)-1e-12
    return dict(nonuniform_zerohead_support_error=ie,nonuniform_zerohead_weight_error=we,conditional_mean_error=me,conditional_variance_error=ve,fast_CRPS_independent_CDF_error=error,gradient_check=bool(grad),different_support_mixture_CDF_error=em,pureatom=True,future_weight_exact_zero=True,reference_lower_bound_pass=True)


def load_market(D,market):
    path=P/f'forest_prior_round5_{market}.pt';cache=torch.load(path,map_location='cpu',weights_only=False)
    assert cache['market']==market and cache['gamma']==GAMMA and cache['seeds']==SEEDS
    times,first,inverse=np.unique(D['t'],return_index=True,return_inverse=True)
    assert np.array_equal(cache['times'].numpy(),times) and np.array_equal(cache['row_sid'].numpy(),inverse)
    for key in ['t','hi','y','scale','z','tr','va','rf','te']:assert np.array_equal(cache['rows'][key].numpy(),D[key])
    prepared={};audits={}
    for phase,part in [('inner','tr'),('final','rf')]:
        source=cache['phases'][phase];X,banks=R.prepare(D,part)
        boundary=D['audit']['validation_start_index' if phase=='inner' else 'cut_index']
        assert source['phase']==phase and source['part']==part and source['boundary']==boundary
        assert np.array_equal(source['fit_D_indices'].numpy(),D[part])
        assert torch.equal(source['x'],X['x']) and torch.equal(source['row_sid'],X['sid'])
        assert np.array_equal(source['scaler']['mean'].numpy(),D['x_aug'][D[part]].mean(0))
        assert np.array_equal(source['scaler']['std'].numpy(),D['x_aug'][D[part]].std(0)+1e-5)
        assert np.all(D['t'][D[part]]+R.T.HOR[D['hi'][D[part]]]<boundary)
        checks=[]
        for hi,(bank,cached) in enumerate(zip(banks,source['banks'])):
            fit=D[part][D['hi'][D[part]]==hi];nz=fit[np.abs(D['y'][fit])>1e-12];nz=nz[np.argsort(D['z'][nz],kind='stable')]
            assert np.array_equal(cached['bank_D_indices'].numpy(),nz)
            for key in ['z','mature','allowed','count']:assert torch.equal(cached[key],bank[key])
            assert cached['p_static']==bank['p']
            u=cached['u'];assert u.shape==bank['allowed'].shape and torch.isfinite(u).all() and bool(torch.all(u>=0))
            assert bool(torch.all(u[~bank['allowed']]==0))
            good=bank['count']>0;assert torch.allclose(u[good].sum(1),torch.ones(int(good.sum())),atol=2e-6,rtol=0)
            uniform=bank['allowed'].float()/bank['count'].clamp_min(1)[:,None]
            assert bool(torch.all(u+1e-8>=GAMMA*uniform))
            bank['vu']=bank['v0'].clone();reference_moments(bank,u)
            eligible=bank['count']>=32;zero=forest_operator(torch.zeros(int(eligible.sum()),4),bank,torch.nonzero(eligible,as_tuple=True)[0],torch.full((int(eligible.sum()),),.6),'forest_transport')
            ae=float((zero['a']-1).abs().max());be=float(zero['b'].abs().max());assert ae<2e-6 and be<2e-5
            bank['phase']=phase;anchor=None
            if phase=='inner':
                vr=D['va'][D['hi'][D['va']]==hi];vs=X['sid'][torch.tensor(vr)]
                bank['diagnostic_val_sids']=vs;bank['diagnostic_val_y']=torch.tensor(D['y'][vr],dtype=torch.float64);bank['diagnostic_val_scale']=torch.tensor(D['scale'][vr],dtype=torch.float64)
                vo=forest_operator(torch.zeros(len(vr),4),bank,vs,X['local'][torch.tensor(vr)],REFERENCE)
                vq=_BASE_CHEAP(bank['z'],vo).double()*bank['diagnostic_val_scale'][:,None];e=bank['diagnostic_val_y'][:,None]-vq;tau=torch.tensor(R.T.B.TAUS,dtype=torch.float64)
                anchor=float(2*torch.maximum(tau*e,(tau-1)*e).mean());assert anchor>0;bank['diagnostic_anchor_MQS']=anchor
            checks.append(dict(horizon=int(R.T.HOR[hi]),bank_rows=len(nz),uniform_count_min=int(bank['count'][eligible].min()),zerohead_a_error=ae,zerohead_b_error=be,reference_mass_max_error=float((u[good].sum(1)-1).abs().max()),inner_validation_raw_reference_MQS=anchor))
        prepared[phase]=(X,banks);audits[phase]=dict(part=part,boundary=boundary,bank_checks=checks,source_fit_rows=len(D[part]))
    return prepared,audits,dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),times=torch.tensor(times),row_sid=torch.tensor(inverse))


def summaries(markets):
    cells=[c['scores'] for v in markets.values() for c in v['by_horizon'].values()];out={}
    for model in LEARNED+[REFERENCE]:
        refs=[REFERENCE,'rule_qrf','qrf_original','lstm_qr','uniform_seq_transport','uniform_rule']
        if model=='forest_transport':refs+=LEARNED[1:]
        for ref in refs:
            if ref==model:continue
            out[model+'_vs_'+ref]={}
            for metric in ['MQS','CRPS','WIS','MAE']:
                valid=[c for c in cells if c[ref][metric] is not None]
                ratios=[c[model][metric]/c[ref][metric] for c in valid]
                if ratios:out[model+'_vs_'+ref][metric]=dict(geometric_gain_percent=float(100*(1-np.exp(np.log(ratios).mean()))),wins=int(np.sum(np.array(ratios)<1)),cells=len(ratios))
    return out


def save(tag,report,arrays,rows):
    report['summary']=summaries(report['markets']);(P/(tag+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');np.savez_compressed(P/(tag+'_predictions.npz'),**arrays);pd.DataFrame(rows).to_csv(P/(tag+'_cells.csv'),index=False)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--markets',nargs='+',default=list(R.T.B.CUT));ap.add_argument('--seeds',nargs='+',type=int,default=SEEDS);ap.add_argument('--epochs',type=int,default=64);ap.add_argument('--min-epochs',type=int,default=30);ap.add_argument('--patience',type=int,default=16);ap.add_argument('--tag',default='forest_transport_round5');ap.add_argument('--wait-cache',action='store_true');args=ap.parse_args();assert args.tag.startswith('forest_transport_round5')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);geometry=sanity();activate();print('geometry',json.dumps(geometry),flush=True)
    frozen=[]
    for pattern in ['seq_transport_round4*','deep_seq_round4*','qrf_round4*','empirical_tilt_round3*','verify_empirical_affine_round4*']:frozen += [p for p in P.glob(pattern) if p.is_file()]
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen}
    source=np.load(P/'seq_transport_round4_predictions.npz');rule=np.load(P/'qrf_round4_probability_control_predictions.npz');qrf=np.load(P/'qrf_round4_predictions.npz');qr=np.load(P/'deep_seq_round4_predictions.npz');panel=pd.read_csv(R.T.B.PANEL,parse_dates=['date']);started=time.perf_counter()
    report=dict(status='exploratory forest reference embedded before transport; no test model/seed selection',config=vars(args),geometry=geometry,frozen_sha256=hashes,gamma=GAMMA,training='unchanged round4 run_fit: AdamW .001/wd.003, beta² penalty.003, no new delta/ell penalty, min30/max64/pat16, all4h from epoch1, full-precut same-seed refit',reference='u=.95 matured QRF conditional-NZ+.05 matured uniform; forest5seed common reference; inner/final strict separate',probability='branch p fixed rule; actual nonzero p computed from finalF including exact zero collision',ensemble='sort union of each query-specific seed affine support+0, probability mixture exact CRPS/Q',markets={},reference_cache_sha256={})
    arrays={};rows=[];pending=list(args.markets)
    while pending:
        ready=[market for market in pending if (P/f'forest_prior_round5_{market}.pt').is_file()]
        if not ready:
            if not args.wait_cache:raise FileNotFoundError('No completed per-market forest prior cache: '+','.join(pending))
            print('waiting for completed prior cache',','.join(pending),flush=True);time.sleep(10);continue
        market=ready[0];pending.remove(market)
        data=panel[panel.market==market].sort_values('date').reset_index(drop=True);D=R.S.build(data);te=D['te'];hi=D['hi'][te];y=D['y'][te];scale=D['scale'][te]
        for field,value in [('origin',D['t'][te]),('horizon',R.T.HOR[hi]),('target',y),('scale',scale)]:
            for ref in [source,rule,qrf,qr]:assert np.array_equal(value,ref[market+'_'+field])
            arrays[market+'_'+field]=value
        arrays[market+'_origin_date']=source[market+'_origin_date'];prepared,audits,cacheinfo=load_market(D,market);Xi,bi=prepared['inner'];Xf,bf=prepared['final'];states={}
        report['reference_cache_sha256'][cacheinfo['path']]=cacheinfo['sha256'];print('begin',market,'phase caches validated',flush=True)
        refs={'rule_qrf':(rule,'qrf_rule_atom'),'qrf_original':(qrf,'qrf_ensemble'),'uniform_seq_transport':(source,'seq_transport'),'uniform_rule':(source,'uniform_rule')}
        pred={m:ref[market+'_'+key+'_Q'] for m,(ref,key) in refs.items()};cr={m:ref[market+'_'+key+'_CRPS'] for m,(ref,key) in refs.items()};pro={m:ref[market+'_'+key+'_p'] for m,(ref,key) in refs.items()}
        pred['lstm_qr']=qr[market+'_lstm_qr_Q'];cr['lstm_qr']=None;pro['lstm_qr']=None
        fixed=R.TransportNet(REFERENCE);(q,c,p),branch,aa,bb,_,fixed_diag=R.evaluate_seed(fixed,D,Xf,bf);pred[REFERENCE]=q;cr[REFERENCE]=c;pro[REFERENCE]=p
        for suffix,value in [('branch_p',branch),('a',aa),('b',bb)]:arrays[market+'_'+REFERENCE+'_'+suffix]=value
        mres=dict(audit=D['audit'],phase_audit=audits,reference_geometry=fixed_diag,fits={},by_horizon={},strict_origin_target_scale_alignment=True)
        for model in LEARNED:
            allops=[];seedcs=[];fits=[]
            for seed in args.seeds:
                _,info=run_inner(D,Xi,bi,model,seed,args);ep=info['best_epoch'];net,refit=R.run_fit(D,Xf,bf,model,seed,args,'rf',ep)
                (q,c,p),branch,aa,bb,ops,diag=R.evaluate_seed(net,D,Xf,bf);assert np.array_equal(branch,rule[market+'_qrf_rule_atom_p'])
                states[f'{model}_seed{seed}']={k:v.detach().clone() for k,v in net.state_dict().items()};key=f'{market}_{model}_seed{seed}'
                for suffix,value in [('Q',q),('CRPS',c),('p',p),('branch_p',branch),('a',aa),('b',bb)]:arrays[key+'_'+suffix]=value
                allops.append(ops);seedcs.append(c);info.update(seed=seed,refit_epochs=ep,refit=refit,test_geometry=diag,allocated_parameters=sum(p.numel() for p in net.parameters()));fits.append(info)
                for j,h in enumerate(R.T.HOR):
                    mask=hi==j;rows.append(dict(market=market,horizon=int(h),model=model,seed=seed,n=int(mask.sum()),**R.metrics(y[mask],q[mask],c[mask],p[mask])))
                print(market,model,seed,'best',ep,'run',info['epochs_run'],'elapsed',round(time.perf_counter()-started,1),flush=True)
            Q=np.empty((len(te),15));C=np.empty(len(te));PR=np.empty(len(te));convex=0.
            for j,bank in enumerate(bf):
                positions=np.flatnonzero(hi==j);qq,cc,pp,_,_=R.mixture(torch.tensor(D['z'][te[positions]]),bank['z'],[ops[j] for ops in allops]);Q[positions]=qq.numpy()*scale[positions,None];C[positions]=cc.numpy()*scale[positions];PR[positions]=pp.numpy();convex=max(convex,float(np.max(C[positions]-np.mean(seedcs,0)[positions])))
            assert convex<1e-6 and np.isfinite(Q).all() and np.all(np.diff(Q,axis=1)>=-1e-8)
            pred[model]=Q;cr[model]=C;pro[model]=PR;mres['fits'][model]=fits;mres.setdefault('mixture_convexity_error',{})[model]=convex
        checkpoint=dict(state_dicts=states,variants=LEARNED,config=vars(args),reference_cache={k:v for k,v in cacheinfo.items()},phase_audit=audits,context_mean=D['x_aug'][D['rf']].mean(0),context_std=D['x_aug'][D['rf']].std(0)+1e-5,banks=[{k:b[k] for k in ['z','phi2','mature','p','u','allowed','count','m0','v0','vu']} for b in bf],gamma=GAMMA,source_panel_sha256=hashlib.sha256(R.T.B.PANEL.read_bytes()).hexdigest())
        torch.save(checkpoint,P/f'{args.tag}_{market}_checkpoints.pt')
        for j,h in enumerate(R.T.HOR):
            mask=hi==j;ss={m:R.metrics(y[mask],q[mask],None if cr[m] is None else cr[m][mask],None if pro[m] is None else pro[m][mask]) for m,q in pred.items()};mres['by_horizon'][str(h)]=dict(n=int(mask.sum()),scores=ss)
            for m in pred:rows.append(dict(market=market,horizon=int(h),model=m,seed='fixed' if m==REFERENCE else 'ensemble',n=int(mask.sum()),**ss[m]))
        for m in pred:
            arrays[market+'_'+m+'_Q']=pred[m]
            if cr[m] is not None:arrays[market+'_'+m+'_CRPS']=cr[m]
            if pro[m] is not None:arrays[market+'_'+m+'_p']=pro[m]
        report['markets'][market]=mres;report['elapsed_seconds']=time.perf_counter()-started;save(args.tag,report,arrays,rows)
        assert hashlib.sha256((P/cacheinfo['path']).read_bytes()).hexdigest()==cacheinfo['sha256']
        print('market complete',market,'remaining',','.join(pending),flush=True)
    assert all(hashlib.sha256((P/name).read_bytes()).hexdigest()==sha for name,sha in hashes.items());report['frozen_sha_unchanged']=True;report['elapsed_seconds']=time.perf_counter()-started;save(args.tag,report,arrays,rows)
    lines=['# 第五轮：条件森林参考上的LSTM矩归一输运','',f'完成{sum(len(fs) for v in report["markets"].values() for fs in v["fits"].values())}次内层选择/全precut重拟合组合及各市场固定reference，运行{report["elapsed_seconds"]:.2f}秒。没有按test选择模型/seed。','',
    '唯一主要变化为参考u：0.95*成熟QRF conditional NZ+0.05*成熟uniform；head输出位置/尺度/shape2，其余骨干与训练设置复用冻结第四轮。inner只读inner forest与Dtr，final只读final forest与Drf；每phase fit indices/scaler/bank D行顺序/maturity/sid全部严格核对。模型参数2468、64past+36ctx、所有h首轮联合。','',
    'w正比u exp(beta·phi2)，参考条件矩来自u，倾斜矩来自w；位置尺度目标归一后shape不改变指定条件矩。fixed reference_only无NN更新，隔离gamma平滑作用。保持旧beta²正则，没有新增delta/ell penalty。','',
    '所有支持随query/seed变化，最终五seed先合并其概率/affine支持及零atom、排序再算同一F的quantile和完整CRPS。Brier使用完整F非零质量，branch p固定rule且另记碰撞。variance floors与实际minvar/稳定性下界比均另存。','',
    '| 主候选 / 参照 | MQS改善 | 完整CRPS改善 |','|---|---:|---:|']
    for key,values in report['summary'].items():
        if not key.startswith('forest_transport_vs_'):continue
        q=values['MQS'];c=values.get('CRPS');ct='15q不可识别' if c is None else f"{c['geometric_gain_percent']:+.4f}% ({c['wins']}/20)";lines.append(f"| {key} | {q['geometric_gain_percent']:+.4f}% ({q['wins']}/20) | {ct} |")
    lines+=['','正值表示左侧forest_transport更好。JSON含全部三学习组及reference对强基线评分；cells有逐seed/market/h，NPZ保存单seed与真正mixture Q/CRPS/actualp/branchp/a/b。checkpoint包含15模型state、final bank/u/times/sid、scaler及inner/final cache审计与SHA，可重建权重。','',
    '森林/scaler/phi是整个允许fit区间参数；成熟mask限制响应检索，不称每个训练query都使用当时冻结的全部参数。历史测试已开发复用，结果不证明新颖性、显著性或可发表成功。']
    (P/(args.tag+'_report_zh.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in report['summary'].items() if k.startswith('forest_transport_vs_')},indent=2),flush=True);print('elapsed',report['elapsed_seconds'],'frozen unchanged',True,flush=True)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    with threadpool_limits(limits=1):main()
