# -*- coding: utf-8 -*-
"""Prior readout + prefix-reference mechanism experiment; only this prefix writes.
Read-only dependencies keep original geometry, loss, selection, and raw diagnostic.
"""
from __future__ import annotations
import os,sys
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
sys.dont_write_bytecode=True
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
from threadpoolctl import threadpool_limits
import forest_transport_round5 as B
R=B.R;P=Path(__file__).resolve().parent
MODELS=['honest_readout','honest_no_readout','insample_readout','honest_readout_reweight_only']
CACHE_PREFIX={'honest':'forest_prior_honest_round5','insample':'forest_prior_round5'}
SEEDS=[11,22,33,44,55]


def prior_operator(raw,bank,sids,local,variant):
    mapped='forest_reweight_only' if variant=='honest_readout_reweight_only' else B.REFERENCE if variant==B.REFERENCE else 'forest_transport'
    return B.forest_operator(raw,bank,sids,local,mapped)


class ReadoutNet(nn.Module):
    def __init__(self,variant):
        super().__init__();self.variant=variant;self.encoder=nn.LSTM(2,16,batch_first=True)
        self.context=nn.Sequential(nn.Linear(36,16),nn.Tanh());self.prior_encoder=nn.Sequential(nn.Linear(4,8),nn.Tanh())
        self.shared=nn.Sequential(nn.Linear(40,16),nn.Tanh());self.head=nn.Linear(16,4)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        self.prior_weight_nonzero_gradient=False
        def remember(grad):
            if bool(torch.any(grad!=0)):self.prior_weight_nonzero_gradient=True
            return grad
        self.prior_encoder[0].weight.register_hook(remember)
    def prior_input(self,X,idx):
        value=X['prior'][idx]
        return torch.zeros_like(value) if self.variant=='honest_no_readout' else value
    def forward(self,X,idx):
        unique,inverse=torch.unique(X['sid'][idx],return_inverse=True)
        out,_=self.encoder(X['seq'][unique]);h=out[:,-1][inverse]
        joined=torch.cat([h,self.context(X['x'][idx]),self.prior_encoder(self.prior_input(X,idx))],1)
        return self.head(self.shared(joined))


def activate():
    B.activate();R.operator=prior_operator;R.TransportNet=ReadoutNet


def feature_table(bank):
    u=bank['u'].double();mass=u.sum(1);weights=u/mass[:,None].clamp_min(1e-30)
    mean=bank['m0'].double();std=bank['v0'].double().clamp_min(R.FLOOR).sqrt()
    ess=1/weights.square().sum(1).clamp_min(1e-30)
    skew=(weights*((bank['z'].double()-mean[:,None])/std[:,None]).pow(3)).sum(1)
    feats=torch.stack([(mean/std).clamp(-8,8),std.log().clamp(-8,8),torch.log((ess+1)/(bank['count'].double()+1)),skew.clamp(-8,8)],1).float()
    valid=(bank['count']>=32)&(mass>0)&torch.isfinite(feats).all(1)
    feats[~valid]=0
    return feats,valid


def fit_prior_scaler(D,X,banks,part):
    raw=torch.zeros(len(D['t']),4);valid=torch.zeros(len(D['t']),dtype=torch.bool)
    for hi,bank in enumerate(banks):
        rows=torch.tensor(np.flatnonzero(D['hi']==hi));tab,good=feature_table(bank)
        raw[rows]=tab[X['sid'][rows]];valid[rows]=good[X['sid'][rows]]
    fit=torch.tensor(D[part]);fit=fit[valid[fit]]
    expected=R.eligible(D,X,banks,part);assert torch.equal(torch.sort(fit).values,torch.sort(expected).values)
    mean=raw[fit].numpy().mean(0);std=raw[fit].numpy().std(0)+1e-5
    normalized=((raw-torch.tensor(mean))/torch.tensor(std)).clamp(-8,8);normalized[~valid]=0
    assert torch.isfinite(normalized).all() and bool(torch.all(normalized[~valid]==0))
    X['prior']=normalized;X['prior_raw']=raw;X['prior_valid']=valid
    meta=dict(feature_names=['clip(m0/s0,-8,8)','clip(log(s0),-8,8)','log((ESS+1)/(matureNZcount+1))','clip(skew_u,-8,8)'],fit_part=part,fit_eligible_count=len(fit),fit_total_rows=len(D[part]),invalid_total_rows=int((~valid).sum()),mean=mean.tolist(),std=std.tolist(),fit_D_indices=fit.tolist(),fit_D_indices_sha256=hashlib.sha256(fit.numpy().tobytes()).hexdigest(),normalization='fit eligible phase rows only, then clip[-8,8]; invalid rows zero, excluded from fit',source='only current matured reference weights/support, never query target')
    return meta


def small_metadata(value):
    if isinstance(value,torch.Tensor):return dict(shape=list(value.shape),dtype=str(value.dtype),sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest())
    if isinstance(value,dict):return {str(k):small_metadata(v) for k,v in value.items() if k not in ['banks','x','rows','u','forest','u_forest','uniform','mature_mask','allowed']}
    if isinstance(value,(list,tuple)):return [small_metadata(v) for v in value]
    if isinstance(value,np.ndarray):return dict(shape=list(value.shape),sha256=hashlib.sha256(value.tobytes()).hexdigest())
    return value


def load_explicit(D,market,kind):
    prefix=CACHE_PREFIX[kind];path=P/f'{prefix}_{market}.pt';cache=torch.load(path,map_location='cpu',weights_only=False)
    assert cache['market']==market and cache['gamma']==.05 and cache['seeds']==SEEDS
    times,_,inverse=np.unique(D['t'],return_index=True,return_inverse=True)
    assert np.array_equal(cache['times'].numpy(),times) and np.array_equal(cache['row_sid'].numpy(),inverse)
    for key in ['t','hi','y','scale','z','tr','va','rf','te']:assert np.array_equal(cache['rows'][key].numpy(),D[key])
    prepared={};audits={}
    for phase,part in [('inner','tr'),('final','rf')]:
        cached=cache['phases'][phase];X,banks=R.prepare(D,part)
        boundary=D['audit']['validation_start_index' if part=='tr' else 'cut_index']
        assert cached['phase']==phase and cached['part']==part and cached['boundary']==boundary
        assert np.array_equal(cached['fit_D_indices'].numpy(),D[part]) and torch.equal(cached['x'],X['x']) and torch.equal(cached['row_sid'],X['sid'])
        assert np.array_equal(cached['scaler']['mean'].numpy(),D['x_aug'][D[part]].mean(0)) and np.array_equal(cached['scaler']['std'].numpy(),D['x_aug'][D[part]].std(0)+1e-5)
        assert np.all(D['t'][D[part]]+R.T.HOR[D['hi'][D[part]]]<boundary)
        for hi,(bank,cb) in enumerate(zip(banks,cached['banks'])):
            ids=D[part][D['hi'][D[part]]==hi];ids=ids[np.abs(D['y'][ids])>1e-12];ids=ids[np.argsort(D['z'][ids],kind='stable')]
            assert np.array_equal(cb['bank_D_indices'].numpy(),ids)
            for key in ['z','mature','allowed','count']:assert torch.equal(cb[key],bank[key])
            assert cb['p_static']==bank['p'];u=cb['u'];assert u.shape==bank['allowed'].shape and torch.isfinite(u).all() and bool(torch.all(u>=0))
            assert bool(torch.all(u[~bank['allowed']]==0));good=bank['count']>0
            assert torch.allclose(u[good].sum(1),torch.ones(int(good.sum())),atol=2e-6,rtol=0)
            uniform=bank['allowed'].float()/bank['count'].clamp_min(1)[:,None];assert bool(torch.all(u+1e-8>=.05*uniform))
            bank['vu']=bank['v0'].clone();B.reference_moments(bank,u);bank['phase']=phase;bank['bank_D_indices']=torch.tensor(ids)
            eligible=bank['count']>=32;sids=torch.nonzero(eligible,as_tuple=True)[0]
            zero=prior_operator(torch.zeros(len(sids),4),bank,sids,torch.full((len(sids),),.6),'honest_readout');assert float((zero['a']-1).abs().max())<2e-6 and float(zero['b'].abs().max())<2e-5
            if phase=='inner':
                vr=D['va'][D['hi'][D['va']]==hi];vs=X['sid'][torch.tensor(vr)]
                bank['diagnostic_val_sids']=vs;bank['diagnostic_val_y']=torch.tensor(D['y'][vr],dtype=torch.float64);bank['diagnostic_val_scale']=torch.tensor(D['scale'][vr],dtype=torch.float64)
                op=prior_operator(torch.zeros(len(vr),4),bank,vs,X['local'][torch.tensor(vr)],B.REFERENCE);q=B._BASE_CHEAP(bank['z'],op).double()*bank['diagnostic_val_scale'][:,None]
                e=bank['diagnostic_val_y'][:,None]-q;tau=torch.tensor(R.T.B.TAUS,dtype=torch.float64);bank['diagnostic_anchor_MQS']=float(2*torch.maximum(tau*e,(tau-1)*e).mean())
        scaler=fit_prior_scaler(D,X,banks,part)
        prepared[phase]=(X,banks);audits[phase]=dict(cache_kind=kind,explicit_prefix=prefix,path=path.name,phase=phase,part=part,boundary=boundary,fit_rows=len(D[part]),prior_scaler=scaler,cache_phase_metadata=small_metadata(cached))
    info=dict(kind=kind,prefix=prefix,path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),times=torch.tensor(times),row_sid=torch.tensor(inverse),cache_metadata=small_metadata({k:v for k,v in cache.items() if k not in ['phases','rows']}))
    return prepared,audits,info


def validate_eval_equality(D,honest,insample):
    out={}
    for phase,part in [('inner','tr'),('final','rf')]:
        X,banks=honest[phase];_,old=insample[phase];bound=D['audit']['validation_start_index' if part=='tr' else 'cut_index'];outer=X['origin_times']>=bound
        for bank,reference in zip(banks,old):assert torch.equal(bank['u'][outer],reference['u'][outer])
        out[phase]=dict(boundary=bound,outer_sid_count=int(outer.sum()),u_bitwise_equal_old_full_phase=True,fit_eligible_count=len(R.eligible(D,X,banks,part)))
    return out


def sanity():
    geometry=B.sanity();torch.manual_seed(37);read=ReadoutNet('honest_readout');torch.manual_seed(37);no=ReadoutNet('honest_no_readout')
    assert all(torch.equal(v,no.state_dict()[k]) for k,v in read.state_dict().items());assert sum(p.numel() for p in read.parameters())==2636
    X=dict(seq=torch.randn(3,64,2),sid=torch.tensor([0,1,2,0,1,2]),x=torch.randn(6,36),prior=torch.randn(6,4));idx=torch.arange(6)
    assert bool(torch.all(no.prior_input(X,idx)==0)) and torch.equal(read.prior_input(X,idx),X['prior'])
    assert bool(torch.all(read(X,idx)==0)) and bool(torch.all(no(X,idx)==0))
    with torch.no_grad():
        read.head.weight.fill_(.1);no.load_state_dict(read.state_dict());before=read(X,idx);nobefore=no(X,idx);other=dict(X);other['prior']=X['prior']+1.7
        difference=float((read(other,idx)-before).abs().max());nodiff=float((no(other,idx)-nobefore).abs().max());assert difference>1e-6 and nodiff==0
    z=torch.linspace(-2.3,3.1,40,dtype=torch.float64);bank=dict(z=z,phi=torch.tensor(R.T.basis(z.numpy()),dtype=torch.float64),mature=torch.arange(40),p=.71)
    R.add_cache(bank,torch.tensor([31,33,36,39]),torch.float64);bank['vu']=bank['v0'].clone();u=bank['allowed'].double()/bank['count'][:,None];B.reference_moments(bank,u)
    first,_=feature_table(bank);changed=dict(bank);changed['z']=z.clone();changed['z'][32:]+=100;B.reference_moments(changed,u);after,_=feature_table(changed);assert torch.equal(first[0],after[0])
    return dict(geometry=geometry,allocated_parameters=2636,same_seed_same_initial_state=True,zerohead_recovers_reference=True,standardized_no_readout_all_zero=True,readout_change_effect=difference,no_readout_change_effect=nodiff,future_zero_weight_support_does_not_change_prior_features=True)


def summaries(markets):
    cells=[c['scores'] for v in markets.values() for c in v['by_horizon'].values()];out={}
    for model in MODELS:
        refs=[B.REFERENCE,'rule_qrf','qrf_original','lstm_qr','old_forest_transport','uniform_seq_transport']
        if model=='honest_readout':refs+=MODELS[1:]
        for ref in refs:
            out[model+'_vs_'+ref]={}
            for metric in ['MQS','CRPS','WIS','MAE']:
                valid=[c for c in cells if c[ref][metric] is not None];ratios=[c[model][metric]/c[ref][metric] for c in valid]
                if ratios:out[model+'_vs_'+ref][metric]=dict(geometric_gain_percent=float(100*(1-np.exp(np.log(ratios).mean()))),wins=int(np.sum(np.array(ratios)<1)),cells=len(ratios))
    return out


def save(tag,r,arrays,rows):
    r['summary']=summaries(r['markets']);(P/(tag+'.json')).write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8');np.savez_compressed(P/(tag+'_predictions.npz'),**arrays);pd.DataFrame(rows).to_csv(P/(tag+'_cells.csv'),index=False)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--markets',nargs='+',default=list(R.T.B.CUT));ap.add_argument('--seeds',nargs='+',type=int,default=SEEDS);ap.add_argument('--epochs',type=int,default=64);ap.add_argument('--min-epochs',type=int,default=30);ap.add_argument('--patience',type=int,default=16);ap.add_argument('--tag',default='prior_readout_transport_round5');ap.add_argument('--wait-cache',action='store_true');args=ap.parse_args();assert args.tag.startswith('prior_readout_transport_round5')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);check=sanity();activate();print('checks',json.dumps(check),flush=True)
    files=[]
    for pattern in ['forest_transport_round5*','forest_prior_round5*','seq_transport_round4*','deep_seq_round4*','qrf_round4*','empirical_tilt_round3*','verify_empirical_affine_round4*']:files += [p for p in P.glob(pattern) if p.is_file()]
    frozen={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files};old=np.load(P/'forest_transport_round5_predictions.npz');rule=np.load(P/'qrf_round4_probability_control_predictions.npz');qrf=np.load(P/'qrf_round4_predictions.npz');qr=np.load(P/'deep_seq_round4_predictions.npz');seq=np.load(P/'seq_transport_round4_predictions.npz');panel=pd.read_csv(R.T.B.PANEL,parse_dates=['date']);started=time.perf_counter()
    report=dict(status='prior-readout and prefix-reference mechanism experiment, old results remain valid',config=vars(args),checks=check,frozen_sha256=frozen,reference_cache_sha256={},network='LSTM16+ctx16+prior4->8,concat40->shared16->4zerohead;2636allocated',features='current phase reference only, eligible fit-row scaler, invalid0; no-readout standardized4zero',selector='original normalized macro pinball; raw-relative diagnostics retained only',training='unchanged R.run_fit64/min30/pat16/AdamW.001/wd.003/beta penalty.003, same rulep, no new penalties',ensemble='true five-seed query-specific support mixture, exactCRPS and inverseCDF',markets={})
    arrays={};rows=[];pending=list(args.markets)
    while pending:
        ready=[m for m in pending if all((P/f'{prefix}_{m}.pt').is_file() for prefix in CACHE_PREFIX.values())]
        if not ready:
            if not args.wait_cache:raise FileNotFoundError('honest/insample cache pair not ready: '+','.join(pending))
            print('waiting completed honest cache',','.join(pending),flush=True);time.sleep(10);continue
        market=ready[0];pending.remove(market);data=panel[panel.market==market].sort_values('date').reset_index(drop=True);D=R.S.build(data);te=D['te'];hi=D['hi'][te];y=D['y'][te];scale=D['scale'][te]
        for field,val in [('origin',D['t'][te]),('horizon',R.T.HOR[hi]),('target',y),('scale',scale)]:
            for source in [old,rule,qrf,qr,seq]:assert np.array_equal(val,source[market+'_'+field])
            arrays[market+'_'+field]=val
        arrays[market+'_origin_date']=old[market+'_origin_date'];prepared={};audits={};cacheinfo={}
        for kind in ['insample','honest']:
            prepared[kind],audits[kind],cacheinfo[kind]=load_explicit(D,market,kind);report['reference_cache_sha256'][cacheinfo[kind]['path']]=cacheinfo[kind]['sha256']
        equality=validate_eval_equality(D,prepared['honest'],prepared['insample']);print('begin',market,'explicit cache pairs validated',flush=True)
        refs={B.REFERENCE:(old,B.REFERENCE),'old_forest_transport':(old,'forest_transport'),'rule_qrf':(rule,'qrf_rule_atom'),'qrf_original':(qrf,'qrf_ensemble'),'uniform_seq_transport':(seq,'seq_transport')}
        pred={m:s[market+'_'+key+'_Q'] for m,(s,key) in refs.items()};cr={m:s[market+'_'+key+'_CRPS'] for m,(s,key) in refs.items()};pro={m:s[market+'_'+key+'_p'] for m,(s,key) in refs.items()};pred['lstm_qr']=qr[market+'_lstm_qr_Q'];cr['lstm_qr']=None;pro['lstm_qr']=None
        mres=dict(audit=D['audit'],preparation=audits,evaluation_reference_equality=equality,variant_cache_kind={m:'insample' if m=='insample_readout' else 'honest' for m in MODELS},fits={},by_horizon={});states={}
        for model in MODELS:
            kind='insample' if model=='insample_readout' else 'honest';Xi,bi=prepared[kind]['inner'];Xf,bf=prepared[kind]['final'];allops=[];seedcs=[];fits=[]
            for seed in args.seeds:
                inner,info=B.run_inner(D,Xi,bi,model,seed,args);info['prior_weight_nonzero_gradient']=inner.prior_weight_nonzero_gradient;ep=info['best_epoch'];net,refit=R.run_fit(D,Xf,bf,model,seed,args,'rf',ep);refit['prior_weight_nonzero_gradient']=net.prior_weight_nonzero_gradient
                (q,c,p),branch,aa,bb,ops,diag=R.evaluate_seed(net,D,Xf,bf);assert np.array_equal(branch,rule[market+'_qrf_rule_atom_p']);states[f'{model}_seed{seed}']={k:v.detach().clone() for k,v in net.state_dict().items()};key=f'{market}_{model}_seed{seed}'
                for suffix,value in [('Q',q),('CRPS',c),('p',p),('branch_p',branch),('a',aa),('b',bb)]:arrays[key+'_'+suffix]=value
                allops.append(ops);seedcs.append(c);info.update(seed=seed,refit_epochs=ep,refit=refit,test_geometry=diag,cache_kind=kind,inner_cache_path=cacheinfo[kind]['path'],final_cache_path=cacheinfo[kind]['path'],allocated_parameters=sum(p.numel() for p in net.parameters()));fits.append(info)
                for j,h in enumerate(R.T.HOR):
                    mask=hi==j;rows.append(dict(market=market,horizon=int(h),model=model,seed=seed,n=int(mask.sum()),**R.metrics(y[mask],q[mask],c[mask],p[mask])))
                print(market,model,seed,'best',ep,'rawbest',info['best_raw_epoch'],'run',info['epochs_run'],'elapsed',round(time.perf_counter()-started,1),flush=True)
            Q=np.empty((len(te),15));C=np.empty(len(te));PP=np.empty(len(te));convex=0.
            for j,b in enumerate(bf):
                pos=np.flatnonzero(hi==j);qq,cc,pp,_,_=R.mixture(torch.tensor(D['z'][te[pos]]),b['z'],[ops[j] for ops in allops]);Q[pos]=qq.numpy()*scale[pos,None];C[pos]=cc.numpy()*scale[pos];PP[pos]=pp.numpy();convex=max(convex,float(np.max(C[pos]-np.mean(seedcs,0)[pos])))
            assert convex<1e-6 and np.isfinite(Q).all() and np.all(np.diff(Q,axis=1)>=-1e-8);pred[model]=Q;cr[model]=C;pro[model]=PP;mres['fits'][model]=fits;mres.setdefault('mixture_convexity_error',{})[model]=convex
        checkpoint=dict(state_dicts=states,variants=MODELS,config=vars(args),cache_info=cacheinfo,preparation=audits,evaluation_reference_equality=equality,context_mean=D['x_aug'][D['rf']].mean(0),context_std=D['x_aug'][D['rf']].std(0)+1e-5,final_banks={kind:[{k:b[k] for k in ['z','phi2','mature','p','u','allowed','count','m0','v0','vu','bank_D_indices']} for b in prepared[kind]['final'][1]] for kind in prepared},prior_scalers={kind:audits[kind]['final']['prior_scaler'] for kind in prepared},source_panel_sha256=hashlib.sha256(R.T.B.PANEL.read_bytes()).hexdigest())
        torch.save(checkpoint,P/f'{args.tag}_{market}_checkpoints.pt')
        for j,h in enumerate(R.T.HOR):
            mask=hi==j;ss={m:R.metrics(y[mask],q[mask],None if cr[m] is None else cr[m][mask],None if pro[m] is None else pro[m][mask]) for m,q in pred.items()};mres['by_horizon'][str(h)]=dict(n=int(mask.sum()),scores=ss)
            for m in pred:rows.append(dict(market=market,horizon=int(h),model=m,seed='ensemble',n=int(mask.sum()),**ss[m]))
        for m in pred:
            arrays[market+'_'+m+'_Q']=pred[m]
            if cr[m] is not None:arrays[market+'_'+m+'_CRPS']=cr[m]
            if pro[m] is not None:arrays[market+'_'+m+'_p']=pro[m]
        report['markets'][market]=mres;report['elapsed_seconds']=time.perf_counter()-started;save(args.tag,report,arrays,rows)
    assert all(hashlib.sha256((P/name).read_bytes()).hexdigest()==sha for name,sha in frozen.items());assert all(hashlib.sha256((P/name).read_bytes()).hexdigest()==sha for name,sha in report['reference_cache_sha256'].items());report['frozen_sha_unchanged']=True;report['elapsed_seconds']=time.perf_counter()-started;save(args.tag,report,arrays,rows)
    lines=['# 参考读出与前缀森林参考的机制实验','',f'完成{sum(len(fs) for v in report["markets"].values() for fs in v["fits"].values())}configs，运行{report["elapsed_seconds"]:.2f}秒，全部市场/seed/四组保留。','',
    '新网络2636分配参数：LSTM16+context16+prior4经Linear4->8/Tanh后concat40->shared16->4zerohead。prior为clip(m0/s0)、clip(log s0)、log((ESS+1)/(count+1))、clip(skew_u)，仅当前成熟参考；phase fit eligible count>=32行估计mean/std，clip[-8,8]，无效sid置0且不入fit统计。no-readout在标准化后四维置0、同容量/同初始化。','',
    'honest三组显式读取forest_prior_honest_round5缓存，insample组显式读取旧forest_prior_round5缓存；没有替换旧文件。每phase fitindices/scaler/bank-D序/maturity/sid严格核对，t>=boundary的u与旧fullphase逐位相同。prefix是reference-training-bias机制对照，不宣称旧结果无效。','',
    '保持旧conditional transport/shape2/O(M)exactCRPS/samebranchp/64-min30-pat16/AdamW与正则。raw-relative validation轨迹仅作诊断，主normalized selector和refit不改。每seed支持a_s z+b_s合并0atom后排序，完整同F算五seed mixture Q/CRPS与实际非零概率；LSTMQR只有15q无完整CRPS。','', '| honest_readout / 参照 | MQS改善 | 完整CRPS改善 |','|---|---:|---:|']
    for key,vals in report['summary'].items():
        if not key.startswith('honest_readout_vs_'):continue
        q=vals['MQS'];c=vals.get('CRPS');ct='15q不可识别' if c is None else f"{c['geometric_gain_percent']:+.4f}% ({c['wins']}/20)";lines.append(f"| {key} | {q['geometric_gain_percent']:+.4f}% ({q['wins']}/20) | {ct} |")
    lines+=['','正值表示主候选更好。JSON/cells保留各组全部基线、逐seed与raw轨迹；NPZ存每seed及真实mixture Q/CRPS/actualp/branchp/a/b。各市场checkpoint有20模型state、honest/insample各自final bank/u、prior scaler eligible行映射与cache路径SHA。','', '历史测试已开发复用，结果不证明显著性/新颖性或发表成功；没有用测试挑模型/seed，也不把负结果推广为整个方向无效。']
    (P/(args.tag+'_report_zh.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in report['summary'].items() if k.startswith('honest_readout_vs_')},indent=2),flush=True);print('elapsed',report['elapsed_seconds'],'frozen unchanged',True,flush=True)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    with threadpool_limits(limits=1):main()
