# -*- coding: utf-8 -*-
"""Two matched no-readout controls, retaining the frozen25-model reference.
Only prefix_transport_controls_round5* is written; old checkpoints/caches read-only.
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
import prior_readout_transport_round5 as M
R=M.R;B=M.B;P=Path(__file__).resolve().parent
CONTROLS=['matched_insample_no_readout','matched_honest_no_readout_reweight_only']
KINDS={CONTROLS[0]:'insample',CONTROLS[1]:'honest'}
FROZEN='frozen_honest_no_readout'
SEEDS=[11,22,33,44,55]


class MatchedNet(M.ReadoutNet):
    def prior_input(self,X,idx):return torch.zeros_like(X['prior'][idx])


def control_operator(raw,bank,sids,local,variant):
    mapped='forest_reweight_only' if variant==CONTROLS[1] else B.REFERENCE if variant==B.REFERENCE else 'forest_transport'
    return B.forest_operator(raw,bank,sids,local,mapped)


def activate():
    M.activate();R.TransportNet=MatchedNet;R.operator=control_operator


def sanity():
    result=M.sanity();X=dict(seq=torch.randn(3,64,2),sid=torch.tensor([0,1,2,0,1,2]),x=torch.randn(6,36),prior=torch.randn(6,4));idx=torch.arange(6)
    for seed in [11,22,33,44,55]:
        torch.manual_seed(seed);reference=M.ReadoutNet('honest_no_readout')
        for control in CONTROLS:
            torch.manual_seed(seed);net=MatchedNet(control)
            assert all(torch.equal(v,net.state_dict()[k]) for k,v in reference.state_dict().items())
            assert sum(p.numel() for p in net.parameters())==2636 and bool(torch.all(net.prior_input(X,idx)==0))
            assert torch.equal(net(X,idx),reference(X,idx))
    # Inference-only algebra: prior0 -> tanh(prior bias), absorb into shared bias.
    with torch.no_grad():
        net.head.weight.fill_(.1);full=net(X,idx)
        unique,inverse=torch.unique(X['sid'][idx],return_inverse=True);out,_=net.encoder(X['seq'][unique]);features=torch.cat([out[:,-1][inverse],net.context(X['x'][idx])],1)
        constant=torch.tanh(net.prior_encoder[0].bias);linear=net.shared[0]
        effective_bias=linear.bias+linear.weight[:,32:]@constant
        folded=net.head(torch.tanh(torch.nn.functional.linear(features,linear.weight[:,:32],effective_bias)))
        error=float((full-folded).abs().max());assert error<1e-6
    result.update(all5seeds_same_initial_state_as_frozen_architecture=True,both_controls_standardized_prior_zero=True,inference_constant_prior_folding_error=error,training_modules_unchanged=True)
    return result


def summaries(markets):
    cells=[c['scores'] for m in markets.values() for c in m['by_horizon'].values()];out={}
    pairs=[(FROZEN,c) for c in CONTROLS]+[(model,base) for model in [FROZEN]+CONTROLS for base in ['forest_reference_only','rule_qrf','qrf_original','lstm_qr']]
    for model,reference in pairs:
        out[model+'_vs_'+reference]={}
        for metric in ['MQS','CRPS','WIS','MAE']:
            valid=[c for c in cells if c[reference][metric] is not None];ratios=[c[model][metric]/c[reference][metric] for c in valid]
            if ratios:out[model+'_vs_'+reference][metric]=dict(geometric_gain_percent=float(100*(1-np.exp(np.log(ratios).mean()))),wins=int(np.sum(np.array(ratios)<1)),cells=len(ratios))
    return out


def save(tag,r,arrays,rows):
    r['summary']=summaries(r['markets']);(P/(tag+'.json')).write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8');np.savez_compressed(P/(tag+'_predictions.npz'),**arrays);pd.DataFrame(rows).to_csv(P/(tag+'_cells.csv'),index=False)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--markets',nargs='+',default=list(R.T.B.CUT));ap.add_argument('--seeds',nargs='+',type=int,default=SEEDS);ap.add_argument('--epochs',type=int,default=64);ap.add_argument('--min-epochs',type=int,default=30);ap.add_argument('--patience',type=int,default=16);ap.add_argument('--tag',default='prefix_transport_controls_round5');args=ap.parse_args();assert args.tag.startswith('prefix_transport_controls_round5')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);checks=sanity();activate();print('checks',json.dumps(checks),flush=True)
    files=[]
    for pattern in ['prior_readout_transport_round5*','forest_transport_round5*','forest_prior_round5*','forest_prior_honest_round5*','seq_transport_round4*','deep_seq_round4*','qrf_round4*','empirical_tilt_round3*','verify_empirical_affine_round4*']:files += [p for p in P.glob(pattern) if p.is_file()]
    frozen={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files};old=np.load(P/'prior_readout_transport_round5_predictions.npz');forest=np.load(P/'forest_transport_round5_predictions.npz');rule=np.load(P/'qrf_round4_probability_control_predictions.npz');qrf=np.load(P/'qrf_round4_predictions.npz');qr=np.load(P/'deep_seq_round4_predictions.npz');panel=pd.read_csv(R.T.B.PANEL,parse_dates=['date']);started=time.perf_counter()
    report=dict(status='matched mechanisms; original25 honest_no_readout models frozen and not retrained',config=vars(args),checks=checks,frozen_sha256=frozen,reference_cache_sha256={},cache_kind=KINDS,network='same2636 parameters and seed initialization; subclass only forces standardized prior4 input0',protocol='unchanged64/min30/pat16, normselector including0, freshsame-seed RF refit, all4h firstepoch, samep/exactCRPS/true5mix, raw diagnostics only',folding='tanh(prior_bias) is constant at inference; shared.bias may absorb W_prior*tanh(prior_bias); no modules removed or training changed',markets={})
    arrays={};rows=[]
    for market in args.markets:
        data=panel[panel.market==market].sort_values('date').reset_index(drop=True);D=R.S.build(data);te=D['te'];hi=D['hi'][te];y=D['y'][te];scale=D['scale'][te]
        for field,value in [('origin',D['t'][te]),('horizon',R.T.HOR[hi]),('target',y),('scale',scale)]:
            for source in [old,forest,rule,qrf,qr]:assert np.array_equal(value,source[market+'_'+field])
            arrays[market+'_'+field]=value
        arrays[market+'_origin_date']=old[market+'_origin_date'];prepared={};audits={};cacheinfo={}
        for kind in ['insample','honest']:
            prepared[kind],audits[kind],cacheinfo[kind]=M.load_explicit(D,market,kind);report['reference_cache_sha256'][cacheinfo[kind]['path']]=cacheinfo[kind]['sha256']
        equality=M.validate_eval_equality(D,prepared['honest'],prepared['insample'])
        refs={FROZEN:(old,'honest_no_readout'),'forest_reference_only':(forest,'forest_reference_only'),'rule_qrf':(rule,'qrf_rule_atom'),'qrf_original':(qrf,'qrf_ensemble')}
        pred={m:s[market+'_'+key+'_Q'] for m,(s,key) in refs.items()};cr={m:s[market+'_'+key+'_CRPS'] for m,(s,key) in refs.items()};pro={m:s[market+'_'+key+'_p'] for m,(s,key) in refs.items()};pred['lstm_qr']=qr[market+'_lstm_qr_Q'];cr['lstm_qr']=None;pro['lstm_qr']=None
        for seed in SEEDS:
            for suffix in ['Q','CRPS','p','branch_p','a','b']:arrays[f'{market}_{FROZEN}_seed{seed}_{suffix}']=old[f'{market}_honest_no_readout_seed{seed}_{suffix}']
        mres=dict(audit=D['audit'],preparation=audits,evaluation_reference_equality=equality,frozen_reference='old honest_no_readout25 models, direct stored predictions, no retraining',fits={},by_horizon={});states={}
        print('begin',market,'explicit references checked',flush=True)
        for model in CONTROLS:
            kind=KINDS[model];Xi,bi=prepared[kind]['inner'];Xf,bf=prepared[kind]['final'];allops=[];seedcs=[];fits=[]
            for seed in args.seeds:
                inner,info=B.run_inner(D,Xi,bi,model,seed,args);assert not inner.prior_weight_nonzero_gradient;ep=info['best_epoch'];net,refit=R.run_fit(D,Xf,bf,model,seed,args,'rf',ep);assert not net.prior_weight_nonzero_gradient
                (q,c,p),branch,aa,bb,ops,diag=R.evaluate_seed(net,D,Xf,bf);assert np.array_equal(branch,rule[market+'_qrf_rule_atom_p'])
                if model==CONTROLS[1]:assert np.array_equal(aa,np.ones_like(aa)) and np.array_equal(bb,np.zeros_like(bb))
                states[f'{model}_seed{seed}']={k:v.detach().clone() for k,v in net.state_dict().items()};key=f'{market}_{model}_seed{seed}'
                for suffix,value in [('Q',q),('CRPS',c),('p',p),('branch_p',branch),('a',aa),('b',bb)]:arrays[key+'_'+suffix]=value
                allops.append(ops);seedcs.append(c);info.update(seed=seed,refit_epochs=ep,refit=refit,test_geometry=diag,cache_kind=kind,prior_weight_nonzero_gradient=inner.prior_weight_nonzero_gradient,refit_prior_weight_nonzero_gradient=net.prior_weight_nonzero_gradient,allocated_parameters=sum(p.numel() for p in net.parameters()));fits.append(info)
                for j,h in enumerate(R.T.HOR):
                    mask=hi==j;rows.append(dict(market=market,horizon=int(h),model=model,seed=seed,n=int(mask.sum()),**R.metrics(y[mask],q[mask],c[mask],p[mask])))
                print(market,model,seed,'best',ep,'rawbest',info['best_raw_epoch'],'run',info['epochs_run'],'elapsed',round(time.perf_counter()-started,1),flush=True)
            Q=np.empty((len(te),15));C=np.empty(len(te));PP=np.empty(len(te));convex=0.
            for j,b in enumerate(bf):
                pos=np.flatnonzero(hi==j);qq,cc,pp,_,_=R.mixture(torch.tensor(D['z'][te[pos]]),b['z'],[op[j] for op in allops]);Q[pos]=qq.numpy()*scale[pos,None];C[pos]=cc.numpy()*scale[pos];PP[pos]=pp.numpy();convex=max(convex,float(np.max(C[pos]-np.mean(seedcs,0)[pos])))
            assert convex<1e-6 and np.isfinite(Q).all() and np.all(np.diff(Q,axis=1)>=-1e-8)
            pred[model]=Q;cr[model]=C;pro[model]=PP;mres['fits'][model]=fits;mres.setdefault('mixture_convexity_error',{})[model]=convex
        cp=dict(state_dicts=states,variants=CONTROLS,config=vars(args),cache_info=cacheinfo,preparation=audits,evaluation_reference_equality=equality,final_banks={kind:[{k:b[k] for k in ['z','phi2','mature','p','u','allowed','count','m0','v0','vu','bank_D_indices']} for b in prepared[kind]['final'][1]] for kind in prepared},prior_scalers={kind:audits[kind]['final']['prior_scaler'] for kind in prepared},frozen_reference_checkpoints=f'prior_readout_transport_round5_{market}_checkpoints.pt')
        torch.save(cp,P/f'{args.tag}_{market}_checkpoints.pt')
        for j,h in enumerate(R.T.HOR):
            mask=hi==j;ss={m:R.metrics(y[mask],q[mask],None if cr[m] is None else cr[m][mask],None if pro[m] is None else pro[m][mask]) for m,q in pred.items()};mres['by_horizon'][str(h)]=dict(n=int(mask.sum()),scores=ss)
            for m in pred:rows.append(dict(market=market,horizon=int(h),model=m,seed='ensemble',n=int(mask.sum()),**ss[m]))
        for m in pred:
            arrays[market+'_'+m+'_Q']=pred[m]
            if cr[m] is not None:arrays[market+'_'+m+'_CRPS']=cr[m]
            if pro[m] is not None:arrays[market+'_'+m+'_p']=pro[m]
        report['markets'][market]=mres;report['elapsed_seconds']=time.perf_counter()-started;save(args.tag,report,arrays,rows)
    assert all(hashlib.sha256((P/name).read_bytes()).hexdigest()==sha for name,sha in frozen.items());report['frozen_sha_unchanged']=True;report['elapsed_seconds']=time.perf_counter()-started;save(args.tag,report,arrays,rows)
    lines=['# Prefix与moment transport的同初始化匹配机制对照','',f'完成50新configs及原25模型冻结参照，运行{report["elapsed_seconds"]:.2f}秒。旧参照未重训，两个新control各5市场×5seed，全保留。','',
    '两control均新Net2636参数、same-seed初始state与honest_no_readout逐值一致，prior标准化后四维强制0。matched_insample_no_readout只换旧insample训练参考；matched_honest_no_readout_reweight_only仍honest参考，且严格a1/b0，隔离affine。显式缓存路径、原phase与成熟mask/32NZ/64past/36ctx/selector/refit协议全部不变。','',
    '全部visited raw-relative轨迹仅作诊断，主选择仍normalized；没有按test选择seed/epoch。五seed合并query-specific支持与零atom，精确同F计算Q/CRPS/实际非零质量。LSTMQR仅15q无完整CRPS。','',
    '| 冻结honest_no_readout / 对照 | MQS改善 | 完整CRPS改善 |','|---|---:|---:|']
    for key,v in report['summary'].items():
        if not key.startswith(FROZEN+'_vs_'):continue
        q=v['MQS'];c=v.get('CRPS');ct='15q不可识别' if c is None else f"{c['geometric_gain_percent']:+.4f}% ({c['wins']}/20)";lines.append(f"| {key} | {q['geometric_gain_percent']:+.4f}% ({q['wins']}/20) | {ct} |")
    lines+=['','正值表示冻结honest_no_readout更好。参数分配与随机初始化精确匹配后才解释机制，不把原B不同初始化/常量分支差异称为prefix贡献。新control与固定ref/ruleQRF/rawQRF/LSTM的全部分数另存JSON/cells。','',
    'no-readout推断时prior_encoder(0)=tanh(prior_bias)是常量8向量，shared可将W_prior*tanh(prior_bias)并入shared bias；只说明函数等价，未删模块、未改训练/初始化。人工fold误差已记录。','',
    'checkpoint可按显式cache/path/scaler/mapping重建新10state/market；旧25state继续引用原checkpoint。全部旧文件SHA保持，历史测试已复用，正增量仅是开发候选，尚需冻结滚动复核。']
    (P/(args.tag+'_report_zh.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in report['summary'].items() if k.startswith(FROZEN+'_vs_')},indent=2),flush=True);print('elapsed',report['elapsed_seconds'],'frozen unchanged',True,flush=True)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    with threadpool_limits(limits=1):main()
