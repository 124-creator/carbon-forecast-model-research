# -*- coding: utf-8 -*-
"""Scalar response split-target experiment; immutable frozen QRF anchors.
Only shape_split_round5* outputs; no neural fitting and identity transport.
"""
from __future__ import annotations
import os,sys
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
sys.dont_write_bytecode=True
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from scipy.special import ndtri
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestRegressor
from threadpoolctl import threadpool_limits
import qrf_round4 as Q
import qrf_round4_probability_control as PC
import forest_prior_round5 as PR
import forest_transport_round5 as F
R=F.R;T=R.T;P=Path(__file__).resolve().parent
ARMS=['mse_y','mse_abs','rank_score','mse_z2'];SEEDS=[11,22,33,44,55];LEAVES=[10,25,50]
ESS={'CEA':'fixed_0.50','GUANGZHOU':'fixed_0.50','SHANGHAI':'ess_l0.75_k16','HUBEI':'fixed_0.20','EU':'fixed_0.50'}


def transformed(z,arm):
    z=np.asarray(z,np.float32)
    if arm=='mse_y':out=z.copy()
    elif arm=='mse_abs':out=np.abs(z)
    elif arm=='mse_z2':out=z*z
    elif arm=='rank_score':out=ndtri((rankdata(z,method='average')-.5)/len(z)).astype(np.float32)
    else:raise KeyError(arm)
    assert np.isfinite(out).all();return np.ascontiguousarray(out,np.float32)


def predict_raw(x,z,fit,queries,arm,leaf,seed):
    response=transformed(z[fit],arm)
    tree=RandomForestRegressor(min_samples_leaf=int(leaf),random_state=seed,**Q.RF_PARAMS)
    tree.fit(x[fit],response);w,error=Q.leaf_weights(tree,x[fit],x[queries],sanity=True)
    return w,dict(leaf_brute_error=error,response_mean=float(response.mean()),response_std=float(response.std()),response_min=float(response.min()),response_max=float(response.max()),fit_rows=len(fit),query_rows=len(queries),zero_target_count=int(np.sum(z[fit]==0)))


def identity(D,X,bank,rows,raw_nz):
    sids=X['sid'][torch.tensor(rows)];allowed=bank['allowed'][sids].numpy()
    u,forest,uniform,count,fallback=PR.make_reference(raw_nz,allowed,np.ones(len(rows),bool))
    local=dict(bank);local['count']=bank['count'][sids];local['vu']=bank['vu'][sids]
    F.reference_moments(local,torch.tensor(u))
    with torch.no_grad():
        op=F.forest_operator(torch.zeros(len(rows),4),local,torch.arange(len(rows)),X['local'][torch.tensor(rows)],F.REFERENCE)
        q,c,p,v,w=R.mixture(X['z'][torch.tensor(rows)],bank['z'],[op]);independent=R.cdf_integral(X['z'][torch.tensor(rows)].numpy(),v.numpy(),w.numpy())
    error=float(np.max(np.abs(c.numpy()-independent)));assert error<2e-10
    sc=D['scale'][rows];out=(q.numpy()*sc[:,None],c.numpy()*sc,p.numpy())
    return out,op,u,dict(CDF_CRPS_error=error,fallback_queries=int(fallback.sum()),max_mass_error=float(np.max(np.abs(w.sum(1).numpy()-1))),minimum_variance=float(op['v0'].min()))


def equal_conditional(D,X,bank,rows,ops):
    with torch.no_grad():q,c,p,_,_=R.mixture(X['z'][torch.tensor(rows)],bank['z'],ops)
    sc=D['scale'][rows];return q.numpy()*sc[:,None],c.numpy()*sc,p.numpy()


def scores(y,pred):return R.metrics(y,*pred)


def summary(rows):
    lookup={(r['market'],r['horizon'],r['model'],r['seed']):r for r in rows};out=[]
    for model in ARMS+['selected_composite']:
        for seed in [*map(str,SEEDS),'canonical']:
            selected=[r for r in rows if r['model']==model and r['seed']==seed]
            for base in ['mse_y','rule_qrf','qrf_original','forest_reference_only','honest_no_readout','rbf_additive','lstm_qr','ess_selected']:
                if model==base:continue
                bseed='canonical' if base=='mse_y' else 'fixed'
                for metric in ['MQS','CRPS','WIS','MAE']:
                    pairs=[(r,lookup.get((r['market'],r['horizon'],base,bseed))) for r in selected];pairs=[(a,b) for a,b in pairs if b is not None and a[metric] is not None and b[metric] is not None]
                    if not pairs:continue
                    for group in ['all',*dict.fromkeys(a['market'] for a,b in pairs)]:
                        pp=[(a,b) for a,b in pairs if group=='all' or a['market']==group];ratios=np.array([a[metric]/b[metric] for a,b in pp])
                        out.append(dict(model=model,seed=seed,baseline=base,metric=metric,group=group,cells=len(pp),wins=int((ratios<1).sum()),geometric_gain_percent=float(100*(1-np.exp(np.log(ratios).mean())))))
    return out


def save(tag,r,arrays,rows):
    r['summary']=summary(rows);(P/(tag+'.json')).write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8');pd.DataFrame(rows).to_csv(P/(tag+'_cells.csv'),index=False);np.savez_compressed(P/(tag+'_predictions.npz'),**arrays)


class Tee:
    def __init__(self,screen,path):self.screen=screen;self.file=path.open('a',encoding='utf-8')
    def write(self,value):self.screen.write(value);self.file.write(value);self.file.flush()
    def flush(self):self.screen.flush();self.file.flush()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--markets',nargs='+',default=list(T.B.CUT));ap.add_argument('--arms',nargs='+',default=ARMS);ap.add_argument('--tag',default='shape_split_round5');args=ap.parse_args();assert args.tag.startswith('shape_split_round5') and 'mse_y' in args.arms
    sys.stdout=Tee(sys.stdout,P/(args.tag+'.log'));torch.set_num_threads(1);torch.set_num_interop_threads(1)
    manifest=json.loads((P/'shape_split_round5_manifest.json').read_text(encoding='utf-8'));assert manifest['final_seeds']==SEEDS and manifest['gamma']==.05
    files=[]
    for pattern in ['qrf_round4*','mixed_cdf_forest_round4*','forest_prior_round5*','forest_transport_round5*','prior_readout_transport_round5*','prefix_transport_controls_round5*','additive_transport_round5*','ess_shrinkage_round5*','deep_seq_round4*','empirical_tilt_round3*','verify_empirical_affine_round4*']:files += [p for p in P.glob(pattern) if p.is_file()]
    frozen={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files};selection=json.loads((P/'qrf_round4.json').read_text(encoding='utf-8'))
    cached={key:np.load(P/name) for key,name in {'rule':'qrf_round4_probability_control_predictions.npz','forest':'forest_transport_round5_predictions.npz','no':'prior_readout_transport_round5_predictions.npz','add':'additive_transport_round5_predictions.npz','deep':'deep_seq_round4_predictions.npz','ess':'ess_shrinkage_round5_predictions.npz'}.items()}
    panel=pd.read_csv(T.B.PANEL,parse_dates=['date']);started=time.perf_counter();r=dict(manifest=manifest,config=vars(args),frozen_sha256=frozen,markets={},forest_fits=0,anchor_checks=[],ensemble='canonical equal raw probabilities including zero ->conditionNZ ->fixedp/gamma; per-seed and equal-conditioned diagnostic kept distinct',no_neural_training=True);arrays={};rows=[]
    for market in args.markets:
        d=panel[panel.market==market].sort_values('date').reset_index(drop=True);D=R.S.build(d);D['market']=market;te=D['te'];hi=D['hi'][te];y=D['y'][te];prepared,audit,meta=F.load_market(D,market);Xi,bi=prepared['inner'];Xf,bf=prepared['final'];xi=Xi['x'].numpy();xf=Xf['x'].numpy();z=Xi['z'].numpy()
        for field,value in [('origin',D['t'][te]),('horizon',T.HOR[hi]),('target',y),('scale',D['scale'][te])]:
            for source_name,source in cached.items():
                if market+'_'+field in source:assert np.array_equal(value,source[market+'_'+field])
                else:assert source_name=='ess' and field=='scale'
            arrays[market+'_'+field]=value
        boundary=D['audit']['validation_start_index'];cut=D['audit']['cut_index'];ends=D['t']+T.HOR[D['hi']]
        assert boundary==max(T.B.W+50,int(cut*.75)) and np.all(ends[D['tr']]<boundary) and np.all(D['t'][D['va']]>=boundary) and np.all(ends[D['va']]<cut)
        arrays[market+'_origin_date']=cached['forest'][market+'_origin_date'];oldw=np.load(P/f'qrf_round4_{market}_weights.npz');oldleaf={int(h):selection['markets'][market]['selection'][str(h)]['chosen_minleaf'] for h in T.HOR}
        validation={};anchor_h={};chosen={};inner_tables={};mres=dict(audit=D['audit'],phase_cache_audit=audit,reference_cache=meta['path'],reference_cache_sha256=meta['sha256'],old_leaf_by_horizon=oldleaf,selection={},final={},by_horizon={})
        for arm in args.arms:
            candidates=[None] if arm=='mse_y' else LEAVES;table=[]
            for leaf in candidates:
                by_h={}
                for j,h in enumerate(T.HOR):
                    fit=D['tr'][D['hi'][D['tr']]==j];va=D['va'][D['hi'][D['va']]==j];nz=PR.nonzero_ids(D,D['tr'],j);lookup={int(v):k for k,v in enumerate(fit)};cols=np.array([lookup[int(v)] for v in nz]);selected=oldleaf[int(h)] if leaf is None else leaf
                    w,info=predict_raw(xi,z,fit,va,arm,selected,11);r['forest_fits']+=1;pred,_,_,diag=identity(D,Xi,bi[j],va,w[:,cols]);mq=scores(D['y'][va],pred)['MQS'];by_h[str(h)]=dict(MQS=mq,leaf=int(selected),**diag,**info)
                    if arm=='mse_y':anchor_h[str(h)]=mq
                value=float(np.mean([np.log(by_h[str(h)]['MQS']/anchor_h[str(h)]) for h in T.HOR]));table.append(dict(leaf=leaf,macro_log_MQS_anchor_ratio=value,by_horizon=by_h))
            best=min(table,key=lambda v:v['macro_log_MQS_anchor_ratio']);tied=[v['leaf'] for v in table if v['macro_log_MQS_anchor_ratio']==best['macro_log_MQS_anchor_ratio']];chosen[arm]=best['leaf'];validation[arm]=best['macro_log_MQS_anchor_ratio'];inner_tables[arm]=table;mres['selection'][arm]=dict(chosen_leaf=best['leaf'],tied_best_leaves=tied,tie_rule='smaller leaf, candidates10/25/50 in fixed ascending order',fixed_old_leaf_by_horizon=oldleaf if arm=='mse_y' else None,macro_log_MQS_anchor_ratio=validation[arm],candidates=table)
        composite=min(args.arms,key=lambda a:validation[a]);mres['selected_composite_arm']=composite;predictions={};print('inner',market,'chosen_leaf',chosen,'arm_macro',validation,'composite',composite,flush=True)
        for arm in args.arms:
            results={str(seed):(np.empty((len(te),15)),np.empty(len(te)),np.empty(len(te))) for seed in SEEDS};results['canonical']=(np.empty((len(te),15)),np.empty(len(te)),np.empty(len(te)));results['equal_conditional_diagnostic']=(np.empty((len(te),15)),np.empty(len(te)),np.empty(len(te)));arm_info={}
            for j,h in enumerate(T.HOR):
                fit=D['rf'][D['hi'][D['rf']]==j];pos=np.flatnonzero(hi==j);test=te[pos];nz=PR.nonzero_ids(D,D['rf'],j);lookup={int(v):k for k,v in enumerate(fit)};cols=np.array([lookup[int(v)] for v in nz]);order=np.argsort(z[fit],kind='stable');leaf=oldleaf[int(h)] if arm=='mse_y' else chosen[arm]
                total=np.zeros((len(test),len(fit)),np.float32);ops=[];mass=[];seed_diag={};key=f'{market}_{arm}_h{h}';arrays[key+'_support_nz_z']=bf[j]['z'].numpy();arrays[key+'_bank_D_indices']=nz;arrays[key+'_test_positions']=pos
                for seed in SEEDS:
                    w,info=predict_raw(xf,z,fit,test,arm,leaf,seed);r['forest_fits']+=1;total+=w/len(SEEDS)
                    if arm=='mse_y':assert np.array_equal(w[:,order],oldw[f'seed{seed}_weights_h{h}'])
                    pred,op,u,diag=identity(D,Xf,bf[j],test,w[:,cols]);ops.append(op);mass.append(w[:,cols].sum(1,dtype=np.float64));seed_diag[str(seed)]=dict(**info,**diag)
                    for k in range(3):results[str(seed)][k][pos]=pred[k]
                    arrays[key+f'_seed{seed}_u']=u
                pred,_,u,diag=identity(D,Xf,bf[j],test,total[:,cols]);arrays[key+'_canonical_u']=u
                for k in range(3):results['canonical'][k][pos]=pred[k]
                eq=equal_conditional(D,Xf,bf[j],test,ops)
                for k in range(3):results['equal_conditional_diagnostic'][k][pos]=eq[k]
                mass=np.stack(mass,1);alpha=mass/mass.sum(1,keepdims=True).clip(1e-30);alpha[mass.sum(1)<=0]=1/len(SEEDS);arrays[key+'_raw_nz_mass_by_seed']=mass;arrays[key+'_conditional_seed_mixture_weights']=alpha
                arm_info[str(h)]=dict(leaf=leaf,seeds=seed_diag,canonical=diag,seed_raw_NZ_mass_min=float(mass.min()),seed_raw_NZ_mass_max=float(mass.max()),conditional_seed_weight_min=float(alpha.min()),conditional_seed_weight_max=float(alpha.max()),canonical_vs_equal_conditioned_Q_max=float(np.max(np.abs(pred[0]-eq[0]))),canonical_vs_equal_conditioned_CRPS_max=float(np.max(np.abs(pred[1]-eq[1]))))
                if arm=='mse_y':
                    assert np.array_equal(total[:,order],oldw[f'ensemble_weights_h{h}'])
                    probability=(.8*Xf['local'][torch.tensor(test)]+.2*torch.full_like(Xf['local'][torch.tensor(test)],bf[j]['p'])).numpy()
                    torch.set_num_threads(2);support,new,_=PC.replace_probability(z[fit][order],total[:,order],probability.astype(np.float64));q0,c0,e0=PC.evaluate_distribution(D['z'][test],D['scale'][test],support,new);torch.set_num_threads(1)
                    for actual,suffix in [(q0,'Q'),(c0,'CRPS'),(probability,'p')]:assert np.array_equal(actual,cached['rule'][market+'_qrf_rule_atom_'+suffix][pos]),('gamma0',market,int(h),suffix,float(np.max(np.abs(actual-cached['rule'][market+'_qrf_rule_atom_'+suffix][pos]))))
                    for actual,suffix in zip(pred,['Q','CRPS','p']):assert np.array_equal(actual,cached['forest'][market+'_forest_reference_only_'+suffix][pos]),('gamma05',market,int(h),suffix,float(np.max(np.abs(actual-cached['forest'][market+'_forest_reference_only_'+suffix][pos]))))
                    r['anchor_checks'].append(dict(market=market,horizon=int(h),raw_seed_and_mean_weights_bitwise_equal=True,gamma0_Q_CRPS_p_bitwise_equal_rule_QRF=True,gamma05_Q_CRPS_p_bitwise_equal_fixed_reference=True,legacy_independent_CRPS_error=e0))
                print('final',market,arm,int(h),'leaf',leaf,'fits',r['forest_fits'],'elapsed',round(time.perf_counter()-started,1),flush=True)
            predictions[arm]=results;mres['final'][arm]=arm_info
            for seed,pred in results.items():
                name=arm if seed!='equal_conditional_diagnostic' else arm+'_equal_conditional_diagnostic';skey=seed if seed!='equal_conditional_diagnostic' else 'diagnostic';prefix=f'{market}_{arm}_'+('ensemble' if seed=='canonical' else 'seed'+seed if seed in list(map(str,SEEDS)) else seed)
                for suffix,value in zip(['Q','CRPS','p'],pred):arrays[prefix+'_'+suffix]=value
                for j,h in enumerate(T.HOR):
                    mask=hi==j;rows.append(dict(market=market,horizon=int(h),model=name,seed=skey,n=int(mask.sum()),**scores(y[mask],tuple(v[mask] for v in pred))))
        predictions['selected_composite']=predictions[composite]
        for seed in [*map(str,SEEDS),'canonical']:
            pred=predictions[composite][seed];prefix=f'{market}_selected_composite_'+('ensemble' if seed=='canonical' else 'seed'+seed)
            for suffix,value in zip(['Q','CRPS','p'],pred):arrays[prefix+'_'+suffix]=value
            for j,h in enumerate(T.HOR):
                mask=hi==j;rows.append(dict(market=market,horizon=int(h),model='selected_composite',seed=seed,n=int(mask.sum()),**scores(y[mask],tuple(v[mask] for v in pred))))
        bases={'rule_qrf':('rule','qrf_rule_atom'),'qrf_original':('rule','qrf_original'),'forest_reference_only':('forest','forest_reference_only'),'honest_no_readout':('no','honest_no_readout'),'rbf_additive':('add','rbf_prior'),'lstm_qr':('deep','lstm_qr'),'ess_selected':('ess',ESS[market])}
        for name,(src,oldkey) in bases.items():
            q=cached[src][market+'_'+oldkey+'_Q'];c=cached[src][market+'_'+oldkey+'_CRPS'] if market+'_'+oldkey+'_CRPS' in cached[src] else None;p=cached[src][market+'_'+oldkey+'_p'] if market+'_'+oldkey+'_p' in cached[src] else None
            for suffix,v in [('Q',q),('CRPS',c),('p',p)]:
                if v is not None:arrays[market+'_'+name+'_'+suffix]=v
            for j,h in enumerate(T.HOR):
                mask=hi==j;ss=scores(y[mask],(q[mask],None if c is None else c[mask],None if p is None else p[mask]));rows.append(dict(market=market,horizon=int(h),model=name,seed='fixed',n=int(mask.sum()),**ss))
        r['markets'][market]=mres;r['elapsed_seconds']=time.perf_counter()-started;save(args.tag,r,arrays,rows)
    assert all(hashlib.sha256((P/name).read_bytes()).hexdigest()==sha for name,sha in frozen.items());r['frozen_sha_unchanged']=True;r['elapsed_seconds']=time.perf_counter()-started;save(args.tag,r,arrays,rows)
    ss=[x for x in r['summary'] if x['seed']=='canonical' and x['group']=='all' and x['metric'] in ['MQS','CRPS']]
    lines=['# 形状分裂响应：四臂冻结机制实验','',f'完成{r["forest_fits"]}森林拟合，运行{r["elapsed_seconds"]:.2f}秒；没有NN训练。A旧leaf固定，B/C/D仅inner选各market-arm共同leaf，四臂全部保留test。','',
    'A虽然叫mse_y，实际沿用冻结QRF的标准化float32 z响应；B=abs(z)、C=fit-only平均秩正态分数、D=z²。叶内仍用原有符号z的经验分布，未替换mark。仅更改分裂响应，不把单一变换响应MSE称为完整形状/CDF impurity。','',
    '每seed原始leaf概率包含零质量，canonical先等权均值raw概率再条件NZ、套rule p和γ=.05；等价最终条件seed混合权重与其原始NZ质量成正比，不是等权conditional-F平均。每seed独立结果和等权conditional-F诊断另外全保存，不把异质性藏进均值。','',
    'A每h逐seed/mean raw权重与冻结QRF逐位一致；γ0通过旧完整zero支持evaluator逐位复现ruleQRF，γ.05通过旧identity evaluator逐位复现forest_reference_only。失败会终止，未静默容忍锚改变。','', '| 模型 | 基线 | MQS改善 | CRPS改善 |','|---|---|---:|---:|']
    for arm in args.arms:
        for base in ['mse_y','rule_qrf','honest_no_readout','rbf_additive','lstm_qr']:
            dd={s['metric']:s for s in ss if s['model']==arm and s['baseline']==base}
            if 'MQS' not in dd:continue
            q=dd['MQS'];c=dd.get('CRPS');ct='15q不可识别' if c is None else f"{c['geometric_gain_percent']:+.4f}% ({c['wins']}/20)";lines.append(f"| {arm} | {base} | {q['geometric_gain_percent']:+.4f}% ({q['wins']}/20) | {ct} |")
    lines+=['','硬性约束逐项核对：','1. 满足：仅inner Dtr拟合，Dva为原cut前最后25%（保留原lookback下限），seed11的一组80树森林；按四h rawMQS相对A的mean log比，共同leaf固定10/25/50；精确并列取更小leaf并在selection.tied_best_leaves留痕。','2. 满足：A不搜索旧四h leaf；完整多零支持γ0 evaluator与ruleQRF逐位，γ.05 identity evaluator与fixed reference逐位，任何失败均终止。','3. 满足：四臂四h五seed全final；逐seed、canonical和equal-conditional-final三套完整保留。','4. 满足：composite标为reference-only，留在JSON/cells，未进入本主表和机制结论。','5. 满足：inner mean log比>0的B/C/D仍全部跑final，没有改判据/换leaf/中途剔除。','', '正值左侧更好；composite仅按inner预注册规则选臂，是二次选择参考而非主结果。逐market/h与每seed对各基线gains完整保存在JSON summary/cells，NPZ含全部Q/CRPS/p及每h/seed/canonical u、rawNZ质量和条件seed混合权重，可重建F。','',
    '历史test已开发复用；若正向只来自上海或个别市场，应解释为分裂目标未改变当前结论，不用composite掩盖负方向。全部旧文件SHA前后一致，不宣称模型新颖性/显著性或发表成功。']
    (P/(args.tag+'_report_zh.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(ss,ensure_ascii=False,indent=2),flush=True);print('elapsed',r['elapsed_seconds'],'all anchors bitwise pass','frozen unchanged',True,flush=True)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    with threadpool_limits(limits=2):main()
