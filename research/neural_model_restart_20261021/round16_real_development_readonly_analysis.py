"""Read-only comparison of frozen Round16 2025 DEVELOPMENT forecasts.
No CSV is opened, no model is imported/trained/run, and no 2026 observation is read.
Only new analysis artifacts are written alongside the supplied result files.
"""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,math
import numpy as np

ROOT=Path(__file__).resolve().parent
ASSETS=['KRBN','KCCA']
HORIZONS=[1,5,22]
MODELS=['neural_precision','fixed_precision','raw_constant','bias_constant',
        'neural_constant_precision','HAR_geometry','HGB_geometry','MEM_stationary','MEM_integrated']
BASELINES=MODELS[1:]
OUT_JSON=ROOT/'ROUND16_REAL_DEVELOPMENT_COMPARISON.json'
OUT_MD=ROOT/'ROUND16_REAL_DEVELOPMENT_ZH.md'
assert not OUT_JSON.exists() and not OUT_MD.exists(),'Refuse to overwrite an existing analysis'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def safe(x):
    if isinstance(x,dict):return {str(k):safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [safe(v) for v in x]
    if isinstance(x,np.ndarray):return safe(x.tolist())
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,np.floating):return float(x)
    if isinstance(x,np.bool_):return bool(x)
    return x

def fingerprint(a):
    a=np.ascontiguousarray(a)
    return {'shape':list(a.shape),'dtype':str(a.dtype),'raw_array_bytes_sha256':hashlib.sha256(a.tobytes()).hexdigest()}

json_paths={a:ROOT/f'bridge_precision_proxy_round16_{a}_grid512_development_results.json' for a in ASSETS}
npz_paths={a:ROOT/f'bridge_precision_proxy_round16_{a}_grid512_development_validation_predictions.npz' for a in ASSETS}
mem_json_path=ROOT/'mem_baselines_round16_results.json'
mem_npz_path=ROOT/'mem_baselines_round16_predictions.npz'
inputs=list(json_paths.values())+list(npz_paths.values())+[mem_json_path,mem_npz_path]
input_hash_before={p.name:sha(p) for p in inputs}
reports={a:json.loads(json_paths[a].read_text(encoding='utf-8')) for a in ASSETS}
archives={a:np.load(npz_paths[a],allow_pickle=False) for a in ASSETS}
mem_report=json.loads(mem_json_path.read_text(encoding='utf-8'))
mem=np.load(mem_npz_path,allow_pickle=False)
source_hash_checks=[]
# Do NOT open source CSVs, even for hashing: their declared hashes are retained only.
known_csv_hashes={'KRBN':'ff99c14e84153717be0d72506b0c8b581605495288849823e4bf219ab1462f11',
                  'KCCA':'17c93aeca744b91b041a419413fae105c80aca8c9b214c625c7ab6b47d2cae68'}
for a,report in reports.items():
    for kind,entry in report['hashes'].items():
        p=Path(entry['path'])
        if kind=='csv' or p.suffix.lower()=='.csv':
            source_hash_checks.append({'asset':a,'kind':kind,'path':str(p),'declared_sha256':entry['sha256'],
                'csv_not_opened':True,'matches_previously_archived_acquisition_hash':entry['sha256']==known_csv_hashes[a],
                'verification_limit':'Declared hash agrees with previously acquired immutable snapshot record; CSV bytes deliberately not read/rehashed in this task.'})
        else:
            actual=sha(p)
            source_hash_checks.append({'asset':a,'kind':kind,'path':str(p),'declared_sha256':entry['sha256'],
                'current_sha256':actual,'match':actual==entry['sha256'],'use':'Hash bytes only; no code imported or executed.'})
mem_source_actual=sha(ROOT/'mem_baselines_round16.py')
mem_source_check={'path':'mem_baselines_round16.py','declared_sha256':mem_report['script_sha256'],
                  'current_sha256':mem_source_actual,'match':mem_source_actual==mem_report['script_sha256'],
                  'filter_hash_matches_main_reports':all(mem_report['filter_script_sha256']==reports[a]['hashes']['filter']['sha256'] for a in ASSETS)}

align={};preds={};targets={};seed_audits={};score_checks=[];selection={}
for a in ASSETS:
    z=archives[a];y=z['target'];dates=z['date'];ends=z['label_end']
    assert y.shape==(228,3) and len(dates)==228
    assert np.array_equal(z['horizons'],np.asarray(HORIZONS))
    assert np.array_equal(dates,mem[a+'_date'])
    assert np.array_equal(y,mem[a+'_target'])
    assert np.all(np.isfinite(y)) and np.all(y>0)
    assert np.all(dates>=np.datetime64('2025-01-01')) and np.all(dates<np.datetime64('2026-01-01'))
    assert np.all(ends<np.datetime64('2026-01-01')) and np.all(ends>=dates)
    assert np.all(np.diff(dates)>np.timedelta64(0,'D'))
    assert reports[a]['audit']['validation_common_origins']==228
    assert reports[a]['audit']['last_training_label']=='2024-12-31'
    pred={m:(mem[a+'_'+m] if m.startswith('MEM_') else z[m]) for m in MODELS}
    for m,v in pred.items():
        assert v.shape==y.shape and np.all(np.isfinite(v)) and np.all(v>0),(a,m)
        for j,h in enumerate(HORIZONS):
            q=float(np.mean(np.log(v[:,j])+y[:,j]/v[:,j]));mse=float(np.mean((y[:,j]-v[:,j])**2))
            published=mem_report['markets'][a]['scores'][m][str(h)] if m.startswith('MEM_') else reports[a]['scores'][str(h)][m]
            dq=abs(q-published['mean_QLIKE']);dm=abs(mse-published['MSE'])
            assert dq<1e-12 and dm<1e-18,(a,h,m,dq,dm)
            score_checks.append({'asset':a,'h':h,'model':m,'abs_QLIKE_difference_from_source_JSON':dq,'abs_MSE_difference_from_source_JSON':dm})
    seeds=reports[a]['config']['seeds'];s_audit={}
    for mode in ['raw_constant','bias_constant','fixed_precision','neural_precision','neural_constant_precision']:
        variants=np.stack([z[f'{mode}_seed{s}'] for s in seeds])
        ensemble=variants.mean(axis=0)
        discrepancy=float(np.max(np.abs(ensemble-z[mode])))
        assert discrepancy<1e-14,(a,mode,discrepancy)
        deterministic=bool(all(np.array_equal(variants[0],v) for v in variants[1:]))
        if mode in ['raw_constant','bias_constant','fixed_precision']:assert deterministic
        s_audit[mode]={'seeds':seeds,'stored_ensemble_is_mean_of_forecasts':True,'max_absolute_ensemble_difference':discrepancy,
                       'seed_predictions_exactly_identical':deterministic,
                       'not_independent_datasets':True,
                       'seed_mean_QLIKE':{str(s):float(np.mean(np.log(variants[k])+y/variants[k])) for k,s in enumerate(seeds)}}
    seed_audits[a]=s_audit
    align[a]={'n_common_origins':228,'horizons':HORIZONS,'date_first':str(dates[0].astype('datetime64[D]')),
              'date_last':str(dates[-1].astype('datetime64[D]')),'label_end_first':str(ends[0].astype('datetime64[D]')),
              'label_end_last':str(ends[-1].astype('datetime64[D]')),
              'all_origin_and_label_dates_before_2026':True,'dates_equal_MEM_exactly':True,'targets_equal_MEM_exactly':True,
              'target_min':float(y.min()),'target_max':float(y.max()),'zero_or_nonpositive_targets':0,
              'dates_fingerprint':fingerprint(dates),'target_fingerprint':fingerprint(y),
              'origin_index_first_last':[int(z['origin'][0]),int(z['origin'][-1])],
              'MEM_horizon_order_verified_by_reproducing_all_JSON_scores':True}
    selection[a]={mode:[{'seed':f['seed'],'best_epoch':f['best_epoch'],'epochs_run':f['epochs_run']} for f in fits]
                  for mode,fits in reports[a]['fits'].items()}
    preds[a]=pred;targets[a]=y
cross_dates_equal=np.array_equal(archives['KRBN']['date'],archives['KCCA']['date'])
assert cross_dates_equal,'Date calendars differ; must realign before joint bootstrap'
assert np.array_equal(archives['KRBN']['label_end'],archives['KCCA']['label_end'])

# A x T x H x M. Equal asset/h mean gives each of the six cells weight 1/6.
Y=np.stack([targets[a] for a in ASSETS])
V=np.stack([np.stack([preds[a][m] for m in MODELS],axis=-1) for a in ASSETS])
L=np.log(V)+Y[...,None]/V
NQL=L-np.log(Y[...,None])-1
E2=(V-Y[...,None])**2
assert np.all(NQL>=-1e-12)
assert np.max(np.abs((NQL[...,0,None]-NQL[...,1:])-(L[...,0,None]-L[...,1:])))<1e-12

def metrics(l,nq,e):
    return {m:{'mean_QLIKE':float(l[...,k].mean()),'mean_normalized_QLIKE':float(nq[...,k].mean()),
               'MSE':float(e[...,k].mean())} for k,m in enumerate(MODELS)}

per_cell={a:{str(h):metrics(L[ia,:,j,:],NQL[ia,:,j,:],E2[ia,:,j,:]) for j,h in enumerate(HORIZONS)} for ia,a in enumerate(ASSETS)}
per_asset={a:metrics(L[ia],NQL[ia],E2[ia]) for ia,a in enumerate(ASSETS)}
aggregate=metrics(L,NQL,E2)
D=L[...,0,None]-L[...,1:]
DM=E2[...,0,None]-E2[...,1:]
# Every contrast column below shares the SAME date resample in a bootstrap replicate.
columns=[];meta=[]
for ia,a in enumerate(ASSETS):
    for j,h in enumerate(HORIZONS):
        for k,b in enumerate(BASELINES):
            columns.extend([D[ia,:,j,k],DM[ia,:,j,k]])
            meta.extend([('asset_h',a,str(h),b,'QLIKE'),('asset_h',a,str(h),b,'MSE')])
for ia,a in enumerate(ASSETS):
    for k,b in enumerate(BASELINES):
        columns.extend([D[ia,:,:,k].mean(axis=1),DM[ia,:,:,k].mean(axis=1)])
        meta.extend([('asset',a,'all_h_equal',b,'QLIKE'),('asset',a,'all_h_equal',b,'MSE')])
for k,b in enumerate(BASELINES):
    columns.extend([D[:,:,:,k].mean(axis=(0,2)),DM[:,:,:,k].mean(axis=(0,2))])
    meta.extend([('aggregate','both_assets','all_h_equal',b,'QLIKE'),('aggregate','both_assets','all_h_equal',b,'MSE')])
M=np.column_stack(columns)
boot={};B=10000;n=228
for block,seed in [(22,2026102216),(44,2026102217)]:
    rng=np.random.default_rng(seed);nb=math.ceil(n/block)
    starts=rng.integers(0,n,size=(B,nb))
    indices=((starts[:,:,None]+np.arange(block)[None,None,:])%n).reshape(B,-1)[:,:n]
    weights=np.zeros((B,n),dtype=np.int16)
    np.add.at(weights,(np.arange(B)[:,None],indices),1)
    assert np.all(weights.sum(axis=1)==n)
    bm=weights.astype(float)@M/n
    lo,hi=np.quantile(bm,[.025,.975],axis=0)
    boot[block]={'block_length_trading_origins':block,'B':B,'seed':seed,'sampling':'Circular moving block, uniform start, wrap modulo228, ceil(n/L) blocks then truncate to228',
                 'same_indices_for_both_assets_all_horizons_all_models':True,
                 'bootstrap_indices_raw_int64_sha256':hashlib.sha256(indices.tobytes()).hexdigest(),
                 'bootstrap_type':'Percentile95% conditional on frozen 2025-selected models; descriptive, not selection-adjusted confirmation',
                 'contrasts':[{'scope':s,'asset':a,'h':h,'baseline':b,'metric':met,
                               'point_NN_minus_baseline':float(M[:,i].mean()),'CI95':[float(lo[i]),float(hi[i])],
                               'bootstrap_standard_deviation':float(bm[:,i].std(ddof=1))}
                              for i,(s,a,h,b,met) in enumerate(meta)]}

lookup={(block,c['scope'],c['asset'],c['h'],c['baseline'],c['metric']):c for block,x in boot.items() for c in x['contrasts']}
def comparison(scope,a,h,b,metric='QLIKE'):
    c=lookup[(22,scope,a,h,b,metric)]
    return {'point_NN_minus_baseline':c['point_NN_minus_baseline'],
            'circular_block22_CI95':c['CI95'],'circular_block44_CI95':lookup[(44,scope,a,h,b,metric)]['CI95']}
comparisons={a:{str(h):{b:{'QLIKE':comparison('asset_h',a,str(h),b),'MSE':comparison('asset_h',a,str(h),b,'MSE')} for b in BASELINES} for h in HORIZONS} for a in ASSETS}
agg_comparisons={b:{'QLIKE':comparison('aggregate','both_assets','all_h_equal',b),
                    'MSE':comparison('aggregate','both_assets','all_h_equal',b,'MSE'),
                    'asset_h_point_wins_out_of6':int(np.sum(D[:,:,:,k].mean(axis=1)<0))} for k,b in enumerate(BASELINES)}
asset_comparisons={a:{b:{'QLIKE':comparison('asset',a,'all_h_equal',b),'MSE':comparison('asset',a,'all_h_equal',b,'MSE')} for b in BASELINES} for a in ASSETS}
# Ensure the six supplied input artifacts were not changed while this audit ran.
input_hash_after={p.name:sha(p) for p in inputs}
assert input_hash_after==input_hash_before,'Input mutated during analysis; refuse final comparison.'
result={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'2025 DEVELOPMENT ONLY; no 2026 raw data read or models executed',
        'models':MODELS,'horizons':HORIZONS,'assets':ASSETS,
        'inputs_sha256':input_hash_before,'input_hashes_unchanged_during_analysis':True,
        'source_hash_checks':source_hash_checks,'MEM_source_hash_check':mem_source_check,
        'alignment':align,'cross_asset_date_and_label_end_alignment_exact':True,
        'cross_asset_numeric_origin_indices_not_used_for_alignment':True,
        'target':'Same future mean Parkinson intraday-range proxy; not true integrated variance or return RV',
        'loss_definitions':{'QLIKE':'log(v)+y/v, smaller better','normalized_QLIKE':'y/v-log(y/v)-1 = QLIKE-log(y)-1, y>0',
                            'MSE':'mean((y-v)^2), raw proxy scale',
                            'difference_sign':'neural_precision ensemble minus comparator; negative favors neural',
                            'equal_weight_aggregate':'Average 2assets x3h cells; each cell same228 dates and weight1/6',
                            'percentage_policy':'No percentage calculated from negative raw QLIKE means.'},
        'seed_audit':seed_audits,'validation_epoch_selection_from_source':selection,
        'score_reproduction_checks':{'n':len(score_checks),'max_absolute_QLIKE_discrepancy':max(x['abs_QLIKE_difference_from_source_JSON'] for x in score_checks),
                                     'max_absolute_MSE_discrepancy':max(x['abs_MSE_difference_from_source_JSON'] for x in score_checks)},
        'metrics_per_asset_horizon':per_cell,'metrics_equal_horizon_per_asset':per_asset,'metrics_equal_asset_horizon':aggregate,
        'paired_NN_minus_comparator_per_asset_horizon':comparisons,'paired_equal_horizon_per_asset':asset_comparisons,
        'paired_equal_asset_horizon':agg_comparisons,'bootstrap':{str(k):v for k,v in boot.items()},
        'interpretation':{'MEM':'Assess point and dependent-date intervals separately; outperforming MEM alone is not evidence of added neural precision if simpler filter controls do better.',
                          'selection':'Epochs selected on these same2025 data. CIs are descriptive conditional-on-selection and unadjusted for multiple comparisons; not independent confirmation.',
                          'seeds':'Ensemble means forecast values before loss, not mean seed losses. Seeds are training randomness on same data, not independent samples. NonNN seed copies deterministic.',
                          'dependence':'All assets/horizons jointly resampled by the same dates. KRBN/KCCA may share CCA exposures; no independent-asset bootstrap combined.',
                          'recommendation':'Keep2026 sealed; continue development. A small KRBN5-day signal is not consistent added value across assets/horizons/strong controls.'},
        'scope':'Only supplied2025 result JSON/NPZ plus code/reference file bytes for provenance hashes read; no CSV opened; no model imports/training/inference or literature searches.',
        'analysis_script_sha256':sha(Path(__file__))}
with OUT_JSON.open('x',encoding='utf-8') as f:json.dump(safe(result),f,ensure_ascii=False,indent=2,allow_nan=False)

f6=lambda x:f'{x:+.6f}'
ci=lambda q:'['+f'{q[0]:+.6f}, {q[1]:+.6f}'+']'
lines=['# Round16 真实市场开发结果独立复核（仅2025）','',
       '**建议继续开发并保持2026隔离。神经精度集成在等资产/期限平均上优于新增MEM点估计，但没有超过更强的raw_constant/fixed_precision；KCCA的1/5日退化不可忽略。** 本报告只重算已经冻结的开发预测，不运行模型，不读取原始CSV或任何2026行情。','',
       '## 对齐、目标与来源','',
       '- 两资产均为228个共同起点：2025-01-02—2025-11-28，h=1/5/22；最长标签截止2025-12-31。两资产日期和label_end完全一致，origin整数索引因历史长度不同而不同，未用它跨资产配对。',
       '- 对每个资产，MEM与主模型的date及target逐元素完全一致；所有模型预测有限且>0，所有目标>0。MEM三列顺序通过重算每h原JSON分数核实。',
       f"- 54项模型×资产×期限的QLIKE/MSE均复现源JSON；最大QLIKE差={result['score_reproduction_checks']['max_absolute_QLIKE_discrepancy']:.3g}，最大MSE差={result['score_reproduction_checks']['max_absolute_MSE_discrepancy']:.3g}。", 
       '- 六个输入JSON/NPZ的SHA-256已保存，并验证分析前后不变。代码和Brownian参考文件只读字节核对hash；原始CSV连hash也没有重新读取，其声明hash仅与此前归档记录比对。该限制不影响预测对齐与重评分，但本报告不重新证明原始target构造或完整pipeline无泄漏。',
       '- 目标是未来平均Parkinson盘中区间proxy，不是真实IV/QV，也不是收盘收益RV。MEM按相同proxy建模，不能把它误写成return-GARCH。','',
       '## 损失方向与集成','',
       'QLIKE=log(v)+y/v，越低越好；差值统一为NN集成减比较模型，负数有利于NN。raw QLIKE可为负，不能对负的raw mean计算“提升百分比”。目标全>0，因此补充normalized QLIKE=y/v-log(y/v)-1；它与raw QLIKE只差同一目标常数，全部配对差和排名不变。MSE为原proxy单位，跨资产平均是字面等权，其量级仍受资产尺度影响。',
       '集成先平均seed预测值再算非线性损失。seed11/22/33不是三个独立数据集；raw/bias/fixed三种nonNN的seed预测逐元素完全相同，是deterministic复用，只计一套。神经不同seed也不作为独立样本做显著性检验。','',
       '## 等资产×期限汇总（六格各1/6）','',
       '| 模型 | raw QLIKE | normalized QLIKE | MSE | NN−该模型QLIKE | NN−该模型MSE |',
       '|---|---:|---:|---:|---:|---:|']
for m in MODELS:
    v=aggregate[m];dq=aggregate['neural_precision']['mean_QLIKE']-v['mean_QLIKE'];dm=aggregate['neural_precision']['MSE']-v['MSE']
    lines.append(f"| {m} | {v['mean_QLIKE']:.6f} | {v['mean_normalized_QLIKE']:.6f} | {v['MSE']:.6e} | {dq:+.6f} | {dm:+.6e} |")
lines+=['','## 每资产/期限：同框架强控制','',
        '| 资产 | h | NN raw QLIKE | Δfixed | Δraw | Δbias | Δneural_constant |',
        '|---|---:|---:|---:|---:|---:|---:|']
for a in ASSETS:
    for h in HORIZONS:
        cc=comparisons[a][str(h)]
        vals=[cc[b]['QLIKE']['point_NN_minus_baseline'] for b in ['fixed_precision','raw_constant','bias_constant','neural_constant_precision']]
        lines.append(f"| {a} | {h} | {per_cell[a][str(h)]['neural_precision']['mean_QLIKE']:.6f} | "+' | '.join(f6(v) for v in vals)+' |')
lines+=['','## 每资产/期限：几何预测与新增MEM','',
        '| 资产 | h | ΔHAR_geometry | ΔHGB_geometry | ΔMEM_stationary | ΔMEM_integrated |',
        '|---|---:|---:|---:|---:|---:|']
for a in ASSETS:
    for h in HORIZONS:
        cc=comparisons[a][str(h)]
        vals=[cc[b]['QLIKE']['point_NN_minus_baseline'] for b in ['HAR_geometry','HGB_geometry','MEM_stationary','MEM_integrated']]
        lines.append(f'| {a} | {h} | '+' | '.join(f6(v) for v in vals)+' |')
lines+=['','## 每资产/期限MSE','',
        '| 模型 | KRBN h1 | KRBN h5 | KRBN h22 | KCCA h1 | KCCA h5 | KCCA h22 |',
        '|---|---:|---:|---:|---:|---:|---:|']
for m in MODELS:
    lines.append('| '+m+' | '+' | '.join(f"{per_cell[a][str(h)][m]['MSE']:.6e}" for a in ASSETS for h in HORIZONS)+' |')
lines+=['','## 循环日期块配对bootstrap（开发诊断）','',
        'B=10,000；block22/44个连续交易起点，随机块起点并循环接回，拼接后截取228项。每次重抽对KRBN、KCCA、三个h以及所有模型使用完全相同的date indices，保留跨资产/跨期限和底层CCA敞口相关；没有将独立资产bootstrap拼起来。种子分别2026102216/2026102217。95%区间为percentile区间。22/44长度只约对应10/5个块，228起点的有限开发样本限制不小。','',
        '**这些区间条件于已经选好的2025 epoch，未校正epoch选择、模式选择或多重比较，不能解释为独立确认或确认性p值。** MEM参数从训练目标选择，而各神经/滤波模式有本年validation epoch选择；这不是对神经方法不利的调参安排。','',
        '| NN减比较模型（六格等权） | 点差 | block22 95%CI | block44 95%CI | NN点估计胜出/6 |',
        '|---|---:|---|---|---:|']
for b in BASELINES:
    c=agg_comparisons[b]['QLIKE']
    lines.append(f"| {b} | {f6(c['point_NN_minus_baseline'])} | {ci(c['circular_block22_CI95'])} | {ci(c['circular_block44_CI95'])} | {agg_comparisons[b]['asset_h_point_wins_out_of6']} |")
lines+=['','### 核心单格差值与区间','',
        '| 资产/h | 比较模型 | NN−比较模型 | block22 95%CI | block44 95%CI |',
        '|---|---|---:|---|---|']
for a in ASSETS:
    for h in HORIZONS:
        for b in ['fixed_precision','raw_constant','MEM_stationary','MEM_integrated']:
            c=comparisons[a][str(h)][b]['QLIKE']
            lines.append(f"| {a}/{h} | {b} | {f6(c['point_NN_minus_baseline'])} | {ci(c['circular_block22_CI95'])} | {ci(c['circular_block44_CI95'])} |")
lines+=['','## 对是否打开2026的判断','',
        '1. 必须承认NN超过新增MEM的平均点估计。stationary MEM在KRBN22日略好于NN，其余五格NN点估计较好；integrated MEM六格NN均较好。区间及训练/开发选择差异应同时报告，不能把“未建立神经增量”改写成“NN从未超过MEM”。',
        '2. KRBN5日确有小信号：NN相对fixed/raw/bias、神经常数精度、HAR/HGB和两个MEM的点差均为负。但KRBN1日未超过HAR，22日未超过raw/fixed及stationary MEM，不能仅展示最有利期限。',
        '3. KCCA1/5日明显不支持神经精度增量：相对raw/fixed/bias均更差；KCCA22日也未超过raw或神经常数精度。该资产不应被平均值或MEM对照掩盖。',
        '4. 六格平均虽超过MEM，却落后raw/fixed等更强同框架控制。当前证据不足以把已知偏差校正、状态动力学或其他预测效应归因于神经shape/liquidity精度机制；源结果中各模式动力学也分别拟合，不能当作固定动态下的纯R因果识别。',
        '5. 因此建议保持2026未打开、继续开发。下一阶段应先在开发范围冻结目标、比较规则、选择预算和有说服力的强控制；本报告不运行任何新模型，也不建议因某个单格的小优势立即消耗封存期。','',
        '## 可核验文件与来源哈希','',
        '完整所有模型/单格QLIKE、normalized QLIKE、MSE、48组单格配对差及两种块长区间、按资产汇总、六格等权汇总、seed/epoch审计和来源hash均见 `ROUND16_REAL_DEVELOPMENT_COMPARISON.json`。分析脚本为 `round16_real_development_readonly_analysis.py`。本轮仅新建此脚本、比较JSON和本说明，未改任何输入或模型源文件。','',
        '| 输入归档 | SHA-256 |','|---|---|']
for name,s in input_hash_before.items():lines.append(f'| {name} | `{s}` |')
non_csv_mismatch=[r for r in source_hash_checks if 'match' in r and not r['match']]
lines+=['',f'非CSV来源hash当前不一致项：{len(non_csv_mismatch)}；MEM脚本hash匹配：{mem_source_check["match"]}。逐项证据及对CSV未重读的限制保存在比较JSON。','']
with OUT_MD.open('x',encoding='utf-8') as f:f.write('\n'.join(lines))
print(json.dumps({'output_json':OUT_JSON.name,'output_markdown':OUT_MD.name,'alignment':align,
                  'source_hash_mismatch_count':len(non_csv_mismatch),'MEM_source_hash_check':mem_source_check,
                  'aggregate_metrics':aggregate,'aggregate_comparisons':agg_comparisons},ensure_ascii=False,indent=2))
