"""Pure synthetic zero-drift Brownian measurement-reference experiment.
No market data, model predictions, labels, or files outside this directory are read.
"""
from pathlib import Path
from datetime import datetime, timezone
import json, math, hashlib, platform, time
import numpy as np

OUT = Path(__file__).resolve().parent
EDGES = np.linspace(0.0, 1.0, 6)
MIN_CELL = 250
R_FLOOR = 1e-8
BATCH = 1024
CONFIGS = [
    {'name':'grid512_train','steps':512,'n':60000,'seed':2026102116},
    {'name':'grid512_heldout','steps':512,'n':20000,'seed':2026102117},
    {'name':'grid2048_train','steps':2048,'n':20000,'seed':2026102118},
    {'name':'grid2048_heldout','steps':2048,'n':10000,'seed':2026102119},
]
ARTIFACTS = ['brownian_reference_samples.npz', 'brownian_shape_models.json',
             'brownian_reference_results.json', 'experiment_config.json',
             'README_brownian_reference_zh.md', 'artifact_hashes.json']
assert not any((OUT/f).exists() for f in ARTIFACTS), 'Outputs already exist; refuse overwrite.'


def json_safe(x):
    if isinstance(x, dict): return {str(k):json_safe(v) for k,v in x.items()}
    if isinstance(x, (tuple,list)): return [json_safe(v) for v in x]
    if isinstance(x, np.ndarray): return json_safe(x.tolist())
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x) if np.isfinite(x) else None
    return x


def write_json(name, data):
    with (OUT/name).open('x',encoding='utf-8') as f:
        json.dump(json_safe(data), f, ensure_ascii=False, indent=2, allow_nan=False)


def summary(v):
    v = np.asarray(v, dtype=float)
    return {'n':v.size, 'mean':float(v.mean()), 'std_ddof1':float(v.std(ddof=1)),
            'mean_se':float(v.std(ddof=1)/math.sqrt(v.size)),
            'quantiles_01_10_50_90_99':np.quantile(v,[.01,.1,.5,.9,.99]).tolist()}


def simulate(cfg):
    rng = np.random.default_rng(cfg['seed'])
    n, steps = cfg['n'], cfg['steps']
    H,L,C = (np.empty(n) for _ in range(3))
    for lo in range(0,n,BATCH):
        hi = min(n,lo+BATCH)
        path = rng.standard_normal((hi-lo,steps))
        path *= 1/math.sqrt(steps)
        np.cumsum(path,axis=1,out=path)
        # H/L and C come from the SAME sampled path; include the O=0 endpoint.
        H[lo:hi] = np.maximum(0,path.max(axis=1))
        L[lo:hi] = np.minimum(0,path.min(axis=1))
        C[lo:hi] = path[:,-1]
    q = H-L
    assert np.all(q>0)
    eta = C/q
    xi = (H+L-C)/q
    diamond = np.abs(eta)+np.abs(xi)
    assert np.all(diamond<=1+1e-12), 'OHLC diamond geometry violation'
    a = np.abs(eta)
    denom = 1-a
    boundary = denom<=1e-12
    z = np.divide(np.abs(xi),denom,out=np.zeros(n),where=~boundary)
    assert np.all(z>=-1e-12) and np.all(z<=1+1e-10)
    # Only clamp rounding-size coordinate excursions for BIN INDEXING.
    # At |eta|=1, xi=0: choose z=0 explicitly; preserve original OHLC/eta/xi.
    ia = np.clip(np.searchsorted(EDGES,np.clip(a,0,1),side='right')-1,0,4)
    iz = np.clip(np.searchsorted(EDGES,np.clip(z,0,1),side='right')-1,0,4)
    cell = ia*5+iz
    P = q*q/(4*math.log(2))
    y = np.log(P)
    data = {'H':H,'L':L,'C':C,'q':q,'eta':eta,'xi':xi,'a':a,'z':z,
            'P':P,'y':y,'cell':cell,'boundary':boundary}
    sanity = {'config':cfg, 'O':0, 'geometry_max_abs_eta_plus_abs_xi':float(diamond.max()),
              'geometry_violations_over_1e_12':int(np.sum(diamond>1+1e-12)),
              'boundary_abs_eta_equals_1_count':int(boundary.sum()),
              'endpoint':summary(C), 'endpoint_second_moment':float(np.mean(C*C)),
              'P':summary(P),'y':summary(y),
              'same_path_OHLC_order_violation_count':int(np.sum((L>np.minimum(0,C))|(H<np.maximum(0,C))))}
    print('SIMULATED',cfg['name'],'n',n,'steps',steps,flush=True)
    return data,sanity


def predict(data,model):
    cells=data['cell']
    return np.asarray(model['b_by_cell'])[cells],np.asarray(model['R_by_cell'])[cells]


def fit_model(train, label):
    y,cell=train['y'],train['cell']
    b0=float(y.mean()); R0=float(y.var(ddof=1))
    count=np.bincount(cell,minlength=25)
    b=np.full(25,b0); R=np.full(25,R0)
    audit=[]
    for k in range(25):
        v=y[cell==k]
        supported=len(v)>=MIN_CELL
        sample_b=float(v.mean()) if len(v) else None
        sample_R=float(v.var(ddof=1)) if len(v)>1 else None
        if supported:
            b[k]=sample_b; R[k]=max(sample_R,R_FLOOR)
        audit.append({'cell':k,'a_bin':k//5,'z_bin':k%5,'n_train':len(v),
                      'fallback':not supported,'raw_train_cell_mean':sample_b,
                      'raw_train_cell_variance_ddof1':sample_R,
                      'b_used':float(b[k]),'R_used':float(R[k])})
    train_x=y-b[cell]
    # Strong constant-R control: fitted on the SAME shape-bias-corrected residual.
    # Mean-square (Gaussian MLE under mean 0); no heldout fitting or tuning.
    const_R=float(np.mean(train_x*train_x))
    pred_R=R[cell]
    group_edges=np.unique(np.quantile(pred_R,[0,.2,.4,.6,.8,1]))
    model={'label':label,'train_n':len(y),'edges_a':EDGES,'edges_z':EDGES,
           'min_train_cell':MIN_CELL,'R_floor':R_FLOOR,
           'global_b':b0,'global_R_ddof1':R0,
           'shape_bias_constant_R_train_residual_mean_square':const_R,
           'constant_R_definition':'mean((train_y - train_shape_b(cell))^2), fitted only on train',
           'b_by_cell':b,'R_by_cell':R,'train_count_by_cell':count,
           'fallback_cell_count':int(np.sum(count<MIN_CELL)),
           'fallback_training_path_count':int(np.sum(count[count<MIN_CELL])),
           'floor_activated_supported_cells':sum(audit[k]['n_train']>=MIN_CELL and audit[k]['raw_train_cell_variance_ddof1']<R_FLOOR for k in range(25)),
           'cell_audit':audit,'calibration_R_group_edges_from_train':group_edges,
           'train_predicted_R':summary(pred_R),
           'train_R_p90_over_p10':float(np.quantile(pred_R,.9)/np.quantile(pred_R,.1))}
    print('FIT',label,'fallback cells',model['fallback_cell_count'],flush=True)
    return model


def pearson(x,y):
    x=np.asarray(x);y=np.asarray(y)
    if np.std(x)==0 or np.std(y)==0:return None
    return float(np.corrcoef(x,y)[0,1])


def average_ranks(v):
    _,inverse,counts=np.unique(v,return_inverse=True,return_counts=True)
    starts=np.cumsum(counts)-counts
    return (starts+(counts-1)/2)[inverse]


def paired_delta(left,right):
    d=left-right
    se=float(d.std(ddof=1)/math.sqrt(len(d)))
    return {'mean':float(d.mean()),'paired_se':se,
            'paired_normal_95_CI':[float(d.mean()-1.96*se),float(d.mean()+1.96*se)],
            'n':len(d),'negative_is_better_for_left':True}


def evaluate(hold,model,label):
    y=hold['y']; b,R=predict(hold,model)
    x=y-b; x2=x*x
    x0=y-model['global_b']
    R0=model['global_R_ddof1']; Rc=model['shape_bias_constant_R_train_residual_mean_square']
    s0=math.log(R0)+x0*x0/R0
    sc=math.log(Rc)+x2/Rc
    ss=np.log(R)+x2/R
    # B and C use this exact same x array, not two separate bias corrections.
    scores={'global_bias_global_R':summary(s0),
            'shape_bias_constant_R_STRONG_CONTROL':summary(sc),
            'shape_bias_shape_R':summary(ss)}
    edges=np.asarray(model['calibration_R_group_edges_from_train'])
    group=np.searchsorted(edges[1:-1],R,side='right')
    groups=[]
    for g in range(len(edges)-1):
        m=group==g
        if not m.any():continue
        v=x2[m];rv=R[m]
        groups.append({'group':g,'lower_train_R_edge':float(edges[g]),'upper_train_R_edge':float(edges[g+1]),
                       'n_heldout':int(m.sum()),'mean_predicted_R':float(rv.mean()),
                       'mean_error_squared':float(v.mean()),'error_squared_mean_se':float(v.std(ddof=1)/math.sqrt(len(v))),
                       'mean_error_squared_over_mean_R':float(v.mean()/rv.mean()),
                       'mean_corrected_error':float(x[m].mean()),'mean_score_shape_R':float(ss[m].mean())})
    cells=[]
    for k in range(25):
        m=hold['cell']==k
        cells.append({'cell':k,'a_bin':k//5,'z_bin':k%5,
                      'train_count':int(model['train_count_by_cell'][k]),
                      'train_fallback':bool(model['train_count_by_cell'][k]<MIN_CELL),
                      'heldout_count':int(m.sum()),'predicted_R':float(model['R_by_cell'][k]),
                      'mean_corrected_error':float(x[m].mean()) if m.any() else None,
                      'mean_error_squared':float(x2[m].mean()) if m.any() else None,
                      'error_squared_mean_se':float(x2[m].std(ddof=1)/math.sqrt(m.sum())) if m.sum()>1 else None})
    delta=paired_delta(ss,sc)
    result={'label':label,'n_heldout':len(y),'model_fitted_only_on':model['label'],
            'score_definition':'log(R)+x^2/R; smaller better; Gaussian quasi measurement score with latent log variance fixed at 0 in this synthetic reference, no market forecast',
            'raw_logP':summary(y),'raw_P':summary(hold['P']),
            'global_b':model['global_b'],'global_R':R0,'shape_bias_constant_R':Rc,
            'scores':scores,
            'precision_delta_shape_R_minus_STRONG_constant_R':delta,
            'bias_delta_shape_bias_constant_R_minus_global_bias_global_R':paired_delta(sc,s0),
            'combined_delta_shape_R_minus_global_bias_global_R':paired_delta(ss,s0),
            'global_bias_corrected_error':summary(x0),'shape_bias_corrected_error':summary(x),
            'shape_corrected_error_MSE':float(x2.mean()),'global_corrected_error_MSE':float(np.mean(x0*x0)),
            'mean_predicted_shape_R':float(R.mean()),'overall_error_squared_over_mean_shape_R':float(x2.mean()/R.mean()),
            'error_squared_vs_predicted_R_pearson':pearson(x2,R),
            'error_squared_vs_predicted_R_spearman_tie_aware':pearson(average_ranks(x2),average_ranks(R)),
            'R_groups_fixed_from_training_prediction_quantiles':groups,
            'high_vs_low_R_group_empirical_MSE_ratio':groups[-1]['mean_error_squared']/groups[0]['mean_error_squared'] if len(groups)>1 else None,
            'heldout_fallback_path_count':int(np.sum(np.asarray(model['train_count_by_cell'])[hold['cell']]<MIN_CELL)),
            'cell_calibration':cells,
            'heldout_is_diagnostic_only':'No model means/variances/bin edges/group thresholds refitted to heldout.'}
    arrays={'x_global':x0,'x_shape':x,'predicted_R_shape':R,'score_global':s0,
            'score_shape_bias_constant_R':sc,'score_shape_bias_shape_R':ss,'R_group':group}
    print('EVAL',label,'precision delta',delta['mean'],'CI',delta['paired_normal_95_CI'],flush=True)
    return result,arrays


def main():
    started=time.time(); samples={};sanity={};extra={}
    for cfg in CONFIGS:
        data,checks=simulate(cfg);samples[cfg['name']]=data;sanity[cfg['name']]=checks
    m512=fit_model(samples['grid512_train'],'grid512_train')
    m2048=fit_model(samples['grid2048_train'],'grid2048_train')
    e512,a512=evaluate(samples['grid512_heldout'],m512,'512_train_to_independent_512_heldout')
    e2048,a2048=evaluate(samples['grid2048_heldout'],m2048,'2048_train_to_independent_2048_heldout')
    transfer,atr=evaluate(samples['grid2048_heldout'],m512,'512_train_to_independent_2048_heldout_FROZEN')
    for name,data in samples.items():
        for field,arr in data.items():extra[name+'__'+field]=arr
    for label,arrays in [('eval512',a512),('eval2048',a2048),('frozen512_on2048',atr)]:
        for field,arr in arrays.items():extra[label+'__'+field]=arr
    for label,model in [('model512',m512),('model2048',m2048)]:
        for field in ('b_by_cell','R_by_cell','train_count_by_cell'):extra[label+'__'+field]=np.asarray(model[field])
    with (OUT/'brownian_reference_samples.npz').open('xb') as f:np.savez_compressed(f,**extra)
    config={'created_utc':datetime.now(timezone.utc).isoformat(),'pure_synthetic':True,
            'market_data_files_read':0,'model_prediction_or_test_target_files_read':0,
            'process':'W_0=0, increments iid N(0,1/steps), daily variance=1, drift=0',
            'O':0,'q':'H-L','r':'C','eta':'r/q','xi':'(H+L-r)/q',
            'second_shape_coordinate':'abs(xi)/(1-abs(eta)); choose 0 on the finite-grid boundary abs(eta)=1,xi=0',
            'P':'q^2/(4 log 2)','y':'log(P)','true_log_integrated_variance':0,
            'configs':CONFIGS,'batch_paths':BATCH,'bin_edges_a':EDGES,'bin_edges_z':EDGES,
            'min_train_cell':MIN_CELL,'R_floor':R_FLOOR,
            'fallback':'For training cells below 250 paths, use training global mean(y) and var(y,ddof=1); flag cells and paths',
            'symmetry':'Absolute eta and xi quotient is justified only for this zero-drift Brownian reference by path reflection and time reversal, not asserted for drifting/jumping markets.',
            'mesh_limitation':'H/L are same-path extrema on a finite grid including O/C, not exact continuous Brownian extrema. No separate bridge-extremum formulas used.',
            'rng':'numpy default_rng PCG64, independent explicit seeds per split and resolution',
            'environment':{'python':platform.python_version(),'numpy':np.__version__},
            'scope':'Read no files outside this newly created synthetic-experiment directory; no real quotes, scores or labels; no other models or searches.'}
    resolution={'coarse_steps':512,'fine_steps':2048,'independently_simulated':True,
                'fine_has_smaller_training_sample':True,
                'mean_P_by_train_resolution':{'512':sanity['grid512_train']['P']['mean'],'2048':sanity['grid2048_train']['P']['mean']},
                'global_b_fine_minus_coarse':m2048['global_b']-m512['global_b'],
                'global_R_fine_minus_coarse':m2048['global_R_ddof1']-m512['global_R_ddof1'],
                'precision_delta_fine_minus_coarse':e2048['precision_delta_shape_R_minus_STRONG_constant_R']['mean']-e512['precision_delta_shape_R_minus_STRONG_constant_R']['mean'],
                'frozen_coarse_model_on_fine_heldout':transfer,
                'interpretation_limit':'Independent finite samples and unequal train sizes confound pure mesh effects. Frozen transfer assesses sensitivity without fine-heldout fitting; neither resolution establishes continuous-path accuracy.'}
    results={'created_utc':datetime.now(timezone.utc).isoformat(),'simulation_sanity':sanity,
             'primary_grid512':e512,'resolution_grid2048':e2048,'resolution_sensitivity':resolution,
             'elapsed_seconds':time.time()-started,
             'interpretation':'Prerequisite for shape-dependent precision under a unit-variance zero-drift Brownian reference only; not recovered real IV or successful market forecasting.'}
    write_json('experiment_config.json',config)
    write_json('brownian_shape_models.json',{'grid512':m512,'grid2048':m2048})
    write_json('brownian_reference_results.json',results)
    lines=['# Brownian 形状条件精度：纯合成先决条件实验','',
           '本实验只模拟 unit-variance、零漂移 Brownian 单日路径，不读取真实行情、模型测试标签或评分。潜在日方差固定为 1，所以其对数为 0。结果仅用于备选 B 的先决条件判断。','',
           '## 方法','',
           '以固定种子分批生成同一条路径的正态独立增量，H、L、C 来自该路径，O=0。512 步使用训练 60,000 / 独立留出 20,000 条；2048 步使用另一组独立训练 20,000 / 留出 10,000 条。两种分辨率均为有限网格，不是连续路径精确极值，也没有分别抽取两个 bridge 极值。','',
           '定义 q=H-L，r=C，eta=r/q，xi=(H+L-r)/q，验证 |eta|+|xi|<=1。形状坐标 a=|eta|、z=|xi|/(1-|eta|) 各分 5 档，共 25 格；|eta|=1 的有限网格边界按 z=0 处理并记录次数。反射与时间反转允许在本零漂移 Brownian 参考中使用绝对坐标，不推广到任意真实市场。','',
           'P=q²/(4ln2)，y=logP。在训练单元估计 b=E[y|cell]、R=Var[y|cell]（样本方差 ddof=1）；不足 250 条的单元使用训练全局 b/R 并显式标记。所有阈值与模型只由训练样本确定。','',
           '## 必须保留的三项比较','',
           'A：原 logP + 训练全局 b/global R；B：形状 b 校正 + 常数 R；C：同一份形状 b 校正 + 形状 R。B 的 R 取训练形状校正残差的均方，是该残差在均值为 0 的 Gaussian 测量模型下的最佳训练常数。B 和 C 使用完全相同的 x=y-b(shape)，其差异不混入偏差校正收益。','',
           '留出分数为 logR+x²/R，越小越好，是省略常数与 1/2 的 Gaussian 测量准分数。它不包含任何市场预测。下表的差值均为同一留出路径上的配对差，区间使用配对标准误的正态 95% 区间。','',
           '| 独立留出检验 | A 分数 | B 强控制分数 | C 分数 | C−B：纯精度差值及 95% CI |',
           '|---|---:|---:|---:|---|']
    for name,e in [('512 步匹配校准',e512),('2048 步匹配校准',e2048),('512 校准冻结用于 2048 留出',transfer)]:
        sc=e['scores'];d=e['precision_delta_shape_R_minus_STRONG_constant_R'];ci=d['paired_normal_95_CI']
        lines.append(f"| {name} | {sc['global_bias_global_R']['mean']:.6f} | {sc['shape_bias_constant_R_STRONG_CONTROL']['mean']:.6f} | {sc['shape_bias_shape_R']['mean']:.6f} | {d['mean']:.6f} [{ci[0]:.6f}, {ci[1]:.6f}] |")
    lines+=['','## 校准与误差平方相关','',
            '分组边界由训练预测 R 的五分位数确定；留出样本只用于诊断，不重新拟合 R 或分组阈值。R 单元值有并列时边界去重。完整单元、fallback、分组校准及原 logP 分布保存在 JSON。','',
            '| 匹配分辨率 | 误差²与预测 R 的 Pearson | 平均误差²/平均 R | 高 R 组/低 R 组实际误差² |',
            '|---|---:|---:|---:|']
    for name,e in [('512',e512),('2048',e2048)]:
        lines.append(f"| {name} | {e['error_squared_vs_predicted_R_pearson']:.6f} | {e['overall_error_squared_over_mean_shape_R']:.6f} | {e['high_vs_low_R_group_empirical_MSE_ratio']:.6f} |")
    lines+=['','### 512 步留出分组','',
            '| 训练 R 分组 | 留出数 | 平均预测 R | 平均误差² | 误差²/R |',
            '|---|---:|---:|---:|---:|']
    for g in e512['R_groups_fixed_from_training_prediction_quantiles']:
        lines.append(f"| {g['group']} | {g['n_heldout']} | {g['mean_predicted_R']:.6f} | {g['mean_error_squared']:.6f} | {g['mean_error_squared_over_mean_R']:.6f} |")
    lines+=['','## 网格敏感性与限制','',
            f"训练 E[P]：512 步={sanity['grid512_train']['P']['mean']:.6f}，2048 步={sanity['grid2048_train']['P']['mean']:.6f}；训练全局 b 的细网格减粗网格变化={resolution['global_b_fine_minus_coarse']:.6f}。有限采样遗漏连续极值，对 P 产生网格依赖。细网格训练量更小且独立抽样，变化同时包含蒙特卡洛误差及训练量影响，不能当作纯离散误差估计。冻结 512 校准的跨分辨率留出比较避免在 2048 留出上重拟合。",'',
            '即使 C 优于 B，也只表明这个参考过程及这个粗形状网格下存在可利用的条件测量误差方差差异；不能证明实盘 IV 恢复、真实市场预测或一般分布下成立。Gaussian 分布仅为均值/方差准评分模型。若 C−B 的区间跨 0，应按精度收益证据不足报告；不能用 B−A 的偏差收益替代。','',
            '## 产物与复现','',
            '- `brownian_reference_results.json`：全部留出评分、配对差、校准、相关、几何检查与分辨率敏感性。',
            '- `brownian_shape_models.json`：两个分辨率的训练网格、单元统计、fallback 和常数 R。',
            '- `brownian_reference_samples.npz`：各独立样本的 OHLC/形状/P/y/单元，以及留出误差、R 和逐路径评分。未保存全部路径网格，可用固定种子脚本复现。',
            '- `experiment_config.json`：固定种子、样本量、批量、分箱、时步、环境和无真实数据访问范围。',
            '- `run_brownian_reference.py`：完整脚本；仅在自身目录排他新建输出，不读取任何行情文件。',
            '- `artifact_hashes.json`：产物 SHA-256。','']
    with (OUT/'README_brownian_reference_zh.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines))
    hashes={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ARTIFACTS if name!='artifact_hashes.json'}
    hashes['run_brownian_reference.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_json('artifact_hashes.json',hashes)
    print('COMPLETE seconds',round(time.time()-started,2),flush=True)


if __name__=='__main__':main()
