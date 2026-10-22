"""Pure synthetic diagnostic of observation density in a unit Brownian path.

No market or forecasting results are read. All M use the SAME underlying512
path within each sample. Training and heldout streams are independent. The
frozen512 table is queried only on heldout; optional M-specific tables are
measurement calibration summaries, not future forecasting models.
"""
from pathlib import Path
import hashlib
import json
import math
import time
import numpy as np

HERE=Path(__file__).resolve().parent
REFERENCE=HERE/'brownian_reference_round16'/'brownian_shape_models.json'
MS=(1,2,4,8,16,64,512)
FINE=512
BATCH=1024
TRAIN_N,HELD_N=30000,10000
TRAIN_SEED,HELD_SEED=2026102219,2026102220
MIN_CELL=250
R_FLOOR=1e-8


def summary(v):
    v=np.asarray(v,dtype=float)
    return dict(n=len(v),mean=float(v.mean()),variance_ddof1=float(v.var(ddof=1)),
        mean_se=float(v.std(ddof=1)/np.sqrt(len(v))),
        quantiles_01_10_50_90_99=np.quantile(v,[.01,.1,.5,.9,.99]).tolist())


def cells(a,z,model):
    ea,ez=np.asarray(model['edges_a']),np.asarray(model['edges_z'])
    ia=np.clip(np.searchsorted(ea,np.clip(a,0,1),side='right')-1,0,len(ea)-2)
    iz=np.clip(np.searchsorted(ez,np.clip(z,0,1),side='right')-1,0,len(ez)-2)
    return ia*(len(ez)-1)+iz


def simulate(n,seed,reference):
    rng=np.random.default_rng(seed)
    bounds={M:{key:np.empty(n) for key in ('H','L','C')} for M in MS}
    for lo in range(0,n,BATCH):
        hi=min(n,lo+BATCH)
        path=rng.standard_normal((hi-lo,FINE))/np.sqrt(FINE)
        np.cumsum(path,axis=1,out=path)
        for M in MS:
            stride=FINE//M
            observed=path[:,stride-1::stride]
            assert observed.shape[1]==M
            bounds[M]['H'][lo:hi]=np.maximum(0,observed.max(axis=1))
            bounds[M]['L'][lo:hi]=np.minimum(0,observed.min(axis=1))
            bounds[M]['C'][lo:hi]=observed[:,-1]
    output={}
    for M in MS:
        H,L,C=(bounds[M][key] for key in ('H','L','C'))
        q=H-L
        assert (q>0).all()
        eta=C/q;xi=(H+L-C)/q;a=np.abs(eta)
        assert (a+np.abs(xi)<=1+1e-12).all()
        upper=H-np.maximum(0,C);lower=np.minimum(0,C)-L
        assert (upper>=0).all() and (lower>=0).all()
        upper_zero=upper==0;lower_zero=lower==0
        atom=upper_zero & lower_zero
        edge=upper_zero ^ lower_zero
        interior=~(atom|edge)
        denom=1-a;boundary=denom<=1e-12
        z_query=np.divide(np.abs(xi),denom,out=np.zeros(n),where=~boundary)
        # Exactly the frozen512 binning convention, including boundary z=0.
        cell=cells(a,z_query,reference)
        # Stable algebraically equal coordinate for distribution summaries.
        # z=|upper-lower|/(upper+lower); double-zero convention z=0.
        z=np.divide(np.abs(upper-lower),upper+lower,out=np.zeros(n),where=~atom)
        assert (z>=0).all() and (z<=1).all()
        P=q*q/(4*math.log(2));y=np.log(P)
        assert np.isfinite(y).all()
        phase=np.where(atom,0,np.where(edge,1,2))
        output[M]=dict(P=P,y=y,a=a,z=z,z_query=z_query,cell=cell,phase=phase,C=C,
            phase_counts=dict(atom_both_wicks_zero=int(atom.sum()),edge_exactly_one_wick_zero=int(edge.sum()),
                              interior_both_wicks_positive=int(interior.sum())),
            numerical=dict(reference_boundary_count=int(boundary.sum()),
                reference_z_outside_unit_roundoff_count=int(((z_query<0)|(z_query>1)).sum()),
                reference_vs_stable_z_max_abs_difference=float(np.abs(z_query-z).max()),
                reference_vs_stable_cell_mismatches=int((cell!=cells(a,z,reference)).sum())))
        assert sum(output[M]['phase_counts'].values())==n
        np.testing.assert_array_equal(C,bounds[FINE]['C'])
    for left,right in zip(MS[:-1],MS[1:]):
        assert (output[left]['P']<=output[right]['P']).all()
    np.testing.assert_array_equal(output[1]['P'],output[1]['C']**2/(4*math.log(2)))
    assert output[1]['phase_counts']['atom_both_wicks_zero']==n
    return output


def analytic_M1(data):
    n=len(data['P']);gamma=.577215664901532860606512090082402431
    psi_half=-gamma-2*math.log(2)
    expected_P=1/(4*math.log(2))
    b=psi_half+math.log(2)-math.log(4*math.log(2))
    R=math.pi**2/2
    se_P=np.sqrt(2*expected_P**2/n)
    se_b=np.sqrt(R/n)
    # log(chi-square_1) fourth cumulant psi3(1/2)=pi^4.
    se_R=np.sqrt(math.pi**4/n+2*R**2/(n-1))
    observed=dict(E_P=float(data['P'].mean()),E_logP=float(data['y'].mean()),Var_logP=float(data['y'].var(ddof=1)))
    theory=dict(E_P=expected_P,E_logP=b,Var_logP=R)
    se=dict(E_P=float(se_P),E_logP=float(se_b),Var_logP=float(se_R))
    z={key:(observed[key]-theory[key])/se[key] for key in theory}
    assert all(abs(value)<6 for value in z.values()),f'M1 analytic check outside6 MC SE: {z}'
    return dict(theory=theory,observed=observed,analytic_monte_carlo_se=se,z_scores=z,
        passed_within6_standard_errors=True,
        formula='E[P]=1/(4log2); ElogP=digamma(1/2)+log2-log(4log2); VarlogP=pi^2/2',
        exact_sample_identity_P_equals_C_squared_over_4log2=True)


def fit_measurement_table(train):
    y,cell=train['y'],train['cell']
    global_b=float(y.mean());global_R=float(y.var(ddof=1))
    b=np.full(25,global_b);R=np.full(25,global_R)
    audit=[]
    for k in range(25):
        v=y[cell==k];supported=len(v)>=MIN_CELL
        if supported:
            b[k]=v.mean();R[k]=max(float(v.var(ddof=1)),R_FLOOR)
        audit.append(dict(cell=k,a_bin=k//5,z_bin=k%5,train_n=len(v),supported=supported,
            global_fallback=not supported,raw_cell_b=float(v.mean()) if len(v) else None,
            raw_cell_R=float(v.var(ddof=1)) if len(v)>1 else None,b_used=float(b[k]),R_used=float(R[k])))
    residual=y-b[cell]
    return dict(global_b=global_b,global_R=global_R,b_by_cell=b.tolist(),R_by_cell=R.tolist(),
        shape_bias_constant_R=float(np.mean(residual**2)),
        fallback_cells=sum(row['global_fallback'] for row in audit),
        empty_cells=sum(row['train_n']==0 for row in audit),min_train_cell=MIN_CELL,R_floor=R_FLOOR,cell_audit=audit)


def measurement_report(data,b,R):
    residual=data['y']-b
    score=np.log(R)+residual**2/R
    return dict(residual=summary(residual),mean_residual_squared=float(np.mean(residual**2)),
        average_predicted_R=float(np.mean(np.broadcast_to(R,residual.shape))),measurement_score=summary(score)),residual,score


def data_report(data):
    return dict(P=summary(data['P']),logP=summary(data['y']),a=summary(data['a']),z=summary(data['z']),
        a_hist5=np.histogram(data['a'],bins=np.linspace(0,1,6))[0].tolist(),
        z_hist5=np.histogram(data['z'],bins=np.linspace(0,1,6))[0].tolist(),
        cell_counts=np.bincount(data['cell'],minlength=25).tolist(),
        phase_counts=data['phase_counts'],numerical=data['numerical'])


def run(reference):
    train=simulate(TRAIN_N,TRAIN_SEED,reference)
    held=simulate(HELD_N,HELD_SEED,reference)
    checks=dict(train_M1=analytic_M1(train[1]),held_M1=analytic_M1(held[1]),
                same_fine_path_per_M=True,pathwise_P_monotone_in_M=True,endpoint_identical_across_M=True,
                geometry_legal=True,train_held_seeds_independent=True)
    frozen_b=np.asarray(reference['b_by_cell']);frozen_R=np.asarray(reference['R_by_cell'])
    _,base512,_=measurement_report(held[512],frozen_b[held[512]['cell']],frozen_R[held[512]['cell']])
    reports={}
    for M in MS:
        T,H=train[M],held[M];cell=H['cell']
        calibration=fit_measurement_table(T)
        frozen,residual,score512=measurement_report(H,frozen_b[cell],frozen_R[cell])
        b=np.asarray(calibration['b_by_cell'])[cell];R=np.asarray(calibration['R_by_cell'])[cell]
        matched,_,scorematched=measurement_report(H,b,R)
        constant,_,scoreconstant=measurement_report(H,b,np.full(HELD_N,calibration['shape_bias_constant_R']))
        fallback=np.array([row['global_fallback'] for row in calibration['cell_audit']])[cell]
        phase={}
        for code,name in enumerate(('atom','edge','interior')):
            mask=H['phase']==code
            phase[name]=dict(n=int(mask.sum()),frozen512_residual=summary(residual[mask]) if mask.sum()>1 else None,
                implied_additive_log_correction=-float(residual[mask].mean()) if mask.any() else None)
        reports[str(M)]=dict(train=data_report(T),held=data_report(H),frozen512_held=frozen,
            residual_difference_vs_same_path512=summary(residual-base512),phase_held_frozen512=phase,
            M_specific_train_calibration=calibration,M_specific_held=matched,
            M_specific_shape_bias_constant_R_held=constant,
            held_measurement_score_difference_M_specific_minus_frozen512=summary(scorematched-score512),
            held_measurement_score_difference_shapeR_minus_constantR=summary(scorematched-scoreconstant),
            held_fallback_n=int(fallback.sum()),held_fallback_fraction=float(fallback.mean()))
    return dict(checks=checks,levels=reports,
        train_endpoint=summary(train[512]['C']),held_endpoint=summary(held[512]['C']))


def chinese_note(result,json_path):
    lines=['# 观测密度的纯合成参考诊断','',
        '单位零漂移 Brownian 过程先生成512步细路径，再在同一路径上等距保留M个增量的观测点，均包含开端与终端。训练30,000条，独立留出10,000条；batch=1024，seed分别为2026102219和2026102220。各M共用终点，范围估计随观测网格加密逐路径不减。','',
        '| M | 留出E[P] | 留出ElogP | 留出VarlogP | 冻结512校正后均值 | 双wick零 | 单wick零 | 内部 |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for M in MS:
        row=result['levels'][str(M)];h=row['held'];p=h['phase_counts']
        lines.append(f"| {M} | {h['P']['mean']:.6f} | {h['logP']['mean']:.6f} | {h['logP']['variance_ddof1']:.6f} | {row['frozen512_held']['residual']['mean']:.6f} | {p['atom_both_wicks_zero']} | {p['edge_exactly_one_wick_zero']} | {p['interior_both_wicks_positive']} |")
    lines+=['',
        'M=1时P=C²/(4ln2)，C服从标准正态，故E[P]=1/(4ln2)，ElogP=digamma(1/2)+ln2−ln(4ln2)，VarlogP=π²/2。训练与留出均通过解析Monte Carlo误差核对；这是抽样与量纲的单元检查。','',
        '表中“冻结512校正后均值”为留出logP−b512(shape)；负值表示该固定参考校正在单位潜在方差下仍低估log-risk，所需观测均值修正方向为正。该方向与稀疏采样导致遗漏极值的已知机制一致，不能据此认定真实ETF的修正已经由采样密度解释。','',
        '另对每个M仅用训练样本估计同5×5条件b/R；每cell至少250条，不足时明确回退该M的全局训练b/R，空cell不宣称有条件支持。JSON保存所有cell支持数、回退及独立留出的measurement score。这里比较当前measurement的logR+residual²/R，不涉及未来风险预测损失。','',
        'z在双wick零处取0只是约定；其余分布用稳定等价式|upperwick−lowerwick|/(upperwick+lowerwick)。冻结表查询严格沿用旧参考|xi|/(1−|eta|)及其边界/索引裁剪约定，数值差异另有审计。','',
        '成交量不是已知的真实交易次数，等距M不是真实市场成交过程。本实验没有识别ETF的N，也没有验证真实市场的Brownian、零漂移、无微观噪声假设；不把已知有限网格偏差作为新理论。没有读取行情或既有预测结果，没有拟合neural模型、生成未来预测或改变主候选。','',
        f'[完整JSON](<{json_path.resolve().as_posix()}>) · [源代码](<{Path(__file__).resolve().as_posix()}>)','']
    return '\n'.join(lines)


def main():
    json_path=HERE/'sampling_density_reference_diagnostic.json'
    note_path=HERE/'sampling_density_reference_diagnostic_zh.md'
    for path in (json_path,note_path):
        if path.exists():raise FileExistsError(f'Protected existing diagnostic: {path}')
    reference_sha=hashlib.sha256(REFERENCE.read_bytes()).hexdigest()
    reference=json.loads(REFERENCE.read_text(encoding='utf-8'))['grid512']
    started=time.perf_counter();result=run(reference)
    assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest()==reference_sha
    result.update(status='PURE SYNTHETIC MEASUREMENT DENSITY DIAGNOSTIC',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        frozen_reference=dict(path=str(REFERENCE),sha256=reference_sha,unchanged=True),
        config=dict(M=list(MS),fine_increments=FINE,train_n=TRAIN_N,held_n=HELD_N,batch=BATCH,
                    train_seed=TRAIN_SEED,held_seed=HELD_SEED,min_train_cell=MIN_CELL),
        measurement_score='log R+(logP-b)^2/R; omitted Gaussian constants; smaller is better',
        market_data_read=False,existing_forecasting_results_read=False,real2026_data_read=False,
        neural_model_fitted=False,future_prediction_loss_computed=False,
        interpretation='Known finite-grid sampling bias illustration; no market N identification or volume-to-trade-count assumption',
        elapsed_seconds=time.perf_counter()-started)
    json_path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    note_path.write_text(chinese_note(result,json_path),encoding='utf-8')
    for M in MS:
        row=result['levels'][str(M)]
        print(M,'held_E_P',round(row['held']['P']['mean'],6),'held_logP',round(row['held']['logP']['mean'],6),
              'held_R',round(row['held']['logP']['variance_ddof1'],6),
              'frozen512_resid',round(row['frozen512_held']['residual']['mean'],6),
              row['held']['phase_counts'],flush=True)
    print(json.dumps(dict(source_sha256=result['source_sha256'],json=str(json_path),note=str(note_path),
        M1_train_checks=result['checks']['train_M1'],M1_held_checks=result['checks']['held_M1'],
        elapsed_seconds=result['elapsed_seconds']),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
