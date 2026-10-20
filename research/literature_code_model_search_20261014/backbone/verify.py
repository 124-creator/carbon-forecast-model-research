"""Saved-output independent audit and explicitly reused historical baseline."""
import importlib.util,json,hashlib
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('backbone_impl',OUT/'run.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def require(c,s):
    if not c:raise ValueError(s)
def metric(pred,true,years):
    return float(np.mean([np.mean((pred[years==y]-true[years==y])**2,axis=0)/np.mean(true[years==y]**2,axis=0) for y in sorted(set(years))]))
def main():
    gate=json.loads((OUT/'GATE.json').read_text());require(gate['market_allowed'],'control gate')
    dates,p=m.base.read_prefix();initial=int(np.sum(dates<'2020-01-01'))
    z=np.load(OUT/'PREDICTIONS_LOCAL_ONLY.npz',allow_pickle=False);orig=z['origins'];true=z['true_changes'];years=np.array([s[:4] for s in dates[orig]])
    require(len(orig)==451 and np.array_equal(true,p[orig[:,None]+m.base.H]-p[orig,None]),'origins targets')
    histpath=m.ROOT/'research/persistent_residual_search_20261013/PREDICTIONS_LOCAL_ONLY.npz'
    hist=np.load(histpath,allow_pickle=False);require(np.array_equal(hist['origins'],orig) and np.array_equal(hist['true_changes'],true),'historical contract mismatch')
    linear=hist['linear'];comparison={k:metric(z[k],true,years) for k in ['RAW_MLP_seed7','RAW_MLP_seed17','RW','MLE_window480']};comparison['historical_simple_linear_alpha10']=metric(linear,true,years)
    require(abs(comparison['historical_simple_linear_alpha10']-.942705467815)<1e-7,'historical linear')
    logs=json.loads((OUT/'TRAINING_LOGS.json').read_text());maxdiff=0.;perturb=0
    x,loc=m.prepare(p)
    for log in logs:
        if 'seed' not in log:continue
        seed=log['seed'];end=log['fit_end_exclusive'];idx=np.arange(log['min_training_origin'],end-20);require(idx[-1]+20==log['max_training_target']<end,'maturity')
        pos=np.where((orig>=end)&(orig<end+40))[0];theta=z[f'theta_{seed}_{end}'];pr=m.forward(theta,x[orig[pos]],loc[orig[pos]])
        maxdiff=max(maxdiff,float(np.max(abs(pr-z[f'RAW_MLP_seed{seed}'][pos]))))
        val,grad=m.objective(theta,x[idx],loc[idx],p[idx[:,None]+m.base.H]-p[idx,None],z['horizon_scales']);require(abs(val-log['final_objective'])<1e-10,'saved training objective');require(np.isfinite(grad).all() and log['decreased'],'training finite decrease')
        # Future perturbations cannot affect train inputs/labels or current forecast.
        t=int(orig[pos[0]]);changed=p.copy();changed[t+1:]+=1e6;xx,ll=m.prepare(changed)
        require(np.array_equal(xx[:t+1],x[:t+1]) and np.array_equal(ll[:t+1],loc[:t+1]),'future feature leak')
        require(np.array_equal(changed[idx[:,None]+m.base.H]-changed[idx,None],p[idx[:,None]+m.base.H]-p[idx,None]),'future training label leak')
        require(np.array_equal(m.forward(theta,xx[[t]],ll[[t]]),m.forward(theta,x[[t]],loc[[t]])),'future forecast leak');perturb+=1
    require(maxdiff<1e-12,'saved prediction mismatch')
    res=json.loads((OUT/'RESULTS.json').read_text());require(all(abs(metric(z[r['name']],true,years)-r['score'])<1e-12 for r in res['results']),'saved metric mismatch')
    controls=json.loads((OUT/'CONTROLS.json').read_text());cz=np.load(OUT/'CONTROL_LOCAL_ONLY.npz',allow_pickle=False)
    for r in controls:
        stem=f"{r['kind']}_{r['seed']}";require(abs(float(np.mean((cz[stem+'_pred']-cz[stem+'_true'])**2))-r['mse'])<1e-12,'control saved metric')
    mean=float(np.mean([comparison[f'RAW_MLP_seed{s}'] for s in [7,17]]))
    audit=dict(passed=True,origin_count=len(orig),replayed_training_fits=perturb,future_perturbation_checks=perturb,max_prediction_error=maxdiff,comparison=comparison,two_seed_mean_loss_score=mean,historical_linear_reuse=dict(path=str(histpath),sha256=hashlib.sha256(histpath.read_bytes()).hexdigest(),weights_loaded=False,selected_before_this_run=True),no_2022_record_requested=True)
    m.dump('VERIFICATION_OPTIMIZED.json' if not __debug__ else 'VERIFICATION.json',audit)
    lines=['# 可学习非线性骨干：真实能力基线','', '普通60→24 tanh→3全参数可训练MLP，NumPy手写反向传播、SciPy L-BFGS-B；不是RFF，不是首创。固定种子7/17，单一窗口480，单一正则.01，每40点刷新，1/5/20观察点，广州950行许可前缀与451个共同开发起点，没有读取2022边界。输入为60历史增量及因果局部RMS尺度，初始训练期确定步长损失尺度，无外生变量、无分解。','', '|方法|相对RW MSE|','|---|---|']
    lines += [f'|{k}|{v:.9f}|' for k,v in comparison.items()];lines += [f'|固定两种子等权平均损失（不是预测集成）|{mean:.9f}|','','## 控制与训练证据']
    for r in controls:lines.append(f"- {r['kind']}，种子{r['seed']}：验证MSE/RW={r['ratio_rw']:.6f}，MSE/Ridge={r['ratio_ridge']:.6f}；训练目标{r['training']['initial_objective']:.6f}→{r['training']['final_objective']:.6f}。")
    grad=json.loads((OUT/'GRADIENT_CHECK.json').read_text());lines.append(f"- 有限差分45项（坐标和方向），最大相对误差{grad['max_relative_error']:.3g}；两种子非线性正控制通过后才读取市场。")
    train=[r for r in logs if 'seed' in r];lines.append(f"- 市场{len(train)}次训练全部目标下降，求解器未声明成功{sum(not r['solver_success'] for r in train)}次；最大200迭代、240函数调用，没有按市场验证挑选迭代/种子。实际执行{res['seconds']:.3f}秒，预算900秒。参数共{res['parameter_count']}个，训练支持起点数{min(r['n_training'] for r in train)}—{max(r['n_training'] for r in train)}。")
    lines += ['','## 结果解释与限制','网络确实能够学习非线性映射，正控制通过不能推出价格可预测。市场两种子均输给RW/MLE/历史线性修正；无信号控制也明显高于RW，说明这个预算、容量和正则下存在样本外拟合噪声风险。训练下降不等于泛化改善，不能宣称优化完全解决，更不能把市场失败推断为一切非线性结构无效。停止当前固定能力基线，不追加窗口/正则/优秀种子搜索。','MLE窗口480有比共同成熟多步MLP更近期的一步创新；历史简单线性修正具有各步长独立成熟支持和历史发行MLE状态，信息支持并不完全一致。历史线性基线按此前开发选择复用预测并独立重评分，绝不是本次新获得的未见验证。初始误认trend修正保留原结果，见BASELINE_CORRECTION_ZH.md。','2020—2021已被多次使用，只是开发评分，不是独立实证确认。本结果是后续新模型创新所需学习底座，不是新模型创新本身。控制是独立高斯设计的静态非线性能力测试，并非满足市场动态的完整生成过程。','普通及-O读取保存参数重放预测、目标/评分校验和逐块未来扰动检验通过；所有NPZ与训练源码本地隔离，未提交或推送。首次执行仅因numpy布尔值JSON序列化失败，修正为Python bool后重新运行，未改变协议/模型。']
    (OUT/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
