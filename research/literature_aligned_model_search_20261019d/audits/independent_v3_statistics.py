"""Reproduce the frozen V3 audit, without fitting or importing experiment models."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import independent_v1_statistics as core

STUDY=core.STUDY
PRIMARY=["BEIJING_MEAN","TIANJIN_MEAN"]
MARKETS=PRIMARY+["CHONGQING_MEAN"]
FOLD="fixed602020"
NEURAL=["liquidity_full","liquidity_readout","shock_size_only","liquidity_unconstrained",
        "allocation_constant","allocation_readout","gru","lstm"]
CLASSICAL=["rw","drift","ridge_price","ridge_full","hgb_full"]
FAMILIES=NEURAL+CLASSICAL
CONTRASTS=[
    ("primary_full_vs_rw","liquidity_full","rw"),
    ("primary_full_vs_shock_size","liquidity_full","shock_size_only"),
    ("primary_full_vs_constant","liquidity_full","allocation_constant"),
    ("primary_full_vs_liquidity_readout","liquidity_full","liquidity_readout"),
    ("secondary_readout_vs_rw","allocation_readout","rw"),
    ("secondary_readout_vs_constant","allocation_readout","allocation_constant"),
]
COUNTS={"BEIJING_MEAN":(673,319,114,116,453),
        "TIANJIN_MEAN":(457,190,71,72,281),
        "CHONGQING_MEAN":(240,60,28,28,108)}
core.PARAMETERS={"liquidity_full":11,"liquidity_readout":11,"shock_size_only":9,
                 "liquidity_unconstrained":11,"allocation_constant":7,
                 "allocation_readout":169,"gru":4500,"lstm":5780}
B=50000
PLAN_SHA="c025efb1b6ab2bd4493868d6b48830b7b23f4049d8f74766797989b0d2453aad"


def reconstruct(manifest,data_root):
    market=manifest["market"]
    src=manifest["sources"][0]
    path=data_root/Path(src["path"]).name if data_root else Path(src["path"])
    core.check(core.sha(path)==src["sha256"],"V3 target source hash")
    df=pd.read_csv(path)
    core.check(df.date.is_unique and df.date.is_monotonic_increasing,"target date order")
    dates=pd.to_datetime(df.date).to_numpy(dtype="datetime64[D]")
    p=df.price.to_numpy(float);vol=df.volume_tons.to_numpy(float)
    core.check(np.isfinite(p).all() and (p>0).all(),"valid price")
    core.check(np.isfinite(vol).all() and (vol>0).all(),"valid volume")
    # CSV-to-float parsing can shift the workbook's exact identity by a few ulps.
    quantity_gap=float(np.max(np.abs(p-df.turnover_yuan.to_numpy(float)/vol)))
    core.close(p,df.turnover_yuan.to_numpy(float)/vol,"mean-price identity",rtol=1e-12,atol=1e-12)
    n=len(p);A=n*3//5;Z=A+n//5
    core.check(n==COUNTS[market][0],"source row count")
    core.check(manifest["row_boundaries"]==[0,A,Z,n],"fixed sequential60/20/20")
    tokens=[np.r_[0.,np.diff(p)]]
    for i,source in enumerate(manifest["sources"][1:]):
        ep=core.REPO/source["path"]
        core.check(core.sha(ep)==source["sha256"],"EIA source hash")
        ef=pd.read_csv(ep).sort_values("date")
        missing=int(ef.spot_price.isna().sum())
        ef=ef.dropna(subset=["spot_price"])
        ed=pd.to_datetime(ef.date).to_numpy(dtype="datetime64[D]")
        values=ef.spot_price.to_numpy(float)
        ix=np.searchsorted(ed,dates-np.timedelta64(14,"D"),side="right")-1
        core.check((ix>=0).all() and np.isfinite(values[ix]).all(),"EIA availability")
        age=(dates-ed[ix]).astype(int)
        core.check((age>=14).all(),"EIA14day cutoff")
        a=manifest["energy_alignment"][i]
        core.check((int(age.min()),int(age.max()),missing)==
                   (a["minimum_age_days"],a["maximum_age_days"],
                    a["source_missing_price_rows_excluded_before_backward_asof"]),"EIA alignment manifest")
        tokens.append(np.r_[0.,np.diff(values[ix])])
    tokens=np.column_stack(tokens)
    logs=np.log1p(vol)
    context=np.zeros((n,2),dtype="float32")
    for t in range(10,n):
        past=logs[max(0,t-60):t]
        context[t,0]=np.clip((logs[t]-np.median(past))/max(float(np.std(past)),.25),-5,5)
        context[t,1]=1
    core.check(manifest["volume"]["valid_rows"]==n and manifest["volume"]["usable_context_rows"]==n-10,
               "volume context counts")
    origins=np.arange(64,n-20)
    masks={"train":origins+20<A,"val":(origins>=A)&(origins+20<Z),
           "final":origins+20<Z,"test":origins>=Z}
    check_counts={k:int(v.sum()) for k,v in masks.items()}
    core.check(check_counts==manifest["counts"],"fold counts")
    _,tr,va,te,fi=COUNTS[market]
    core.check(check_counts=={"train":tr,"val":va,"final":fi,"test":te},"planned counts")
    for k,v in masks.items():
        core.check(manifest["origin_ranges"][k]==[int(origins[v][0]),int(origins[v][-1])],"origin range")
    target=np.column_stack([p[origins+h]-p[origins] for h in core.HORIZONS])
    rms=np.maximum(np.sqrt((tokens[1:A]**2).mean(axis=0)),1e-7)
    variance=np.maximum((target[masks["train"]]**2).mean(axis=0),rms[0]**2*1e-6)
    core.close(rms,manifest["training_rms"],"V3 RMS")
    core.close(variance,manifest["train_target_second_moment"],"V3 weighted loss")
    times={"first_train_origin":dates[origins[masks["train"]][0]],
           "last_train_label":dates[origins[masks["train"]][-1]+20],
           "first_validation_origin":dates[origins[masks["val"]][0]],
           "last_validation_label":dates[origins[masks["val"]][-1]+20],
           "first_test_origin":dates[origins[masks["test"]][0]],
           "last_test_label":dates[origins[masks["test"]][-1]+20]}
    for k,v in times.items():core.check(str(v)==manifest[k],"V3 dates")
    core.check(origins[masks["train"]][-1]+20<origins[masks["val"]][0],"train/val maturity")
    core.check(origins[masks["val"]][-1]+20<origins[masks["test"]][0],"val/test maturity")
    rw=float(np.mean(target[masks["val"]]**2/variance))
    drift=[]
    for window in [20,60,120,None]:
        pred=np.array([tokens[max(1,t-window+1) if window else 1:t+1,0].mean()*core.HORIZONS
                       for t in origins[masks["val"]]])
        drift.append((float(np.mean((pred-target[masks["val"]])**2/variance)),str(window)))
    return {"observed":target[masks["test"]],"origins":origins[masks["test"]],
            "dates":dates[origins[masks["test"]]],"validation_rw":rw,"drift_choice":min(drift),
            "counts":check_counts,"source_path_resolved":str(path),"source_sha256":src["sha256"],
            "context_reconstruction_sha256":core.hashlib.sha256(context.tobytes()).hexdigest(),
            "pandas_mean_price_identity_max_absolute_gap":quantity_gap}


def block_intervals(cell,length,rng):
    n=len(cell["observed"])
    core.check(length<n,"block length must be smaller than sample")
    # Utility creates circular block frequency counts; the grid here is observed origins, not dates.
    w=core.calendar_weights(rng,B,n,length)
    den=w@(cell["observed"]**2)
    valid=(den>0).all(axis=1)
    invalid=int((~valid).sum())
    results={}
    for name,candidate,comparator in CONTRASTS:
        numerator=w@(cell["loss"][comparator]-cell["loss"][candidate])
        if invalid:
            # Ratio statistic is undefined for these empirical resamples.
            # Do not omit those draws and silently present a conditional CI.
            ci95=ci975=ci_joint=None
        else:
            draw=(numerator/den).mean(axis=1)
            ci95=np.quantile(draw,[.025,.975]).tolist()
            ci975=np.quantile(draw,[.0125,.9875]).tolist() if cell["market"] in PRIMARY else None
            ci_joint=np.quantile(draw,[.0020833333333333333,.9979166666666667]).tolist() if cell["market"] in PRIMARY else None
        results[name]={"ci95_marginal":ci95,"ci97_5_for_macro_marginal":ci975,
                       "ci99_583333_for_joint_six_family":ci_joint,
                       "undefined_ratio_resamples":invalid,"B":B,"block_observed_origins":length,
                       "coverage_status":"approximate" if invalid==0 else "CI unavailable: some bootstrap RW denominators zero"}
    return results


def combine_bounds(intervals):
    if any(x is None for x in intervals):return None
    return np.mean(np.array(intervals),axis=0).tolist()


def self_test():
    # Different fixed effects at the two markets must average equally despite different lengths.
    effects=[]
    rng=np.random.default_rng(20261019)
    for n,difference in [(116,.1),(72,.3)]:
        y=np.column_stack([np.ones(n),2*np.ones(n),3*np.ones(n)])
        weights=core.calendar_weights(rng,100,n,20)
        effect=((weights@(difference*y*y))/(weights@(y*y))).mean(axis=1)
        core.close(effect,difference,"known paired block effect")
        effects.append(np.quantile(effect,[.0125,.9875]).tolist())
    core.close(combine_bounds(effects),[.2,.2],"equal-market interval endpoint combination")
    return {"status":"PASS","checks":["paired observed-origin blocks","equal-market endpoint combination despite unequal sample sizes"]}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",type=Path,default=Path("/tmp/carbon-literature-aligned-runs-20261019d/new_market_v3"))
    parser.add_argument("--data-root",type=Path,default=STUDY/"data/sciencedb_819_v1/prepared")
    parser.add_argument("--out",type=Path,default=STUDY/"audits")
    parser.add_argument("--plan",type=Path,default=STUDY/"audits/V3_STATISTICAL_AUDIT_PLAN.json")
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    synthetic=self_test()
    if args.self_test:print(json.dumps(synthetic));return
    protocol=core.read_json(args.run/"PROTOCOL_BEFORE_RUN.json")
    complete=core.read_json(args.run/"EXECUTION_COMPLETE.json")
    core.check(complete["status"]=="completed" and complete["source_hashes_unchanged"] and complete["statistics_plan_hash_unchanged"],
               "V3 complete and frozen identities")
    core.check(complete["jobs"]==27 and complete["summary_rows"]==39,"V3 task counts")
    core.check(core.sha(args.plan)==PLAN_SHA==protocol["statistics_plan_sha256"],"pre-result statistical plan")
    core.check(protocol["neural_families"]==NEURAL and protocol["markets"]==MARKETS,"frozen families/population")
    for name,digest in protocol["source_sha256"].items():
        core.check(core.sha(STUDY/"source"/name)==digest,"V3 frozen source")
    stored=core.read_json(args.run/"ALL_SUMMARIES.json")
    sums={(r["market"],r["year"],r["family"]):r for r in stored}
    core.check(len(sums)==39,"unique V3 summary records")
    cells=[];ledger=[];identity=[];epochs=[];selection=[];native=[]
    for market in MARKETS:
        cd=args.run/f"{market}_{FOLD}"
        manifest=core.read_json(cd/"INPUT_MANIFEST.json")
        core.check(manifest["fold"]==FOLD and manifest["legacy_year_field_is_fold_id"],"fold is not calendar year")
        d=reconstruct(manifest,args.data_root)
        validations={}
        for family in FAMILIES:
            fd=cd/family
            if family in NEURAL:
                score,epoch=core.audit_receipts(fd,family,d,ledger)
                validations[family]=score
                epochs.append({"market":market,"fold":FOLD,"family":family,"selected_epoch":epoch})
            else:
                r=core.read_json(fd/"SELECTION.json");validations[family]=float(r["validation_loss"])
                if family=="rw":core.close(r["validation_loss"],d["validation_rw"],"V3 RW validation")
                elif family=="drift":
                    core.close(r["validation_loss"],d["drift_choice"][0],"V3 drift validation")
                    core.check(r["window"]==d["drift_choice"][1],"V3 drift window")
                elif family.startswith("ridge"):
                    best=min(r["trials"]);core.close(r["validation_loss"],best[0],"V3 ridge selection")
                    core.check(r["alpha"]==best[1],"V3 ridge alpha")
                else:
                    best=min(r["trials"]);core.close(r["validation_loss"],best[0],"V3 HGB selection")
                    core.check([r["max_leaf_nodes"],r["l2_regularization"]]==best[1:],"V3 HGB configuration")
        strong=min(core.POOL,key=lambda f:(validations[f],core.POOL.index(f)))
        selection.append({"market":market,"family":strong,"validation_loss":validations[strong],
                          "pool_validation_losses":{f:validations[f] for f in core.POOL}})
        c={"market":market,**d,"metrics":{},"loss":{},"strong":strong}
        for family in FAMILIES:
            path=cd/family/"predictions.npz"
            with np.load(path,allow_pickle=False) as z:
                core.check(np.array_equal(z["origins"],d["origins"]),"V3 NPZ origins")
                core.check(np.array_equal(z["observed"],d["observed"]),"V3 NPZ targets")
                core.check(np.array_equal(z["horizons"],core.HORIZONS),"V3 NPZ horizons")
                seeds=core.SEEDS if family in NEURAL else [-1]
                core.check(np.array_equal(z["seeds"],seeds),"V3 NPZ seeds")
                pred=np.array(z["predictions"],dtype=float)
            core.check(pred.shape==(len(seeds),len(d["observed"]),3),"V3 NPZ shape")
            metric,loss=core.score(pred,d["observed"])
            if family=="rw":core.check(np.all(pred==0),"V3 RW prediction zero")
            if family in ["gru","lstm"]:
                ep=next(r["selected_epoch"] for r in epochs if r["market"]==market and r["family"]==family)
                if ep==0:core.check(np.all(pred==0),"V3 recurrent epoch0 exact RW")
            for saved in [core.read_json(path.parent/"SUMMARY.json"),sums[(market,FOLD,family)]]:
                core.check(saved["n_origins"]==len(d["observed"]),"V3 n_origins")
                for key in ["mean_seed_mse_over_rw","seed_mse_over_rw_by_horizon","rw_mse_by_horizon",
                            "ensemble_mse_over_rw","ensemble_mse_over_rw_by_horizon"]:
                    core.close(metric[key],saved[key],"V3 summary reconciliation")
            c["metrics"][family]=metric;c["loss"][family]=loss
            identity.append({"market":market,"fold":FOLD,"family":family,"path":str(path),
                             "npz_sha256":core.sha(path),"n_origins":len(d["observed"])})
            for j,h in enumerate(core.HORIZONS):
                native.append({"market":market,"fold":FOLD,"family":family,"horizon":int(h),
                               "role":"primary" if market in PRIMARY else "stress","n_origins":len(d["observed"]),
                               "mean_seed_mse_over_rw":float(np.mean(np.array(metric["seed_mse_over_rw_by_horizon"])[:,j])),
                               "mean_seed_mse":metric["native_mean_seed_mse_by_horizon"][j],
                               "mean_seed_rmse":metric["native_mean_seed_rmse_by_horizon"][j],
                               "rmse_of_mean_seed_mse":metric["native_rmse_of_mean_seed_mse_by_horizon"][j],
                               "mean_seed_mae":metric["native_mean_seed_mae_by_horizon"][j],
                               "ensemble_mse":metric["native_ensemble_mse_by_horizon"][j]})
        c["metrics"]["validation_selected_strong"]=c["metrics"][strong]
        c["loss"]["validation_selected_strong"]=c["loss"][strong]
        cells.append(c)
    core.check(len(ledger)==120 and len(identity)==39,"V3 inspected outputs/receipts")
    rng=np.random.default_rng(20261019)
    intervals={}
    for c in cells:
        intervals[c["market"]]={"20":block_intervals(c,20,rng)}
        if c["market"] in PRIMARY:intervals[c["market"]]["40"]=block_intervals(c,40,rng)
    macro={f:{"primary_mean_seed_mse_over_rw":float(np.mean([c["metrics"][f]["mean_seed_mse_over_rw"] for c in cells if c["market"] in PRIMARY])),
              "secondary_ensemble_mse_over_rw":float(np.mean([c["metrics"][f]["ensemble_mse_over_rw"] for c in cells if c["market"] in PRIMARY]))}
           for f in FAMILIES+["validation_selected_strong"]}
    effects={}
    for name,candidate,comparator in CONTRASTS:
        per={c["market"]:c["metrics"][comparator]["mean_seed_mse_over_rw"]-c["metrics"][candidate]["mean_seed_mse_over_rw"] for c in cells}
        cm=macro[comparator]["primary_mean_seed_mse_over_rw"];pm=macro[candidate]["primary_mean_seed_mse_over_rw"]
        effects[name]={"candidate":candidate,"comparator":comparator,"macro_improvement_rw_units":cm-pm,
                       "macro_relative_reduction":1-pm/cm,"per_market_improvement_rw_units":per,
                       "primary_market_wins_of2":sum(per[m]>0 for m in PRIMARY)}
        for length in ["20","40"]:
            effects[name]["block"+length]={
                "macro_ci95_without_market_independence":combine_bounds([
                    intervals[m][length][name]["ci97_5_for_macro_marginal"] for m in PRIMARY]),
                "macro_simultaneous95_joint_six_family":combine_bounds([
                    intervals[m][length][name]["ci99_583333_for_joint_six_family"] for m in PRIMARY])}
    refs=[r for r in ledger if r["mode"]=="final"]
    counts={"neural_tasks":120,"selection_tasks":48,"refit_tasks":72,
            "tasks_with_positive_gradient_steps":sum(r["gradient_positive_steps"]>0 for r in ledger),
            "refits_with_positive_gradient_steps":sum(r["gradient_positive_steps"]>0 for r in refs),
            "zero_epoch_refits":sum(r["epochs_executed"]==0 for r in refs),
            "selected_epoch_zero_family_markets":sum(r["selected_epoch"]==0 for r in epochs),
            "total_executed_epochs":sum(r["epochs_executed"] for r in ledger),
            "total_logged_positive_gradient_steps":sum(r["gradient_positive_steps"] for r in ledger),
            "prediction_files":39,"primary_market_origins":188,"stress_market_origins":28,
            "primary_origin_horizon_targets":564,"stress_origin_horizon_targets":84}
    result={"title":"V3 三市场固定复制独立统计审查","status":"PASS_INTEGRITY",
            "created_utc":datetime.now(timezone.utc).isoformat(),
            "stage":protocol["stage"],"frozen_plan_sha256":PLAN_SHA,"script_sha256":core.sha(__file__),
            "utility_script_sha256":core.sha(core.__file__),"source_sha256":protocol["source_sha256"],
            "resolved_run":str(args.run),"resolved_data_root":str(args.data_root),"resolved_out":str(args.out),
            "self_test":synthetic,"execution_counts":counts,"primary_macro_scores_bj_tj":macro,
            "six_prespecified_effects":effects,"marketwise_bootstrap":intervals,
            "cell_metrics":[{"market":c["market"],"counts":c["counts"],"dates":[str(c["dates"][0]),str(c["dates"][-1])],
                            "source_path_resolved":c["source_path_resolved"],"source_sha256":c["source_sha256"],
                            "volume_context_reconstruction_sha256":c["context_reconstruction_sha256"],
                            "pandas_mean_price_identity_max_absolute_gap":c["pandas_mean_price_identity_max_absolute_gap"],
                            "metrics":c["metrics"]} for c in cells],
            "validation_selected_strong":selection,"selected_epochs":epochs,"training_receipt_ledger":ledger,
            "npz_identity":identity,
            "limitations":[
                "First identified model evaluation on these three markets followed prior source QC and partitioning. After this run their test performance is exposed.",
                "Authors cleaned daily transaction mean prices; these are not close prices or certified historical vintages.",
                "BJ/TJ primary macro excludes CQ. Unequal sample counts do not change equal-market weights.",
                "Four primary contrasts plus two secondary allocation_readout contrasts comprise one six-contrast multiplicity family.",
                "Macro intervals combine separate-market endpoints using a union bound; no cross-market independence assumption. Short dependent samples still make percentile coverage approximate.",
                "Length20 and40 results both remain. CQ40 is deliberately omitted because40>=28; CQ20 is only descriptive.",
                "If any bootstrap denominator is zero, that market/length CI is unavailable rather than silently dropping undefined draws.",
                "Epoch0 is the native initialization, sometimes already a useful forecast. Positive total gain is not automatically a learned gain.",
                "Softplus constrained coefficients and affine unconstrained coefficients have unequal initial effective update derivatives; sign restrictions are not isolated.",
                "Strong validation selector is context, not a replacement for fixed RW/structural comparisons.",
                "Later SWEWT-GRU supplemental baselines are outside this frozen run/family and are not added to its hypothesis family."
            ]}
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"V3_INDEPENDENT_STATISTICAL_AUDIT_ZH.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
    with (args.out/"V3_NATIVE_METRICS.csv").open("w",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=list(native[0]));wr.writeheader();wr.writerows(native)
    def ci(x):
        return "不可估（有零分母抽样）" if x is None else f"[{100*x[0]:+.5f}, {100*x[1]:+.5f}]"
    lines=["# V3 三市场固定复制：独立统计审查","",
           "**完整性 PASS。** 按09:46:10 UTC冻结的统计方案独立重建源数据、目标与39个预测文件。主模型保持liquidity_full，次候选保持v1 allocation_readout；北京/天津构成主要复制，重庆单列压力测试。本次审查没有拟合模型。",
           "",
           "## 固定评价范围","",
           "北京116、天津72个测试起点进入主结果；重庆28个不进入主平均。H1/H5/H20共用起点，先平均seed平方误差、各市场/horizon除以自身RW MSE，再等权汇总市场和horizon。原JSON中的year=fixed602020是fold标签，不是年度。",
           "",
           "## 各模型主指标","",
           "| 家族 | 北京 | 天津 | BJ/TJ等权主指标 | 重庆压力测试 |","|---|---:|---:|---:|---:|"]
    for f in sorted(macro,key=lambda x:macro[x]["primary_mean_seed_mse_over_rw"]):
        values=[c["metrics"][f]["mean_seed_mse_over_rw"] for c in cells]
        lines.append(f"| {f} | {values[0]:.9f} | {values[1]:.9f} | {macro[f]['primary_mean_seed_mse_over_rw']:.9f} | {values[2]:.9f} |")
    lines+=["","## 六项预指定比较","",
           "改善为comparator−candidate，正值更好，单位为RW归一化MSE百分点。四个full主对比及两个预指定readout次候选对比共同构成六项多重比较家族。",
           "",
           "| 比较 | BJ/TJ改善百分点 | 北京/天津获胜 | 20块宏观95% | 20块六项同时95% | 40块宏观95% | 40块六项同时95% |",
           "|---|---:|---:|---|---|---|---|"]
    for name,e in effects.items():
        lines.append(f"| {name} | {100*e['macro_improvement_rw_units']:+.6f} | {e['primary_market_wins_of2']}/2 | {ci(e['block20']['macro_ci95_without_market_independence'])} | {ci(e['block20']['macro_simultaneous95_joint_six_family'])} | {ci(e['block40']['macro_ci95_without_market_independence'])} | {ci(e['block40']['macro_simultaneous95_joint_six_family'])} |")
    lines+=["",
           "每市场按观测起点循环移动块配对重采样，两个块长各50,000次，随机种子20261019；同一抽样用于全部模型和horizon，并重算RW分母。先计算每市场97.5%区间再等权合并端点，得到单对比宏观约95%区间；每市场每对比99.583333%区间合并后构成六项宏观约同时95%区间。该Bonferroni构造不假定两市场独立。没有把seeds视为独立样本，也没有按结果挑块长。",
           "",
           "## 每市场边际区间","",
           "| 比较 | 北京20块95% | 天津20块95% | 重庆20块95%（仅描述） |","|---|---|---|---|"]
    for name,_,_ in CONTRASTS:
        lines.append("| "+name+" | "+" | ".join(ci(intervals[m]["20"][name]["ci95_marginal"]) for m in MARKETS)+" |")
    lines+=["",
           "116、72、28个起点约只能排入5、3、1个不重叠20起点块；这不是正式有效样本量估计。尤其重庆的区间仅作描述。40大于重庆样本数28，会退化为整段旋转，故按原方案不报告重庆40块区间。全部区间仍依赖短序列上的近似及固定已训练模型，不能转述为充分的新市场总体证据。",
           "",
           "## 执行核数","",
           f"共120个神经网络任务：48次选择、72次最终重训；记录正梯度步骤的任务{counts['tasks_with_positive_gradient_steps']}个，最终重训中{counts['refits_with_positive_gradient_steps']}个；{counts['zero_epoch_refits']}次最终重训为epoch0。39个NPZ中的origin、horizon、seed与源价格目标逐一一致，SUMMARY主指标及集成指标均复算通过。5个源码文件和冻结统计方案哈希保持一致。",
           "",
           "本轮目标是作者清洗的日成交均价，存在月内参考处理及历史版本限制；均价和量假定当前报告已知。固定先验可在epoch0产生非零预测，所有对RW的收益都不能自动归于学习。无约束对照与softplus约束对照的有效更新导数不同，仍不把差异全归于经济符号限制。",
           "",
           "原单位逐市场/horizon的MSE、RMSE、MAE保存在V3_NATIVE_METRICS.csv；机器记录含完整训练收据、39个NPZ哈希、各市场指标及所有区间。后续补充SWEWT-GRU对照按新增研究描述，不改变这份已冻结的主候选或六项统计家族。",
           ""]
    (args.out/"V3_INDEPENDENT_STATISTICAL_AUDIT_ZH.md").write_text("\n".join(lines))
    print(json.dumps({"counts":counts,"primary_macro":macro,"effects":effects,
                      "market95":{m:{k:v["ci95_marginal"] for k,v in intervals[m]["20"].items()} for m in MARKETS}},
                     ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()

