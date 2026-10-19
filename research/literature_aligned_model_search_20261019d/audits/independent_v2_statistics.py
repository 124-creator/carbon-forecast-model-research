"""Independent V2 audit, reusing arithmetic-only V1 audit utilities.

No experiment source is modified/imported and no model fitting/checkpoint loading
occurs. The V1 audit utility parameter-count metadata is changed in this process
only to describe the six frozen V2 architectures.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import csv
import argparse
import numpy as np
import pandas as pd
import independent_v1_statistics as core

STUDY=core.STUDY
RUN=Path("/tmp/carbon-literature-aligned-runs-20261019d/liquidity_v2")
V1=Path("/tmp/carbon-literature-aligned-runs-20261019d/main_v1")
AUXILIARY_ROOT=None
NEURAL=["liquidity_full","liquidity_readout","shock_size_only","liquidity_unconstrained","gru","lstm"]
CLASSICAL=["rw","drift","ridge_price","ridge_full","hgb_full"]
FAMILIES=NEURAL+CLASSICAL
TARGET="liquidity_full"
COMPARATORS=["rw","liquidity_readout","shock_size_only","liquidity_unconstrained","v1_allocation_constant"]
core.PARAMETERS={"liquidity_full":11,"liquidity_readout":11,"shock_size_only":9,
                 "liquidity_unconstrained":11,"gru":4500,"lstm":5780}


def reconstruct_volume(manifest):
    src=manifest["volume_source"]
    path=AUXILIARY_ROOT/Path(src["path"]).name if AUXILIARY_ROOT else Path(src["path"])
    core.check(core.sha(path)==src["sha256"],"volume source hash")
    aux=pd.read_csv(path)
    carbon=pd.read_csv(core.REPO/manifest["sources"][0]["path"]).sort_values("date").reset_index(drop=True)
    dates=pd.to_datetime(aux.date).to_numpy(dtype="datetime64[D]")
    core.check(np.array_equal(dates,pd.to_datetime(carbon.date).to_numpy(dtype="datetime64[D]")),"volume/price dates")
    amount=aux.volume.to_numpy(float)
    valid=(aux.valid_volume.to_numpy()==1)&np.isfinite(amount)
    core.check((amount[valid]>=0).all(),"negative valid volume")
    logs=np.full(len(aux),np.nan);logs[valid]=np.log1p(amount[valid])
    context=np.zeros((len(aux),2),dtype="float32")
    for t in range(len(aux)):
        preceding=logs[max(0,t-60):t]
        preceding=preceding[np.isfinite(preceding)]
        if valid[t] and len(preceding)>=10:
            context[t,0]=np.clip((logs[t]-np.median(preceding))/max(float(np.std(preceding)),.25),-5,5)
            context[t,1]=1
    core.check(len(aux)==src["rows"],"volume rows")
    core.check(int(valid.sum())==src["valid_reported_volume_rows"],"reported volume count")
    core.check(int(context[:,1].sum())==src["usable_context_rows"],"usable context count")
    core.check(np.isfinite(context).all() and (abs(context[:,0])<=5).all(),"finite volume context")
    core.check(np.all(context[context[:,1]==0,0]==0),"missing volume context")
    return {"market":manifest["market"],"year":manifest["year"],
            "path":str(path),"original_manifest_path":src["path"],"source_sha256":src["sha256"],
            "rows":len(aux),"reported_valid":int(valid.sum()),"usable":int(context[:,1].sum()),
            "reconstructed_context_float32_sha256":core.hashlib.sha256(context.tobytes()).hexdigest(),
            "interpretation":"Current reported volume plus only prior60targetrecords; after-current-report availability assumed, vintage/unit not certified"}


def bootstrap(cells,days):
    rng=np.random.default_rng(20261019);B=20000
    draws=np.zeros((B,len(COMPARATORS)))
    for year in [2022,2023]:
        start=np.datetime64(f"{year}-01-01")
        n=int((np.datetime64(f"{year+1}-01-01")-start).astype(int))
        w=core.calendar_weights(rng,B,n,days)
        for c in cells:
            if c["year"]!=year:continue
            ww=w[:,(c["dates"]-start).astype(int)]
            den=ww@(c["observed"]**2)
            core.check((den>0).all(),"bootstrap RW denominator")
            for j,comp in enumerate(COMPARATORS):
                draws[:,j]+=((ww@(c["loss"][comp]-c["loss"][TARGET]))/den).mean(axis=1)/len(cells)
    return {name:{"ci95":np.quantile(draws[:,j],[.025,.975]).tolist(),
                  "ci99_bonferroni5":np.quantile(draws[:,j],[.005,.995]).tolist(),
                  "positive_fraction_not_pvalue":float((draws[:,j]>0).mean()),
                  "B":B,"block_calendar_days":days} for j,name in enumerate(COMPARATORS)}


def main():
    global RUN,V1,AUXILIARY_ROOT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=RUN)
    parser.add_argument("--v1-run",type=Path,default=V1)
    parser.add_argument("--auxiliary-root",type=Path,
                        help="Relocate the two hash-pinned MARKET_auxiliary.csv files")
    parser.add_argument("--v1-audit",type=Path,default=STUDY/"audits/V1_INDEPENDENT_STATISTICAL_AUDIT_ZH.json")
    parser.add_argument("--out",type=Path,default=STUDY/"audits")
    args=parser.parse_args()
    RUN,V1,AUXILIARY_ROOT=args.run,args.v1_run,args.auxiliary_root
    protocol=core.read_json(RUN/"PROTOCOL_BEFORE_RUN.json")
    done=core.read_json(RUN/"EXECUTION_COMPLETE.json")
    core.check(done["status"]=="completed" and done["source_hashes_unchanged"],"V2 completed/hash flag")
    core.check(done["jobs"]==28 and done["summary_rows"]==44,"V2 jobs")
    core.check(protocol["markets"]==["GUANGZHOU","HUBEI"] and protocol["years"]==[2022,2023],"V2 population")
    for name,digest in protocol["source_sha256"].items():
        core.check(core.sha(STUDY/"source"/name)==digest,"frozen V2 source")
    v1audit=core.read_json(args.v1_audit)
    v1hash={(r["market"],r["year"],r["family"]):r["npz_sha256"] for r in v1audit["npz_identity"]}
    allsum=core.read_json(RUN/"ALL_SUMMARIES.json")
    sums={(r["market"],r["year"],r["family"]):r for r in allsum}
    core.check(len(sums)==44,"V2 summary uniqueness")
    cells=[];ledger=[];identity=[];volume=[];epochs=[];selected=[];native=[]
    for market in protocol["markets"]:
        for year in protocol["years"]:
            cd=RUN/f"{market}_{year}"
            manifest=core.read_json(cd/"INPUT_MANIFEST.json")
            original_manifest=core.read_json(V1/f"{market}_{year}"/"INPUT_MANIFEST.json")
            for key in ["sources","counts","training_rms","train_target_second_moment",
                        "last_train_label","first_validation_origin","last_validation_label",
                        "first_test_origin","last_test_label","energy_alignment"]:
                core.check(manifest[key]==original_manifest[key],"V1/V2 manifest same target information")
            d=core.reconstruct(manifest);volume.append(reconstruct_volume(manifest))
            validation={}
            for family in FAMILIES:
                fd=cd/family
                if family in NEURAL:
                    vs,epoch=core.audit_receipts(fd,family,d,ledger)
                    validation[family]=vs
                    epochs.append({"market":market,"year":year,"family":family,"selected_epoch":epoch})
                else:
                    r=core.read_json(fd/"SELECTION.json");validation[family]=r["validation_loss"]
                    if family=="rw":
                        core.close(r["validation_loss"],d["validation_rw"],"V2 RW validation")
                    elif family=="drift":
                        core.close(r["validation_loss"],d["drift_choice"][0],"V2 drift validation")
                        core.check(r["window"]==d["drift_choice"][1],"V2 drift choice")
                    elif family.startswith("ridge"):
                        best=min(r["trials"]);core.close(r["validation_loss"],best[0],"V2 ridge validation")
                        core.check(r["alpha"]==best[1],"V2 ridge alpha")
                    else:
                        best=min(r["trials"]);core.close(r["validation_loss"],best[0],"V2 HGB validation")
                        core.check((r["max_leaf_nodes"],r["l2_regularization"])==tuple(best[1:]),"V2 HGB choice")
            strong=min(core.POOL,key=lambda f:(validation[f],core.POOL.index(f)))
            selected.append({"market":market,"year":year,"family":strong,"validation_loss":validation[strong],
                             "pool_validation_losses":{f:validation[f] for f in core.POOL}})
            cell={"market":market,"year":year,**d,"loss":{},"metrics":{},"strong":strong}
            for family in FAMILIES+["v1_allocation_constant"]:
                path=(V1/f"{market}_{year}"/"allocation_constant"/"predictions.npz"
                      if family=="v1_allocation_constant" else cd/family/"predictions.npz")
                if family=="v1_allocation_constant":
                    core.check(core.sha(path)==v1hash[(market,year,"allocation_constant")],"reused V1 constant hash")
                with np.load(path,allow_pickle=False) as z:
                    pred=np.array(z["predictions"],dtype=float);obs=np.array(z["observed"],dtype=float)
                    core.check(np.array_equal(z["origins"],d["origins"]),"V2 NPZ origin")
                    core.check(np.array_equal(obs,d["observed"]),"V2 NPZ target")
                    core.check(np.array_equal(z["horizons"],core.HORIZONS),"V2 horizons")
                    expected_seeds=core.SEEDS if family in NEURAL+["v1_allocation_constant"] else [-1]
                    core.check(np.array_equal(z["seeds"],expected_seeds),"V2 seeds")
                core.check(pred.shape==(len(expected_seeds),len(obs),3),"V2 shape")
                metrics,loss=core.score(pred,obs)
                if family=="rw":core.check(np.all(pred==0),"V2 RW zero")
                if family in ["gru","lstm"]:
                    ep=next(r["selected_epoch"] for r in epochs if (r["market"],r["year"],r["family"])==(market,year,family))
                    if ep==0:core.check(np.all(pred==0),"V2 recurrent epoch0")
                if family!="v1_allocation_constant":
                    for saved in [core.read_json(path.parent/"SUMMARY.json"),sums[(market,year,family)]]:
                        for key in ["mean_seed_mse_over_rw","seed_mse_over_rw_by_horizon","rw_mse_by_horizon",
                                    "ensemble_mse_over_rw","ensemble_mse_over_rw_by_horizon"]:
                            core.close(metrics[key],saved[key],"V2 stored summary")
                cell["metrics"][family]=metrics;cell["loss"][family]=loss
                identity.append({"market":market,"year":year,"family":family,"path":str(path),"npz_sha256":core.sha(path),
                                 "n_origins":len(obs),"reused_v1":family=="v1_allocation_constant"})
                for j,h in enumerate(core.HORIZONS):
                    native.append({"market":market,"year":year,"family":family,"horizon":int(h),"n_origins":len(obs),
                        "mean_seed_mse_over_rw":float(np.mean(np.array(metrics["seed_mse_over_rw_by_horizon"])[:,j])),
                        "mean_seed_mse":metrics["native_mean_seed_mse_by_horizon"][j],
                        "mean_seed_rmse":metrics["native_mean_seed_rmse_by_horizon"][j],
                        "rmse_of_mean_seed_mse":metrics["native_rmse_of_mean_seed_mse_by_horizon"][j],
                        "mean_seed_mae":metrics["native_mean_seed_mae_by_horizon"][j],
                        "ensemble_mse":metrics["native_ensemble_mse_by_horizon"][j]})
            cell["metrics"]["validation_selected_strong"]=cell["metrics"][strong]
            cell["loss"]["validation_selected_strong"]=cell["loss"][strong]
            cells.append(cell)
    core.check(len(ledger)==120 and len(identity)==48,"V2 audit count incl4 reused V1 files")
    family_names=FAMILIES+["v1_allocation_constant","validation_selected_strong"]
    macro={f:{"primary_mean_seed_mse_over_rw":float(np.mean([c["metrics"][f]["mean_seed_mse_over_rw"] for c in cells])),
              "secondary_ensemble_mse_over_rw":float(np.mean([c["metrics"][f]["ensemble_mse_over_rw"] for c in cells]))}
           for f in family_names}
    b42,b84=bootstrap(cells,42),bootstrap(cells,84)
    effects={}
    for comp in COMPARATORS:
        cm=macro[comp]["primary_mean_seed_mse_over_rw"];fm=macro[TARGET]["primary_mean_seed_mse_over_rw"]
        per=[{"market":c["market"],"year":c["year"],
              "improvement_rw_units":c["metrics"][comp]["mean_seed_mse_over_rw"]-c["metrics"][TARGET]["mean_seed_mse_over_rw"]}
             for c in cells]
        effects[comp]={"improvement_rw_units":cm-fm,"relative_reduction_in_macro_ratio":1-fm/cm,
                       "cell_wins_of_4":sum(r["improvement_rw_units"]>0 for r in per),"per_cell":per,
                       "calendar42":b42[comp],"calendar84":b84[comp],
                       "by_market":{m:float(np.mean([r["improvement_rw_units"] for r in per if r["market"]==m]))
                                    for m in protocol["markets"]}}
    refs=[r for r in ledger if r["mode"]=="final"];sels=[r for r in ledger if r["mode"]=="train"]
    counts={"neural_tasks":len(ledger),"selection_tasks":len(sels),"refit_tasks":len(refs),
            "tasks_with_positive_gradient_steps":sum(r["gradient_positive_steps"]>0 for r in ledger),
            "refit_with_positive_gradient_steps":sum(r["gradient_positive_steps"]>0 for r in refs),
            "zero_epoch_refits":sum(r["epochs_executed"]==0 for r in refs),
            "selected_epoch_zero_family_cells":sum(r["selected_epoch"]==0 for r in epochs),
            "total_executed_epochs":sum(r["epochs_executed"] for r in ledger),
            "total_logged_positive_gradient_steps":sum(r["gradient_positive_steps"] for r in ledger),
            "v2_prediction_files":44,"reused_v1_constant_files":4,
            "unique_origin_count":sum(len(c["observed"]) for c in cells),
            "origin_horizon_targets":sum(c["observed"].size for c in cells)}
    caveats=[
        "V2 was designed after partial V1 development results. All2022/2023 outcomes remain development; no fresh holdout claim.",
        "Full remains the V2 target. Other variants cannot replace it after seeing rankings.",
        "Volume surprise uses the current author's reported amount after the report and a strictly prior60targetrecord reference; exact unit, release timestamp and vintage are not certified.",
        "Return tokens have64record lookback; each volume token has up to60 earlier records in its normalization, expanding underlying volume information span.",
        "Full/readout/unconstrained each have11parameters; shock_size_only9; V1constant7. Full/readout have different nonzero initial forecast priors despite matching parameter count.",
        "Constrained coefficients use softplus(raw), derivative~0.09516 at0.1; unconstrained coefficients use an affine map with derivative1. Same raw learning rate thus changes effective update scale, so this ablation does not isolate sign restriction alone.",
        "ridge_full uses64x6inputs, HGB20x6, recurrent64x6; ridge_price deliberately remains20target-only. Historical length and capacity are not globally matched.",
        "V1constant is reused only after source,fold,scale,target and NPZ identity checks; V1eightcells and V2fourcells are not pooled.",
        "Bootstrap averages seed losses within each origin, synchronizes calendar blocks by year across markets, and recomputes RW denominator. It conditions on trained models and selected families.",
        "99% intervals adjust the five fixed core contrasts approximately by Bonferroni, not earlier searches or design selection.95% intervals are marginal.",
        "No model fitting, checkpoint loading or optimizer replay by this audit. Neural/ridge/HGB validation values are audited through source and receipts; RW/drift are reconstructed arithmetically.",
        "EIA14day alignment uses public historical data, not certified vintages."
    ]
    result={"title":"V2独立统计审查","status":"PASS_INTEGRITY","stage":"development_only",
            "resolved_arguments":{"run":str(RUN),"v1_run":str(V1),
                                  "auxiliary_root":str(AUXILIARY_ROOT) if AUXILIARY_ROOT else None,
                                  "v1_audit":str(args.v1_audit),"out":str(args.out)},
            "created_utc":datetime.now(timezone.utc).isoformat(),"source_sha256":protocol["source_sha256"],
            "script_sha256":core.sha(__file__),"utility_script_sha256":core.sha(core.__file__),
            "plan_sha256":core.sha(STUDY/"audits/V2_STATISTICAL_AUDIT_PLAN.json"),
            "execution_counts":counts,"macro_scores":macro,"core_effects":effects,
            "seed_primary_scores_order_17_29_43":{f:np.mean([np.array(c["metrics"][f]["seed_mse_over_rw_by_horizon"]).mean(axis=1)
                                                          for c in cells],axis=0).tolist() for f in NEURAL},
            "selected_epochs":epochs,"validation_selected_strong":selected,
            "volume_reconstruction":volume,"training_receipt_ledger":ledger,"npz_identity":identity,
            "cell_metrics":[{"market":c["market"],"year":c["year"],"metrics":c["metrics"]} for c in cells],
            "limitations":caveats}
    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    (out/"V2_INDEPENDENT_STATISTICAL_AUDIT_ZH.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
    with (out/"V2_NATIVE_METRICS.csv").open("w",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=list(native[0]));wr.writeheader();wr.writerows(native)
    def ci(x):return f"[{100*x[0]:+.5f}, {100*x[1]:+.5f}]"
    lines=["# V2 独立统计审查","",
       "**完整性 PASS；仅为 development。** 本审查固定 liquidity_full 为主模型，从全部44个新 NPZ 逐行复算，并核对4个复用的 v1 constant 文件。v2 形成于看过部分 v1 开发结果之后；未重新定义主模型，也未把两个研究的单元混合。",
       "",
       "## 主指标","",
       "先平均 seed 的平方误差，按相同市场/年度/horizon 的 RW MSE 归一化，再对两市场、两年度、三个 horizon 等权。集成预测仅列次要指标。",
       "",
       "| 家族 | 主 mean-seed MSE/RW | 次要 ensemble MSE/RW |","|---|---:|---:|"]
    for f in sorted(family_names,key=lambda x:macro[x]["primary_mean_seed_mse_over_rw"]):
        lines.append(f"| {f} | {macro[f]['primary_mean_seed_mse_over_rw']:.9f} | {macro[f]['secondary_ensemble_mse_over_rw']:.9f} |")
    lines+=["","## 固定的五项核心比较","",
       "改善为 comparator−full；正值更好。下表单位为 RW 归一化损失的百分点；相对降低另外除以 comparator 的宏观指标。",
       "",
       "| 对照 | 改善（RW 百分点） | 相对降低 | 获胜 /4 | 42 日95% | 42 日99% | 84 日95% |",
       "|---|---:|---:|---:|---|---|---|"]
    for comp,e in effects.items():
        lines.append(f"| {comp} | {100*e['improvement_rw_units']:+.6f} | {100*e['relative_reduction_in_macro_ratio']:+.6f}% | {e['cell_wins_of_4']} | {ci(e['calendar42']['ci95'])} | {ci(e['calendar42']['ci99_bonferroni5'])} | {ci(e['calendar84']['ci95'])} |")
    lines+=["",
       "同步循环日历块按年抽取，42 日为主、84 日为敏感性，各20,000次，随机种子20261019。同日权重共享于广州/湖北、主模型/所有对照、RW分母；每次重新计算分母，保持单元和 horizon 等权。seeds先折为每个origin的平均损失，不是独立样本。99%区间只近似调整本轮五项比较，不能修正此前开发筛选；95%为边际区间。",
       "",
       "## 执行与数据核验","",
       f"核对120个神经网络任务：48次选择、72次重训；其中{counts['tasks_with_positive_gradient_steps']}个有正梯度步骤，{counts['refit_with_positive_gradient_steps']}次重训执行正梯度，{counts['zero_epoch_refits']}次最终重训为epoch0。24个家族/单元选择中{counts['selected_epoch_zero_family_cells']}个为epoch0。",
       "",
       f"四单元共{counts['unique_origin_count']}个origin、{counts['origin_horizon_targets']}个origin×horizon目标。源码四文件哈希、源数据哈希、标签/日期/分割、尺度、EIA14日对齐、已记录验证选择和checkpoint字节哈希均通过；所有NPZ主指标/集成指标与SUMMARY一致。",
       "",
       "成交量由源文件日期与规范价格精确配对。当前log1p(volume)只用前60个目标记录的有效成交量作参考，至少10个前期有效观测；缺失上下文为(0,0)，不代表成交量为零。该量假设在当前日报发布后可用，不能声称掌握真实历史发布时间或vintage。",
       "",
       "## 解释边界","",
       "- full/readout/unconstrained 均11参数，shock-size9、constant7。v2门初始化非零，full/readout的初始预测先验不相同；epoch0收益不能自动归于训练。",
       "- constrained 在0.1系数初始化处的softplus导数约0.09516，而unconstrained仿射映射导数为1。同一raw学习率改变了有效系数的局部更新尺度，符号约束消融也包含优化参数化差异。",
       "- ridge_full和神经网络使用64×6输入，HGB用20×6；ridge_price是20个价格记录的有意控制。64个成交量上下文还各自引用更早60记录，完整原始信息跨度不是只有64条。",
       "- v1constant只复用于源价格、输入来源、尺度、切分和targets均相同的4单元；旧8单元主指标不能直接和本轮4单元主指标比较。",
       "- 验证选择的强基线另列为次要参照，不能用胜过它抵消五个核心控制中的不足。",
       "",
       "完整机器证据见 V2_INDEPENDENT_STATISTICAL_AUDIT_ZH.json；原单位逐市场/horizon的MSE、RMSE、MAE见 V2_NATIVE_METRICS.csv；本次事后独立审查范围见 V2_STATISTICAL_AUDIT_PLAN.json。",
       ""]
    (out/"V2_INDEPENDENT_STATISTICAL_AUDIT_ZH.md").write_text("\n".join(lines))
    print(json.dumps({"counts":counts,"macro":macro,"effects":{k:{x:y for x,y in v.items() if x not in ['per_cell','by_market']}
                                                             for k,v in effects.items()}},ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()

