"""Post-ranking exploratory readout analysis. Does not revise V1's full primary."""
from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np
from independent_v1_statistics import STUDY, read_json, sha, check, score, reconstruct, calendar_weights

RUN = Path("/tmp/carbon-literature-aligned-runs-20261019d/main_v1")
COMPARATORS = ["rw", "allocation_constant"]
TARGET = "allocation_readout"


def bootstrap(cells, days):
    rng=np.random.default_rng(20261019)
    B=20000
    draws=np.zeros((B,2))
    for year in [2022,2023]:
        start=np.datetime64(f"{year}-01-01")
        n=int((np.datetime64(f"{year+1}-01-01")-start).astype(int))
        w=calendar_weights(rng,B,n,days)
        for c in cells:
            if c["year"] != year: continue
            ww=w[:,(c["dates"]-start).astype(int)]
            denominator=ww@(c["observed"]**2)
            check((denominator>0).all(),"positive RW denominator")
            for j,comp in enumerate(COMPARATORS):
                draws[:,j]+=((ww@(c["loss"][comp]-c["loss"][TARGET]))/denominator).mean(axis=1)/len(cells)
    return {comp:{"ci95":np.quantile(draws[:,j],[.025,.975]).tolist(),
                  "ci97_5_bonferroni2":np.quantile(draws[:,j],[.0125,.9875]).tolist(),
                  "positive_fraction_not_pvalue":float((draws[:,j]>0).mean()),
                  "B":B,"block_calendar_days":days}
             for j,comp in enumerate(COMPARATORS)}


def main():
    original=read_json(STUDY/"audits/V1_INDEPENDENT_STATISTICAL_AUDIT_ZH.json")
    hashes={(r["market"],r["year"],r["family"]):r["npz_sha256"] for r in original["npz_identity"]}
    protocol=read_json(RUN/"PROTOCOL_BEFORE_RUN.json")
    for name,digest in protocol["source_sha256"].items():
        check(sha(STUDY/"source"/name)==digest,"frozen source")
    cells=[]
    for market in protocol["markets"]:
        for year in protocol["years"]:
            base=RUN/f"{market}_{year}"
            inputs=reconstruct(read_json(base/"INPUT_MANIFEST.json"))
            c={"market":market,"year":year,**inputs,"metrics":{},"loss":{}}
            for family in [TARGET]+COMPARATORS:
                path=base/family/"predictions.npz"
                check(sha(path)==hashes[(market,year,family)],"original NPZ unchanged")
                with np.load(path,allow_pickle=False) as z:
                    check(np.array_equal(z["origins"],inputs["origins"]),"origin")
                    check(np.array_equal(z["observed"],inputs["observed"]),"observed")
                    met,loss=score(z["predictions"],z["observed"])
                c["metrics"][family]=met;c["loss"][family]=loss
            cells.append(c)
    macro={f:float(np.mean([c["metrics"][f]["mean_seed_mse_over_rw"] for c in cells]))
           for f in [TARGET]+COMPARATORS}
    b42,b84=bootstrap(cells,42),bootstrap(cells,84)
    effects={}
    for comp in COMPARATORS:
        per=[{"market":c["market"],"year":c["year"],
              "improvement_rw_units":c["metrics"][comp]["mean_seed_mse_over_rw"]-
                                     c["metrics"][TARGET]["mean_seed_mse_over_rw"],
              "readout_score":c["metrics"][TARGET]["mean_seed_mse_over_rw"],
              "comparator_score":c["metrics"][comp]["mean_seed_mse_over_rw"]} for c in cells]
        effects[comp]={"improvement_rw_units":macro[comp]-macro[TARGET],
                       "relative_reduction_in_macro_ratio":1-macro[TARGET]/macro[comp],
                       "cell_wins_of_8":sum(r["improvement_rw_units"]>0 for r in per),
                       "per_cell":per,
                       "leave_one_market_out":{m:float(np.mean([r["improvement_rw_units"] for r in per if r["market"]!=m]))
                                                for m in protocol["markets"]},
                       "calendar42":b42[comp],"calendar84":b84[comp]}
    result={
        "title":"V1 readout 开发排名后的发现性诊断",
        "status":"EXPLORATORY_POST_SELECTION_ONLY",
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "target":TARGET,"original_primary_remains":"allocation_full",
        "selection_reason":"This diagnostic was requested only after development rankings showed readout ahead of the original full candidate. It is not preregistered confirmation.",
        "macro_primary_scores":macro,"effects":effects,
        "readout_seed_scores_order_17_29_43":np.mean([
            np.array(c["metrics"][TARGET]["seed_mse_over_rw_by_horizon"]).mean(axis=1) for c in cells],axis=0).tolist(),
        "source_sha256":protocol["source_sha256"],"script_sha256":sha(__file__),
        "plan_sha256":sha(STUDY/"audits/V1_READOUT_DISCOVERY_PLAN.json"),
        "limitations":[
            "All 2022/2023 cells are already exposed development data.",
            "The target arm was selected after viewing rankings. Intervals do not adjust for that selection or previous research search.",
            "97.5% intervals only apply Bonferroni to the two diagnostic contrasts, without resolving arm-selection bias.",
            "Synchronous calendar blocks preserve shared-date dependence across markets; circular block and fixed-model assumptions remain.",
            "Seed losses are averaged within origin before resampling; seeds are not independent observations.",
            "No model refitting occurred and the original four-contrast full audit is unchanged."
        ]}
    out=STUDY/"audits"
    (out/"V1_READOUT_DISCOVERY_ZH.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
    def ci(x):return f"[{100*x[0]:+.5f}, {100*x[1]:+.5f}]"
    lines=["# V1 readout：开发排名出现后的发现性诊断","",
       "**本文件属于事后发现性分析。** 在 v1 原四项审查看到 readout 排名后，才选择该臂检查它相对 RW 和 constant 的价值。原候选仍为 full，原审查结论与四项对比不变。所有 2022/2023 数据都已暴露，以下区间不能称为新确认。",
       "",
       f"readout 的 mean-seed MSE/RW 为 {macro[TARGET]:.9f}；RW 为 {macro['rw']:.9f}，constant 为 {macro['allocation_constant']:.9f}。",
       "",
       "| 对照 | 改善（RW 百分点） | 相对降低 | 获胜单元 / 8 | 42 日 95% 区间 | 42 日 97.5% 区间 | 84 日 95% 区间 |",
       "|---|---:|---:|---:|---|---|---|"]
    for name,e in effects.items():
        lines.append(f"| {name} | {100*e['improvement_rw_units']:+.5f} | {100*e['relative_reduction_in_macro_ratio']:+.5f}% | {e['cell_wins_of_8']} | {ci(e['calendar42']['ci95'])} | {ci(e['calendar42']['ci97_5_bonferroni2'])} | {ci(e['calendar84']['ci95'])} |")
    lines+=["",
       "改善为 comparator−readout，正值更好；百分点的分母是各单元 RW MSE，相对降低则以 comparator 的宏观指标作分母。三 seed 的 readout 主指标依次为 "+", ".join(f"{v:.9f}" for v in result["readout_seed_scores_order_17_29_43"])+"。",
       "",
       "方法沿用原独立审查：逐 origin 先平均 seed 平方误差；按年度完整日历同步抽取循环连续块，同一天的权重用于全部市场、readout、对照及 RW 分母。主块长 42 日，敏感性 84 日，各 20,000 次，随机种子 20261019。每个 replicate 重算单元损失比，并保持市场/年度/horizon 等权。",
       "",
       "95% 区间为边际区间；97.5% 只对本文件两项对比做 Bonferroni 处理，**没有修正从开发排名中选中 readout 的选择偏差**，也没有涵盖此前研究尝试。它们用于判断该臂是否值得后续保留，不用于宣告已经达到发表或确认标准。",
       "",
       "## 逐单元结果","",
       "| 市场 | 年度 | readout | RW | constant |",
       "|---|---:|---:|---:|---:|"]
    for c in cells:
        lines.append(f"| {c['market']} | {c['year']} | {c['metrics'][TARGET]['mean_seed_mse_over_rw']:.9f} | {c['metrics']['rw']['mean_seed_mse_over_rw']:.9f} | {c['metrics']['allocation_constant']['mean_seed_mse_over_rw']:.9f} |")
    lines+=["","## 保留与判读","",
       "readout 原来是控制臂，这一身份不决定它自身是否有预测价值。此处按真实排名如实保留它的表现；后续若发展该臂，应另立模型主张和评价边界，并保留这次事后发现的时间顺序。不能用该臂结果重命名原 full 候选，或抹去原 full 未获优势支持的结果。",
       "",
       "原始 NPZ 的哈希与 v1 独立审查完全一致，本次未重新拟合市场模型。机器记录见 V1_READOUT_DISCOVERY_ZH.json，预先写明的本次事后诊断范围见 V1_READOUT_DISCOVERY_PLAN.json。",
       ""]
    (out/"V1_READOUT_DISCOVERY_ZH.md").write_text("\n".join(lines))
    print(json.dumps({"macro":macro,"effects":{k:{x:y for x,y in v.items() if x not in ['per_cell','leave_one_market_out']} for k,v in effects.items()}},ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()

