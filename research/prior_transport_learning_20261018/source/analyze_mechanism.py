#!/usr/bin/env python3
"""Post-run descriptive analysis of this study's approved JSON summaries/traces.

Reads only results/D1/*_SUMMARY.json, the two named independent-audit JSON files,
and matching *_TRACE.json for the nine mechanism arms. It imports no learner,
model, torch, market reader or metrics/selection code. It does not read CSV/NPZ,
model checkpoints or confirmation data, refit, choose a new model, or bootstrap.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np


MECHANISMS = (
    "local_profile", "uniform_profile", "aux_one", "aux_five", "profile",
    "transport", "transport_uniform", "covariance_profile", "diagonal_profile",
)
NEURAL = MECHANISMS + ("local_gru", "pooled_gru")
ADAPTIVE = ("aux_one", "aux_five", "profile", "transport", "covariance_profile", "diagonal_profile")


def stats(values):
    x = np.asarray(list(values), dtype=np.float64)
    if not len(x):
        return {"n": 0}
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite descriptive input")
    return {"n": len(x), "min": float(x.min()), "p05": float(np.quantile(x, .05)),
            "mean": float(x.mean()), "median": float(np.median(x)),
            "p95": float(np.quantile(x, .95)), "max": float(x.max())}


def require(condition, label):
    if not condition:
        raise AssertionError(label)


def full_weights(row):
    source = np.asarray(row["source_weights"], dtype=np.float64)
    null = 1.0-float(source.sum())
    if "null_weight" in row:
        require(abs(null-row["null_weight"]) < 1e-12, "null weight identity")
    return np.r_[null, source]


def weight_change(initial, current):
    delta = current-initial
    result = {"full_probabilities": current.tolist(),
              "source_mass": float(current[1:].sum()), "null_mass": float(current[0]),
              "max_abs_source_probability_change": float(np.abs(delta[1:]).max()),
              "full_probability_l1_change": float(np.abs(delta).sum()),
              "null_probability_change": float(delta[0])}
    if (initial > 0).all() and (current > 0).all():
        centered_delta = np.log(current)-np.log(initial)
        centered_delta -= centered_delta.mean()
        odds_delta = np.log(current[1:]/current[0])-np.log(initial[1:]/initial[0])
        result["centered_logit_change"] = centered_delta.tolist()
        result["max_abs_centered_logit_change"] = float(np.abs(centered_delta).max())
        result["max_abs_source_to_null_log_odds_change"] = float(np.abs(odds_delta).max())
    return result


def prior_snapshot(row, initial_isotropic=False):
    if row["step"] == 0:
        if not initial_isotropic:
            return None
        return {"trace": .8, "min_eigenvalue": .1, "max_eigenvalue": .1,
                "spectral_condition": 1.0, "trace_ratio_to_initial_isotropic": 1.0,
                "origin": "known fixed initial 0.1I, not a newly read checkpoint"}
    result = {"trace": float(row["precision_trace"]),
              "min_eigenvalue": float(row["precision_min_eigenvalue"]),
              "max_eigenvalue": float(row["precision_max_eigenvalue"])}
    require(result["min_eigenvalue"] > 0, "positive recorded prior eigenvalue")
    result["spectral_condition"] = result["max_eigenvalue"]/result["min_eigenvalue"]
    if initial_isotropic:
        result["trace_ratio_to_initial_isotropic"] = result["trace"]/.8
    return result


def run(study, trace_dir):
    accesses = []

    def read(path, role):
        raw = path.read_bytes()
        record = {"file": path.name, "role": role, "bytes": len(raw),
                  "sha256": hashlib.sha256(raw).hexdigest()}
        accesses.append(record)
        return json.loads(raw), record["sha256"]

    summaries = {}
    for path in sorted((study/"results/D1").glob("*_SUMMARY.json")):
        data, _ = read(path, "public_job_summary")
        require(path.name == data["job"]+"_SUMMARY.json", "summary filename/job identity")
        require(data["status"] == "COMPLETE", "all source summaries complete")
        require(data["job"] not in summaries, "unique jobs")
        summaries[data["job"]] = data
    require(len(summaries) == 350, "fixed 350 completed flows")
    independent, _ = read(study/"audits/D1_independent/INDEPENDENT_SELECTION.json", "independent_frozen_selection")
    independent_audit, _ = read(study/"audits/D1_independent/INDEPENDENT_METRIC_AUDIT.json", "independent_metric_audit")
    require(independent_audit["status"] == "PASS_INDEPENDENT_METRICS_AND_SELECTION", "independent metric PASS")

    grouped = defaultdict(list)
    by_key = {}
    for item in summaries.values():
        grouped[item["model"]].append(item)
        key = (item["model"], item["target"], item["budget"], item["seed"])
        require(key not in by_key, "unique model/target/budget/seed")
        by_key[key] = item
    ranking_by_name = {x["model"]: x for x in independent["ranking"]}
    overall = []
    largest_metric_difference = 0.0
    for model, rows in sorted(grouped.items()):
        score = float(np.mean([r["evaluation"]["score"] for r in rows]))
        error = abs(score-ranking_by_name[model]["score"])
        largest_metric_difference = max(largest_metric_difference, error)
        require(error < 1e-12, "summary aggregation vs independent metric")
        overall.append({"model":model,"score":score,"jobs":len(rows),
                        "budget100":float(np.mean([r["evaluation"]["score"] for r in rows if r["budget"]==100])),
                        "budget400":float(np.mean([r["evaluation"]["score"] for r in rows if r["budget"]==400]))})

    stopping = {}
    for model in NEURAL:
        rows = grouped[model]
        receipts = [r["fit_receipt"] for r in rows]
        if model in MECHANISMS:
            steps = [r["steps"] for r in receipts]
            selected = [r["selected_step"] for r in receipts]
            reasons = Counter(r["status"] for r in receipts)
        else:
            steps = [r["steps_completed"] for r in receipts]
            selected = [r["best_step"] for r in receipts]
            reasons = Counter(r["stop_reason"] for r in receipts)
        stopping[model] = {"fits":len(rows),"executed_steps":stats(steps),
                           "selected_steps":stats(selected),"selected_step_zero":sum(x==0 for x in selected),
                           "selected_step_at_600":sum(x==600 for x in selected),
                           "executed_steps_at_600":sum(x==600 for x in steps),
                           "stop_reasons":dict(reasons),"total_executed_steps":sum(steps)}
    require(sum(x["selected_step_zero"] for x in stopping.values()) == independent_audit["resource_counts"]["selected_step_zero_neural_flows"], "step zero total matches independent audit")
    require(sum(x["total_executed_steps"] for x in stopping.values()) == independent_audit["resource_counts"]["neural_optimizer_steps_completed"], "step total matches independent audit")

    mechanisms = {}
    by_model_detail = defaultdict(list)
    pooled_head = defaultdict(list)
    pooled_head_ratio = defaultdict(list)
    for job, summary in sorted(summaries.items()):
        model = summary["model"]
        if model not in MECHANISMS:
            continue
        trace_path = trace_dir/(job+"_TRACE.json")
        trace, trace_sha = read(trace_path,"approved_current_full_training_trace")
        require(trace_sha == summary["private_full_trace_sha256"], "full trace matches emitted summary hash")
        receipt = summary["fit_receipt"]
        require([x["step"] for x in trace] == list(range(receipt["steps"]+1)), "trace has every completed step")
        require(receipt["eligible"] and receipt["failure"] is None, "eligible mechanism fit")
        selected_step = receipt["selected_step"]
        selected, terminal = trace[selected_step], trace[-1]
        require(selected["selected"], "stored selected step was a selected checkpoint")
        require(abs(selected["validation_score"]-summary["validation"]["score"])<1e-12,"selected validation metric agreement")
        initial_weight = full_weights(trace[0])
        weights = np.asarray([full_weights(x) for x in trace])
        selected_change = weight_change(initial_weight,full_weights(selected))
        terminal_change = weight_change(initial_weight,full_weights(terminal))
        path_changes = [weight_change(initial_weight,w) for w in weights]
        item = {"job":job,"model":model,"target":summary["target"],"budget":summary["budget"],"seed":summary["seed"],
                "test_score":summary["evaluation"]["score"],"validation_score":summary["validation"]["score"],
                "initial_validation_score":trace[0]["validation_score"],
                "terminal_validation_score":terminal["validation_score"],
                "selected_step":selected_step,"completed_steps":receipt["steps"],"source_names":receipt["source_names"],
                "initial_source_weights":initial_weight[1:].tolist(),
                "selected_weights":selected_change,"terminal_weights":terminal_change,
                "path_max_abs_source_probability_change":float(np.abs(weights[:,1:]-initial_weight[None,1:]).max()),
                "path_max_abs_null_probability_change":float(np.abs(weights[:,0]-initial_weight[0]).max()),
                "selected_prior":prior_snapshot(selected,model!="covariance_profile"),
                "terminal_prior":prior_snapshot(terminal,model!="covariance_profile"),
                "target_gradient_norm":stats(x["target_gradient_norm"] for x in trace[1:]),
                "combined_gradient_norm":stats(x["combined_gradient_norm"] for x in trace[1:]),
                "source_gradient_norm":stats(v for x in trace[1:] for v in x["source_gradient_norms"].values()),
                "meta_gradient_norm_positive_steps":sum(x["meta_gradient_norm"] is not None and x["meta_gradient_norm"]>0 for x in trace[1:]),
                "recorded_meta_gradient_norm":stats(x["meta_gradient_norm"] for x in trace[1:] if x["meta_gradient_norm"] is not None)}
        if "max_abs_centered_logit_change" in selected_change:
            item["path_max_abs_centered_logit_change"] = max(x["max_abs_centered_logit_change"] for x in path_changes)
        if model in ("transport","transport_uniform"):
            prior_rows = [prior_snapshot(x,True) for x in trace]
            item["prior_path"] = {"trace_min":min(x["trace"] for x in prior_rows),
                                  "trace_max":max(x["trace"] for x in prior_rows),
                                  "minimum_eigenvalue":min(x["min_eigenvalue"] for x in prior_rows),
                                  "maximum_eigenvalue":max(x["max_eigenvalue"] for x in prior_rows),
                                  "maximum_spectral_condition":max(x["spectral_condition"] for x in prior_rows),
                                  "transport_relative_residual":stats(x["transport_relative_residual"] for x in trace[1:]),
                                  "old_feature_condition":stats(x["transport_old_feature_condition"] for x in trace[1:])}
        if model == "covariance_profile":
            calibration = receipt["covariance_calibration"]
            df = 8.0/(1.0+calibration["mu"])
            require(abs(df-calibration["initial_df"]) < 1e-12, "covariance effective df algebra")
            item["covariance_df"] = {"initial_recorded_df":calibration["initial_df"],"mu":calibration["mu"],
                                     "analytic_df_at_any_full_rank_step":df,
                                     "scope":"identity for Lambda=mu*C with fixed mu and rank8; no feature matrix was read or recomputed"}
        if model == "diagonal_profile":
            rs = np.asarray([np.zeros(8) if x["step"]==0 else x["diagonal_log_precision"] for x in trace])
            item["diagonal_prior"] = {"selected_r":rs[selected_step].tolist(),"terminal_r":rs[-1].tolist(),
                                     "selected_max_abs_r":float(np.abs(rs[selected_step]).max()),
                                     "path_max_abs_r":float(np.abs(rs).max()),
                                     "path_clamp_coordinate_count":int((np.abs(rs)>=4).sum())}
        if model in ("aux_one","aux_five"):
            residuals = [x["relative_normal_equation_residual"] for x in trace[1:]]
            ratios = [x["new_to_old_max_hessian_ratio"] for x in trace[1:]]
            pooled_head[model].extend(residuals)
            pooled_head_ratio[model].extend(ratios)
            item["head_approximation"] = {"equation_residual":stats(residuals),
                                           "new_to_old_max_hessian_ratio":stats(ratios),
                                           "new_to_old_ratio_at_least_two_steps":sum(x>=2 for x in ratios)}
        mechanisms[job] = item
        by_model_detail[model].append(item)

    family = {}
    for model, rows in sorted(by_model_detail.items()):
        data = {"fits":len(rows),
                "selected_max_abs_source_probability_change":stats(x["selected_weights"]["max_abs_source_probability_change"] for x in rows),
                "terminal_max_abs_source_probability_change":stats(x["terminal_weights"]["max_abs_source_probability_change"] for x in rows),
                "path_max_abs_source_probability_change":stats(x["path_max_abs_source_probability_change"] for x in rows),
                "selected_null_mass":stats(x["selected_weights"]["null_mass"] for x in rows),
                "terminal_null_mass":stats(x["terminal_weights"]["null_mass"] for x in rows),
                "selected_source_mass":stats(x["selected_weights"]["source_mass"] for x in rows),
                "initial_validation_score":stats(x["initial_validation_score"] for x in rows),
                "selected_validation_score":stats(x["validation_score"] for x in rows),
                "terminal_validation_score":stats(x["terminal_validation_score"] for x in rows),
                "meta_gradient_norm_step_maximum":max(x["recorded_meta_gradient_norm"].get("max",0) for x in rows),
                "meta_gradient_norm_positive_steps":sum(x["meta_gradient_norm_positive_steps"] for x in rows),
                "source_gradient_norm_maximum":max(x["source_gradient_norm"]["max"] for x in rows),
                "combined_gradient_norm_maximum":max(x["combined_gradient_norm"]["max"] for x in rows)}
        if "max_abs_centered_logit_change" in rows[0]["selected_weights"]:
            data["selected_max_abs_centered_logit_change"] = stats(x["selected_weights"]["max_abs_centered_logit_change"] for x in rows)
            data["path_max_abs_centered_logit_change"] = stats(x["path_max_abs_centered_logit_change"] for x in rows)
        if model in ("transport","transport_uniform"):
            data["selected_prior_trace_ratio"] = stats(x["selected_prior"]["trace_ratio_to_initial_isotropic"] for x in rows)
            data["terminal_prior_trace_ratio"] = stats(x["terminal_prior"]["trace_ratio_to_initial_isotropic"] for x in rows)
            data["selected_prior_min_eigenvalue"] = stats(x["selected_prior"]["min_eigenvalue"] for x in rows)
            data["selected_prior_max_eigenvalue"] = stats(x["selected_prior"]["max_eigenvalue"] for x in rows)
            data["selected_prior_spectral_condition"] = stats(x["selected_prior"]["spectral_condition"] for x in rows)
            data["selected_prior_trace_above_initial"] = sum(x["selected_prior"]["trace"]>.8+1e-12 for x in rows)
            data["selected_prior_trace_below_initial"] = sum(x["selected_prior"]["trace"]<.8-1e-12 for x in rows)
            data["selected_prior_trace_at_initial"] = sum(abs(x["selected_prior"]["trace"]-.8)<=1e-12 for x in rows)
            data["prior_path_minimum_eigenvalue"] = min(x["prior_path"]["minimum_eigenvalue"] for x in rows)
            data["prior_path_maximum_eigenvalue"] = max(x["prior_path"]["maximum_eigenvalue"] for x in rows)
            data["prior_path_maximum_spectral_condition"] = max(x["prior_path"]["maximum_spectral_condition"] for x in rows)
            data["prior_path_maximum_feature_condition"] = max(x["prior_path"]["old_feature_condition"]["max"] for x in rows)
            data["equal_job_mean_transport_residual"] = stats(x["prior_path"]["transport_relative_residual"]["mean"] for x in rows)
            data["maximum_transport_residual"] = max(x["prior_path"]["transport_relative_residual"]["max"] for x in rows)
        if model == "covariance_profile":
            data["analytic_effective_df"] = stats(x["covariance_df"]["analytic_df_at_any_full_rank_step"] for x in rows)
            data["covariance_mu"] = stats(x["covariance_df"]["mu"] for x in rows)
        if model == "diagonal_profile":
            data["selected_max_abs_r"] = stats(x["diagonal_prior"]["selected_max_abs_r"] for x in rows)
            data["path_max_abs_r"] = stats(x["diagonal_prior"]["path_max_abs_r"] for x in rows)
            data["path_clamp_coordinate_count"] = sum(x["diagonal_prior"]["path_clamp_coordinate_count"] for x in rows)
        if model in pooled_head:
            data["step_weighted_head_equation_residual"] = stats(pooled_head[model])
            data["equal_job_mean_head_equation_residual"] = stats(x["head_approximation"]["equation_residual"]["mean"] for x in rows)
            data["new_to_old_hessian_ratio"] = stats(pooled_head_ratio[model])
            data["hessian_ratio_at_least_two_steps"] = sum(x>=2 for x in pooled_head_ratio[model])
        family[model] = data

    def pair(left,right):
        paired = []
        for r in grouped[left]:
            mate = by_key[(right,r["target"],r["budget"],r["seed"])]
            diff = r["evaluation"]["score"]-mate["evaluation"]["score"]
            hdiff = np.asarray(r["evaluation"]["all20_relative_mse"])-np.asarray(mate["evaluation"]["all20_relative_mse"])
            ldetail, rdetail = mechanisms.get(r["job"]), mechanisms.get(mate["job"])
            paired.append({"target":r["target"],"budget":r["budget"],"seed":r["seed"],"score_difference":diff,
                           "selected_step_equal":bool(ldetail["selected_step"]==rdetail["selected_step"]) if ldetail and rdetail else None,
                           "maximum_all20_horizon_ratio_difference":float(np.abs(hdiff).max())})
        case_groups = defaultdict(list)
        for r in paired:case_groups[(r["target"],r["budget"])].append(r["score_difference"])
        case_diffs = [float(np.mean(v)) for v in case_groups.values()]
        sleft, sright = ranking_by_name[left]["score"], ranking_by_name[right]["score"]
        return {"left":left,"right":right,"left_minus_right_score":sleft-sright,
                "relative_improvement_of_left_percent":100*(sright-sleft)/sright,
                "matched_job_score_difference":stats(x["score_difference"] for x in paired),
                "matched_job_absolute_score_difference":stats(abs(x["score_difference"]) for x in paired),
                "jobs_left_better":sum(x["score_difference"]<0 for x in paired),
                "jobs_left_worse":sum(x["score_difference"]>0 for x in paired),
                "selected_step_equal_jobs":sum(x["selected_step_equal"] is True for x in paired),
                "seed_averaged_target_budget_cases_left_better":sum(x<0 for x in case_diffs),
                "seed_averaged_target_budget_cases_left_worse":sum(x>0 for x in case_diffs),
                "case_count":len(case_diffs),"all20_horizon_ratio_absolute_difference_maximum":max(x["maximum_all20_horizon_ratio_difference"] for x in paired),
                "paired_rows":paired,
                "inference":"descriptive matched differences only; overlapping budgets, horizons, related series and seeds are not independent replication"}
    comparisons = {left+"__vs__"+right:pair(left,right) for left,right in (
        ("transport","profile"),("transport","covariance_profile"),("transport","transport_uniform"),
        ("profile","uniform_profile"),("aux_one","profile"),("aux_five","profile"),
        ("diagonal_profile","profile"))}

    by_target = []
    for target in sorted({s["target"] for s in summaries.values()}):
        row = {"target":target}
        for model in grouped:
            row[model] = float(np.mean([r["evaluation"]["score"] for r in grouped[model] if r["target"]==target]))
        by_target.append(row)
    source_rows = []
    for item in by_model_detail["transport"]:
        for j,name in enumerate(item["source_names"]):
            source_rows.append({"target":item["target"],"source":name,"budget":item["budget"],"seed":item["seed"],
                                "selected_weight":item["selected_weights"]["full_probabilities"][j+1],
                                "initial_weight":item["initial_source_weights"][j],
                                "selected_change":item["selected_weights"]["full_probabilities"][j+1]-item["initial_source_weights"][j]})

    return {"schema":"post-run-mechanism-description-v1","analysis_timing":"post hoc, after complete D1 and independent scoring; no new selection or training",
            "created_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
            "scope":{"public_job_summaries_read":len(summaries),"current_full_mechanism_traces_read":len(mechanisms),
                     "independent_audit_json_files_read":2,"csv_reads":0,"npz_reads":0,"checkpoint_reads":0,
                     "confirmation_reads":0,"market_fits":0,"optimizer_steps":0,"new_prediction_inference":0},
            "input_identity_checks":{"summary_aggregation_max_error_against_independent":largest_metric_difference,
                                      "all_full_trace_hashes_match_summary":True,
                                      "all_trace_steps_complete":True,"source_selection_gates":independent["gates"]},
            "global_scores":overall,"stopping":stopping,"mechanism_families":family,"matched_comparisons":comparisons,
            "by_target_scores":by_target,"transport_source_identity_rows":source_rows,"mechanism_jobs":mechanisms,
            "independent_resource_counts":independent_audit["resource_counts"],
            "limitations":["No new independent prediction scoring: public aggregate identities are compared with the existing independent audit.",
                           "Centered logits are reconstructed from softmax probabilities; absolute logits are not directly read.",
                           "Prior trace/eigenvalues do not identify effective degrees of freedom without the current uncentered feature second-moment matrix; no transport df trajectory is claimed.",
                           "Covariance-control df is an algebraic identity under the full-rank condition, using its recorded fixed mu.",
                           "Selected checkpoint and terminal training iterate are distinct and reported separately.",
                           "These exposed-development associations and ablations do not identify a unique cause of failure; no p-values, bootstrap or causal conclusion is produced."],
            "input_files":accesses}


def write_report(result,path):
    score={x["model"]:x for x in result["global_scores"]}
    family=result["mechanism_families"]
    stopping=result["stopping"]
    c=result["matched_comparisons"]
    tr=family["transport"]
    def f(x):return f"{x:.6g}"
    lines=["# D1 完成后的机制与消融描述", "", "## 1. 结论与证据范围", "",
           "本说明在正式 D1 完成、独立指标审核通过后编写，属于事后描述。它不重新选择模型，不改变停止门槛，不追加训练、预测或超参数搜索。所有输入仅为本轮350份公开 SUMMARY、270份获准读取的机制臂 TRACE 和两个独立审核 JSON；没有读取报价 CSV、NPZ、模型检查点或确认留出。", "",
           f"候选 transport 的主分数为 **{score['transport']['score']:.9f}**；相对 profile 降低 {c['transport__vs__profile']['relative_improvement_of_left_percent']:.4f}%，但相对 covariance_profile 高 {-c['transport__vs__covariance_profile']['relative_improvement_of_left_percent']:.4f}%，相对 RW 高 {(score['transport']['score']-1)*100:.4f}%。原有停止结论 **advance=false** 保持不变。优于固定岭先验这个对照，不能抵消对更直接的未中心化二阶矩对照和 RW 的失败。", "",
           "主分数是五序列×两预算×h=1/5/20 的等权相对 MSE，神经模型先平均17/29/43三个种子的损失。这是已暴露的2022开发区；预算相互嵌套，相关欧盟序列、重叠期限和种子不构成独立重复。下文没有置信区间或显著性结论。来源：[独立选择](../audits/D1_independent/INDEPENDENT_SELECTION.json)、[独立审核](../audits/D1_independent/INDEPENDENT_METRIC_AUDIT.json)。", "",
           "## 2. 原有各臂结果，未作新选择", "",
           "| 方法 | 主分数 | 预算100 | 预算400 | 输出流数 |", "|---|---:|---:|---:|---:|"]
    for name in ("rw","local_profile","uniform_profile","aux_one","aux_five","profile","transport","transport_uniform","covariance_profile","diagonal_profile","local_gru","pooled_gru","ridge"):
        r=score[name];lines.append(f"| {name} | {r['score']:.9f} | {r['budget100']:.9f} | {r['budget400']:.9f} | {r['jobs']} |")
    lines += ["", "这些数值直接继承已验证输出。分析脚本仅将350份摘要等权聚合，与既有独立选择核对，最大差为 "+f(result['input_identity_checks']['summary_aggregation_max_error_against_independent'])+"；没有读取预测数组重新计算市场分数。", "",
              "## 3. 源权重究竟改变了多少", "",
              "每个机制 fit 的源顺序取该 fit 回执。初始空源质量为0.5，其余源平分0.5；比较均相对该 fit 自己的初始值，避免把三个源与四个源的初始权重混在一起。表中每个 fit 先取源坐标的最大绝对概率变化，再汇总30次 fit；数值是概率单位，乘100才是百分点。", "",
              "| 方法 | 选中状态最大源权重变化：中位数 | 选中状态：全体最大 | 整条训练路径：全体最大 | 选中空源质量：均值 |", "|---|---:|---:|---:|---:|"]
    for name in ("aux_one","aux_five","profile","transport","covariance_profile","diagonal_profile","uniform_profile","transport_uniform"):
        r=family[name];lines.append(f"| {name} | {f(r['selected_max_abs_source_probability_change']['median'])} | {f(r['selected_max_abs_source_probability_change']['max'])} | {f(r['path_max_abs_source_probability_change']['max'])} | {r['selected_null_mass']['mean']:.9f} |")
    lines += ["", f"对主候选，选中状态的最大源概率偏离不超过 **{tr['selected_max_abs_source_probability_change']['max']:.9f}**（{100*tr['selected_max_abs_source_probability_change']['max']:.6f}个百分点）；末次迭代的对应上界是 {tr['terminal_max_abs_source_probability_change']['max']:.9f}。选中空源质量范围为 [{tr['selected_null_mass']['min']:.9f}, {tr['selected_null_mass']['max']:.9f}]。这表明实际迁移总量和源间比例都靠近初始设置，但不能据此把非零更新说成未执行。", "",
              f"TRACE 没有直接保存各步绝对 logits。利用softmax概率，分析恢复中心化差值 Δaᵢ=log(wᵢ/wᵢ⁰)−meanⱼ log(wⱼ/wⱼ⁰)，其中包括空源。主候选选中状态的最大绝对中心化logit变化为 {f(tr['selected_max_abs_centered_logit_change']['max'])}，全路径最大为 {f(tr['path_max_abs_centered_logit_change']['max'])}。它是由权重重建的、去除了任意共同平移的量，不是声称读取了未授权检查点。", "",
              f"transport 与 transport_uniform 的主分数差仅 **{c['transport__vs__transport_uniform']['left_minus_right_score']:.12g}**，相对改善 {c['transport__vs__transport_uniform']['relative_improvement_of_left_percent']:.8f}%；两者有 {c['transport__vs__transport_uniform']['selected_step_equal_jobs']}/30 个匹配 fit 选择同一步。30个匹配fit中，单fit主分数绝对差最大为 {f(c['transport__vs__transport_uniform']['matched_job_absolute_score_difference']['max'])}。这组结果不支持把本轮收益主要归于学习到了实质性的源任务取舍；也不能反向确定是哪一个损失尺度或学习率造成变化幅度小。", "",
              "## 4. 运输先验有变化，日志不足以重建其有效自由度", "",
              "transport 初始先验为0.1I，8维迹为0.8。下表区分最终实际用于预测的验证选中状态与训练终止时刻。", "",
              "| 量 | 选中状态：中位数 | 选中状态：最小 | 选中状态：最大 |", "|---|---:|---:|---:|"]
    for key,label in (("selected_prior_trace_ratio","trace(Λ)/0.8"),("selected_prior_min_eigenvalue","最小特征值"),("selected_prior_max_eigenvalue","最大特征值"),("selected_prior_spectral_condition","谱条件数")):
        s=tr[key];lines.append(f"| {label} | {f(s['median'])} | {f(s['min'])} | {f(s['max'])} |")
    lines += ["", f"30个选中状态中，先验迹高于初值 {tr['selected_prior_trace_above_initial']} 个、低于初值 {tr['selected_prior_trace_below_initial']} 个、保持初值 {tr['selected_prior_trace_at_initial']} 个。训练终止时 trace(Λ)/0.8 的中位数为 {f(tr['terminal_prior_trace_ratio']['median'])}，最大为 {f(tr['terminal_prior_trace_ratio']['max'])}。整条候选轨迹记录到的最小先验特征值为 {f(tr['prior_path_minimum_eigenvalue'])}，最大谱条件数为 {f(tr['prior_path_maximum_spectral_condition'])}；所有已读取记录的先验最小特征值为正。这里没有独立重放每步QR或从检查点重算矩阵。", "",
              f"逐步记录的相对表示残差 ||E||F/||X⁺||F，先在每个fit内求均值、再跨fit汇总，其中位数为 {f(tr['equal_job_mean_transport_residual']['median'])}，全体单步最大值为 {f(tr['maximum_transport_residual'])}。非零残差和谱变化表明先验更新并非数值上恒等于保持0.1I；它们不证明新增了可泛化信息。", "",
              "有效自由度需要 tr[C(C+Λ)⁻¹]。本次允许读取的TRACE没有逐步保存C及其与Λ的联合方向，所以仅凭Λ的迹、极端特征值或条件数，不能计算运输的有效自由度轨迹，也不能把迹的上升直接解释为拟合更弱。", "",
              f"未中心化二阶矩对照的固定μ使 Λ=μC；在满秩条件下，有效自由度恒为8/(1+μ)。其30个回执所报告、由该恒等式复核的自由度范围为 [{f(family['covariance_profile']['analytic_effective_df']['min'])}, {f(family['covariance_profile']['analytic_effective_df']['max'])}]，中位数 {f(family['covariance_profile']['analytic_effective_df']['median'])}。该值在各自fit内按定义恒定，来自记录的μ及满秩前提，不是新读取特征数据得到的实测逐步曲线。", "",
              "此前已经证明，该运输是预测空间二次型的继承与正交补惩罚，存在路径依赖，某些回路可改变正则强度。本轮记录支持‘先验几何确实改变’这一弱结论；未中心化二阶矩对照更优，且没有存下可识别复杂度的全部量，因此不能把性能差异归因于确定的正则漂移机制或新增表征价值。", "",
              "## 5. 一步、五步头适配与精确profile", "",
              "aux_one/aux_five从当前精确岭头温启动，在同类虚拟编码器提议上作1/5步头GD；它们是本方匹配近似控制，并非原作者Auto-Lambda的完整联合更新复现。残差定义为 ||(C⁺+0.1I)Wapprox−B⁺||F/||B⁺||F，衡量岭头法方程的满足程度，不等于预测误差。", "",
              "| 近似 | 对profile主分数差 | 同选中步fit数 | 法方程残差：全部步中位数 | 法方程残差：全部步最大 |", "|---|---:|---:|---:|---:|"]
    for name in ("aux_one","aux_five"):
        s=family[name]["step_weighted_head_equation_residual"];p=c[name+"__vs__profile"]
        lines.append(f"| {name} | {p['left_minus_right_score']:.12g} | {p['selected_step_equal_jobs']}/30 | {f(s['median'])} | {f(s['max'])} |")
    lines += ["", "残差表按所有实际更新步汇总，较长轨迹贡献更多观测；JSON同时保存等权fit内均值的分布。两臂与profile独立运行，后续编码器轨迹可以略有不同，因此不把跨臂残差当作同一个矩阵上的逐步收敛证明。", "",
              f"在30个匹配fit中，aux_one/profile的主分数绝对差最大为 {f(c['aux_one__vs__profile']['matched_job_absolute_score_difference']['max'])}，aux_five/profile为 {f(c['aux_five__vs__profile']['matched_job_absolute_score_difference']['max'])}。就这次精确头温启动、固定小提议的设置而言，充分头适配与少步近似没有形成实质预测差距；先前‘错误头可能翻转超梯度’的解析反例并未因此成为本轮成功创新证据。", "",
              f"对角先验控制选中状态的max|r|中位数为 {f(family['diagonal_profile']['selected_max_abs_r']['median'])}，全体训练路径max|r|为 {f(family['diagonal_profile']['path_max_abs_r']['max'])}，触及固定±4边界的坐标记录数为 {family['diagonal_profile']['path_clamp_coordinate_count']}。它的结果并未优于固定profile；这只是该预先固定对照的结果，不能由此推出所有可学习正则方法无效。", "",
              "## 6. 早停与step 0", "",
              "| 方法 | fit数 | 实际更新步中位数 | 选中步中位数 | 选中step0 | 跑到600步 |", "|---|---:|---:|---:|---:|---:|"]
    for name in NEURAL:
        r=stopping[name]
        lines.append(f"| {name} | {r['fits']} | {f(r['executed_steps']['median'])} | {f(r['selected_steps']['median'])} | {r['selected_step_zero']} | {r['executed_steps_at_600']} |")
    lines += ["", f"全部330个神经fit中，有 {sum(r['selected_step_zero'] for r in stopping.values())} 个采用step0检查点；这是验证选择结果，不表示这些fit没有执行后续优化。实际神经更新总数为 {sum(r['total_executed_steps'] for r in stopping.values())}，与既有独立审核一致。MLP的step0已经在随机特征上精确拟合了目标岭头，通常不等于RW；GRU的step0因零输出头才等于RW。二者不能混为‘全都选了随机游走’。", "",
              f"主候选有 {stopping['transport']['selected_step_zero']}/30 个step0选择，其选中验证分数均值为 {tr['selected_validation_score']['mean']:.9f}，2022开发分数为 {score['transport']['score']:.9f}。验证与后续年度表现的差异没有识别出单一原因，不能以它为依据重新设置patience、学习率、正则或留出范围。", "",
              "## 7. 各目标序列，完整呈现异质性", "",
              "以下均在两个预算和三个种子上等权平均；没有选择有利市场/预算作为新主结果。", "",
              "| 目标序列 | profile | transport | transport_uniform | covariance_profile | RW |", "|---|---:|---:|---:|---:|---:|"]
    for row in result["by_target_scores"]:
        lines.append(f"| {row['target']} | {row['profile']:.9f} | {row['transport']:.9f} | {row['transport_uniform']:.9f} | {row['covariance_profile']:.9f} | 1 |")
    lines += ["", "## 8. 可支持与不可支持的创新论断", "",
              "可支持：本轮按固定方案完成了真实、有界的局部头/先验/源权重比较；代码中的运输改变了先验，源权重元更新也实际运行；主候选优于固定各向同性岭先验profile，但没有通过预定未中心化二阶矩和RW门槛。", "",
              "不可支持：已经学会有效识别源任务、先验运输增加了稳定可迁移信息、充分头适配解决了本轮瓶颈，或者这一失败已经被某个确定机制解释。候选与固定源权重运输几乎同分、一步/五步/精确profile近似同分、未中心化二阶矩对照更优，共同限制了正向创新解释。", "",
              "本说明不把这些观察转成新增调参指令。原候选保持STOP，原数据与运行冻结不变，新来源和确认数据不因这份事后分析而开放。逐fit描述、全部输入SHA、重建logits定义及统计分母见 [MECHANISM_ANALYSIS.json](../results/D1/MECHANISM_ANALYSIS.json)；复算入口为 [analyze_mechanism.py](../source/analyze_mechanism.py)。", ""]
    path.write_text("\n".join(lines),encoding="utf-8")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study",type=Path,required=True)
    p.add_argument("--trace-dir",type=Path,required=True)
    p.add_argument("--output-json",type=Path,required=True)
    p.add_argument("--output-md",type=Path,required=True)
    args=p.parse_args()
    if args.output_json.exists() or args.output_md.exists():
        raise FileExistsError("Post-run output paths must both be new")
    result=run(args.study.resolve(),args.trace_dir.resolve())
    result["analysis_source_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output_json.parent.mkdir(parents=True,exist_ok=True)
    args.output_md.parent.mkdir(parents=True,exist_ok=True)
    # Report writer and JSON use only the same in-memory approved evidence.
    write_report(result,args.output_md)
    result["report_sha256"]=hashlib.sha256(args.output_md.read_bytes()).hexdigest()
    with args.output_json.open("x",encoding="utf-8") as handle:
        json.dump(result,handle,ensure_ascii=False,indent=2,allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status":"COMPLETE_POST_HOC_DESCRIPTION","scope":result["scope"],
                      "mechanism_json_sha256":hashlib.sha256(args.output_json.read_bytes()).hexdigest(),
                      "report_sha256":result["report_sha256"],"source_sha256":result["analysis_source_sha256"]},ensure_ascii=False))


if __name__=="__main__":
    main()
