"""Audit the completed SWEWT supplement without fitting or model imports.

Six new NPZ files are rescored independently. Existing V3 descriptive metrics
are reused only after its 39 NPZ byte identities have been checked. No old
bootstrap is rerun and no new confidence intervals or p-values are produced.
"""
from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import independent_v1_statistics as core
import independent_v3_statistics as v3

STUDY = core.STUDY
ARMS = ["swewt_gru_price", "swewt_gru_full"]
PARAMETERS = {"swewt_gru_price": 4308, "swewt_gru_full": 4788}
DISPLAY = ["liquidity_full", "allocation_readout", "allocation_constant",
           "liquidity_readout", "shock_size_only", "liquidity_unconstrained",
           *ARMS, "rw", "gru", "lstm", "ridge_price", "ridge_full",
           "hgb_full", "drift", "validation_selected_strong"]
SUMMARY_KEYS = ["mean_seed_mse_over_rw", "seed_mse_over_rw_by_horizon",
                "rw_mse_by_horizon", "ensemble_mse_over_rw",
                "ensemble_mse_over_rw_by_horizon"]


def execution_details(fd, family, data, ledger):
    """Validate receipt arithmetic; do not replay gradients or unpickle state."""
    start = len(ledger)
    selected_loss, epoch = core.audit_receipts(fd, family, data, ledger)
    for row in ledger[start:]:
        receipt = core.read_json(row["path"])
        core.check(receipt["family"] == family, "receipt family identity")
        count = data["counts"][receipt["mode"]]
        updates = receipt["epochs_executed"] * math.ceil(count / 128)
        positive = receipt["gradient_positive_steps"]
        core.check(isinstance(positive, int) and 0 <= positive <= updates,
                   "logged positive-gradient steps bounded by actual batch count")
        norm = receipt["max_gradient_norm_before_clip"]
        change = receipt["optimization_parameter_change_l2_before_selection"]
        core.check(np.isfinite(norm) and norm >= 0 and np.isfinite(change) and change >= 0,
                   "finite gradient and parameter-change receipts")
        core.check((positive > 0) == (norm > 0), "maximum-gradient/positive-step agreement")
        for step, history in enumerate(receipt["history"], start=1):
            core.check(history["epoch"] == step and np.isfinite(history["train_loss"]),
                       "executed training history identity")
            if receipt["mode"] == "final":
                core.check(history["validation_loss"] is None,
                           "fixed-length refit does not reselect on validation")
        if receipt["mode"] == "train":
            core.close(receipt["initial_validation_loss"], data["validation_rw"],
                       "SWEWT zero-head initial validation is independently reconstructed RW")
            best, stale, stop = receipt["initial_validation_loss"], 0, None
            for step, history in enumerate(receipt["history"], start=1):
                if history["validation_loss"] < best - 1e-7:
                    best, stale = history["validation_loss"], 0
                else:
                    stale += 1
                if step >= 15 and stale >= 12:
                    stop = step
                    break
            core.check(len(receipt["history"]) == (stop if stop is not None else 80),
                       "first allowed early-stop boundary")
        row.update({"market": fd.parent.name.rsplit("_fixed602020", 1)[0],
                    "training_rows": count, "optimizer_steps_from_epochs_and_batches": updates,
                    "max_gradient_norm_before_clip": norm})
    choice = core.read_json(fd / "CHOICE_BEFORE_TEST.json")
    return {"selected_validation_loss": selected_loss, "selected_epoch": epoch,
            "selected_learning_rate": choice["learning_rate"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path(
        "/tmp/carbon-literature-aligned-runs-20261019d/swewt_gru_supplement"))
    parser.add_argument("--reference-v3-run", type=Path, default=Path(
        "/tmp/carbon-literature-aligned-runs-20261019d/new_market_v3"))
    parser.add_argument("--v3-audit", type=Path,
                        default=STUDY / "audits/V3_INDEPENDENT_STATISTICAL_AUDIT_ZH.json")
    parser.add_argument("--data-root", type=Path,
                        default=STUDY / "data/sciencedb_819_v1/prepared")
    parser.add_argument("--ewt-module", type=Path,
                        help="Optional relocated copy of the recorded EWT module, for byte verification only")
    parser.add_argument("--out", type=Path, default=STUDY / "audits")
    args = parser.parse_args()
    for name in ["run", "reference_v3_run", "v3_audit", "data_root", "out"]:
        setattr(args, name, getattr(args, name).resolve())

    # Read this guard before any prediction or model-performance file.
    complete_path = args.run / "EXECUTION_COMPLETE.json"
    core.check(complete_path.is_file(), "supplement must be fully complete before results audit")
    complete = core.read_json(complete_path)
    core.check(complete["status"] == "completed" and complete["jobs"] == 6
               and complete["summary_rows"] == 6 and complete["source_hashes_unchanged"]
               and complete["ewtpy_module_hash_unchanged"], "completed six-job supplement")
    protocol_path = args.run / "PROTOCOL_BEFORE_RUN.json"
    protocol = core.read_json(protocol_path)
    reference_protocol_path = args.reference_v3_run / "PROTOCOL_BEFORE_RUN.json"
    reference = core.read_json(reference_protocol_path)
    reference_complete = core.read_json(args.reference_v3_run / "EXECUTION_COMPLETE.json")
    core.check(reference_complete["status"] == "completed"
               and reference_complete["source_hashes_unchanged"]
               and reference_complete["statistics_plan_hash_unchanged"], "completed frozen V3 reference")
    core.check(protocol["markets"] == v3.MARKETS and protocol["families"] == ARMS,
               "unchanged two arms and three markets")
    core.check(protocol["primary_markets"] == v3.PRIMARY
               and protocol["separate_stress_market"] == reference["separate_stress_market"]
               and protocol["separate_stress_market"].split()[0] == "CHONGQING_MEAN",
               "primary and stress populations")
    core.check(protocol["fold"] == v3.FOLD and protocol["planned_neural_tasks"] == 30,
               "fixed fold and planned task count")
    core.check(protocol["nominal_parameter_counts"] == PARAMETERS, "two model parameter counts")
    core.check(protocol["learning_rates"] == [0.003, 0.01]
               and protocol["selection_seed"] == 17 and protocol["refit_seeds"] == core.SEEDS,
               "fixed search/refit seeds and learning rates")
    core.check(protocol["maximum_epochs"] == 80 and protocol["batch_size"] == 128
               and protocol["minimum_epochs_before_stopping"] == 15 and protocol["patience"] == 12,
               "unchanged training budget")
    core.check(core.sha(reference_protocol_path) == protocol["reference_v3_protocol_sha256"],
               "V3 protocol byte identity")
    core.check(reference["statistics_plan_sha256"] == v3.PLAN_SHA ==
               core.sha(STUDY / "audits/V3_STATISTICAL_AUDIT_PLAN.json"), "original statistics plan unchanged")
    for name, digest in protocol["source_sha256"].items():
        core.check(core.sha(STUDY / "source" / name) == digest, "supplement frozen source identity")
    for name, digest in reference["source_sha256"].items():
        core.check(protocol["source_sha256"][name] == digest, "original V3 source identity")
    module_path = args.ewt_module or Path(protocol["runtime"]["ewtpy_module_path"])
    module_verified = module_path.is_file()
    if module_verified:
        core.check(core.sha(module_path) == protocol["runtime"]["ewtpy_module_sha256"],
                   "EWT module byte identity")
    elif args.ewt_module is not None:
        raise FileNotFoundError(module_path)

    prior = core.read_json(args.v3_audit)
    core.check(prior["status"] == "PASS_INTEGRITY" and prior["frozen_plan_sha256"] == v3.PLAN_SHA,
               "completed independent V3 audit")
    core.check(prior["source_sha256"] == reference["source_sha256"], "audited V3 source identity")
    prior_metrics = {c["market"]: c["metrics"] for c in prior["cell_metrics"]}
    prior_id = {(r["market"], r["family"]): r for r in prior["npz_identity"]}
    core.check(len(prior_id) == 39, "all previously audited reference prediction identities")
    reference_identities = []
    for (market, family), receipt in prior_id.items():
        path = args.reference_v3_run / f"{market}_{v3.FOLD}" / family / "predictions.npz"
        core.check(core.sha(path) == receipt["npz_sha256"], "previously audited V3 NPZ unchanged")
        reference_identities.append({"market": market, "family": family, "path_resolved": str(path),
                                     "npz_sha256": receipt["npz_sha256"]})
    stored = core.read_json(args.run / "ALL_SUMMARIES.json")
    summaries = {(r["market"], r["year"], r["family"]): r for r in stored}
    core.check(len(summaries) == 6 == len(stored), "six unique supplemental summaries")
    core.PARAMETERS.update(PARAMETERS)
    cells, ledger, identities, choices, native = [], [], [], [], []
    for market in v3.MARKETS:
        cd = args.run / f"{market}_{v3.FOLD}"
        manifest_path = cd / "INPUT_MANIFEST.json"
        manifest = core.read_json(manifest_path)
        reference_manifest_path = args.reference_v3_run / f"{market}_{v3.FOLD}" / "INPUT_MANIFEST.json"
        core.check(core.sha(reference_manifest_path) == protocol["reference_v3_manifest_sha256"][market],
                   "original input manifest byte identity")
        original_manifest = core.read_json(reference_manifest_path)
        core.check(manifest["swewt_adaptation"]["original_v3_manifest"] == original_manifest,
                   "nested original scales, masks, source identities and label rules unchanged")
        core.check({k: v for k, v in manifest.items() if k != "swewt_adaptation"} == original_manifest,
                   "top-level original input manifest unchanged")
        data = v3.reconstruct(original_manifest, args.data_root)
        first, a, z, n = original_manifest["row_boundaries"]
        core.check(first == 0, "initial raw boundary")
        origin_all = np.arange(64, n - 20)
        masks = {"train": origin_all + 20 < a, "val": (origin_all >= a) & (origin_all + 20 < z),
                 "final": origin_all + 20 < z, "test": origin_all >= z}
        mask_sha = {key: core.hashlib.sha256(value.tobytes()).hexdigest() for key, value in masks.items()}
        metrics = dict(prior_metrics[market])
        for family in ARMS:
            fd = cd / family
            choice = execution_details(fd, family, data, ledger)
            choices.append({"market": market, "fold": v3.FOLD, "family": family, **choice})
            path = fd / "predictions.npz"
            with np.load(path, allow_pickle=False) as packed:
                core.check(np.array_equal(packed["observed"], data["observed"]), "new NPZ exact raw-price targets")
                core.check(np.array_equal(packed["origins"], data["origins"]), "new NPZ exact V3 origins")
                core.check(np.array_equal(packed["horizons"], core.HORIZONS), "new NPZ H1/H5/H20")
                core.check(np.array_equal(packed["seeds"], core.SEEDS), "new NPZ refit seeds")
                pred = np.array(packed["predictions"], dtype=float)
            core.check(pred.shape == (3, len(data["observed"]), 3), "new NPZ dimensions")
            metric, _ = core.score(pred, data["observed"])
            if choice["selected_epoch"] == 0:
                core.check(np.all(pred == 0), "epoch-zero SWEWT final forecasts exactly RW")
            for saved in [core.read_json(fd / "SUMMARY.json"), summaries[(market, v3.FOLD, family)]]:
                core.check(saved["n_origins"] == len(data["observed"])
                           and saved["parameters"] == PARAMETERS[family], "summary counts and parameter identity")
                core.check(saved["selected_epoch"] == choice["selected_epoch"]
                           and saved["selected_lr"] == choice["selected_learning_rate"], "summary selection identity")
                for key in SUMMARY_KEYS:
                    core.close(metric[key], saved[key], "independent new NPZ summary reconstruction")
            metrics[family] = metric
            identities.append({"market": market, "fold": v3.FOLD, "family": family,
                               "path": str(path), "npz_sha256": core.sha(path),
                               "n_origins": len(data["observed"]),
                               "targets_float64_sha256": core.hashlib.sha256(data["observed"].tobytes()).hexdigest()})
        for family in DISPLAY:
            metric = metrics[family]
            ratio_h = np.mean(np.array(metric["seed_mse_over_rw_by_horizon"]), axis=0)
            for j, horizon in enumerate(core.HORIZONS):
                native.append({"market": market, "fold": v3.FOLD, "family": family,
                               "role": "BJ_TJ_primary" if market in v3.PRIMARY else "CQ_separate_stress",
                               "source": "six_new_npz_recomputed" if family in ARMS else "prior_independent_v3_audit_hash_verified",
                               "horizon": int(horizon), "n_origins": len(data["observed"]),
                               "mean_seed_mse_over_rw": float(ratio_h[j]),
                               "mean_seed_mse": metric["native_mean_seed_mse_by_horizon"][j],
                               "mean_seed_rmse": metric["native_mean_seed_rmse_by_horizon"][j],
                               "rmse_of_mean_seed_mse": metric["native_rmse_of_mean_seed_mse_by_horizon"][j],
                               "mean_seed_mae": metric["native_mean_seed_mae_by_horizon"][j],
                               "ensemble_mse": metric["native_ensemble_mse_by_horizon"][j]})
        cells.append({"market": market, "counts": data["counts"],
                      "test_date_range": [str(data["dates"][0]), str(data["dates"][-1])],
                      "original_manifest_sha256": core.sha(reference_manifest_path),
                      "supplement_manifest_sha256": core.sha(manifest_path),
                      "source_path_resolved": data["source_path_resolved"],
                      "original_masks_reconstructed_sha256": mask_sha,
                      "training_rms": original_manifest["training_rms"],
                      "train_target_second_moment": original_manifest["train_target_second_moment"],
                      "metrics": metrics})
    core.check(len(ledger) == 30 and len(identities) == 6 and len(choices) == 6,
               "all supplemental tasks and outputs audited")
    macro = {}
    for family in DISPLAY:
        selected = [c["metrics"][family] for c in cells if c["market"] in v3.PRIMARY]
        macro[family] = {
            "primary_mean_seed_mse_over_rw": float(np.mean([m["mean_seed_mse_over_rw"] for m in selected])),
            "mean_seed_mse_over_rw_by_horizon": np.mean([
                np.mean(m["seed_mse_over_rw_by_horizon"], axis=0) for m in selected], axis=0).tolist(),
            "secondary_ensemble_mse_over_rw": float(np.mean([m["ensemble_mse_over_rw"] for m in selected]))}
    refits = [r for r in ledger if r["mode"] == "final"]
    counts = {"supplement_neural_tasks": len(ledger), "selection_tasks": len(ledger) - len(refits),
              "refit_tasks": len(refits), "supplement_prediction_files": len(identities),
              "prior_v3_prediction_hashes_verified_only": len(reference_identities),
              "tasks_with_positive_gradient_steps": sum(r["gradient_positive_steps"] > 0 for r in ledger),
              "refits_with_positive_gradient_steps": sum(r["gradient_positive_steps"] > 0 for r in refits),
              "selected_epoch_zero_family_markets": sum(r["selected_epoch"] == 0 for r in choices),
              "zero_epoch_refits": sum(r["epochs_executed"] == 0 for r in refits),
              "total_executed_epochs": sum(r["epochs_executed"] for r in ledger),
              "optimizer_steps_from_epochs_and_batches": sum(r["optimizer_steps_from_epochs_and_batches"] for r in ledger),
              "total_logged_positive_gradient_steps": sum(r["gradient_positive_steps"] for r in ledger),
              "primary_market_origins": 188, "stress_market_origins": 28,
              "primary_origin_horizon_targets": 564, "stress_origin_horizon_targets": 84,
              "new_fits_by_auditor": 0, "checkpoint_unpickles_by_auditor": 0,
              "new_bootstrap_or_significance_tests": 0}
    result = {"title": "SWEWT-GRU 补充对照独立结果审查", "status": "PASS_INTEGRITY_DESCRIPTIVE_ONLY",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "stage": protocol["stage"], "registration_status": protocol["registration_status"],
              "scope": "Six new NPZ files independently rescored; 30 new training receipts/checkpoint byte hashes audited; prior V3 metrics reused only after 39 prediction byte identities checked. No new model or significance tests.",
              "script_sha256": core.sha(__file__),
              "utility_script_sha256": {Path(core.__file__).name: core.sha(core.__file__),
                                        Path(v3.__file__).name: core.sha(v3.__file__)},
              "resolved_arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "completion_marker_sha256": core.sha(complete_path), "protocol_sha256": core.sha(protocol_path),
              "reference_protocol_sha256": core.sha(reference_protocol_path),
              "reference_v3_audit_sha256": core.sha(args.v3_audit), "source_sha256": protocol["source_sha256"],
              "frozen_v3_statistics_plan_sha256_unchanged": v3.PLAN_SHA,
              "runtime_ewt_module": {"recorded_sha256": protocol["runtime"]["ewtpy_module_sha256"],
                                     "resolved_path": str(module_path), "current_bytes_verified": module_verified,
                                     "note": "Module not imported. Its presence is optional for arithmetic-only audit of archived predictions."},
              "execution_counts": counts, "primary_macro_scores_bj_tj": macro,
              "cell_metrics": cells, "selected_epochs": choices, "training_receipt_ledger": ledger,
              "new_npz_identity": identities, "reference_v3_npz_byte_identity": reference_identities,
              "reference_validation_selected_strong": prior["validation_selected_strong"],
              "limitations": [
                  "Supplement added after partial V3 outcomes; both arms are descriptive and remain separate from the frozen six-comparison V3 family.",
                  "Original primary candidate remains liquidity_full; allocation_readout remains the original secondary candidate.",
                  "BJ/TJ use equal market and horizon weights; CQ remains a separate 28-origin stress test.",
                  "Primary score averages seed squared errors, not seed predictions. Ensemble scores are secondary.",
                  "Gradient counts are hash-matched execution-log counts checked against batch/epoch bounds, not an independent backward-pass replay.",
                  "Validation selection is reconstructed from complete saved traces and initial RW loss; trained checkpoint validation forecasts are not replayed.",
                  "Checkpoint files are verified only as bytes and are never unpickled.",
                  "Future-information qualification is from the separate source/synthetic preflight; this result audit does not reconstruct or run EWT features.",
                  "Controlled multi-step adaptation of published SWEWT-GRU, not author-code or numerical reproduction; fixed EWT/GRU/budget and 17 auxiliary tokens differ from the source study and V3 GRU history.",
                  "Author-cleaned transaction-mean target and non-vintage historical exogenous-data limitations remain.",
                  "No new confidence intervals, p-values, test-based selection between supplement arms, or alteration of V3 planned comparisons."
              ]}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "SWEWT_INDEPENDENT_RESULT_AUDIT_ZH.json").write_text(core.json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    with (args.out / "SWEWT_NATIVE_METRICS.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(native[0]))
        writer.writeheader()
        writer.writerows(native)
    lines = ["# SWEWT-GRU 补充对照：独立结果审查", "",
             "**完整性 PASS；全部结果只作描述性补充。** 在 EXECUTION_COMPLETE 出现后，从六份新 NPZ 逐行复算误差，核对三十项训练任务和 checkpoint 字节身份。没有新增拟合、checkpoint 反序列化、bootstrap 或显著性检验。", "",
             "这两个 SWEWT-GRU 臂在部分 V3 北京成绩被查看后增加。原主候选仍为 liquidity_full，原次候选仍为 allocation_readout；原 V3 六项统计比较家族及冻结计划未改动。", "",
             "## 同一目标上的描述性并列", "",
             "主指标先在每个市场/步长平均三个种子的平方误差，再除以同一目标上的 RW MSE。北京 116 个、天津 72 个共同 H1/H5/H20 原点等权汇总，重庆 28 个原点单列；数值越小越好。旧对照指标沿用已完成的独立 V3 NPZ 复算，并重新核对其全部 39 份预测文件字节哈希。没有重跑旧统计区间。", "",
             "| 模型 | 北京 | 天津 | BJ/TJ 等权主指标 | 重庆压力测试 |", "|---|---:|---:|---:|---:|"]
    for family in DISPLAY:
        values = [c["metrics"][family]["mean_seed_mse_over_rw"] for c in cells]
        lines.append(f"| {family} | {values[0]:.9f} | {values[1]:.9f} | {macro[family]['primary_mean_seed_mse_over_rw']:.9f} | {values[2]:.9f} |")
    lines += ["", "## 新增两臂逐市场、逐步长", "",
              "| 市场 | 对照臂 | H1 MSE/RW | H5 MSE/RW | H20 MSE/RW | 三种子集成 MSE/RW（次要） |", "|---|---|---:|---:|---:|---:|"]
    for cell in cells:
        for family in ARMS:
            metric = cell["metrics"][family]
            h = np.mean(metric["seed_mse_over_rw_by_horizon"], axis=0)
            lines.append(f"| {cell['market']} | {family} | {h[0]:.9f} | {h[1]:.9f} | {h[2]:.9f} | {metric['ensemble_mse_over_rw']:.9f} |")
    lines += ["", "## 验证选择与执行核数", "",
              "| 市场 | 对照臂 | 验证选择 LR | 验证选择 epoch | 验证损失 |", "|---|---|---:|---:|---:|"]
    for row in choices:
        lines.append(f"| {row['market']} | {row['family']} | {row['selected_learning_rate']} | {row['selected_epoch']} | {row['selected_validation_loss']:.9f} |")
    lines += ["",
              f"共 {counts['supplement_neural_tasks']} 项任务：{counts['selection_tasks']} 次学习率/epoch 选择、{counts['refit_tasks']} 次固定 epoch 三种子重训；其中 {counts['tasks_with_positive_gradient_steps']} 项记录正梯度步骤，最终重训中 {counts['refits_with_positive_gradient_steps']} 项有正梯度。{counts['selected_epoch_zero_family_markets']} 个市场×对照臂选择 epoch 0，对应 {counts['zero_epoch_refits']} 份零 epoch 重训；这些输出已逐点确认等于 RW。",
              "",
              f"训练收据共记录 {counts['total_executed_epochs']} 个实际 epoch、{counts['total_logged_positive_gradient_steps']} 个正梯度步骤；按原 batch=128 和每个任务实际样本数/epoch 独立核算 {counts['optimizer_steps_from_epochs_and_batches']} 次优化器调用。正梯度次数来自哈希匹配执行收据并通过非负性、范数一致性和实际 batch 上限检查；不把它称为独立反向传播回放。",
              "",
              "已从完整验证 trace 重演 1e-7 改善阈值、最早停止条件与两学习率选择，确认 seed 17 选择、17/29/43 固定 epoch 从头重训；初始零输出头的验证损失独立重算为 RW。所有 checkpoint 只核验字节 SHA，不加载参数。训练后的 checkpoint 验证预测未在此脚本重放。",
              "",
              "## 身份、信息与解释边界", "",
              "补充 manifest 的原 V3 字段及嵌套原 manifest 完全一致。独立按源 CSV 重建固定 60/20/20 边界、H20 标签成熟掩码、训练 RMS 与各步长训练二阶矩；六个 NPZ 的目标、原点、步长、种子均一致。全部原/新源码和现存 EWT 模块字节已按冻结记录核验。EWT 输入特征的信息时点由单列的静态/合成预检核实，本次结果审计没有重新生成分解特征。",
              "",
              "这是一项受控多步改编，固定 EWT 参数、GRU32、零初始多步输出头、RMS 和预算均已与原论文区分。full 仅取末 17 个辅助 token，短于原 V3 GRU/LSTM 的 64 个；作者清洗成交均价与历史非 vintage 外生变量限制继续适用。两个新臂都完整报告，不按测试成绩选择保留其中一臂。",
              "",
              "逐市场/步长原单位 MSE、两种明确定义的 RMSE、MAE 与集成误差在 SWEWT_NATIVE_METRICS.csv。JSON 保留三种子逐步长风险、全部训练收据与预测身份、参考审计哈希、实际 CLI 路径和审计脚本 SHA。", ""]
    (args.out / "SWEWT_INDEPENDENT_RESULT_AUDIT_ZH.md").write_text("\n".join(lines))
    print(core.json.dumps({"status": result["status"], "counts": counts,
                           "macro_bj_tj": macro, "choices": choices}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
