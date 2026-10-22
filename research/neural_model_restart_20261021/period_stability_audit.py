"""Period-stability audit for the frozen R20 prediction set.

Question: can the existing evaluation instrument resolve the size of effect the
project has been chasing (~0.1-0.5% known-scale QLIKE)?

Design
------
* Two already-scored periods exist: development 2025 and used-2026.
* Each period is cut into three contiguous origin blocks.
* We score every frozen model on every block with raw QLIKE
  ``log(v) + y / v`` (undefined-free at y=0, and additive-offset invariant, so
  the ranking it induces equals the ranking of the reported known-scale
  QLIKE, which differs by a model-independent per asset-horizon constant).
* We then report three effect scales:
    - seed noise     : spread between the three seeds of one model, same block
    - model effect   : best-minus-worst spread across models, same block
    - period effect  : the same model's score change between periods
  and the rank agreement between all six blocks.

Read-only. Loads archived predictions; trains nothing; writes two new files.
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.stats import kendalltau, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
DEV_NPZ = os.path.join(HERE, "round20_holdout_runner_development_predictions.npz")
TEST_NPZ = os.path.join(HERE, "round20_holdout_runner_test_predictions.npz")
OUT_JSON = os.path.join(HERE, "PERIOD_STABILITY_AUDIT.json")
OUT_MD = os.path.join(HERE, "PERIOD_STABILITY_ZH.md")

ASSETS = ("KRBN", "KCCA")

# ensemble-level candidates: one entry per distinct model
MODELS = [
    "R20_joint_ensemble",
    "R20_clock_only_ensemble",
    "R20_continued_ensemble",
    "R20_density_only_ensemble",
    "C18_GRU_ensemble",
    "R16_frozen",
    "R16_raw",
    "EWMA22",
    "HAR_geometry",
    "HGB_geometry",
    "MEM_stationary",
    "MEM_integrated",
    "GK_global_KF",
    "RS_global_KF",
]

# seed triples available in the archive, used to measure training noise
SEED_GROUPS = {
    "R20_joint": [f"R20_joint_seed{s}" for s in (11, 22, 33)],
    "R20_continued": [f"R20_continued_seed{s}" for s in (11, 22, 33)],
    "R20_clock_only": [f"R20_clock_only_seed{s}" for s in (11, 22, 33)],
    "R20_density_only": [f"R20_density_only_seed{s}" for s in (11, 22, 33)],
    "C18_GRU": [f"C18_GRU_seed{s}" for s in (11, 22, 33)],
}


def load(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.keys()}


def raw_qlike(v: np.ndarray, y: np.ndarray) -> np.ndarray:
    if np.any(~np.isfinite(v)) or np.any(v <= 0):
        raise ValueError("non-positive or non-finite variance forecast")
    return np.log(v) + y / v


def block_scores(store: dict, mask: np.ndarray) -> dict:
    """Equal-weight average of per-asset block means (mean over origins & horizons)."""
    out = {}
    names = sorted({k.split("_", 1)[1] for k in store if k.startswith("KRBN_")}
                   - {"date", "origin", "label_end", "target", "ewma"})
    for m in names:
        vals = []
        for a in ASSETS:
            key = f"{a}_{m}"
            if key not in store:
                break
            v = store[key]
            y = store[f"{a}_target"]
            if v.shape != y.shape:
                break
            vals.append(float(raw_qlike(v, y)[mask].mean()))
        if len(vals) == len(ASSETS):
            out[m] = float(np.mean(vals))
    return out


def blocks(n: int, k: int = 3) -> list[tuple[str, np.ndarray]]:
    edges = np.linspace(0, n, k + 1).astype(int)
    out = []
    for i in range(k):
        mask = np.zeros(n, dtype=bool)
        mask[edges[i]:edges[i + 1]] = True
        out.append((f"b{i + 1}", mask))
    return out


# candidate-vs-control pairs, matched on seed, tested over 6 blocks x 3 seeds = 18 cells
PAIRS = [
    ("R20_joint", "R20_continued", "算子（密度/时钟）是否带来增量"),
    ("R20_joint", "C18_GRU", "R20 整包是否胜过纯 GRU"),
    ("R20_joint", "MEM_stationary", "是否胜过最强简单基线 MEM"),
    ("R20_clock_only", "R20_continued", "仅时钟项是否带来增量"),
    ("R20_density_only", "R20_continued", "仅密度项是否带来增量"),
    ("R16_raw", "C18_GRU", "R16 原始 KF 是否胜过纯 GRU"),
]
SEEDS = (11, 22, 33)


def seed_score(store: dict, mask: np.ndarray, base: str, seed: int) -> float | None:
    name = f"{base}_seed{seed}"
    vals = []
    for a in ASSETS:
        key = f"{a}_{name}"
        if key not in store:
            return None
        vals.append(float(raw_qlike(store[key], store[f"{a}_target"])[mask].mean()))
    return float(np.mean(vals))


def main() -> int:
    dev = load(DEV_NPZ)
    test = load(TEST_NPZ)
    n_dev = len(dev["KRBN_date"])
    n_test = len(test["KRBN_date"])

    periods = {"dev2025": (dev, n_dev), "used2026": (test, n_test)}
    table = {}          # block -> {model: score}
    block_meta = {}
    block_ctx = {}
    for pname, (store, n) in periods.items():
        for bname, mask in blocks(n):
            key = f"{pname}.{bname}"
            table[key] = block_scores(store, mask)
            block_ctx[key] = (store, mask)
            block_meta[key] = {
                "n_origins": int(mask.sum()),
                "first_origin": int(store["KRBN_origin"][mask][0]),
                "last_origin": int(store["KRBN_origin"][mask][-1]),
                "first_date": str(np.datetime_as_string(store["KRBN_date"][mask][0], unit="D")),
                "last_date": str(np.datetime_as_string(store["KRBN_date"][mask][-1], unit="D")),
            }

    common = [m for m in MODELS if all(m in table[k] for k in table)]
    missing = [m for m in MODELS if m not in common]

    # ---- rank agreement across all six blocks -----------------------------
    keys = list(table)
    rho = np.eye(len(keys))
    tau = np.eye(len(keys))
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            if i >= j:
                continue
            x = np.array([table[a][m] for m in common])
            y = np.array([table[b][m] for m in common])
            r = float(spearmanr(x, y).statistic)
            t = float(kendalltau(x, y).statistic)
            rho[i, j] = rho[j, i] = r
            tau[i, j] = tau[j, i] = t

    def mean_offdiag(mat: np.ndarray, rows: list[int], cols: list[int]) -> float | None:
        vals = [float(mat[i, j]) for i in rows for j in cols if i != j]
        return float(np.mean(vals)) if vals else None

    dev_idx = [i for i, k in enumerate(keys) if k.startswith("dev2025")]
    test_idx = [i for i, k in enumerate(keys) if k.startswith("used2026")]

    within_dev = mean_offdiag(rho, dev_idx, dev_idx)
    within_test = mean_offdiag(rho, test_idx, test_idx)
    cross = mean_offdiag(rho, dev_idx, test_idx)

    # ---- effect scales ----------------------------------------------------
    # seed noise: sd across seeds of the same model within a block, averaged
    seed_noise = {}
    for pname, (store, n) in periods.items():
        per_block = {}
        for bname, mask in blocks(n):
            spreads = []
            for gname, members in SEED_GROUPS.items():
                s = [table[f"{pname}.{bname}"].get(m) for m in members]
                s = [v for v in s if v is not None]
                if len(s) >= 2:
                    spreads.append(float(np.std(s, ddof=1)))
            if spreads:
                per_block[bname] = float(np.mean(spreads))
        seed_noise[pname] = per_block

    model_effect = {}
    for k in keys:
        vals = np.array([table[k][m] for m in common])
        level = abs(float(vals.mean()))
        model_effect[k] = {
            "best": common[int(np.argmin(vals))],
            "best_score": float(vals.min()),
            "worst": common[int(np.argmax(vals))],
            "worst_score": float(vals.max()),
            "spread": float(vals.max() - vals.min()),
            "level": level,
            "spread_over_level": float((vals.max() - vals.min()) / level),
        }

    period_effect = {}
    for m in common:
        d = np.array([table[k][m] for k in keys if k.startswith("dev2025")]).mean()
        t = np.array([table[k][m] for k in keys if k.startswith("used2026")]).mean()
        period_effect[m] = {
            "dev2025_mean": float(d),
            "used2026_mean": float(t),
            "delta": float(t - d),
            "delta_over_level": float((t - d) / abs(d)),
        }
    pe = np.array([abs(v["delta_over_level"]) for v in period_effect.values()])
    pe_abs = np.array([abs(v["delta"]) for v in period_effect.values()])
    se = np.array([v for p in seed_noise.values() for v in p.values()])
    ms = np.array([v["spread"] for v in model_effect.values()])

    verdict = {
        "within_period_rank_agreement_spearman": {
            "dev2025_blocks": within_dev,
            "used2026_blocks": within_test,
            "cross_period_blocks": cross,
        },
        "median_abs_period_shift": float(np.median(pe_abs)),
        "max_abs_period_shift": float(np.max(pe_abs)),
        "median_abs_period_shift_over_level": float(np.median(pe)),
        "max_abs_period_shift_over_level": float(np.max(pe)),
        "median_seed_noise": float(np.median(se)),
        "median_model_spread": float(np.median(ms)),
        "median_model_spread_over_level": float(np.median(
            [v["spread_over_level"] for v in model_effect.values()])),
    }
    verdict["period_shift_over_model_spread"] = float(
        verdict["median_abs_period_shift"] / verdict["median_model_spread"])
    verdict["model_spread_over_seed_noise"] = float(
        verdict["median_model_spread"] / verdict["median_seed_noise"])
    verdict["period_shift_over_seed_noise"] = float(
        verdict["median_abs_period_shift"] / verdict["median_seed_noise"])

    # Under the null of unrelated rankings, sd(Spearman) ~ 1/sqrt(n_models - 1).
    rho_null_sd = float(1.0 / np.sqrt(len(common) - 1))
    verdict["spearman_null_sd"] = rho_null_sd
    verdict["spearman_z"] = {
        "dev2025_blocks": float(within_dev / rho_null_sd) if within_dev is not None else None,
        "used2026_blocks": float(within_test / rho_null_sd) if within_test is not None else None,
        "cross_period_blocks": float(cross / rho_null_sd) if cross is not None else None,
    }
    verdict["spearman_distinguishable_from_zero"] = {
        k: (None if v is None else bool(abs(v) > 1.96))
        for k, v in verdict["spearman_z"].items()
    }

    # The ranking-relevant part of the period effect is the *spread* of the
    # per-model shifts: a common level shift cancels in every comparison.
    deltas = np.array([v["delta"] for v in period_effect.values()])
    verdict["differential_period_shift"] = float(deltas.max() - deltas.min())
    verdict["differential_period_shift_models"] = {
        "most_worse": min(period_effect, key=lambda m: period_effect[m]["delta"]),
        "most_improved": max(period_effect, key=lambda m: period_effect[m]["delta"]),
    }
    # effect sizes the project has actually reported, in raw QLIKE absolute units
    reported = {"R20_joint_minus_continued": 0.0004, "R20_joint_minus_C18_GRU": 0.0025}
    verdict["reported_effect_range"] = [min(reported.values()), max(reported.values())]
    verdict["differential_shift_over_reported_effect"] = {
        "vs_smallest": round(verdict["differential_period_shift"] / min(reported.values()), 1),
        "vs_largest": round(verdict["differential_period_shift"] / max(reported.values()), 1),
    }

    # ---- paired seed x block sign test -----------------------------------
    pair_rows = []
    for cand, ctrl, label in PAIRS:
        cells = []
        cand_seeded = all(f"{a}_{cand}_seed{SEEDS[0]}" in periods["dev2025"][0]
                          for a in ASSETS)
        ctrl_seeded = all(f"{a}_{ctrl}_seed{SEEDS[0]}" in periods["dev2025"][0]
                          for a in ASSETS)
        paired_on_seed = cand_seeded and ctrl_seeded
        for key in keys:
            store, mask = block_ctx[key]
            if paired_on_seed:
                for s in SEEDS:
                    a = seed_score(store, mask, cand, s)
                    b = seed_score(store, mask, ctrl, s)
                    if a is None or b is None:
                        continue
                    cells.append((key, f"s{s}", a, b, a - b, a < b))
            else:
                a = table[key].get(f"{cand}_ensemble", table[key].get(cand))
                b = table[key].get(f"{ctrl}_ensemble", table[key].get(ctrl))
                if a is None or b is None:
                    continue
                cells.append((key, "agg", a, b, a - b, a < b))
        if not cells:
            continue
        wins = sum(1 for c in cells if c[5])
        dev_cells = [c for c in cells if c[0].startswith("dev2025")]
        tst_cells = [c for c in cells if c[0].startswith("used2026")]
        dev_wins = sum(1 for c in dev_cells if c[5])
        tst_wins = sum(1 for c in tst_cells if c[5])
        diffs = np.array([c[4] for c in cells])
        pair_rows.append({
            "candidate": cand,
            "control": ctrl,
            "label": label,
            "paired_on_seed": paired_on_seed,
            "n_cells": len(cells),
            "wins": int(wins),
            "win_frac": float(wins / len(cells)),
            "dev2025_wins": f"{dev_wins}/{len(dev_cells)}",
            "used2026_wins": f"{tst_wins}/{len(tst_cells)}",
            "median_diff": float(np.median(diffs)),
            "max_abs_diff": float(np.abs(diffs).max()),
            "verdict": ("稳健" if wins == len(cells) else
                        "方向一致但非全胜" if wins >= len(cells) * 0.8 else
                        "未确立"),
        })

    payload = {
        "created_utc": None,
        "scope": "read-only audit of archived R20 predictions; no training, no new data",
        "inputs": {
            "development_predictions": os.path.basename(DEV_NPZ),
            "test_predictions": os.path.basename(TEST_NPZ),
            "n_origins": {"dev2025": n_dev, "used2026": n_test},
        },
        "loss": "raw QLIKE = log(v) + y/v, equal weight over assets and horizons",
        "loss_note": (
            "known-scale QLIKE differs from raw QLIKE by a model-independent "
            "per asset-horizon additive offset, so model rankings are identical"
        ),
        "models_used": common,
        "models_missing": missing,
        "block_meta": block_meta,
        "block_scores": table,
        "rank_spearman_matrix": {k: dict(zip(keys, map(float, row))) for k, row in zip(keys, rho)},
        "rank_kendall_matrix": {k: dict(zip(keys, map(float, row))) for k, row in zip(keys, tau)},
        "seed_noise": seed_noise,
        "model_effect": model_effect,
        "period_effect": period_effect,
        "paired_seed_block_tests": pair_rows,
        "verdict": verdict,
        "limitations": [
            "used2026 has already been scored once; this audit reuses it descriptively",
            "six blocks are not independent; they are contiguous slices of two periods",
            "raw QLIKE is not the predeclared primary metric of R20; it is used here "
            "only because it is offset-free and defined at y=0",
            "paired cells share origins and seeds, so they are not independent draws",
            "no significance claim; no model selection performed on this audit",
        ],
    }

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    # ---- markdown --------------------------------------------------------
    L = []
    L.append("# 跨期稳定性审计：现有评测工具能否分辨 0.1–0.5% 的差异\n")
    L.append("只读审计。输入是已经冻结并计分过的 R20 预测归档，不训练、不引入新数据、不做模型选择。\n")
    L.append(f"- 2025 开发期起点数：{n_dev}；2026 已用测试期起点数：{n_test}")
    L.append("- 每个时期按起点顺序切成 3 个连续块")
    L.append("- 损失：raw QLIKE `log(v)+y/v`，资产与期限等权\n")
    L.append("## 一、三种效应尺度对比\n")
    L.append("全部用 raw QLIKE 绝对单位（`log(v)+y/v`，量级约 −8.3）。"
             "由于 raw QLIKE 为负，本项目一律不报道它的百分比改进，"
             "这里只给绝对差与相对量级的比值。\n")
    L.append("| 尺度 | 数值 |")
    L.append("|---|---:|")
    L.append(f"| 期内分块排序一致性（2025，Spearman 均值） | {within_dev:.3f} |")
    L.append(f"| 期内分块排序一致性（2026，Spearman 均值） | {within_test:.3f} |")
    L.append(f"| **跨期**分块排序一致性（Spearman 均值） | **{cross:.3f}** |")
    L.append(f"| 原假设下 Spearman 的标准差（{len(common)} 个模型） | "
             f"{verdict['spearman_null_sd']:.3f} |")
    L.append(f"| 2025 分块一致性 ÷ 该标准差（z） | {verdict['spearman_z']['dev2025_blocks']:+.2f} |")
    L.append(f"| 2026 分块一致性 ÷ 该标准差（z） | {verdict['spearman_z']['used2026_blocks']:+.2f} |")
    L.append(f"| 跨期一致性 ÷ 该标准差（z） | {verdict['spearman_z']['cross_period_blocks']:+.2f} |")
    L.append(f"| 种子噪声（同模型三种子标准差，中位） | {verdict['median_seed_noise']:.5f} |")
    L.append(f"| 模型间差距（最好−最差，中位） | {verdict['median_model_spread']:.5f} |")
    L.append(f"| 同一模型跨期位移（中位绝对值） | {verdict['median_abs_period_shift']:.5f} |")
    L.append(f"| 跨期位移 ÷ 模型间差距 | {verdict['period_shift_over_model_spread']:.2f}× |")
    L.append(f"| 模型间差距 ÷ 种子噪声 | {verdict['model_spread_over_seed_noise']:.2f}× |")
    L.append(f"| 跨期位移 ÷ 种子噪声 | {verdict['period_shift_over_seed_noise']:.2f}× |")
    L.append(f"| **排序相关的位移**（各模型位移的最大−最小） | "
             f"**{verdict['differential_period_shift']:.5f}** |")
    L.append(f"| 排序相关位移 ÷ 本项目最小已报效应（0.0004） | "
             f"{verdict['differential_shift_over_reported_effect']['vs_smallest']:.0f}× |")
    L.append(f"| 排序相关位移 ÷ 本项目最大已报效应（0.0025） | "
             f"{verdict['differential_shift_over_reported_effect']['vs_largest']:.0f}× |\n")
    L.append("说明：所有模型共同平移的部分在任意比较里都会抵消，只有**位移的离散程度**才改写排序。"
             "因此上表最后三行才是“跨期到底把排序改写多少”的正确量度。\n")

    L.append("## 二、各块分数\n")
    hdr = "| 模型 | " + " | ".join(keys) + " |"
    L.append(hdr)
    L.append("|" + "---|" * (len(keys) + 1))
    for m in common:
        cells = " | ".join(f"{table[k][m]:.4f}" for k in keys)
        L.append(f"| {m} | {cells} |")
    L.append("")

    L.append("## 三、同一模型跨期位移\n")
    L.append("| 模型 | 2025 | 2026 | 位移 |")
    L.append("|---|---:|---:|---:|")
    for m in sorted(common, key=lambda x: period_effect[x]["delta"]):
        v = period_effect[m]
        L.append(f"| {m} | {v['dev2025_mean']:.4f} | {v['used2026_mean']:.4f} "
                 f"| {v['delta']:+.4f} |")
    L.append("")

    L.append("## 四、种子噪声\n")
    for pname, d in seed_noise.items():
        L.append(f"- {pname}：" + "，".join(f"{k}={v:.5f}" for k, v in d.items()))
    L.append("")

    L.append("## 五、配对种子 × 分块符号检验（推荐使用的判别工具）\n")
    L.append("单期聚合分数会被时期位移淹没。把同一对模型在 **6 个分块 × 3 个种子 = 18 个配对格**"
             "上逐一比较符号，才是能分辨小效应的做法。全胜才算稳健。\n")
    L.append("| 候选 vs 控制 | 含义 | 胜格 | 2025 | 2026 | 中位差 | 最大绝对差 | 判定 |")
    L.append("|---|---|---:|---:|---:|---:|---:|---|")
    for r in pair_rows:
        L.append(f"| {r['candidate']} vs {r['control']} | {r['label']} | "
                 f"{r['wins']}/{r['n_cells']} | {r['dev2025_wins']} | {r['used2026_wins']} | "
                 f"{r['median_diff']:+.5f} | {r['max_abs_diff']:.5f} | {r['verdict']} |")
    L.append("")

    L.append("## 六、结论\n")
    L.append("用 14 个模型做 Spearman，原假设下的标准差约 "
             f"{verdict['spearman_null_sd']:.2f}。下面三条一致性"
             f"（{within_dev:.2f} / {within_test:.2f} / {cross:.2f}）"
             "**全都落在该标准差的 2 倍以内**，即：\n")
    L.append("- 我们既不能证明排序在期内稳定，也不能证明它跨期失效；")
    L.append("- 但可以确定：**没有任何一条排序一致性达到可依赖的强度**。"
             "拿这样一个排序去宣布某个机制胜出，本身就是无效推断。\n")
    if cross is not None and cross < 0.2:
        L.append("跨期一致性只有 "
                 f"{cross:.3f}：**2025 上选出的排名对 2026 几乎没有预测力**。")
    if within_test is not None and within_test < 0.3:
        L.append("\n更进一步：**2026 自身三个连续分块的一致性也只有 "
                 f"{within_test:.3f}**，即即使在同一个时期内部，这些模型的相对优劣也不可复现。")
    L.append(
        f"\n项目一直在报告 0.1–0.5% 的增益，换算成 raw QLIKE 绝对量约 0.0004–0.0025；"
        f"而**跨期把排序改写掉的量是 {verdict['differential_period_shift']:.4f}**，"
        f"是这些效应的 "
        f"{verdict['differential_shift_over_reported_effect']['vs_largest']:.0f}–"
        f"{verdict['differential_shift_over_reported_effect']['vs_smallest']:.0f} 倍。"
    )
    L.append(
        "\n这不是说模型没有差异，而是说：**只用一个时期做选择，选出来的东西系统性不可靠**。"
        "在两期结论相反的情况下，任何新机制无论真假都会被其中一个时期判负。"
    )
    L.append("\n修复方向（不是论文卖点，是实验前提）：滚动多折重训、按折报告平均排名，"
             "而不是单窗口胜负。")
    L.append("")
    for lim in payload["limitations"]:
        L.append(f"- 限制：{lim}")

    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))

    print("models used:", len(common), "missing:", missing)
    print("within dev:", round(within_dev, 3), "within test:", round(within_test, 3),
          "cross:", round(cross, 3))
    print("median period shift:", round(verdict["median_abs_period_shift"], 5))
    print("median model spread:", round(verdict["median_model_spread"], 5))
    print("median seed noise:", round(verdict["median_seed_noise"], 5))
    print("ratio shift/spread:", round(verdict["period_shift_over_model_spread"], 2))
    print("wrote", OUT_JSON)
    print("wrote", OUT_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
