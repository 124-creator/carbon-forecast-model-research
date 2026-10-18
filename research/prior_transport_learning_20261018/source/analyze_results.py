"""Post-run descriptive tables and scientific figures; no fitting or new selection.

Reads only this study's public summaries and independently recomputed cell table.
The pre-fit protocol and its gate remain authoritative. Seed spread describes
initialization sensitivity on the same quotes, not sampling uncertainty.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics as st
from collections import defaultdict
from pathlib import Path


def dump_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(study: Path, plots: bool = True):
    protocol = json.loads((study / "provenance/PROTOCOL.json").read_text())
    results = study / "results/D1"
    rows = [json.loads(p.read_text()) for p in sorted(results.glob("*_SUMMARY.json"))]
    assert len(rows) == 350 and all(r["status"] == "COMPLETE" for r in rows)
    expected = {r["job"] for r in protocol["tasks"]}
    assert {r["job"] for r in rows} == expected
    with (study / "audits/D1_independent/INDEPENDENT_SEED_AVERAGED_CELLS.csv").open() as f:
        cells = list(csv.DictReader(f))
    assert len(cells) == 2600
    for c in cells:
        for k in ("budget", "h", "seeds"):
            c[k] = int(c[k])
        for k in ("mean_seed_mse", "rw_mse", "relative_mse"):
            c[k] = float(c[k])
    targets = ["EEX_EUA_T3PA", "GUANGZHOU", "HUBEI", "SHANGHAI", "EU_REPORTED"]
    models = protocol["models"]
    lookup = {(c["model"], c["target"], c["budget"], c["h"]): c["relative_mse"] for c in cells}
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    main_rows, seed_rows, resource_rows = [], [], []
    for model in models:
        rr = by_model[model]
        is_nn = model in protocol["neural_models"]
        ss = []
        for seed in sorted({r["seed"] for r in rr}):
            zz = [r for r in rr if r["seed"] == seed]
            score = st.mean(r["evaluation"]["score"] for r in zz)
            seed_rows.append(dict(model=model, seed=seed, score=score,
                                  budget100=st.mean(r["evaluation"]["score"] for r in zz if r["budget"] == 100),
                                  budget400=st.mean(r["evaluation"]["score"] for r in zz if r["budget"] == 400),
                                  stochastic_neural_seed=is_nn))
            ss.append(score)
        selected = [r["fit_receipt"].get("selected_step", r["fit_receipt"].get("best_step")) for r in rr]
        steps = [r["fit_receipt"].get("steps", r["fit_receipt"].get("steps_completed", 0)) for r in rr]
        main_rows.append(dict(
            model=model, score=st.mean(ss),
            budget100=st.mean(r["evaluation"]["score"] for r in rr if r["budget"] == 100),
            budget400=st.mean(r["evaluation"]["score"] for r in rr if r["budget"] == 400),
            seed_sd=st.stdev(ss) if is_nn else None,
            seed_min=min(ss) if is_nn else None, seed_max=max(ss) if is_nn else None,
            validation_score=st.mean(r["validation"]["score"] for r in rr),
            forecast_flows=len(rr), selected_step_zero=sum(s == 0 for s in selected),
        ))
        resource_rows.append(dict(
            model=model, forecast_flows=len(rr), neural_fits=len(rr) if is_nn else 0,
            neural_updates=sum(steps),
            author_source_ridge_calls=sum(r["fit_receipt"].get("author_ridge_calls", 0) for r in rr),
            ridge_alpha_solves=sum(r["fit_receipt"].get("ridge_fits", 0) for r in rr),
            selected_step_zero=sum(s == 0 for s in selected),
            reached_max600=sum(s == 600 for s in steps),
            selected_step_median=st.median(selected) if is_nn else None,
            total_job_wall_seconds=sum(r["wall_seconds"] for r in rr),
            median_job_wall_seconds=st.median(r["wall_seconds"] for r in rr),
        ))
    main_rows.sort(key=lambda r: r["score"])
    market_rows, horizon_rows, market_budget_rows = [], [], []
    for model in models:
        for target in targets:
            market_rows.append(dict(model=model, target=target,
                                    score=st.mean(lookup[model, target, b, h] for b in (100, 400) for h in (1, 5, 20))))
            for budget in (100, 400):
                market_budget_rows.append(dict(model=model, target=target, budget=budget,
                                               score=st.mean(lookup[model, target, budget, h] for h in (1, 5, 20)),
                                               h1=lookup[model, target, budget, 1],
                                               h5=lookup[model, target, budget, 5],
                                               h20=lookup[model, target, budget, 20]))
        for h in range(1, 21):
            horizon_rows.append(dict(model=model, h=h,
                                     score=st.mean(lookup[model, t, b, h] for t in targets for b in (100, 400)),
                                     budget100=st.mean(lookup[model, t, 100, h] for t in targets),
                                     budget400=st.mean(lookup[model, t, 400, h] for t in targets)))
    scores = {r["model"]: r["score"] for r in main_rows}
    comparisons = []
    for other in ("profile", "covariance_profile", "transport_uniform", "rw", "local_profile", "diagonal_profile"):
        diffs = [lookup["transport", t, b, h] - lookup[other, t, b, h]
                 for t in targets for b in (100, 400) for h in (1, 5, 20)]
        comparisons.append(dict(
            candidate="transport", comparator=other,
            candidate_score=scores["transport"], comparator_score=scores[other],
            relative_score_change_percent=100 * (scores["transport"] / scores[other] - 1),
            absolute_score_difference=scores["transport"] - scores[other],
            cell_wins=sum(x < 0 for x in diffs), cell_ties=sum(x == 0 for x in diffs),
            cell_losses=sum(x > 0 for x in diffs), total_cells=30,
            interpretation="descriptive correlated cells, not binomial trials or significance",
        ))
    complete = json.loads((results / "COMPLETE.json").read_text())
    independent = json.loads((study / "audits/D1_independent/INDEPENDENT_SELECTION.json").read_text())
    for r in independent["ranking"]:
        assert abs(scores[r["model"]] - r["score"]) < 1e-12
    assert sum(r["neural_updates"] for r in resource_rows) == 84010
    counts = dict(
        model_origin_test_rows=sum(r["evaluation"]["n_origins"] for r in rows),
        model_origin_validation_rows=sum(r["validation"]["n_origins"] for r in rows),
        unique_series_origin_pairs_2022=sum(next(r["evaluation"]["n_origins"] for r in rows if r["target"] == t) for t in targets),
        formal_wall_seconds=complete["wall_seconds"],
        sum_job_wall_seconds=sum(r["wall_seconds"] for r in rows),
        neural_updates=sum(r["neural_updates"] for r in resource_rows),
        selected_step_zero=sum(r["selected_step_zero"] for r in resource_rows),
        description="model-origin rows repeat the same quotes across methods/budgets/seeds; not new data",
    )
    counts["model_horizon_test_scalars"] = counts["model_origin_test_rows"] * 20
    out = results / "analysis"
    out.mkdir(exist_ok=True)
    for name, table in (("MAIN", main_rows), ("BY_SEED", seed_rows), ("RESOURCES", resource_rows),
                        ("BY_MARKET", market_rows), ("BY_MARKET_BUDGET", market_budget_rows),
                        ("ALL20_HORIZONS", horizon_rows), ("CANDIDATE_COMPARISONS", comparisons)):
        dump_csv(out / (name + ".csv"), table)
    payload = dict(status="POST_RUN_DESCRIPTIVE_ANALYSIS", main=main_rows, by_seed=seed_rows,
                   resources=resource_rows, by_market=market_rows, by_market_budget=market_budget_rows,
                   all20_horizons=horizon_rows, comparisons=comparisons, counts=counts,
                   frozen_gate=independent["gates"],
                   provenance=dict(protocol_sha256=hashlib.sha256((study / "provenance/PROTOCOL.json").read_bytes()).hexdigest(),
                                   analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
                   scope=dict(training_calls=0, market_csv_reads=0, private_prediction_reads=0,
                              bootstrap_or_p_values=0, new_candidate_selection=False))
    (out / "ANALYSIS.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    if plots:
        make_figures(study, main_rows, seed_rows, lookup, targets, horizon_rows)
    print(json.dumps(dict(counts=counts, comparisons=comparisons, main=main_rows,
                          by_market_budget=[r for r in market_budget_rows if r["model"] in ("transport", "covariance_profile")],
                          by_seed=[r for r in seed_rows if r["model"] in ("transport", "covariance_profile", "profile")]), indent=2))


def make_figures(study, main_rows, seed_rows, lookup, targets, horizon_rows):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white", "pdf.fonttype": 42, "svg.fonttype": "none"})
    figures = study / "figures"
    figures.mkdir(exist_ok=True)
    palette = {"transport": "#0f766e", "covariance_profile": "#2563eb", "rw": "#17212f"}

    def save(fig, name):
        for ext in ("png", "pdf", "svg"):
            fig.savefig(figures / f"{name}.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 7.0), sharey=True, gridspec_kw={"width_ratios": [1.12, 1]})
    for i, r in enumerate(main_rows):
        model = r["model"]
        values = [s["score"] for s in seed_rows if s["model"] == model]
        color = palette.get(model, "#64748b")
        axes[0].scatter(values, [i] * len(values), s=35, marker="|", color="#8594a6", zorder=2)
        axes[0].scatter(r["score"], i, s=65 if model == "transport" else 40, color=color, zorder=3)
        axes[0].text(1.484, i, f"{r['score']:.6f}", va="center", ha="right", color=color, fontsize=9)
        axes[1].plot([r["budget100"], r["budget400"]], [i, i], color="#c5ced8", lw=1.4)
        axes[1].scatter(r["budget100"], i, marker="s", s=35, color="#c76b12", label="100 windows" if i == 0 else None, zorder=3)
        axes[1].scatter(r["budget400"], i, s=35, color="#2563eb", label="400 windows" if i == 0 else None, zorder=3)
        if model == "transport":
            for ax in axes:
                ax.axhspan(i - .43, i + .43, color="#e5f4ef", zorder=0)
    axes[0].set_yticks(range(len(main_rows)), [r["model"] for r in main_rows])
    axes[0].invert_yaxis()
    for ax in axes:
        ax.axvline(1, color="#17212f", lw=1, ls="--")
        ax.set_xlim(.98, 1.50)
        ax.grid(axis="x", color="#e8edf2")
        ax.set_axisbelow(True)
        ax.set_xlabel("Native-price MSE / same-cell random-walk MSE")
    axes[0].set_title("A  Frozen primary score; ticks show initialization seeds", loc="left", fontsize=11)
    axes[1].set_title("B  Same source histories; two supervision budgets", loc="left", fontsize=11)
    axes[1].legend(loc="upper right", frameon=False)
    fig.suptitle("Prior transport: all 13 methods on exposed 2022 development quotes", fontsize=15, x=.06, ha="left", y=.99)
    fig.text(.06, .005, "Lower is better. Neural scores average seed losses; seed ticks are not confidence intervals. RW = 1.\n350 completed forecast flows; the pre-fit resource gate is not passed.", fontsize=10, color="#475569")
    fig.tight_layout(rect=[0, .065, 1, .95])
    save(fig, "01_all_methods")

    selected_models = ("transport", "covariance_profile")
    matrices = [np.array([[lookup[m, t, b, h] for b in (100, 400) for h in (1, 5, 20)] for t in targets]) for m in selected_models]
    low = min(float(a.min()) for a in matrices)
    high = max(float(a.max()) for a in matrices)
    extent = max(abs(low - 1), abs(high - 1), .001)
    norm = TwoSlopeNorm(vmin=1 - extent, vcenter=1, vmax=1 + extent)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), sharey=True)
    for ax, model, matrix in zip(axes, selected_models, matrices):
        im = ax.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")
        ax.set_xticks(range(6), ["100 / 1", "100 / 5", "100 / 20", "400 / 1", "400 / 5", "400 / 20"], rotation=40, ha="right")
        ax.set_yticks(range(5), ["EEX EUA", "Guangzhou", "Hubei", "Shanghai", "EU reported"])
        ax.axvline(2.5, color="white", lw=3)
        for i in range(5):
            for j in range(6):
                value = matrix[i, j]
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=10,
                        color="white" if norm(value) > .82 or norm(value) < .15 else "#162536")
        ax.set_title(model, loc="left", fontsize=12)
        ax.set_xlabel("Supervision budget / observed-quote horizon")
    fig.subplots_adjust(left=.10, right=.88, bottom=.24, top=.84, wspace=.18)
    fig.colorbar(im, cax=fig.add_axes([.905, .25, .015, .57]), label="Relative MSE (RW = 1)")
    fig.suptitle("All 30 primary cells: candidate and strongest non-RW control", x=.1, ha="left", fontsize=14)
    fig.text(.10, .01, "Each cell averages the same three initialization-seed losses. Correlated cells are descriptive, not independent trials.\nBoth budgets retain identical role endpoints and source histories; no subgroup changes the frozen STOP decision.", fontsize=9, color="#475569")
    save(fig, "02_primary_cells")

    keys = ["transport", "covariance_profile", "profile", "local_gru", "rw"]
    colors = ["#0f766e", "#2563eb", "#b45309", "#a855f7", "#17212f"]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), sharey=True)
    for ax, budget in zip(axes, (100, 400)):
        for model, color in zip(keys, colors):
            zz = sorted([r for r in horizon_rows if r["model"] == model], key=lambda r: r["h"])
            ax.plot([r["h"] for r in zz], [r[f"budget{budget}"] for r in zz], color=color,
                    ls="--" if model == "rw" else "-", lw=1.8, label=model)
        ax.set_title(f"{budget} retained supervision windows", loc="left")
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.set_xlabel("Observed-quote horizon")
        ax.grid(color="#e8edf2")
    axes[0].set_ylabel("Native-price MSE / RW MSE")
    axes[1].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Descriptive horizon profile; the primary score remains h = 1, 5, 20", fontsize=14)
    fig.text(.075, .01, "All 20 horizons and all 13 methods are exported in ALL20_HORIZONS.csv. No horizon was chosen after observing results.", fontsize=9, color="#475569")
    fig.tight_layout(rect=[0, .055, 1, .93])
    save(fig, "03_horizon_profile")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    main(args.study, not args.no_plots)
