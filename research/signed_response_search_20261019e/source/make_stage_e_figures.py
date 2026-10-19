"""Scientific plots from exported aggregate experiment evidence only."""
from pathlib import Path
import json
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/carbon-stage-e-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STUDY = Path(__file__).resolve().parents[1]
FAMILIES = ["legacy_fixed_response", "fixed_response", "static_response", "signed_shock_only",
            "amplitude_context", "signed_context_response", "free_horizon_context", "ridge_price",
            "ridge_interaction", "raw_gru", "swewt_gru", "rw"]
LABELS = ["Legacy fixed prior", "Fixed rates + fitted readout", "Learned static rates",
          "Shock-only duration", "Conditional amplitude", "Conditional duration (primary)",
          "Free horizon readouts", "Price Ridge", "Interaction Ridge", "Raw GRU", "SWEWT-GRU", "Random walk"]


def main():
    records = []
    for r, f in [("core", "ALL_SUMMARIES.json"), ("neural_references", "SUMMARY.json")]:
        records.extend(json.loads((STUDY/"results"/r/f).read_text()))
    data = {(r["market"], r["family"]): r for r in records}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(16, 8.5), sharey=True)
    colors = ["#24639a" if f == "signed_context_response" else
              "#408775" if f in ("raw_gru", "swewt_gru") else "#aebac5" for f in FAMILIES]
    for ax, market, title in zip(axes, ["TIANJIN_OFFICIAL_MEAN", "CEA_OFFICIAL_MEAN"],
                                  ["Tianjin: previously exposed development", "National CEA: new scoring evaluation"]):
        q = np.array([data[market, f]["mean_mse_over_rw"] for f in FAMILIES])
        gain = 100*(1-q)
        ax.barh(np.arange(len(FAMILIES)), gain, color=colors, height=.66)
        for i, (value, score) in enumerate(zip(gain, q)):
            display_value = 0. if abs(value) < .00005 else value
            ax.text(value+.28, i, f"{display_value:.2f}  (Q={score:.4f})", va="center", fontsize=9)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
        ax.set_xlim(-.2, 25.7)
        ax.axvline(0, color="#52606d", linewidth=.7)
        ax.set_xlabel("Reduction vs random walk (RW percentage points)", labelpad=13)
        ax.grid(axis="x", alpha=.18)
        ax.set_axisbelow(True)
    axes[0].set_yticks(np.arange(len(FAMILIES)), LABELS)
    axes[0].invert_yaxis()
    fig.suptitle("Signed conditional responses: complete, predeclared model comparison", fontsize=17, fontweight="bold", y=.98)
    fig.text(.02, .025, "Q = mean over H=1,5,20 of MSE / RW-MSE; lower Q is better.  GRUs average seed losses.\n"
             "Blue: primary candidate. Green: neural references. The two market roles remain separate; no pooled confirmation claim.",
             fontsize=10, color="#354553")
    fig.tight_layout(rect=(0, .085, 1, .94))
    out = STUDY/"figures"
    out.mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(out/f"01_complete_model_comparison.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    budget_path = STUDY/"results/budget320/ALL_SUMMARIES.json"
    if budget_path.exists():
        budget = {(r["market"], r["family"]): r for r in json.loads(budget_path.read_text())}
        families = ["static_response", "signed_shock_only", "amplitude_context",
                    "signed_context_response", "free_horizon_context"]
        labels = [LABELS[FAMILIES.index(f)] for f in families]
        fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)
        for ax, market, title in zip(axes, ["TIANJIN_OFFICIAL_MEAN", "CEA_OFFICIAL_MEAN"],
                                     ["Tianjin: development", "National CEA: development sensitivity"]):
            for i, family in enumerate(families):
                old = data[market, family]["mean_mse_over_rw"]
                new = budget[market, family]["mean_mse_over_rw"]
                ax.plot([old, new], [i, i], color="#aebac5", lw=3)
                ax.scatter(old, i, s=80, facecolors="white", edgecolors="#52606d", zorder=3,
                           label="Original cap 80" if i == 0 else None)
                ax.scatter(new, i, s=70, color="#24639a", zorder=4,
                           label="Supplement cap 320" if i == 0 else None)
                ax.text(.928 if market.startswith("TIANJIN") else .8795, i,
                        f"{old:.4f} → {new:.4f}", va="center", fontsize=10)
            ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
            ax.set_xlabel("Q (lower is better)", labelpad=12)
            ax.grid(axis="x", alpha=.18)
            ax.set_axisbelow(True)
            ax.legend(loc="lower left", fontsize=9, frameon=False)
            ax.set_ylim(5.1, -.6)
            ax.set_xlim((.887, .955) if market.startswith("TIANJIN") else (.849, .906))
        axes[0].set_yticks(range(len(families)), labels)
        fig.suptitle("One predeclared budget sensitivity, after original scores were observed", fontsize=16, fontweight="bold", y=.98)
        fig.text(.02, .035, "Only the epoch cap changed: 80 → 320. All five families and both markets are retained.\n"
                 "This supplement does not replace the original evaluation or its uncertainty intervals.", fontsize=10, color="#354553")
        fig.tight_layout(rect=(0, .10, 1, .93))
        for suffix in ("png", "pdf"):
            fig.savefig(out/f"02_budget_sensitivity.{suffix}", dpi=180, bbox_inches="tight")
        plt.close(fig)
    print("Created aggregate-only PNG and PDF")


if __name__ == "__main__":
    main()
