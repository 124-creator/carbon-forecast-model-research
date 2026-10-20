"""Post-result descriptive figures; never selects or refits any model."""
from pathlib import Path
import csv
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/carbon-geometry-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = json.loads((ROOT/"results/synthetic/SUMMARY.json").read_text())
    index = {(r["stratum"], r["control"], r["metric"]): r for r in summary["comparisons"]}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "axes.titleweight": "bold"})
    controls = [("validation_selected_competitor", "Validation-selected competitor"),
                ("tempered_bayes", "Tempered Bayes"),
                ("future_variance", "Future-variance geometry"),
                ("current_cdf", "Current-CDF geometry"),
                ("rotated_future_cdf", "Rotated future-CDF geometry"),
                ("crps_kl", "Current CRPS + KL"),
                ("future_cdf_ratio_preserving", "Ratio-preserving ablation")]
    fig, ax = plt.subplots(figsize=(10.3, 6.3))
    fig.subplots_adjust(left=.31, right=.96, top=.79, bottom=.24)
    rows = []
    for y, (control, label) in enumerate(controls):
        r = index["positive_scenarios_equal_weight", control, "expected_primary"]
        mean, lo, hi = [100*r[k] for k in ("mean_relative_minus_one", "bootstrap95_lower", "bootstrap95_upper")]
        color = "#a33d2e" if mean > 0 else "#1a5e71"
        ax.errorbar(mean, y, xerr=[[mean-lo], [hi-mean]], fmt="o", color=color,
                    capsize=3, linewidth=1.6, markersize=6)
        ax.text(.235, y, f"{mean:+.3f}%", va="center", fontsize=9, color=color)
        rows.append({"control": control, "label": label, "relative_percent": mean,
                     "lower95_percent": lo, "upper95_percent": hi})
    ax.set_yticks(np.arange(len(controls)), [s[1] for s in controls])
    ax.tick_params(axis="y", length=0, pad=10)
    ax.set_xlim(-.55, .34)
    ax.invert_yaxis()
    ax.axvline(0, linestyle="--", linewidth=1, color="#888888")
    ax.grid(axis="x", color="#e8e8e8", linewidth=.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Candidate / control - 1 (%)     |     Negative values favor the candidate", labelpad=12)
    fig.suptitle("Future-target geometry: matched mechanism comparisons", x=.025, y=.97,
                 ha="left", fontsize=16, fontweight="bold")
    fig.text(.025, .9, "Two prespecified shock scenarios • Expected observed-return CRPS • Horizons 5 and 20",
             fontsize=10, color="#444444")
    fig.text(.025, .05, "Nominal 95% paired intervals: 20 independent seeds, 9,999 bootstrap draws.\n"
             "These are synthetic comparisons; no carbon-market performance is claimed.", fontsize=9, color="#555555")
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT/f"mechanism_comparisons.{ext}", dpi=200, facecolor="white")
    plt.close(fig)
    with (OUT/"mechanism_comparisons.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    strata = [("clean_correct", "Clean, correctly specified"),
              ("rare_large_shocks", "Rare large shocks"),
              ("frequent_moderate_shocks", "Frequent moderate shocks"),
              ("wrong_transition_geometry", "Misspecified transition"),
              ("positive_scenarios_equal_weight", "Two shock scenarios, equal weight")]
    fig, ax = plt.subplots(figsize=(10.3, 5.3))
    fig.subplots_adjust(left=.33, right=.94, top=.78, bottom=.2)
    for y, (stratum, label) in enumerate(strata):
        for offset, metric, color, marker in [(-.12, "expected_primary", "#1a5e71", "o"),
                                               (.12, "empirical_primary", "#8b7353", "s")]:
            r = index[stratum, "validation_selected_competitor", metric]
            mean, lo, hi = [100*r[k] for k in ("mean_relative_minus_one", "bootstrap95_lower", "bootstrap95_upper")]
            ax.errorbar(mean, y+offset, xerr=[[mean-lo], [hi-mean]], fmt=marker,
                        color=color, capsize=3, linewidth=1.4, markersize=5,
                        label="Expected CRPS" if metric=="expected_primary" and y==0 else
                              "Empirical CRPS" if y==0 else None)
    ax.set_yticks(np.arange(len(strata)), [s[1] for s in strata])
    ax.tick_params(axis="y", length=0, pad=10)
    ax.invert_yaxis(); ax.axvline(0, color="#888888", linestyle="--", linewidth=1)
    ax.grid(axis="x", color="#e8e8e8", linewidth=.6); ax.set_axisbelow(True)
    ax.set_xlabel("Candidate / validation-selected competitor - 1 (%)", labelpad=12)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -.2), ncol=2, fontsize=9)
    fig.suptitle("All frozen scenarios remain in the result", x=.025, y=.97,
                 ha="left", fontsize=16, fontweight="bold")
    fig.text(.025, .9, "Competitors selected before test • Both scoring views retained • No post-test tuning",
             fontsize=10, color="#444444")
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT/f"scenario_comparisons.{ext}", dpi=200, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
