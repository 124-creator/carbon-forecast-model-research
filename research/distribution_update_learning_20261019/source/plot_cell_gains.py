"""Render the complete primary-versus-matched-raw cell matrix from public aggregates.

This is a post-analysis visualization only. It does not read private predictions,
change aggregation, fit a model, or recalculate any confidence interval.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
import numpy as np


def main(root: Path) -> None:
    result_dir = root / "results" / "market"
    summary = json.loads((result_dir / "SUMMARY.json").read_text())
    with (result_dir / "CELL_METRICS.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    primary, raw = summary["primary"], summary["primary_comparator"]
    scores = {(r["market"], int(r["year"]), int(r["horizon"]), r["method"]):
              float(r["CRPS_normalized"]) for r in rows}
    markets = ["EUA_LONG", "GUANGZHOU", "HUBEI", "SHANGHAI"]
    labels = ["EUA", "Guangzhou", "Hubei", "Shanghai"]
    columns = [(year, horizon) for year in (2021, 2022, 2023) for horizon in (5, 20)]
    matrix = np.full((len(markets), len(columns)), np.nan)
    for i, market in enumerate(markets):
        for j, (year, horizon) in enumerate(columns):
            key = (market, year, horizon)
            if (*key, primary) in scores:
                matrix[i, j] = 100 * (scores[(*key, primary)] / scores[(*key, raw)] - 1)
    finite = matrix[np.isfinite(matrix)]
    assert len(finite) == summary["counts"]["market_year_horizon_cells"]
    assert abs(float(np.mean(finite)) / 100 - summary["primary_vs_raw"]["mean_cell_relative_difference"]) < 1e-12

    plt.rcParams.update({"font.size": 11, "savefig.dpi": 180})
    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    cmap = LinearSegmentedColormap.from_list("improvement", ["#277462", "#f4f7f6"])
    cmap.set_bad("#e8e8e8")
    im = ax.imshow(matrix, cmap=cmap, norm=Normalize(vmin=-1.1, vmax=0), aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isnan(value):
                ax.add_patch(Rectangle((j-.5, i-.5), 1, 1, fill=False, hatch="///", edgecolor="#b6b6b6", linewidth=0))
                text, color = "Excluded\n(data rule)", "#444444"
            elif abs(value) <= 1e-8:
                text, color = "0.000%\nTie", "#42564f"
            else:
                text, color = f"{value:+.3f}%", "white" if value < -.5 else "#222222"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=10.5)
    ax.set_xticks(np.arange(len(columns)), [f"{year}\nH{horizon}" for year, horizon in columns])
    ax.set_yticks(np.arange(len(markets)), labels)
    ax.tick_params(length=0, pad=9)
    ax.set_xticks(np.arange(-.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(markets), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("The path-control gain is confined to one development cell\nPrimary normalized CRPS relative to the same-joint-budget raw update", loc="left", pad=20)
    colorbar = fig.colorbar(im, ax=ax, fraction=.035, pad=.03)
    colorbar.set_label("CRPS change (%) — lower is better")
    fig.text(.12, .027, "Each cell averages losses from 2 backbones × 3 seeds. 1 improved, 21 tied; no independent-period confirmation.", fontsize=9.2, color="#444444")
    fig.subplots_adjust(left=.12, right=.91, top=.76, bottom=.19)
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(figures / f"market_cell_mechanism_gain.{suffix}")
    plt.close(fig)
    print(json.dumps({"cells": int(len(finite)), "mean_relative_percent": float(np.mean(finite)), "figure": "market_cell_mechanism_gain"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    main(parser.parse_args().root)
