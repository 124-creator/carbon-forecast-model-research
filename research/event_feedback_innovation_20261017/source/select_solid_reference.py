"""Apply frozen B1 family selection; an existing-method suitability comparison."""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def table(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError("No overwrite or silent reselection")
    root = args.study.resolve()
    cfg = json.loads((root / "provenance/SOLID_B1_PROTOCOL.json").read_bytes())
    rows = table(root / "results/SOLID_B1/ALL_METRICS.csv")
    d1 = [r for r in table(root / "results/D1/ALL_METRICS.csv")
          if r["config_id"] in ("static_gru", "rw")]
    expected = {(j["market"], j["seed"], s["id"], y, h)
                for j in cfg["jobs"] for s in cfg["configs"]
                for y in (2022, 2023) for h in (1, 5, 20)}
    seen = set()
    for r in rows:
        key = (r["market"], int(r["seed"]), r["config_id"], int(r["year"]), int(r["horizon"]))
        if key in seen or not math.isfinite(float(r["relative_mse"])):
            raise ValueError("Duplicate or nonfinite metric")
        seen.add(key)
    if seen != expected:
        raise ValueError("Incomplete fixed grid")
    support = defaultdict(set)
    agg, byseries, families = defaultdict(list), defaultdict(list), {}
    for r in rows + d1:
        c, y, m = r["config_id"], int(r["year"]), r["market"]
        support[(m, int(r["seed"]), y, int(r["horizon"]))].add((int(r["origins"]), float(r["rw_mse"])))
        agg[(c, y)].append(float(r["relative_mse"]))
        byseries[(c, m, y)].append(float(r["relative_mse"]))
        families[c] = r["family"]
    if any(len(v) != 1 for v in support.values()):
        raise ValueError("Different query support or RW denominator")
    scores = {k: math.fsum(v)/len(v) for k, v in agg.items()}
    series = {k: math.fsum(v)/len(v) for k, v in byseries.items()}
    selected = {}
    for family in ("solid_distance", "solid_recency"):
        ids = [s["id"] for s in cfg["configs"]
               if ("solid_recency" if s["remove_distance"] else "solid_distance") == family]
        selected[family] = min(ids, key=lambda c: (scores[(c, 2022)], c))
    output = []
    for c in sorted(families):
        output.append({"config_id": c, "family": families[c],
                       "score_2022": scores[(c, 2022)], "score_2023": scores[(c, 2023)],
                       "selected_2022": c in selected.values()})
    result = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "protocol_sha256": hashlib.sha256((root/"provenance/SOLID_B1_PROTOCOL.json").read_bytes()).hexdigest(),
              "metric_sha256": hashlib.sha256((root/"results/SOLID_B1/ALL_METRICS.csv").read_bytes()).hexdigest(),
              "selection_year": 2022, "evaluation_year": 2023, "selected": selected,
              "family_comparison": [{"family": family, "config_id": c,
                    "score_2022": scores[(c, 2022)], "score_2023": scores[(c, 2023)]}
                    for family, c in [*selected.items(), ("static_gru", "static_gru"), ("rw", "rw")]],
              "selected_series_2023": [{"config_id": c, "market": m,
                    "score": series[(c, m, 2023)]}
                    for c in [*selected.values(), "static_gru", "rw"]
                    for m in sorted({j["market"] for j in cfg["jobs"]})],
              "new_algorithm_claim": False, "confirmation_data_read": False,
              "advance_to_confirmation": False,
              "interpretation": "Existing-method backup reference on exposed development years. No outcome revives RRC or certifies a novel contribution."}
    args.output.mkdir(parents=True)
    (args.output / "SELECTION.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    with (args.output / "ALL_CONFIGS.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader(); writer.writerows(output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
