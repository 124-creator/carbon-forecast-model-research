"""Apply the prewritten 2022 selection / 2023 advancement rules once."""
from __future__ import annotations
import argparse,csv,json,hashlib
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

BASELINES={"er","masked","dsof","self_td","static_gru","static_teacher_student","online_gru","rw"}

def evaluate(protocol_path,metrics_path):
    pp=Path(protocol_path);cfg=json.loads(pp.read_text());mp=Path(metrics_path)
    with mp.open() as f:raw=list(csv.DictReader(f))
    expected_configs={"rw","static_gru","static_teacher_student","online_gru"}
    expected_configs|={c["id"] for c in cfg["student_configs"]}
    expected_configs|={"output_"+c["id"] for c in cfg["calibrators"] if not c["old_target"]}
    expected={(m["name"],s,y,h,c) for m in cfg["markets"] for s in cfg["seeds"] for y in cfg["scored_years"] for h in cfg["score_horizons"] for c in expected_configs}
    seen=set();score=defaultdict(list);market_scores=defaultdict(list);families={};supports=defaultdict(set)
    for r in raw:
        key=(r["market"],int(r["seed"]),int(r["year"]),int(r["horizon"]),r["config_id"])
        if key in seen:raise ValueError("Duplicate metric cell")
        seen.add(key);v=float(r["relative_mse"])
        if not np.isfinite(v):raise ValueError("Nonfinite metric")
        score[(r["config_id"],key[2])].append(v)
        market_scores[(r["config_id"],key[0],key[2])].append(v)
        families[r["config_id"]]=r["family"]
        supports[key[:4]].add((int(r["origins"]),float(r["rw_mse"])))
    if seen!=expected:raise ValueError(f"Coverage differs: missing{len(expected-seen)},extra{len(seen-expected)}")
    if any(len(s)!=1 for s in supports.values()):raise ValueError("Forecast supports or RW denominators differ")
    agg={key:float(np.mean(vals)) for key,vals in score.items()}
    ms={key:float(np.mean(vals)) for key,vals in market_scores.items()}
    byfamily=defaultdict(list)
    for c,f in families.items():byfamily[f].append(c)
    selected={f:min(ids,key=lambda c:(agg[c,2022],c)) for f,ids in byfamily.items()}
    rrc=selected["rrc"]
    baseline=min((selected[f] for f in BASELINES),key=lambda c:(agg[c,2022],c))
    r,b=agg[rrc,2023],agg[baseline,2023]
    breadth=[m["name"] for m in cfg["markets"] if ms[rrc,m["name"],2023]<ms[baseline,m["name"],2023]]
    no_prefix=selected["no_prefix"];output=selected["output_only"];dsof=selected["dsof"]
    flags={"primary":r<=.99*b and r<1.,"breadth":len(breadth)>=3,
           "mechanism":r<=agg[no_prefix,2023]-.0001 and r<=agg[output,2023]-.0001,
           "EEX_relevance":ms[rrc,"EEX_EUA_T3PA",2023]<1. and ms[rrc,"EEX_EUA_T3PA",2023]<ms[dsof,"EEX_EUA_T3PA",2023],
           "finite_complete":True}
    comparisons=[dict(family=f,selected_config=c,selection_2022=agg[c,2022],advancement_2023=agg[c,2023]) for f,c in sorted(selected.items())]
    all_configs=[dict(config_id=c,family=families[c],score_2022=agg[c,2022],score_2023=agg[c,2023],selected_within_family=c==selected[families[c]]) for c in sorted(expected_configs)]
    all_markets=[dict(config_id=c,market=m["name"],year=y,score=ms[c,m["name"],y]) for c in sorted(expected_configs) for m in cfg["markets"] for y in cfg["scored_years"]]
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=hashlib.sha256(pp.read_bytes()).hexdigest(),metrics_sha256=hashlib.sha256(mp.read_bytes()).hexdigest(),
                selection_only_year=2022,advancement_year=2023,metric_cells=len(raw),configurations=len(expected_configs),selected_rrc=rrc,
                selected_baseline=baseline,selected_baseline_family=families[baseline],rrc_2023=r,baseline_2023=b,
                relative_improvement_2023=1-r/b,breadth_better_series=breadth,gate_flags=flags,advance_to_confirmation=all(flags.values()),
                selected_no_prefix=no_prefix,selected_output_only=output,selected_dsof=dsof,
                no_prefix_2023=agg[no_prefix,2023],output_only_2023=agg[output,2023],
                EEX_rrc_2023=ms[rrc,"EEX_EUA_T3PA",2023],EEX_dsof_2023=ms[dsof,"EEX_EUA_T3PA",2023],
                family_comparison=comparisons,confirmation_data_read=False,
                thresholds_are_research_advancement_heuristics_not_journal_standards=True,
                outcome="advance under separately frozen confirmation protocol" if all(flags.values()) else "stop this fixed candidate; keep EEX2024/2025 sealed")
    return result,all_configs,all_markets

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--protocol",required=True);ap.add_argument("--metrics",required=True);ap.add_argument("--out",required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    result,configs,markets=evaluate(a.protocol,a.metrics)
    (out/"SELECTION_AND_GATE.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    for name,rows in [("ALL_CONFIGS.csv",configs),("ALL_MARKET_SCORES.csv",markets)]:
        with (out/name).open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(result,ensure_ascii=False))
