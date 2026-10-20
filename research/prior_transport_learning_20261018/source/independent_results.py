"""Independent NumPy audit of the frozen prior-transport D1 predictions.

No imports from production data/run/learning/baselines/scoring; no CSV or weights.
Only explicitly expected new-run JSON and NPZ files are read. All losses are
recomputed in raw-price-increment units. Seed losses are averaged before the
equal market/budget/horizon statistic; predictions are never ensembled.
"""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

MARKETS = ("EEX_EUA_T3PA", "GUANGZHOU", "HUBEI", "SHANGHAI", "EU_REPORTED")
PROFILE = ("local_profile", "uniform_profile", "aux_one", "aux_five", "profile",
           "transport", "transport_uniform", "covariance_profile", "diagonal_profile")
GRU = ("local_gru", "pooled_gru")
STATIC = ("ridge", "rw")
MODELS = PROFILE + GRU + STATIC
BUDGETS = (100, 400)
PRIMARY_H = (1, 5, 20)
ATOL, RTOL = 5e-10, 2e-10


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


class Check:
    def __init__(self):
        self.comparisons = 0
        self.max_error = 0.0
        self.maximum_label = None
        self.errors = []

    def require(self, value, label):
        self.comparisons += 1
        if not bool(value):
            raise AssertionError(label)

    def same(self, actual, expected, label):
        a, b = np.asarray(actual), np.asarray(expected)
        self.require(a.shape == b.shape, label+"/shape")
        self.comparisons += a.size
        if a.dtype.kind in "bifu" and b.dtype.kind in "bifu":
            self.require(np.isfinite(a).all() and np.isfinite(b).all(), label+"/finite")
            error = float(np.max(np.abs(a.astype(float)-b.astype(float)))) if a.size else 0.0
            if error > self.max_error:
                self.max_error, self.maximum_label = error, label
            self.require(np.allclose(a, b, atol=ATOL, rtol=RTOL), label+"/numeric")
        else:
            self.require(np.array_equal(a, b), label+"/exact")

    def bitwise(self, actual, expected, label):
        a, b = np.asarray(actual), np.asarray(expected)
        self.comparisons += a.size
        self.require(a.dtype == b.dtype and a.shape == b.shape and
                     np.array_equal(a, b), label+"/paired_exact")


def planned(protocol):
    if int(protocol["year"]) != 2022:
        raise ValueError("This audit is frozen for exposed 2022 development only")
    if tuple(protocol["models"]) != MODELS:
        raise ValueError("Unexpected family list")
    seeds = tuple(int(s) for s in protocol.get("seeds", [protocol["config"].get("seed", 17)]))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Invalid neural seeds")
    static_seed = int(protocol.get("static_seed", seeds[0]))
    tasks = []
    for market in MARKETS:
        for budget in BUDGETS:
            for model in MODELS:
                for seed in ((static_seed,) if model in STATIC else seeds):
                    job = f"{market}_b{budget}_{model}_s{seed}"
                    tasks.append(dict(job=job, target=market, budget=budget, model=model, seed=seed))
    if "tasks" in protocol:
        by_job = {t["job"]: t for t in protocol["tasks"]}
        if len(by_job) != len(protocol["tasks"]) or set(by_job) != {t["job"] for t in tasks}:
            raise ValueError("Protocol explicit tasks disagree with declared design")
        for t in tasks:
            for key in ("target", "budget", "model", "seed"):
                if key in by_job[t["job"]] and by_job[t["job"]][key] != t[key]:
                    raise ValueError("Explicit task identity mismatch: "+t["job"]+"/"+key)
    return tasks, seeds, static_seed


def raw_score(prediction, delta, sigma):
    p, y, s = map(lambda v: np.asarray(v, dtype=np.float64), (prediction, delta, sigma))
    if p.ndim != 2 or p.shape[1] != 20 or p.shape != y.shape or p.shape != s.shape or len(p) == 0:
        raise ValueError("Nonempty matching [N,20] arrays required")
    if not (np.isfinite(p).all() and np.isfinite(y).all() and np.isfinite(s).all() and (s > 0).all()):
        raise ValueError("Invalid finite forecasts, labels or strictly positive sigma")
    cells = []
    for j in range(20):
        errors = [float(p[i,j])*float(s[i,j])-float(y[i,j]) for i in range(len(p))]
        mse = math.fsum(e*e for e in errors) / len(p)
        rw = math.fsum(float(y[i,j])**2 for i in range(len(p))) / len(p)
        if rw <= 0:
            raise ValueError("Zero RW denominator, with no floor or dropped cell")
        cells.append(dict(h=j+1, mse=mse, rw_mse=rw, relative_mse=mse/rw,
                          rmse=math.sqrt(mse), mae=math.fsum(abs(e) for e in errors)/len(p)))
    score = math.fsum(cells[h-1]["relative_mse"] for h in PRIMARY_H) / 3
    return dict(score=score, n_origins=len(p), all20=cells)


def compare_public_metrics(check, computed, public, label):
    check.same(computed["score"], public["score"], label+"/score")
    check.same(computed["n_origins"], public["n_origins"], label+"/n")
    check.same([r["relative_mse"] for r in computed["all20"]],
               public["all20_relative_mse"], label+"/all20")
    rows = {int(row["h"]): row for row in public["horizons"]}
    check.require(set(rows) == set(PRIMARY_H), label+"/horizons")
    for h in PRIMARY_H:
        for key in ("mse", "rw_mse", "relative_mse", "rmse", "mae"):
            check.same(computed["all20"][h-1][key], rows[h][key], label+f"/h{h}/"+key)


def load_prediction(path, check):
    wanted = {"prediction", "delta", "sigma", "origin", "date", "last_label_date",
              "val_prediction", "val_delta", "val_sigma", "val_origin", "val_date",
              "val_last_label_date"}
    with np.load(path, allow_pickle=False) as source:
        check.require(wanted <= set(source.files), str(path.name)+"/required_npz_fields")
        out = {key: source[key].copy() for key in wanted}
    for prefix in ("", "val_"):
        n = len(out[prefix+"origin"])
        check.require(n > 0, prefix+"nonempty")
        for key in ("prediction", "delta", "sigma"):
            check.require(out[prefix+key].shape == (n,20), prefix+key+"/shape")
        for key in ("origin", "date", "last_label_date"):
            check.require(out[prefix+key].shape == (n,), prefix+key+"/shape")
        check.require(np.all(np.diff(out[prefix+"origin"]) > 0), prefix+"ordered_origins")
        check.require(np.all(out[prefix+"date"][1:] > out[prefix+"date"][:-1]), prefix+"ordered_dates")
        check.require(np.all(out[prefix+"last_label_date"] > out[prefix+"date"]), prefix+"future_label_endpoint")
        for date in np.r_[out[prefix+"date"], out[prefix+"last_label_date"]]:
            dt.date.fromisoformat(str(date))
    check.require(np.all(out["date"] >= "2022-01-01") and np.all(out["date"] < "2023-01-01"),
                  "test_origin_in_2022")
    check.require(np.all(out["last_label_date"] < "2023-01-01"), "complete_H20_in_2022")
    check.require(np.all(out["val_last_label_date"] < "2022-01-01"), "validation_pre2022")
    return out


def schedule_hash(sources, max_steps, seed):
    rng = np.random.default_rng(int(seed))
    h = hashlib.sha256()
    # Production construction follows the approved series order, not sorted
    # gradients. This independent replay uses only public source counts.
    names = [name for name in MARKETS if name in sources]
    for _ in range(int(max_steps)):
        for name in names:
            n = int(sources[name]["n"])
            q0 = int(rng.integers(84, n-32+1))
            support = np.array([q0-84+j for j in range(64)], dtype="<i8")
            query = np.array([q0+j for j in range(32)], dtype="<i8")
            h.update(name.encode())
            h.update(support.tobytes())
            h.update(query.tobytes())
    return h.hexdigest()


def development_selection(grouped, complete):
    table = {row["model"]: row["score"] for row in grouped}
    gates = {"all_jobs_completed": bool(complete)}
    if all(k in table for k in ("transport", "profile", "covariance_profile", "rw")):
        gates.update(
            beats_profile_by_half_percent=table["transport"] < .995*table["profile"],
            beats_covariance_by_half_percent=table["transport"] < .995*table["covariance_profile"],
            does_not_lose_to_rw=table["transport"] <= 1.0)
    gates["advance"] = len(gates) == 4 and all(gates.values())
    return gates


def sampler_hash(n, seed, steps, batch_size):
    rng = np.random.default_rng(int(seed))
    h = hashlib.sha256()
    for _ in range(int(steps)):
        selected = rng.choice(int(n), size=int(batch_size), replace=int(n)<int(batch_size))
        h.update(np.asarray(selected,dtype='<i8').tobytes())
    return h.hexdigest()


def validate_checkpoint_choice(check, summary, config, selected_step, selected_score):
    model, job = summary['model'], summary['job']
    points = []
    for row in summary['validation_traces']:
        value = row.get('validation_score') if model in PROFILE else (row.get('validation') or {}).get('score')
        if value is not None:
            points.append((int(row['step']),float(value),bool(row['selected'])))
    check.require(bool(points) and points[0][0]==0,job+'/initial_validation_present')
    check.require(all(points[i][0]>points[i-1][0] for i in range(1,len(points))),job+'/ordered_validation_checks')
    best_step,best_score,_ = points[0]
    check.require(points[0][2],job+'/initial_validation_selected')
    for step,value,flag in points[1:]:
        improved = value < best_score-float(config['min_delta'])
        check.require(flag==improved,job+f'/checkpoint_selection_flag_step{step}')
        if improved:
            best_step,best_score=step,value
    check.same(best_step,selected_step,job+'/independent_selected_step')
    check.same(best_score,selected_score,job+'/independent_selected_validation')


def audit(protocol_path, public_dir, private_dir, output_dir, compare_root=True):
    protocol_path, public_dir, private_dir = map(Path, (protocol_path, public_dir, private_dir))
    protocol = read_json(protocol_path)
    tasks, seeds, static_seed = planned(protocol)
    check = Check()
    expected = {t["job"]: t for t in tasks}
    summaries = {p.name.removesuffix("_SUMMARY.json"): p for p in public_dir.glob("*_SUMMARY.json")}
    failures = {p.name.removesuffix("_FAILED.json"): p for p in public_dir.glob("*_FAILED.json")}
    check.require(not (set(summaries) & set(failures)), "no_complete_and_failed_duplicate")
    check.require(set(summaries) | set(failures) == set(expected), "exact_frozen_task_coverage")
    check.require(not ((set(summaries) | set(failures))-set(expected)), "no_unregistered_task")
    complete = not failures and len(summaries) == len(tasks)
    scores, metadata, cell_rows = {}, {}, []
    paired, case_sources, encoder_hashes = {}, {}, {}
    replay_cache, sampler_cache = {}, {}
    counters = dict(neural_fits_started=0, neural_fits_completed=0, ridge_forecast_flows=0,
                    ridge_candidate_solves=0, rw_forecast_flows=0,
                    neural_optimizer_steps_completed=0, author_source_ridge_calls=0,
                    expected_source_ridge_calls_completed_steps=0,
                    selected_step_zero_neural_flows=0, test_forward_calls=0)
    records = []
    for task in tasks:
        job, model, target, budget, seed = (task[k] for k in ("job","model","target","budget","seed"))
        if model not in STATIC:
            counters["neural_fits_started"] += 1
        if job in failures:
            failed = read_json(failures[job])
            check.require(failed["job"] == job and failed["status"] == "FAILED", job+"/failure_identity")
            receipt = failed.get("fit_receipt") or {}
            counters["author_source_ridge_calls"] += int(receipt.get("author_ridge_calls", 0))
            counters["expected_source_ridge_calls_completed_steps"] += int(receipt.get("source_ridge_calls_expected_completed_steps", 0))
            counters["neural_optimizer_steps_completed"] += int(receipt.get("steps", receipt.get("steps_completed", 0)))
            records.append(dict(**task, status="FAILED", exception=failed.get("exception"),
                                fit_receipt_present=bool(receipt)))
            continue
        summary = read_json(summaries[job])
        for key in ("job", "model", "target", "budget", "seed"):
            check.same(summary[key], task[key], job+"/"+key)
        check.require(summary["status"] == "COMPLETE", job+"/complete")
        prediction_path = private_dir/(job+"_PREDICTIONS.npz")
        check.same(digest(prediction_path), summary["private_prediction_sha256"], job+"/npz_sha")
        data = load_prediction(prediction_path, check)
        s_test = raw_score(data["prediction"], data["delta"], data["sigma"])
        s_val = raw_score(data["val_prediction"], data["val_delta"], data["val_sigma"])
        compare_public_metrics(check, s_test, summary["evaluation"], job+"/test")
        compare_public_metrics(check, s_val, summary["validation"], job+"/validation")
        data_receipt, fit_receipt = summary["data_receipt"], summary["fit_receipt"]
        for key, want in (("target",target),("budget",budget),("year",2022)):
            check.same(data_receipt[key], want, job+"/data_"+key)
        roles = data_receipt["roles"]
        sizes = {"support":60 if budget==100 else 240,
                 "meta":20 if budget==100 else 80, "val":20 if budget==100 else 80}
        for role, n in sizes.items():
            check.same(roles[role]["n"], n, job+"/"+role+"_budget")
        check.same(sum(roles[r]["n"] for r in sizes), budget, job+"/unique_disjoint_role_budget")
        check.require(roles["support"]["last_origin"]+20 < roles["meta"]["first_origin"], job+"/S_Q_purge")
        check.require(roles["meta"]["last_origin"]+20 < roles["val"]["first_origin"], job+"/Q_V_purge")
        check.require(roles["support"]["last_label_date"] < roles["meta"]["first_date"], job+"/S_Q_calendar")
        check.require(roles["meta"]["last_label_date"] < roles["val"]["first_date"], job+"/Q_V_calendar")
        for role, prefix in (("test",""),("val","val_")):
            r = roles[role]
            for field, want in (("n",len(data[prefix+"origin"])),
                                ("first_origin",int(data[prefix+"origin"][0])),
                                ("last_origin",int(data[prefix+"origin"][-1])),
                                ("first_date",str(data[prefix+"date"][0])),
                                ("last_date",str(data[prefix+"date"][-1])),
                                ("last_label_date",str(data[prefix+"last_label_date"][-1]))):
                check.same(r[field], want, job+"/"+role+"/"+field)
        check.require(data_receipt["source_cutoff_exclusive"] == roles["meta"]["first_date"],
                      job+"/source_cut_is_meta_start")
        expected_sources = [m for m in MARKETS if m != target and {m,target} != {"EEX_EUA_T3PA","EU_REPORTED"}]
        sources = data_receipt["sources"]
        check.same(sorted(sources), sorted(expected_sources), job+"/source_set")
        for name, row in sources.items():
            check.require(row["last_label_date"] < data_receipt["source_cutoff_exclusive"], job+"/"+name+"/mature")
            check.same(row["last_origin"]-row["first_origin"]+1, row["n"], job+"/"+name+"/contiguous_source")
        schedule_key = (target, tuple((name,int(sources[name]["n"])) for name in expected_sources),
                        int(protocol["config"]["max_steps"]), int(data_receipt["episode_seed"]))
        if schedule_key not in replay_cache:
            replay_cache[schedule_key] = schedule_hash(sources, schedule_key[-2], schedule_key[-1])
        check.same(data_receipt["episode_schedule_sha256"], replay_cache[schedule_key], job+"/schedule_replay")
        source_identity = json.dumps({"sources":sources,"cut":data_receipt["source_cutoff_exclusive"],
                                      "schedule":data_receipt["episode_schedule_sha256"]},sort_keys=True)
        if target in case_sources:
            check.same(source_identity, case_sources[target], job+"/same_sources_all_budgets_seeds_models")
        else:
            case_sources[target] = source_identity

        # Evaluation labels/scales/dates are identical across budgets, seeds and
        # all arms. Validation is paired per budget and nested across budgets.
        for role, prefix, pair_key in (("test","",(target,"test")),
                                      ("val","val_",(target,budget,"val"))):
            keys = ("delta","sigma","origin","date","last_label_date")
            selected = {key:data[prefix+key] for key in keys}
            if pair_key not in paired:
                paired[pair_key] = {key:value.copy() for key,value in selected.items()}
            else:
                for key in keys:
                    check.bitwise(selected[key], paired[pair_key][key], job+"/"+role+"/paired_"+key)
        counters["test_forward_calls"] += int(fit_receipt.get("test_forward_calls",0))
        if model in PROFILE:
            check.require(fit_receipt.get("eligible") is True, job+"/eligible")
            check.same(fit_receipt["selected_validation_score"], s_val["score"], job+"/selected_val_receipt")
            completed_steps = int(fit_receipt["steps"])
            check.require(0 <= int(fit_receipt["selected_step"]) <= completed_steps <= protocol["config"]["max_steps"],
                          job+"/selected_and_completed_step")
            counters["neural_fits_completed"] += 1
            counters["neural_optimizer_steps_completed"] += completed_steps
            counters["selected_step_zero_neural_flows"] += int(fit_receipt["selected_step"] == 0)
            expected_calls = completed_steps * len(expected_sources)
            check.same(fit_receipt["source_ridge_calls_expected_completed_steps"],expected_calls,job+"/source_call_formula")
            check.same(fit_receipt["author_ridge_calls"],expected_calls,job+"/actual_author_calls")
            counters["expected_source_ridge_calls_completed_steps"] += expected_calls
            counters["author_source_ridge_calls"] += int(fit_receipt["author_ridge_calls"])
            check.require(fit_receipt["test_target_values_used_in_fit"] is False, job+"/declared_test_exclusion")
            validate_checkpoint_choice(check,summary,protocol['config'],fit_receipt['selected_step'],s_val['score'])
            group, enc = ("profile",seed), fit_receipt["initial_encoder_hash"]
        elif model in GRU:
            check.same(fit_receipt["best_validation_score"],s_val["score"],job+"/selected_val_receipt")
            completed_steps = int(fit_receipt["steps_completed"])
            check.require(0 <= int(fit_receipt["best_step"]) <= completed_steps <= protocol["config"]["max_steps"],
                          job+"/selected_and_completed_step")
            counters["neural_fits_completed"] += 1
            counters["neural_optimizer_steps_completed"] += completed_steps
            counters["selected_step_zero_neural_flows"] += int(fit_receipt["best_step"] == 0)
            check.require(fit_receipt["test_labels_accessed"] is False,job+"/declared_test_exclusion")
            check.require(fit_receipt["initial_predictions_equal_rw"] is True,job+"/declared_initial_rw")
            validate_checkpoint_choice(check,summary,protocol['config'],fit_receipt['best_step'],s_val['score'])
            active_sources = sorted(sources) if model=='pooled_gru' else []
            sampling_names = [target]+active_sources
            check.same(sorted(fit_receipt['sampler_seeds']),sorted(sampling_names),job+'/sampler_names')
            base_seed=int(protocol['config']['episode_seed'])
            for name in sampling_names:
                if name==target:
                    want_seed=base_seed
                    n=sizes['support']+sizes['meta']
                else:
                    key=f'{base_seed}|baseline_source|{name}'.encode()
                    want_seed=int.from_bytes(hashlib.sha256(key).digest()[:8],'big')
                    n=int(sources[name]['n'])
                check.same(fit_receipt['sampler_seeds'][name],want_seed,job+'/'+name+'/sampler_seed')
                cache_key=(n,want_seed,completed_steps,64)
                if cache_key not in sampler_cache:
                    sampler_cache[cache_key]=sampler_hash(*cache_key)
                check.same(fit_receipt['sampled_indices_sha256'][name],sampler_cache[cache_key],job+'/'+name+'/actual_sampler_replay')
            enc = fit_receipt.get("initial_encoder_sha256",fit_receipt.get("initial_encoder_hash"))
            check.require(isinstance(enc,str) and len(enc)==64,job+"/encoder_only_hash_present")
            group = ("gru",seed)
        elif model == "ridge":
            counters["ridge_forecast_flows"] += 1
            counters["ridge_candidate_solves"] += int(fit_receipt["ridge_fits"])
            check.same(fit_receipt["ridge_fits"],5,job+"/fixed_five_alpha_solves")
            check.same(fit_receipt["validation_score"],s_val["score"],job+"/selected_val_receipt")
            candidates = summary["validation_traces"]
            check.same([c["alpha"] for c in candidates],[.01,.1,1.,10.,100.],job+"/ridge_alpha_order")
            chosen = min(enumerate(candidates),key=lambda item:(item[1]["validation_score"],item[0]))[1]
            check.same(chosen["alpha"],fit_receipt["selected_alpha"],job+"/ridge_selection")
            check.same(chosen["validation_score"],s_val["score"],job+"/ridge_selected_score")
            group = None
        else:
            counters["rw_forecast_flows"] += 1
            check.require(np.array_equal(data["prediction"],np.zeros_like(data["prediction"])),job+"/RW_zero")
            check.require(np.array_equal(data["val_prediction"],np.zeros_like(data["val_prediction"])),job+"/val_RW_zero")
            check.same(s_test["score"],1.,job+"/RW_score")
            group = None
        if group is not None:
            if group in encoder_hashes:
                check.same(enc,encoder_hashes[group],job+"/paired_encoder_initial_hash")
            else:
                encoder_hashes[group] = enc
        scores[job] = s_test
        metadata[job] = task
        records.append(dict(**task,status="COMPLETE",score=s_test["score"],validation_score=s_val["score"]))
        for split, metric in (("test",s_test),("validation",s_val)):
            for row in metric["all20"]:
                cell_rows.append(dict(**task,split=split,n_origins=metric["n_origins"],**row))

    for market in MARKETS:
        small, big = paired.get((market,100,"val")), paired.get((market,400,"val"))
        if small is not None and big is not None:
            lookup = {int(origin):i for i,origin in enumerate(big["origin"])}
            check.require(all(int(origin) in lookup for origin in small["origin"]),market+"/validation_nested")
            at = np.array([lookup[int(origin)] for origin in small["origin"]])
            for key in ("delta","sigma","date","last_label_date"):
                check.bitwise(small[key],big[key][at],market+"/validation_nested_"+key)
            check.bitwise(small["origin"][[0,-1]],big["origin"][[0,-1]],market+"/validation_equal_span")
    grouped, seed_averaged_rows = [], []
    for model in MODELS:
        completed = [t for t in tasks if t["model"]==model and t["job"] in scores]
        if not completed:
            continue
        by_budget = {}
        for budget in BUDGETS:
            values = [scores[t["job"]]["score"] for t in completed if t["budget"]==budget]
            by_budget[budget] = math.fsum(values)/len(values) if values else None
        # Strictly form every market/budget/h cell from mean seed LOSS, then
        # ratio to its common RW denominator. No prediction averaging occurs.
        group_cells, groups_complete = [], True
        for market in MARKETS:
            for budget in BUDGETS:
                ts = [t for t in completed if t["target"]==market and t["budget"]==budget]
                required = 1 if model in STATIC else len(seeds)
                if len(ts) != required:
                    groups_complete = False
                    continue
                for h in range(1,21):
                    cs = [scores[t["job"]]["all20"][h-1] for t in ts]
                    for c in cs[1:]:
                        check.same(c["rw_mse"],cs[0]["rw_mse"],model+"/"+market+"/same_seed_RW")
                    mean_mse = math.fsum(c["mse"] for c in cs)/len(cs)
                    rw = cs[0]["rw_mse"]
                    row = dict(model=model,target=market,budget=budget,h=h,seeds=len(ts),
                               mean_seed_mse=mean_mse,rw_mse=rw,relative_mse=mean_mse/rw)
                    seed_averaged_rows.append(row)
                    if h in PRIMARY_H:
                        group_cells.append(row["relative_mse"])
        score = (math.fsum(group_cells)/len(group_cells) if groups_complete else
                 math.fsum(scores[t["job"]]["score"] for t in completed)/len(completed))
        grouped.append(dict(model=model,score=score,budget100=by_budget[100],budget400=by_budget[400],
                            jobs=len(completed),full_design_support=groups_complete))
    gates = development_selection(grouped,complete)
    if compare_root:
        root = read_json(public_dir/"AGGREGATE.json")
        check.same(root["completed_jobs"],len(summaries),"root/completed_jobs")
        check.same(root["failed_jobs"],len(failures),"root/failed_jobs")
        check.require(root["resource_gate"]==gates,"root/gates")
        root_rows = {r["model"]:r for r in root["ranking"]}
        check.require(set(root_rows)=={r["model"] for r in grouped},"root/model_set")
        for row in grouped:
            for key in ("score","budget100","budget400","jobs"):
                if row[key] is None:
                    check.require(root_rows[row["model"]][key] is None,"root/"+row["model"]+"/"+key)
                else:
                    check.same(row[key],root_rows[row["model"]][key],"root/"+row["model"]+"/"+key)
    receipt = dict(status="PASS_INDEPENDENT_METRICS_AND_SELECTION",
        completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        independent_source_sha256=digest(__file__),protocol_sha256=digest(protocol_path),
        expected_prediction_flows=len(tasks),completed_prediction_flows=len(summaries),failed_flows=len(failures),
        declared_neural_seeds=list(seeds),static_seed_tag=static_seed,comparisons=check.comparisons,
        maximum_absolute_error=check.max_error,maximum_error_label=check.maximum_label,
        validation_and_test_npz_count=len(summaries),all20_split_cell_rows=len(cell_rows),
        seed_averaged_cell_rows=len(seed_averaged_rows),resource_counts=counters,
        initial_encoder_hash_groups={f"{kind}_seed{seed}":value for (kind,seed),value in encoder_hashes.items()},
        episode_schedule_replays=len(replay_cache),supervised_sampler_replays=len(sampler_cache),root_aggregate_compared=compare_root,
        runtime=dict(python=sys.version.split()[0],numpy=np.__version__),
        scope=dict(market_csv_reads=0,checkpoint_reads=0,old_npz_reads=0,confirmation_reads=0,
                   optimizer_steps=0,production_module_imports=0),
        limitations=[
            "Raw quote CSVs and model weights are not read; label/scale equality is verified among outputs, not re-derived from original quotes here.",
            "Initialization hashes are compared as emitted receipts, not recalculated from checkpoint tensor bytes.",
            "Numerical scoring agreement does not by itself prove training causality; see the separate data and runner reviews.",
            "Full source gradient/target head/QR operations are not independent fitted forecasting models.",
            "This is exposed development evidence, not a publication or new independent-confirmation certificate.",
            "Failed-flow counters use available receipts only; unreported partial operations cannot be reconstructed."
        ])
    selection = dict(status="ADVANCE_DEVELOPMENT" if gates["advance"] else "STOP_OR_RESOURCE_FAILURE",
                     estimand="equal market x budget x h{1,5,20} raw-price MSE/RW-MSE, seed loss averaged",
                     gates=gates,ranking=sorted(grouped,key=lambda r:r["score"]),
                     no_per_budget_extra_gate=True,no_bootstrap_or_p_value_computed=True)
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    dump(output_dir/"INDEPENDENT_METRIC_AUDIT.json",receipt)
    dump(output_dir/"INDEPENDENT_SELECTION.json",selection)
    dump(output_dir/"INDEPENDENT_JOB_LEDGER.json",records)
    for name,rows in (("INDEPENDENT_ALL_CELLS.csv",cell_rows),("INDEPENDENT_SEED_AVERAGED_CELLS.csv",seed_averaged_rows)):
        with (output_dir/name).open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]) if rows else ["model"])
            writer.writeheader();writer.writerows(rows)
    return receipt,selection


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--protocol",type=Path,required=True)
    parser.add_argument("--public-dir",type=Path,required=True)
    parser.add_argument("--private-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    receipt,selection=audit(args.protocol,args.public_dir,args.private_dir,args.output_dir)
    print(json.dumps({"status":receipt["status"],"flows":receipt["completed_prediction_flows"],
                      "comparisons":receipt["comparisons"],"maximum_absolute_error":receipt["maximum_absolute_error"],
                      "gates":selection["gates"]},ensure_ascii=False))


if __name__=="__main__":
    main()
