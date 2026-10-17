"""Independent SOLID shim/runner audit. Synthetic arrays and static reads only.

No market files, market checkpoints, prior prediction arrays, or confirmation
data are read. The actual author's routine performs explicitly counted head
backwards on synthetic examples; no backbone fit is run. NumPy independently
reconstructs GRU latents, neighbor choices, gradients, heads and raw forecasts.
"""
from __future__ import annotations
import argparse
import ast
import copy
from datetime import date, timedelta, datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    study = args.study.resolve()
    output = args.output or study / "backup_review/SOLID_EXECUTION_INDEPENDENT_AUDIT.json"
    sys.path.insert(0, str(study / "source"))
    import torch
    import rrc_core as core
    import solid_author_adapter as adapter
    import run_solid_reference as runner
    core.configure()
    start = time.perf_counter()
    checks, comparisons, largest = [], 0, 0.
    runs, synthetic_queries, synthetic_backwards = [], 0, 0

    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def require(label, condition):
        if not bool(condition):
            raise AssertionError(label)
        checks.append(dict(name=label, status="PASS"))

    def equal(label, observed, expected, atol=2e-11):
        nonlocal comparisons, largest
        a, b = np.asarray(observed), np.asarray(expected)
        if a.shape != b.shape:
            raise AssertionError((label, a.shape, b.shape))
        err = float(np.max(np.abs(a-b))) if a.size else 0.
        comparisons += int(a.size)
        largest = max(largest, err)
        if not np.isfinite(err) or err > atol:
            raise AssertionError((label, err, atol))
        checks.append(dict(name=label, status="PASS", values=int(a.size), max_abs_error=err))

    def expect_error(label, kind, function):
        try:
            function()
        except kind:
            require(label, True)
        else:
            raise AssertionError(label + ": error not raised")

    protocol_path = study / "provenance/SOLID_B1_PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text())
    source_hashes = {rel: sha(study / rel) for rel in protocol["source_sha256"]}
    require("frozen_source_identities", source_hashes == protocol["source_sha256"])
    require("formal_protocol_identity", sha(protocol_path) == "b44357f30ce5c7f50f2023ebb0fd3fcc17e1598038724738b0b64b56c7ca76fd")
    require("five_series_two_seeds_eight_configs", len(protocol["jobs"]) == 10 and
            len({j["market"] for j in protocol["jobs"]}) == 5 and
            set(j["seed"] for j in protocol["jobs"]) == {17, 29} and
            len(protocol["configs"]) == 8)
    require("formal_support_and_no_backbone_fits", protocol["scored_years"] == [2022, 2023] and
            protocol["counts"]["new_backbone_pretraining_fits"] == 0 and
            protocol["counts"]["streams"] == 60 and protocol["counts"]["metric_cells"] == 480 and
            protocol["counts"]["prediction_series"] == 80)
    derived = [s for s in protocol["configs"] if "equivalent_prediction_from" in s]
    specs = {s["id"]: s for s in protocol["configs"]}
    require("exact_two_recency_aliases_per_job", len(derived) == 2 and all(s["remove_distance"] and
            specs[s["equivalent_prediction_from"]]["remove_distance"] and
            s["learning_rate"] == specs[s["equivalent_prediction_from"]]["learning_rate"] and
            min(s["context_samples"], specs[s["equivalent_prediction_from"]]["context_samples"]) >=
            protocol["neighbors"] for s in derived))
    require("pre_execution_equivalence_addendum", protocol["equivalence_addendum"]["before_formal_market_streams"] == 0)
    require("private_numeric_artifacts_not_public", not ({"diagnostics", "query_origins",
            "normalized_predictions", "raw_predictions"} & set(protocol["public_adapter_receipt_keys"])))
    require("only_development_paths_in_formal_jobs", all("development" in j["data_path"].lower()
            and "confirmation" not in j["data_path"].lower() for j in protocol["jobs"]))
    require("all_fixed_pretrain_checkpoints", all("/local_only/D1/" in j["checkpoint_path"] and
            Path(j["checkpoint_path"]).name.startswith("pretrain_shrink") for j in protocol["jobs"]))
    # Parse code rather than invoking state_and_data/worker, which read markets.
    runner_source = (study / "source/run_solid_reference.py").read_text()
    tree = ast.parse(runner_source)
    funcs = {x.name: x for x in tree.body if isinstance(x, ast.FunctionDef)}
    state_code = ast.get_source_segment(runner_source, funcs["state_and_data"])
    worker_code = ast.get_source_segment(runner_source, funcs["worker"])
    require("runner_loader_pre2024_and_offline2021", '"2024-01-01"' in state_code and
            '"2021-01-01"' in state_code and 'key.startswith("T.")' in state_code)
    require("runner_checkpoint_and_offline_state_hash_checks", 'sha(entry["checkpoint_path"])' in state_code
            and 'digest_arrays(arrays(teacher))' in state_code)
    require("runner_actual_shim_called_with_fixed_cache", "run_solid_author(data, teacher, origins" in worker_code
            and "cache_latents=True" in worker_code and "artifact_sink=artifacts.append" in worker_code)
    require("runner_finite_shape_and_base_state_checks", "pred.shape != (len(origins), 20)" in worker_code
            and "not np.isfinite(pred).all()" in worker_code and "digest_arrays(arrays(teacher))" in worker_code)
    require("pilot_score_branch_is_disabled", 'if not cfg["timing_only"]:' in worker_code and
            'origins = origins[:cfg["pilot_query_count"]]' in worker_code)
    require("runner_has_no_training_entrypoint", not any(isinstance(x, ast.Call) and
            isinstance(x.func, ast.Name) and x.func.id in {"pretrain", "run_stream", "run_teacher"}
            for x in ast.walk(tree)))

    rng = np.random.default_rng(926071)
    n = 1050
    days = np.cumsum(rng.choice([1, 1, 1, 2], n)) + date(2020, 1, 1).toordinal()
    dates = np.array([date.fromordinal(int(d)).isoformat() for d in days])
    prices = 80. + np.cumsum(rng.normal(.015, .14, n))
    cov = np.exp(rng.normal(1., .3, (n, 3)))
    market = core.Market("SYNTHETIC_ONLY", dates, days, prices, cov,
                         ("a", "b", "c"), "synthetic_not_from_a_file")
    data = core.CausalData(market, dates[350])
    torch.manual_seed(3417)
    teacher = core.Teacher(5, hidden=4)
    with torch.no_grad():
        teacher.head.weight.copy_(torch.tensor(rng.normal(0., .1, (20, 4))))
        teacher.head.bias.copy_(torch.tensor(rng.normal(0., .1, 20)))
    teacher.eval()
    state = core.arrays(teacher)
    initial_hash = core.digest_arrays(state)
    sqrt_h = np.sqrt(np.arange(1, 21, dtype=float))

    def latent_numpy(x):
        h = np.zeros(4)
        for row in x:
            ai = state["encoder.weight_ih_l0"] @ row + state["encoder.bias_ih_l0"]
            ah = state["encoder.weight_hh_l0"] @ h + state["encoder.bias_hh_l0"]
            ir, iz, inn = np.split(ai, 3)
            hr, hz, hn = np.split(ah, 3)
            r, z = 1./(1.+np.exp(-(ir+hr))), 1./(1.+np.exp(-(iz+hz)))
            nn = np.tanh(inn + r*hn)
            h = (1.-z)*nn + z*h
        return h

    def manual(d, query, context, recency, lr):
        candidates = np.arange(max(60, query-20-context+1), query-20+1)
        qx, qs = d.at(query)
        # Exact epsilon convention of the author's F.pairwise_distance.
        distances = [np.linalg.norm(qx.ravel()-d.at(int(s))[0].ravel()+1e-6) for s in candidates]
        if recency:
            selected = candidates[::-1][:10]
        else:
            selected = np.array([candidates[i] for i in sorted(range(len(candidates)), key=lambda i: distances[i])[:10]])
        gw = np.zeros_like(state["head.weight"])
        gb = np.zeros_like(state["head.bias"])
        for s in selected:
            xx, ss = d.at(int(s))
            z = latent_numpy(xx)
            y = (d.market.price[s+np.arange(1, 21)]-d.market.price[s])/ss/sqrt_h
            error = state["head.weight"] @ z + state["head.bias"] - y
            gw += np.outer(2.*error/20., z)
            gb += 2.*error/20.
        w = state["head.weight"]-lr*gw
        b = state["head.bias"]-lr*gb
        normalized = w @ latent_numpy(qx) + b
        raw = d.market.price[query]+qs*sqrt_h*normalized
        return dict(origin=query, selected_origins=selected, normalized_prediction=normalized,
                    raw_prediction=raw, summed_weight_gradient=gw, summed_bias_gradient=gb,
                    adapted_head_weight=w, adapted_head_bias=b)

    def run_case(label, d, queries, context=128, recency=False, lr=.01, cache=True):
        nonlocal synthetic_queries, synthetic_backwards
        artifacts = []
        pred, receipt = adapter.run_solid_author(d, teacher, queries, context_samples=context,
             neighbors=10, learning_rate=lr, remove_distance=recency, cache_latents=cache,
             artifact_sink=artifacts.append)
        synthetic_queries += len(queries)
        synthetic_backwards += receipt["neighbor_backward_calls"]
        require(label+"_count", receipt["head_updates"] == len(queries) and
                receipt["neighbor_backward_calls"] == len(queries)*10 and receipt["checkpoint_requests"] == 1)
        require(label+"_no_query_truth_or_backbone_mutation", not receipt["query_future_labels_supplied"] and
                receipt["author_query_scoring_disabled"] and receipt["initial_offline_state_sha256"] ==
                receipt["final_offline_state_sha256"] == initial_hash and
                core.digest_arrays(core.arrays(teacher)) == initial_hash)
        for i, (q, art, diag) in enumerate(zip(queries, artifacts, receipt["diagnostics"])):
            expected = manual(d, int(q), context, recency, lr)
            for key in expected:
                equal(label+f"_q{q}_"+key, art[key], expected[key])
            equal(label+f"_q{q}_returned_raw", pred[i], expected["raw_prediction"])
            require(label+f"_q{q}_mature_boundary", diag["candidate_last"] == q-20 and
                    diag["candidate_first"] == max(60, q-20-context+1) and
                    diag["candidate_count"] == min(context, q-20-60+1) and
                    diag["maximum_label_index"] <= q and diag["encoder_unchanged"])
        runs.append(dict(label=label, queries=len(queries), cache=cache, context=context,
                         recency=recency, learning_rate=lr, backwards=receipt["neighbor_backward_calls"]))
        return pred, artifacts, receipt

    queries = np.array([400, 407, 420])
    cases = {}
    for recency in [False, True]:
        for context in [128, 512]:
            label = f"recency{int(recency)}_c{context}"
            cached = run_case(label+"_cached", data, queries, context, recency, cache=True)
            direct = run_case(label+"_direct", data, queries, context, recency, cache=False)
            equal(label+"_cache_forward_equivalence", cached[0], direct[0], 0.)
            for i in range(len(queries)):
                for key in cached[1][i]:
                    equal(label+f"_cache_artifact_{i}_"+key, cached[1][i][key], direct[1][i][key], 0.)
            cases[(recency, context)] = cached
    equal("recency_context128_512_same_effective_neighbors", cases[(True, 128)][0], cases[(True, 512)][0], 0.)
    for q, joint in zip(queries, cases[(False, 128)][0]):
        single = run_case(f"fresh_single_q{q}", data, [q], context=128)
        equal(f"query_reset_q{q}", single[0][0], joint, 0.)

    changed_market = copy.deepcopy(market)
    changed_market.price[421:] += 500. + np.arange(n-421)
    changed_market.covariates[421:] *= 25.
    changed_market.days[421:] += 50
    changed = core.CausalData(changed_market, dates[350])
    alternate = run_case("future_suffix_perturbation", changed, queries, context=512)
    equal("all_queries_unchanged_by_future_suffix", alternate[0], cases[(False, 512)][0], 0.)

    class GuardedArray:
        def __init__(self, a, cutoff):
            self.a, self.cutoff, self.read_max = a, cutoff, -1
        def __len__(self):
            return len(self.a)
        def __getitem__(self, key):
            indices = np.arange(len(self.a))[key]
            if np.size(indices):
                hi = int(np.max(indices))
                if hi > self.cutoff:
                    raise AssertionError(f"Forbidden future numerical access: {hi}>{self.cutoff}")
                self.read_max = max(self.read_max, hi)
            return self.a[key]

    guarded_market = copy.deepcopy(market)
    guarded = core.CausalData(guarded_market, dates[350])
    for name in ["price", "days", "covariates"]:
        setattr(guarded_market, name, GuardedArray(getattr(guarded_market, name), 400))
    guarded_run = run_case("guarded_prefix_only", guarded, [400], context=512)
    require("guarded_price_reads_reach_present_but_never_future", guarded_market.price.read_max == 400)
    require("guarded_day_covariate_reads_no_future", guarded_market.days.read_max <= 400 and
            guarded_market.covariates.read_max <= 400)

    early = SimpleNamespace(start=60, at=data.at, batch=data.batch)
    expect_error("insufficient_mature_pool_stops", ValueError,
                 lambda: adapter.run_solid_author(early, teacher, [85], context_samples=512))
    expect_error("past_deadline_stops", TimeoutError,
                 lambda: adapter.run_solid_author(data, teacher, [400], deadline=time.monotonic()-1))
    expect_error("non_increasing_queries_rejected", ValueError,
                 lambda: adapter.run_solid_author(data, teacher, [401, 400]))
    zero = run_case("zero_learning_rate_identity", data, [400], lr=0.)
    qx, qs = data.at(400)
    baseline = prices[400]+qs*sqrt_h*(state["head.weight"]@latent_numpy(qx)+state["head.bias"])
    equal("zero_lr_equals_offline_numpy_forecast", zero[0][0], baseline)

    # Support and metric tests use synthetic dates/quotes only.
    selected = runner.origins_for(data, [2022, 2023])
    expected = np.array([i for i in range(data.start, n-20) if dates[i][:4] in {"2022", "2023"}
                         and dates[i+20][:4] == dates[i][:4]])
    equal("same_year_completeH20_origin_selection", selected, expected, 0.)
    fake_pred = prices[selected, None] + .017*np.arange(1, 21)[None, :]
    metric_rows = core.metric_cells(data, [{"origin":int(i)} for i in selected], fake_pred, [2022, 2023])
    require("two_year_three_horizon_rows", len(metric_rows) == 6)
    for row in metric_rows:
        inds = np.array([j for j, i in enumerate(selected) if dates[i][:4] == str(row["year"])])
        oo, h = selected[inds], row["horizon"]
        y, rw, pp = prices[oo+h], prices[oo], fake_pred[inds, h-1]
        vals = dict(mse=np.mean((pp-y)**2), rw_mse=np.mean((rw-y)**2),
                    relative_mse=np.mean((pp-y)**2)/np.mean((rw-y)**2),
                    mae=np.mean(np.abs(pp-y)), directional_accuracy=np.mean(np.sign(pp-rw)==np.sign(y-rw)),
                    origins=len(oo))
        for key, value in vals.items():
            equal(f"metric_{row['year']}_h{h}_"+key, row[key], value, 0.)

    require("sources_unchanged_during_audit", {rel: sha(study/rel) for rel in source_hashes} == source_hashes)
    report = dict(status="PASS", created_utc=datetime.now(timezone.utc).isoformat(),
        scope="Independent static runner/protocol plus synthetic adapter execution only",
        protocol_sha256=sha(protocol_path), source_sha256=source_hashes,
        author_identity=adapter.verify_author(), audit_script_sha256=sha(__file__),
        check_count=len(checks), numeric_values_compared=comparisons, maximum_absolute_error=largest,
        checks=checks, synthetic_runs=runs, synthetic_query_adaptations=synthetic_queries,
        synthetic_neighbor_backward_calls=synthetic_backwards,
        market_files_read=0, market_checkpoint_files_read=0, market_prediction_files_read=0,
        confirmation_numerical_reads=0, market_fits=0, backbone_fits=0,
        actual_formal_result_metrics_audited=False,
        limits=["Code/protocol guard review does not execute formal jobs or verify formal market results.",
                "The loader parses only development paths under a pre-2024 numerical cutoff; file identity hashing reads approved development bytes.",
                "Deadline is cooperative per-query/per-forward; checkpoint reads and artifact I/O are not force-killed.",
                "Recency context128 and512 have identical selected ten origins when both pools have at least ten.",
                "Actual author retrieval/update statements are unchanged; full benchmark, data transform and GRU architecture are study adaptations.",
                "This execution audit supplies no evidence of model novelty or forecasting benefit."],
        environment=dict(python=sys.version.split()[0], numpy=np.__version__, torch=torch.__version__,
                         torch_threads=torch.get_num_threads(), dtype=str(torch.get_default_dtype())),
        execution_history=["First audit invocation stopped at the old protocol SHA guard before any synthetic or market operation because the parent had finalized the prior-to-formal exact-recency-cache addendum. Audit was updated to the new frozen SHA and exact execution counts; no model/source/protocol edits were made by this auditor."],
        seconds=time.perf_counter()-start)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k:report[k] for k in ["status", "check_count", "numeric_values_compared",
          "maximum_absolute_error", "synthetic_query_adaptations", "synthetic_neighbor_backward_calls",
          "market_files_read", "market_fits", "seconds"]}, indent=2))


if __name__ == "__main__":
    main()
