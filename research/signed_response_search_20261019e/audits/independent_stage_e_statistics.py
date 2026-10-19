"""Independent post-completion audit of the frozen stage-e experiment.

No model training or alternative model selection. NumPy independently rebuilds
canonical features, response bases, all logged fixed-theta ridge solutions and
paired statistics. PyTorch is used only to read checkpoint bytes and replay the
two already fitted GRU references; that replay is not an independent GRU solver.
Only aggregated metrics and hash ledgers are written, not private source rows.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from datetime import datetime, timezone

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np

STUDY = Path(__file__).resolve().parents[1]
PLAN_SHA = "b6f33c9c328f37f9a283275d6cc36e4e48153bc3322d45e2b8deb74e3d1d987c"
SOURCE_SHA = {
    "signed_response_models.py": "5ceb7f04c567be32f1a399cbf01dafdfc6894d3840a158b6faae4153f09eb2f8",
    "run_signed_response_search.py": "905da0ca33607799b1ca63d9efe6710df173ec33c5c44e137071e024906d990c",
    "run_neural_references.py": "489249eb417e54611123da2b1508f946f416c7f5813124e2ddd2eedfe8b63b11",
}
H = np.array([1, 5, 20])
CORE = ("rw", "ridge_price", "ridge_interaction", "legacy_fixed_response", "fixed_response",
        "static_response", "signed_shock_only", "amplitude_context", "signed_context_response", "free_horizon_context")
RESPONSES = ("fixed_response", "static_response", "signed_shock_only", "amplitude_context", "signed_context_response", "free_horizon_context")
REFERENCES = ("raw_gru", "swewt_gru")
PARAMETERS = {"fixed_response": (0, 3), "static_response": (2, 3), "signed_shock_only": (4, 3),
              "amplitude_context": (6, 3), "signed_context_response": (6, 3), "free_horizon_context": (6, 9)}
COMPARATORS = ("rw", "ridge_price", "static_response", "fixed_response", "signed_shock_only",
               "amplitude_context", "free_horizon_context", "ridge_interaction")
COMPARISON_IDS = ("full_vs_rw", "full_vs_ridge_price", "full_vs_static", "full_vs_fixed",
                  "full_vs_no_volume", "full_vs_amplitude", "full_vs_free_horizon", "full_vs_interaction")
ALPHAS = np.array([1e-5, 1e-4, 1e-3, .01, .1, 1., 10., 100.])
PRICE_ALPHAS = np.array([.01, .1, 1., 10., 100., 1000., 10000., 100000.])
LRS = (.003, .01)
SEEDS = (17, 29, 43)
MARKETS = {
    "TIANJIN_OFFICIAL_MEAN": {"role": "development", "n": 1077, "a": 646, "b": 861, "seed": 2026101901,
                              "sha": "85ca07648855db00fbf67c4e1c083db715f62ab829f74c143c3b3f67da21252a"},
    "CEA_OFFICIAL_MEAN": {"role": "confirmation", "n": 1251, "a": 750, "b": 1000, "seed": 2026101902,
                          "sha": "f9f0e09b7500b2bdb4ec8b16c61d5d5fa2b64de63a24c0cf7bd1d269aa2be75d"},
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(ok, message):
    if not bool(ok):
        raise AssertionError(message)


def close(a, b, message, atol=1e-9, rtol=1e-8):
    a, b = np.asarray(a), np.asarray(b)
    check(a.shape == b.shape, f"{message}: shape {a.shape} != {b.shape}")
    gap = float(np.max(np.abs(a-b))) if a.size else 0.
    check(np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=False), f"{message}: maximum difference {gap}")
    return gap


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def initial_rates_raw():
    return np.array([math.log(.5/(.98-.5)), math.log((.9-.5)/(.98-.9))])


def rates(raw):
    sig = np.exp(-np.logaddexp(0., -np.asarray(raw, dtype=float)))
    fast = .98*sig[..., 0]
    slow = fast+(.98-fast)*sig[..., 1]
    return np.stack((fast, slow), axis=-1)


def input_data(spec, data_root=None):
    path = Path(spec["file"]) if data_root is None else Path(data_root)/Path(spec["file"]).name
    check(sha(path) == spec["sha256"], "Canonical source bytes")
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        check(reader.fieldnames == ["date", "price", "volume_tons", "turnover_yuan"], "Canonical CSV schema")
        rows = list(reader)
    dates = np.array([r["date"] for r in rows], dtype="datetime64[D]")
    def numeric(name):
        return np.array([float(r[name]) if r[name] else np.nan for r in rows])
    p, v, turnover = numeric("price"), numeric("volume_tons"), numeric("turnover_yuan")
    check(np.all(np.diff(dates).astype(int) > 0), "Strictly increasing unique dates")
    check(np.isfinite(p).all() and (p > 0).all(), "Retained positive official prices")
    info = MARKETS[spec["market"]]
    n, a, b = len(p), len(p)*3//5, len(p)*3//5+len(p)//5
    check((n, a, b) == (info["n"], info["a"], info["b"]), "Frozen split boundaries")
    check(spec["role"] == info["role"] and spec["sha256"] == info["sha"], "Frozen source role and digest")
    r = np.concatenate(([0.], np.diff(p)))
    scale = max(float(np.sqrt(np.mean(r[1:a]**2))), 1e-7)
    valid = np.isfinite(v) & (v >= 0)
    logs = np.where(valid, np.log1p(np.maximum(np.where(valid, v, 0.), 0.)), 0.)
    surprise, usable = np.zeros(n), np.zeros(n)
    for t in range(n):
        lo = max(0, t-60)
        hist = logs[lo:t][valid[lo:t]]
        if valid[t] and len(hist) >= 10:
            surprise[t] = np.clip((logs[t]-np.median(hist))/max(float(np.std(hist)), .25), -5, 5)
            usable[t] = 1.
    origin = np.arange(64, n-20)
    token = np.column_stack((r/scale, surprise, usable))
    x = np.stack([token[t-63:t+1] for t in origin])
    y = p[origin[:, None]+H]-p[origin, None]
    masks = {"train": origin+20 < a, "val": (origin >= a) & (origin+20 < b),
             "final": origin+20 < b, "test": origin >= b}
    variance = np.maximum(np.mean(y[masks["train"]]**2, axis=0), scale**2*1e-6)
    ratio_ok = valid & (v > 0) & np.isfinite(turnover)
    ratio_gap = float(np.max(np.abs(p[ratio_ok]-turnover[ratio_ok]/v[ratio_ok])))
    check(ratio_gap <= .005000001, "Official two-decimal mean agrees within rounding, without replacement")
    return {"path": path, "p": p, "v": v, "dates": dates, "x": x, "y": y, "scale": scale,
            "variance": variance, "yn": y/scale, "vn": variance/scale**2, "origins": origin,
            "masks": masks, "n": n, "a": a, "b": b, "ratio_gap": ratio_gap,
            "valid_rows": int(valid.sum()), "usable_context_rows": int(usable.sum())}


def audit_manifest(m, d, spec):
    check(m["market"] == spec["market"] and m["role"] == spec["role"], "Manifest identity")
    check(m["row_boundaries"] == [0, d["a"], d["b"], d["n"]], "Manifest boundaries")
    check(m["counts"] == {k: int(v.sum()) for k, v in d["masks"].items()}, "Manifest mature counts")
    check(m["origin_ranges"] == {k: [int(d["origins"][v][0]), int(d["origins"][v][-1])] for k, v in d["masks"].items()}, "Manifest origin indices")
    check(m["raw_date_range"] == [str(d["dates"][0]), str(d["dates"][-1])], "Raw date range")
    fields = {"first_train_origin": ("train", 0, 0), "last_train_label": ("train", -1, 20),
              "first_validation_origin": ("val", 0, 0), "last_validation_label": ("val", -1, 20),
              "first_test_origin": ("test", 0, 0), "last_test_origin": ("test", -1, 0), "last_test_label": ("test", -1, 20)}
    for key, (part, end, lead) in fields.items():
        check(m[key] == str(d["dates"][d["origins"][d["masks"][part]][end]+lead]), "Manifest "+key)
    close(m["training_rms"], [d["scale"]], "Fixed initial-train price RMS")
    close(m["train_target_second_moment"], d["variance"], "Fixed train horizon normalization")
    close(m["observed_price_turnover_ratio_max_gap_qc_only"], d["ratio_gap"], "Rounding QC gap")
    check(m["price_retained_without_turnover_division"] is True and m["energy_inputs"] is False, "Source target retention and no energy")


def initial_state(family):
    return {"rate_raw": initial_rates_raw(), "gate": np.zeros((2, 1 if family == "signed_shock_only" else 2))}


def basis(x, family, state):
    rho = rates(state["rate_raw"])
    # Independent recurrence over exactly the supplied64 observations.
    v = np.zeros((len(x), 2))
    for i in range(64):
        v = rho*v+(1-rho)*x[:, i, 0, None]
    z = np.column_stack((np.log1p(np.abs(x[:, -1, 0])), x[:, -1, 1]*x[:, -1, 2]))
    if family == "signed_shock_only":
        z = z[:, :1]
    q = 2*np.tanh(z@np.asarray(state["gate"]).T)
    if family in ("signed_context_response", "free_horizon_context", "signed_shock_only"):
        lam = rates(np.asarray(state["rate_raw"])[None]+q)
    else:
        lam = np.broadcast_to(rho, (len(x), 2))
    amp = np.exp(q) if family == "amplitude_context" else np.ones_like(q)
    sums = np.stack([sum(lam**j for j in range(1, int(h)+1)) for h in H], axis=1)
    b = np.concatenate((np.broadcast_to(H[None, :, None], (len(x), 3, 1)), sums*(v*amp)[:, None]), axis=-1)
    if family == "free_horizon_context":
        out = np.zeros((len(x), 3, 9))
        for h in range(3):
            out[:, h, h*3:(h+1)*3] = b[:, h]
        return out
    return b


def direct_interaction_basis(x):
    r = x[:, -20:, 0]
    z = np.column_stack((np.log1p(np.abs(x[:, -1, 0])), x[:, -1, 1]*x[:, -1, 2]))
    f = np.column_stack((np.ones(len(x)), r, z, r*z[:, :1], r*z[:, 1:]))
    check(f.shape[1] == 63, "62 interactions plus intercept")
    out = np.zeros((len(x), 3, 189))
    for h in range(3):
        out[:, h, h*63:(h+1)*63] = f
    return out


def ridge_system(phi, y, variance, alpha, saved_beta=None, saved_scale=None, qr=False):
    a = (phi/np.sqrt(variance)[None, :, None]).reshape(-1, phi.shape[-1])
    yw = (y/np.sqrt(variance)[None]).reshape(-1)
    s = np.maximum(np.sqrt(np.mean(a*a, axis=0)), 1e-8)
    aa = a/s
    n = len(yw)
    gram, rhs = aa.T@aa/n, aa.T@yw/n
    matrix = gram+alpha*np.eye(len(s))
    if qr:
        aug_a = np.vstack((aa/np.sqrt(n), np.sqrt(alpha)*np.eye(len(s))))
        aug_y = np.concatenate((yw/np.sqrt(n), np.zeros(len(s))))
        gamma = np.linalg.lstsq(aug_a, aug_y, rcond=None)[0]
    else:
        gamma = np.linalg.solve(matrix, rhs)
    beta = gamma/s
    pred = phi@beta
    loss = float(np.mean((pred-y)**2/variance))
    obj = loss+float(alpha*np.sum(gamma**2))
    used_beta = beta if saved_beta is None else np.asarray(saved_beta)
    residual = matrix@(used_beta*s)-rhs
    kkt = float(np.max(np.abs(residual)))
    scaled_kkt = kkt/(1+float(np.max(np.abs(rhs))))
    check(scaled_kkt < 1e-8, "Independent normal-equation residual")
    gap = close(used_beta, beta, "Independent linear coefficients", atol=1e-8, rtol=2e-7)
    sgap = 0. if saved_scale is None else close(saved_scale, s, "Weighted basis column RMS", atol=1e-10)
    return beta, {"data_loss": loss, "profiled_objective": obj, "normal_equation_max_residual": kkt,
                  "scaled_normal_equation_residual": scaled_kkt, "coefficient_max_difference": gap,
                  "column_rms_max_difference": sgap, "augmented_lstsq_used": bool(qr)}, s


def metrics(pred, y):
    check(pred.ndim == 3 and pred.shape[1:] == y.shape, "Seed, origin and horizon shapes")
    check(np.isfinite(pred).all(), "Finite predictions")
    den = np.mean(y*y, axis=0)
    check((den > 0).all(), "Positive observed RW denominators")
    losses = np.mean((pred-y[None])**2, axis=0)
    mse = np.mean(losses, axis=0)
    ens_mse = np.mean((pred.mean(0)-y)**2, axis=0)
    return {"Q": float(np.mean(mse/den)), "mse_over_rw_by_horizon": (mse/den).tolist(),
            "mse_by_horizon": mse.tolist(), "rmse_by_horizon": np.sqrt(mse).tolist(),
            "mae_by_horizon": np.mean(np.abs(pred-y[None]), axis=(0, 1)).tolist(),
            "rw_mse_by_horizon": den.tolist(), "seed_count": len(pred),
            "ensemble_Q_descriptive": float(np.mean(ens_mse/den))}, losses


def load_npz(path, d, seed_count, ledger):
    ledger.append({"kind": "prediction_npz", "path": str(path), "sha256": sha(path), "bytes": path.stat().st_size})
    with np.load(path, allow_pickle=False) as z:
        p, y = z["predictions"].copy(), z["observed"].copy()
        check(p.shape == (seed_count, int(d["masks"]["test"].sum()), 3), "Declared prediction tensor size")
        check(np.array_equal(z["origins"], d["origins"][d["masks"]["test"]]), "Exact common origins")
        check(np.array_equal(z["horizons"], H), "Exact common horizons")
        close(y, d["y"][d["masks"]["test"]], "Independent target reconstruction", atol=1e-11, rtol=1e-11)
        check(bool(z["deterministic"][0]) == (seed_count == 1), "Determinism disclosure")
        if seed_count == 3:
            check(np.array_equal(z["seeds"], SEEDS), "Reference seed ordering")
        init = z["fixed_dynamics_refit_readout_predictions"].copy() if "fixed_dynamics_refit_readout_predictions" in z.files else None
    return p, y, init


def checkpoint(path, json_state, ledger, tensor_state=False):
    import torch
    before = sha(path)
    obj = torch.load(path, map_location="cpu", weights_only=True)
    state = obj if tensor_state else obj["state_dict"]
    check(set(state) == set(json_state), "Checkpoint/state receipt keys")
    gaps = []
    for k, v in state.items():
        arr = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
        gaps.append(close(arr, json_state[k], "Checkpoint bytes versus JSON state "+k, atol=0., rtol=0.))
    check(sha(path) == before, "Checkpoint unchanged while read")
    ledger.append({"kind": "checkpoint", "path": str(path), "sha256": before, "bytes": path.stat().st_size,
                   "json_state_max_difference": max(gaps, default=0.)})


def epoch_selection(history, tolerance, initial=None):
    best, chosen, stale = (float("inf"), 0, 0) if initial is None else (initial, 0, 0)
    first_allowed_stop = None
    for row in history:
        e, val = row["epoch"], row["validation_loss"]
        check(np.isfinite(val), "Finite recorded validation objective")
        if val < best-tolerance:
            best, chosen, stale = val, e, 0
        else:
            stale += 1
        if e >= 15 and stale >= 12 and first_allowed_stop is None:
            first_allowed_stop = e
    return chosen, best, first_allowed_stop


def audit_response(dest, family, d, ledger):
    choice = read(dest/"SELECTION.json")
    trials = choice["trials"]
    check(len(trials) == (8 if family == "fixed_response" else 16), "Complete alpha/LR grid")
    expected = {(float(a), None if family == "fixed_response" else lr) for a in ALPHAS for lr in ((None,) if family == "fixed_response" else LRS)}
    check({(r["alpha"], r["learning_rate"]) for r in trials} == expected, "Exact response grid entries")
    m = d["masks"]
    audit_rows = []
    for trial in trials:
        path = dest/trial["receipt"]
        r = read(path)
        check(r["family"] == family and r["deterministic"] is True, "Response receipt identity")
        h = r["history"]
        check([z["epoch"] for z in h] == list(range(r["epochs_executed"]+1)), "Complete epoch0..last trace")
        e, best, stop = epoch_selection(h, r["config"]["selection_tolerance"])
        check(e == r["selected_epoch"] == trial["selected_epoch"], "Own validation-selected epoch")
        close(best, r["validation_loss"], "History-selected loss")
        close(best, trial["validation_loss"], "Trial-selected loss")
        if family == "fixed_response":
            check(r["epochs_executed"] == 0, "Fixed response has no nonlinear updates")
        else:
            check(r["epochs_executed"] == (min(stop, 80) if stop is not None else 80), "Prespecified early stopping")
        gradients = [v["gradient_norm_before_clip"] for v in h if "gradient_norm_before_clip" in v]
        check(len(gradients) == r["epochs_executed"] and all(np.isfinite(g) and g >= 0 for g in gradients), "Nonlinear gradient trace")
        check(sum(g > 0 for g in gradients) == r["positive_gradient_steps"] == trial["positive_gradient_steps"], "Count actual logged positive gradients")
        check(r["linear_readout_fitted_at_epoch0"] is True, "Fitted epoch0 disclosed")
        state = r["state_dict"]
        checkpoint(path.with_suffix(".pt"), state, ledger)
        phi = basis(d["x"], family, state)
        _, linear, _ = ridge_system(phi[m["train"]], d["yn"][m["train"]], d["vn"], trial["alpha"], state["coefficients"], state["head_column_scale"])
        vl = float(np.mean((phi[m["val"]]@np.asarray(state["coefficients"])-d["yn"][m["val"]])**2/d["vn"]))
        vgap = close(vl, best, "Independent actual validation forecast loss", atol=1e-9, rtol=2e-8)
        close(linear["data_loss"], h[e]["training_data_loss"], "Chosen checkpoint training loss")
        close(linear["profiled_objective"], h[e]["training_profiled_objective"], "Chosen checkpoint profiled objective")
        close(rates(state["rate_raw"]), h[e]["historical_rates"], "Chosen historical rates")
        audit_rows.append({"kind": "response_selection", "family": family, "receipt": str(path), "alpha": trial["alpha"], "learning_rate": trial["learning_rate"],
                           "selected_epoch": e, "epochs_executed": r["epochs_executed"], "optimizer_steps": len(gradients),
                           "positive_gradient_steps": r["positive_gradient_steps"], "linear_readout_solves": len(h),
                           "validation_loss_max_difference": vgap, **linear})
    selected = min(trials, key=lambda r: (r["validation_loss"], r["alpha"], r["learning_rate"] or 0, r["selected_epoch"]))
    check(selected == choice["selected"], "Exact frozen alpha/LR/epoch tie rule")
    r = read(dest/"REFIT.json")
    check(r["selected_epoch"] == r["epochs_executed"] == selected["selected_epoch"], "Final refit retains exact selected number of updates")
    check(r["validation_loss"] is None and all(v["validation_loss"] is None for v in r["history"]), "No refit validation reselection")
    check([v["epoch"] for v in r["history"]] == list(range(r["epochs_executed"]+1)), "Refit complete trace")
    gradients = [v["gradient_norm_before_clip"] for v in r["history"] if "gradient_norm_before_clip" in v]
    check(len(gradients) == r["epochs_executed"] and sum(g > 0 for g in gradients) == r["positive_gradient_steps"], "Refit actual gradient counts")
    state = r["state_dict"]
    checkpoint(dest/"refit.pt", state, ledger)
    phi = basis(d["x"], family, state)
    _, linear, _ = ridge_system(phi[m["final"]], d["yn"][m["final"]], d["vn"], selected["alpha"], state["coefficients"], state["head_column_scale"], qr=True)
    close(linear["data_loss"], r["history"][-1]["training_data_loss"], "Final training loss")
    close(linear["profiled_objective"], r["history"][-1]["training_profiled_objective"], "Final profiled loss")
    pred = phi[m["test"]]@np.asarray(state["coefficients"])*d["scale"]
    init_phi = basis(d["x"], family, initial_state(family))
    init_beta, _, _ = ridge_system(init_phi[m["final"]], d["yn"][m["final"]], d["vn"], selected["alpha"], qr=True)
    init_pred = init_phi[m["test"]]@init_beta*d["scale"]
    nonlin, lin = PARAMETERS[family]
    check(r["parameter_counts"] == {"nonlinear_optimizer_parameters": nonlin, "exactly_fitted_linear_coefficients": lin, "total_fitted_scalars": nonlin+lin}, "Fitted parameter counts")
    init_state = initial_state(family)
    for k, logged in r["selected_nonlinear_parameter_max_change"].items():
        close(logged, float(np.max(np.abs(np.asarray(state[k])-init_state[k]))), "Retained nonlinear parameter movement")
    audit_rows.append({"kind": "response_final", "family": family, "receipt": str(dest/"REFIT.json"), "alpha": selected["alpha"],
                       "learning_rate": selected["learning_rate"], "selected_epoch": r["selected_epoch"],
                       "epochs_executed": r["epochs_executed"], "optimizer_steps": len(gradients),
                       "positive_gradient_steps": r["positive_gradient_steps"], "linear_readout_solves": len(r["history"]), **linear})
    return pred, init_pred, audit_rows, {"alpha": selected["alpha"], "learning_rate": selected["learning_rate"],
            "selected_epoch": selected["selected_epoch"], "historical_rates": rates(state["rate_raw"]).tolist(),
            "gate": state["gate"], "coefficients": state["coefficients"], "fitted_parameters": nonlin+lin}


def price_ridge(features, y, train, evaluate, alpha):
    f = features[train]
    mu, s = f.mean(0), f.std(0)
    s = np.where(s > 0, s, 1.)
    a, ae = (f-mu)/s, (features[evaluate]-mu)/s
    xoff, yoff = a.mean(0), y[train].mean(0)
    ac, yc = a-xoff, y[train]-yoff
    matrix, rhs = ac.T@ac+alpha*np.eye(a.shape[1]), ac.T@yc
    b = np.linalg.solve(matrix, rhs)
    intercept = yoff-xoff@b
    residual = matrix@b-rhs
    return ae@b+intercept, {"mean": mu, "scale": s, "coefficients": b.T, "intercept": intercept,
                           "normal_equation_max_residual": float(np.max(np.abs(residual)))}


def legacy_numpy(x):
    # Independently implement the frozen initial transient-allocation formula.
    # The production legacy usesfloat32; this mathematical replay usesfloat64
    # and is compared with an explicit native-unit rounding tolerance.
    rho, w = np.array([.5, .9]), np.array([.05, .05])
    state = np.zeros((len(x), 2))
    for t in range(64):
        innovation = x[:, t, 0]-np.sum(state*(rho-1), axis=1)
        state = rho*state+w*innovation[:, None]
    return state@(rho[None]**H[:, None]-1).T


def audit_classical(dest, family, d):
    choice, m = read(dest/"SELECTION.json"), d["masks"]
    if family == "rw":
        return np.zeros_like(d["y"][m["test"]]), [], {"fitted_parameters": 0}
    if family == "legacy_fixed_response":
        check(choice["never_refitted"] is True and choice["fitted_parameters"] == 0, "Legacy formula never refitted")
        return legacy_numpy(d["x"][m["test"]])*d["scale"], [], choice
    trials = choice["trials"]
    expected = PRICE_ALPHAS if family == "ridge_price" else ALPHAS
    check(np.array_equal([t["alpha"] for t in trials], expected), "Classical exact alpha grid")
    rows = []
    phi = direct_interaction_basis(d["x"]) if family == "ridge_interaction" else None
    for t in trials:
        if family == "ridge_price":
            pv, fit = price_ridge(d["x"][:, -20:, 0], d["y"], m["train"], m["val"], t["alpha"])
            score = float(np.mean((pv-d["y"][m["val"]])**2/d["variance"]))
        else:
            beta, fit, _ = ridge_system(phi[m["train"]], d["yn"][m["train"]], d["vn"], t["alpha"])
            score = float(np.mean((phi[m["val"]]@beta-d["yn"][m["val"]])**2/d["vn"]))
        gap = close(score, t["validation_loss"], "Independent classical validation score", atol=1e-8, rtol=2e-7)
        rows.append({"kind": "classical_selection", "family": family, "alpha": t["alpha"], "optimizer_steps": 0,
                     "positive_gradient_steps": 0, "linear_readout_solves": 1, "validation_loss_max_difference": gap,
                     "normal_equation_max_residual": fit["normal_equation_max_residual"]})
    selected = min(trials, key=lambda z: (z["validation_loss"], z["alpha"]))
    check(selected == choice["selected"], "Classical validation-selected alpha")
    if family == "ridge_price":
        pred, fit = price_ridge(d["x"][:, -20:, 0], d["y"], m["final"], m["test"], selected["alpha"])
        for field, key in (("final_column_mean", "mean"), ("final_column_scale", "scale"), ("coefficients", "coefficients"), ("intercept", "intercept")):
            close(choice[field], fit[key], "Classical final "+field, atol=1e-8, rtol=2e-7)
    else:
        beta, fit, _ = ridge_system(phi[m["final"]], d["yn"][m["final"]], d["vn"], selected["alpha"], choice["coefficients"], choice["column_scale"], qr=True)
        pred = phi[m["test"]]@beta*d["scale"]
    rows.append({"kind": "classical_final", "family": family, "alpha": selected["alpha"], "optimizer_steps": 0,
                 "positive_gradient_steps": 0, "linear_readout_solves": 1, "normal_equation_max_residual": fit["normal_equation_max_residual"]})
    return pred, rows, {"alpha": selected["alpha"], "validation_loss": selected["validation_loss"]}


def reference_sequences(d, family, dependency):
    if family == "raw_gru":
        return d["x"].astype("float32")
    import inspect
    from ewtpy import EWT1D
    check(sha(inspect.getsourcefile(EWT1D)) == dependency["ewtpy_source_sha256"], "Frozen EWT library source")
    out = []
    for i, t in enumerate(d["origins"]):
        raw = d["p"][t-63:t+1].copy()
        bands = EWT1D(raw, N=3, log=0, detect="locmax", completion=1, reg="average", lengthFilter=10, sigmaFilter=5)[0]
        check(bands.shape == (64, 3) and np.isfinite(bands).all(), "Past-only EWT output")
        channels = np.column_stack((raw[-17:]-d["p"][t], bands[-17:, 0]-d["p"][t], bands[-17:, 1:]))/d["scale"]
        out.append(np.column_stack((channels.astype("float32"), d["x"][i, -17:, 1:].astype("float32"))))
    return np.asarray(out, dtype="float32")


def reference_model(family, seed, saved=None):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    class ReplayGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.GRU(3 if family == "raw_gru" else 6, 32, batch_first=True)
            self.head = nn.Linear(32, 3)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        def forward(self, x):
            z, _ = self.encoder(x)
            return self.head(z[:, -1])
    model = ReplayGRU()
    initial = {k: v.detach().numpy().copy() for k, v in model.state_dict().items()}
    if saved is not None:
        model.load_state_dict({k: torch.as_tensor(v, dtype=torch.float32) for k, v in saved.items()})
    model.eval()
    return model, initial


def reference_prediction(model, x):
    import torch
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(x[i:i+256])).numpy() for i in range(0, len(x), 256)])


def audit_reference(dest, family, d, dependency, ledger):
    x = reference_sequences(d, family, dependency)
    choice, m = read(dest/"SELECTION.json"), d["masks"]
    check([r["learning_rate"] for r in choice["trials"]] == list(LRS), "Reference two-LR budget")
    yn, vn = d["yn"].astype("float32"), d["vn"].astype("float32")
    audit_rows = []
    for t in choice["trials"]:
        path = dest/t["receipt"]
        r = read(path)
        check(r["family"] == family and r["seed"] == 17 and r["selection"] is True and r["subset"] == "train", "Reference selection identity")
        initial_val = float(np.mean(yn[m["val"]]**2/vn))
        close(initial_val, r["initial_validation_loss"], "Reference true zero-head initial validation", atol=1e-7)
        e, best, stop = epoch_selection(r["history"], 1e-7, initial=initial_val)
        check([z["epoch"] for z in r["history"]] == list(range(1, r["epochs_executed"]+1)), "Reference complete history")
        check(r["epochs_executed"] == (min(stop, 80) if stop is not None else 80), "Reference early stopping")
        check(e == r["selected_epoch"] == t["selected_epoch"], "Reference own selected epoch")
        close(best, t["validation_loss"], "Reference history-selected loss", atol=1e-7)
        checkpoint(path.with_suffix(".pt"), r["state_dict"], ledger, tensor_state=True)
        model, _ = reference_model(family, 17, r["state_dict"])
        pv = reference_prediction(model, x[m["val"]])
        score = float(np.mean((pv-yn[m["val"]])**2/vn))
        gap = close(score, best, "Reference checkpoint validation replay", atol=3e-7, rtol=3e-7)
        steps = math.ceil(int(m["train"].sum())/128)*r["epochs_executed"]
        check(0 <= r["positive_gradient_steps"] <= steps, "Reference positive steps bounded by executed batches")
        audit_rows.append({"kind": "reference_selection", "family": family, "receipt": str(path), "seed": 17,
                           "selected_epoch": e, "epochs_executed": r["epochs_executed"], "optimizer_steps": steps,
                           "positive_gradient_steps": r["positive_gradient_steps"], "linear_readout_solves": 0,
                           "validation_loss_max_difference": gap})
    selected = min(choice["trials"], key=lambda z: (z["validation_loss"], z["learning_rate"], z["selected_epoch"]))
    check(selected == choice["selected"], "Reference LR/epoch rule")
    pred = []
    for seed in SEEDS:
        path = dest/f"refit_seed{seed}.json"
        r = read(path)
        check(r["seed"] == seed and r["subset"] == "final" and r["selection"] is False, "Reference refit identity")
        check(r["epochs_executed"] == r["selected_epoch"] == selected["selected_epoch"], "Reference final exact epoch budget")
        check(r["validation_loss"] is None and all(z["validation_loss"] is None for z in r["history"]), "Reference no final validation selection")
        checkpoint(path.with_suffix(".pt"), r["state_dict"], ledger, tensor_state=True)
        model, initial = reference_model(family, seed, r["state_dict"])
        pp = reference_prediction(model, x[m["test"]])*d["scale"]
        if r["selected_epoch"] == 0:
            check(np.array_equal(pp, np.zeros_like(pp)), "Reference epoch0 is actual RW")
        pred.append(pp)
        steps = math.ceil(int(m["final"].sum())/128)*r["epochs_executed"]
        check(0 <= r["positive_gradient_steps"] <= steps, "Reference final gradient-step bounds")
        movement = float(np.sqrt(sum(np.sum((np.asarray(r["state_dict"][k], dtype="float32")-v).astype(float)**2) for k, v in initial.items())))
        close(movement, r["attempted_parameter_change_l2"], "Reference actual final parameter change", atol=1e-8, rtol=1e-7)
        audit_rows.append({"kind": "reference_final", "family": family, "receipt": str(path), "seed": seed,
                           "selected_epoch": r["selected_epoch"], "epochs_executed": r["epochs_executed"],
                           "optimizer_steps": steps, "positive_gradient_steps": r["positive_gradient_steps"],
                           "linear_readout_solves": 0, "parameter_change_l2": movement})
    return np.asarray(pred), audit_rows, {"learning_rate": selected["learning_rate"], "selected_epoch": selected["selected_epoch"]}


def intervals(observed, losses, block, rng, replicates=50000):
    names = list(losses)
    n, p = len(observed), len(names)
    loss_matrix = np.stack([losses[k] for k in names], axis=1).reshape(n, p*3)
    rw = observed**2
    draws = np.empty((replicates, p))
    invalid = 0
    for start in range(0, replicates, 500):
        count = min(500, replicates-start)
        starts = rng.integers(0, n, size=(count, math.ceil(n/block)))
        ix = ((starts[..., None]+np.arange(block)) % n).reshape(count, -1)[:, :n]
        weights = np.zeros((count, n))
        np.add.at(weights, (np.arange(count)[:, None], ix), 1.)
        den = weights@rw/n
        mse = (weights@loss_matrix/n).reshape(count, p, 3)
        bad = np.any(den <= 0, axis=1)
        invalid += int(bad.sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            draws[start:start+count] = np.mean(mse/den[:, None], axis=2)
    main = names.index("signed_context_response")
    result = {}
    for c in COMPARATORS+REFERENCES:
        if c not in names:
            continue
        diff = draws[:, names.index(c)]-draws[:, main]
        result[c] = {"ci95_marginal": None if invalid else np.quantile(diff, [.025, .975], method="linear").tolist(),
                     "ci99375_core_family95": None if invalid or c in REFERENCES else np.quantile(diff, [.003125, .996875], method="linear").tolist()}
    return {"block_length": block, "replicates": replicates, "invalid_rw_denominator_draws": invalid,
            "comparisons": result}


def self_test():
    y = np.array([[1., 1., 1.], [2., 2., 2.]])
    p = np.stack((y+1, y-1))
    m, _ = metrics(p, y)
    close(m["Q"], .4, "Mean seed loss is not ensemble loss")
    close(m["ensemble_Q_descriptive"], 0., "Ensemble diagnostic")
    y2 = np.repeat(np.arange(1., 32.)[:, None], 3, axis=1)
    losses = {"rw": y2**2, "signed_context_response": .8*y2**2}
    for block in (20, 40):
        r = intervals(y2, losses, block, np.random.default_rng(123), 1200)
        close(r["comparisons"]["rw"]["ci95_marginal"], [.2, .2], "Known paired block effect")
        close(r["comparisons"]["rw"]["ci99375_core_family95"], [.2, .2], "Known family interval")
    rng = np.random.default_rng(713)
    x = rng.normal(size=(17, 64, 3))
    x[:, :, 2] = 1.
    st = initial_state("signed_context_response")
    b = basis(x, "signed_context_response", st)
    rho = rates(st["rate_raw"])
    explicit_states = x[:, :, 0]@((1-rho)[None]*rho[None]**np.arange(63, -1, -1)[:, None])
    for j, h in enumerate(H):
        close(b[:, j, 1:], explicit_states*sum(rho**k for k in range(1, int(h)+1)), "Independent recurrent finite-window basis")
    yy = rng.normal(size=(17, 3))
    beta, result, s = ridge_system(b, yy, np.array([1., 2., 3.]), .1, qr=True)
    ridge_system(b, yy, np.array([1., 2., 3.]), .1, beta, s)
    bad = intervals(np.zeros((4, 3)), {"rw": np.zeros((4, 3)), "signed_context_response": np.zeros((4, 3))}, 20, rng, 20)
    check(bad["invalid_rw_denominator_draws"] == 20 and bad["comparisons"]["rw"]["ci95_marginal"] is None, "No silent discard of zero denominators")
    return {"status": "PASS_SYNTHETIC_ARRAYS_ONLY", "market_fits": 0, "market_scores": 0,
            "tests": ["mean-seed squared error versus ensemble", "paired circular blocks20/40", "eight-family quantiles", "finite-window recurrence versus explicit lag sum", "augmented least squares and KKT", "zero-denominator policy"],
            "synthetic_kkt_max_residual": result["normal_equation_max_residual"]}


def audit(args):
    import torch
    torch.set_num_threads(1)
    core, neural = Path(args.core_run), Path(args.neural_run) if args.neural_run else None
    plan_path, frozen_path = Path(args.plan), STUDY/"provenance/FROZEN_EXECUTION_MANIFEST.json"
    check(sha(plan_path) == PLAN_SHA, "Frozen statistical plan bytes")
    plan, frozen = read(plan_path), read(frozen_path)
    check(frozen["statistical_plan_sha256"] == PLAN_SHA and frozen["source_sha256"] == SOURCE_SHA, "Local pre-fit freeze identity")
    check({k: sha(STUDY/"source"/k) for k in SOURCE_SHA} == SOURCE_SHA, "All frozen core/reference sources unchanged")
    complete, protocol = read(core/"EXECUTION_COMPLETE.json"), read(core/"PROTOCOL_BEFORE_RUN.json")
    check(complete["status"] == "completed" and complete["summary_rows"] == 20 and complete["source_hashes_unchanged"] and complete["plan_hash_unchanged"], "Core completion gate")
    check(protocol["frozen_analysis_plan_sha256"] == PLAN_SHA, "Core execution used frozen plan")
    check(set(protocol["families"]) == set(CORE) and protocol["deterministic"] and protocol["seed_repeats"] == 0, "Ten deterministic core arms")
    check(protocol["source_sha256"] == {k: SOURCE_SHA[k] for k in ("signed_response_models.py", "run_signed_response_search.py")}, "Core execution source identity")
    check(protocol["datasets"] == frozen["datasets"], "Source identity freeze versus execution")
    check([c["comparator"] for c in plan["eight_comparisons_one_family_within_each_market"]] == list(COMPARATORS), "Unchanged eight core comparisons")
    check([c["id"] for c in plan["eight_comparisons_one_family_within_each_market"]] == list(COMPARISON_IDS), "Comparison IDs")
    check(plan["uncertainty"]["replicates_per_market_and_block"] == 50000, "Frozen bootstrap budget")
    if neural:
        nc, np_ = read(neural/"EXECUTION_COMPLETE.json"), read(neural/"PROTOCOL_BEFORE_RUN.json")
        check(nc["status"] == "completed" and nc["summary_rows"] == 4 and nc["source_hashes_unchanged"] and nc["plan_hash_unchanged"], "Reference completion gate")
        check(np_["source_sha256"] == SOURCE_SHA and np_["statistical_plan_sha256"] == PLAN_SHA, "Reference source and plan freeze")
        check(np_["datasets"] == protocol["datasets"] and np_["families"] == list(REFERENCES), "Reference same data and fixed families")
    else:
        np_ = None
    ledger, fit_rows, markets, native = [], [], {}, []
    summary_index = {(r["market"], r["family"]): r for r in read(core/"ALL_SUMMARIES.json")}
    if neural:
        summary_index.update({(r["market"], r["family"]): r for r in read(neural/"SUMMARY.json")})
    for spec in protocol["datasets"]:
        market = spec["market"]
        d = input_data(spec, args.data_root)
        cell = f"{market}_fixed602020"
        manifest = read(core/cell/"INPUT_MANIFEST.json")
        audit_manifest(manifest, d, spec)
        if neural:
            check(read(neural/cell/"INPUT_MANIFEST.json") == manifest, "References and core identical input manifests")
        metric, loss, selections, learning, replay_gaps = {}, {}, {}, {}, {}
        all_families = CORE+(REFERENCES if neural else ())
        for family in all_families:
            root = neural if family in REFERENCES else core
            dest = root/cell/family
            p, y, init = load_npz(dest/"predictions.npz", d, 3 if family in REFERENCES else 1, ledger)
            met, los = metrics(p, y)
            for saved in (read(dest/"SUMMARY.json"), summary_index[(market, family)]):
                close(met["Q"], saved["mean_mse_over_rw"], "Summary Q from NPZ")
                for k in ("mse_by_horizon", "mse_over_rw_by_horizon", "rw_mse_by_horizon"):
                    close(met[k], saved[k], "Summary "+k)
            if family in RESPONSES:
                pred, init_pred, rows, sel = audit_response(dest, family, d, ledger)
                check(init is not None, "Fixed-dynamics diagnostic present")
                close(init, init_pred[None], "Independent fixed-dynamics final-readout diagnostic", atol=2e-8)
                imet, _ = metrics(init, y)
                learning[family] = {"fixed_dynamics_same_alpha_final_readout_Q": imet["Q"], "fitted_Q": met["Q"],
                                    "nonlinear_fit_increment_RW_pp": 100*(imet["Q"]-met["Q"]),
                                    "interpretation": "Same selected alpha, initial rates/gates, final supervised readout; diagnostic only, no extra candidate selection or test."}
                replay = pred[None]
            elif family in REFERENCES:
                replay, rows, sel = audit_reference(dest, family, d, np_["dependencies"], ledger)
            else:
                pred, rows, sel = audit_classical(dest, family, d)
                replay = pred[None]
            gap = close(p, replay, "Saved actual forecast replay "+market+"/"+family,
                        atol=1e-5 if family == "legacy_fixed_response" else (3e-6 if family in REFERENCES else 2e-8), rtol=3e-7 if family in REFERENCES else 2e-8)
            replay_gaps[family] = gap
            for row in rows:
                row["market"] = market
            fit_rows.extend(rows)
            metric[family], loss[family], selections[family] = met, los, sel
            for j, h in enumerate(H):
                native.append({"market": market, "role": spec["role"], "family": family, "horizon": int(h), "n_origins": len(y),
                               "MSE": met["mse_by_horizon"][j], "RMSE": met["rmse_by_horizon"][j], "MAE": met["mae_by_horizon"][j],
                               "MSE_over_RW": met["mse_over_rw_by_horizon"][j], "equal_horizon_Q": met["Q"]})
        rng = np.random.default_rng(MARKETS[market]["seed"])
        ci20 = intervals(d["y"][d["masks"]["test"]], loss, 20, rng)
        ci40 = intervals(d["y"][d["masks"]["test"]], loss, 40, rng)
        effects = {}
        for family in COMPARATORS+(REFERENCES if neural else ()):
            delta = metric[family]["Q"]-metric["signed_context_response"]["Q"]
            effects[family] = {"effect_RW_units": delta, "effect_RW_percentage_points": 100*delta,
                               "relative_risk_reduction": 1-metric["signed_context_response"]["Q"]/metric[family]["Q"],
                               "classification": "secondary_reference_marginal_only" if family in REFERENCES else "fixed_core_family",
                               "block20": ci20["comparisons"][family], "block40": ci40["comparisons"][family]}
        markets[market] = {"role": spec["role"], "source_sha256": spec["sha256"], "price_definition": spec["price_definition"],
                           "input_manifest": manifest, "metrics": metric, "selections": selections, "learning_diagnostics": learning,
                           "forecast_replay_max_differences": replay_gaps, "effects": effects,
                           "bootstrap": {"20": {k: v for k, v in ci20.items() if k != "comparisons"}, "40": {k: v for k, v in ci40.items() if k != "comparisons"}}}
        print(json.dumps({"audited_market": market, "role": spec["role"], "primary_Q": metric["signed_context_response"]["Q"],
                          "primary_vs_RW_pp": effects["rw"]["effect_RW_percentage_points"],
                          "core_family95_RW_ci20_pp": [100*x for x in effects["rw"]["block20"]["ci99375_core_family95"]]}, ensure_ascii=False), flush=True)
    kinds = {k: sum(r["kind"] == k for r in fit_rows) for k in sorted({r["kind"] for r in fit_rows})}
    check(kinds["response_selection"] == 176 and kinds["response_final"] == 12 and kinds["classical_selection"] == 32 and kinds["classical_final"] == 4, "All224 core fits audited")
    if neural:
        check(kinds["reference_selection"] == 8 and kinds["reference_final"] == 12, "All20 reference fits audited")
    check(sum(v["kind"] == "prediction_npz" for v in ledger) == (24 if neural else 20), "All prediction artifacts")
    for v in ledger:
        check(sha(v["path"]) == v["sha256"], "Output artifact unchanged during audit")
    check(sha(plan_path) == PLAN_SHA and {k: sha(STUDY/"source"/k) for k in SOURCE_SHA} == SOURCE_SHA, "Frozen identity at audit end")
    counts = {"fitted_tasks_by_kind": kinds, "total_logged_fit_tasks_audited": len(fit_rows), "fixed_formula_arms_not_fits": 4,
              "diagnostic_initial_dynamics_final_readout_solves": 12, "prediction_npz_count": sum(v["kind"] == "prediction_npz" for v in ledger),
              "checkpoint_count": sum(v["kind"] == "checkpoint" for v in ledger),
              "nonlinear_optimizer_steps": sum(r["optimizer_steps"] for r in fit_rows),
              "logged_positive_gradient_steps": sum(r["positive_gradient_steps"] for r in fit_rows),
              "linear_readout_solves_in_logged_fits": sum(r["linear_readout_solves"] for r in fit_rows),
              "linear_readout_solves_including_diagnostics": 12+sum(r["linear_readout_solves"] for r in fit_rows),
              "selected_zero_nonlinear_epoch_response_refits": sum(r["kind"] == "response_final" and r["selected_epoch"] == 0 for r in fit_rows),
              "selected_zero_epoch_neural_refits": sum(r["kind"] == "reference_final" and r["selected_epoch"] == 0 for r in fit_rows),
              "gradient_count_limitation": "VP counts checked against every per-epoch logged norm. GRU positive-gradient counts are receipt values bounded by exact batch/epoch update counts; no backward retraining was performed."}
    maximums = {key: max((float(r.get(key, 0.)) for r in fit_rows), default=0.) for key in
                ("normal_equation_max_residual", "scaled_normal_equation_residual", "coefficient_max_difference", "column_rms_max_difference", "validation_loss_max_difference")}
    result = {"status": "PASS_INDEPENDENT_NUMPY_RECONSTRUCTION_AND_FIXED_STATISTICS", "created_utc": datetime.now(timezone.utc).isoformat(),
              "statistical_plan_sha256": PLAN_SHA, "source_sha256": SOURCE_SHA, "audit_script_sha256": sha(__file__),
              "core_run": str(core.resolve()), "neural_run": str(neural.resolve()) if neural else None,
              "independence": "Canonical parsing, rolling context, finite filter basis, standardized ridge equations, KKT and statistics independently reconstructed in NumPy. GRU weights are replayed with PyTorch's GRU operator, and EWT with the frozen library, without training; these are not independent algorithm implementations.",
              "scope": "Tianjin development and CEA confirmation remain separate. No extra candidates, seeds, block lengths, tests or fits are selected by this audit.",
              "counts": counts, "maximum_numerical_discrepancies": maximums, "markets": markets, "fit_audit_rows": fit_rows, "artifact_ledger": ledger,
              "limitations": ["One CEA source tail cannot establish generality across all carbon markets.", "H20 targets overlap; circular-block percentile intervals are approximate and both20/40 sensitivities are retained.", "Eight core intervals target family95 within each market. Two neural reference intervals are marginal95 only, outside that family.", "Current public historical source snapshots are not certified per-day vintage archives.", "VP epoch0 includes a supervised linear readout, unlike the completely fixed legacy formula.", "The fixed-dynamics same-alpha diagnostic is an after-fit attribution record, not another selected competitor."]}
    out = Path(args.out)
    dump(out/"INDEPENDENT_STATISTICAL_AUDIT_ZH.json", result)
    with (out/"NATIVE_METRICS.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(native[0])); w.writeheader(); w.writerows(native)
    render_report(out/"INDEPENDENT_STATISTICAL_AUDIT_ZH.md", result)
    return result


def fmt_ci(v):
    return "不可计算" if v is None else f"[{100*v[0]:+.4f}, {100*v[1]:+.4f}]"


def render_report(path, result):
    lines = ["# 阶段 e 独立统计及数值审计", "", f"状态：{result['status']}。", "",
             "从完整运行的NPZ逐行复算，并用独立NumPy重建输入、指数状态、岭回归方程及验证选择。GRU检查点使用PyTorch只读回放，EWT使用冻结库；不将这两项回放称为独立算法重实现。未新增拟合、候选选择或显著性检验。", "",
             "## 指标与角色", "", "Q为H1/H5/H20各自MSE除以同期限RW MSE后的等权平均；Q越低越好，RW=1。神经参考先对种子平方误差取平均，ensemble另列。天津为已暴露开发，CEA为这轮固定设计的单来源确认，二者不合并。", ""]
    for market, data in result["markets"].items():
        lines += [f"## {market}：{data['role']}", "", f"目标：{data['price_definition']}。", "",
                  f"测试原点{data['input_manifest']['counts']['test']}个，{data['input_manifest']['first_test_origin']}至{data['input_manifest']['last_test_origin']}；最末H20目标{data['input_manifest']['last_test_label']}。", "",
                  "| 模型 | Q | H1 MSE/RW | H5 MSE/RW | H20 MSE/RW |", "|---|---:|---:|---:|---:|"]
        for name, m in data["metrics"].items():
            h = m["mse_over_rw_by_horizon"]
            lines.append(f"| {name} | {m['Q']:.9f} | {h[0]:.9f} | {h[1]:.9f} | {h[2]:.9f} |")
        lines += ["", "### 原生价格单位误差", "", "神经参考RMSE由种子平均MSE开方，MAE对种子/原点取平均。", "",
                  "| 模型 | H1 RMSE / MAE | H5 RMSE / MAE | H20 RMSE / MAE |", "|---|---:|---:|---:|"]
        for name, m in data["metrics"].items():
            cells = [f"{m['rmse_by_horizon'][i]:.6f} / {m['mae_by_horizon'][i]:.6f}" for i in range(3)]
            lines.append("| "+name+" | "+" | ".join(cells)+" |")
        lines += ["", "### 预先指定的八项核心比较", "", "效应为比较模型Q−主模型Q，乘100后是RW归一化百分点；正值表示主模型更好。核心家族95区间由每项99.375% Bonferroni区间构成。", "",
                  "| 比较模型 | 主模型改善pp | 20块边际95 | 20块家族95 | 40块边际95 | 40块家族95 |", "|---|---:|---:|---:|---:|---:|"]
        for name in COMPARATORS:
            e = data["effects"][name]
            lines.append(f"| {name} | {e['effect_RW_percentage_points']:+.6f} | {fmt_ci(e['block20']['ci95_marginal'])} | {fmt_ci(e['block20']['ci99375_core_family95'])} | {fmt_ci(e['block40']['ci95_marginal'])} | {fmt_ci(e['block40']['ci99375_core_family95'])} |")
        if all(k in data["effects"] for k in REFERENCES):
            lines += ["", "### 评分前固定的两项次要参考", "", "以下只报告边际95，不属于八项核心家族，不声称这十个区间共同有95%覆盖率。", "",
                      "| 比较模型 | 主模型改善pp | 20块边际95 | 40块边际95 |", "|---|---:|---:|---:|"]
            for name in REFERENCES:
                e = data["effects"][name]
                lines.append(f"| {name} | {e['effect_RW_percentage_points']:+.6f} | {fmt_ci(e['block20']['ci95_marginal'])} | {fmt_ci(e['block40']['ci95_marginal'])} |")
        lines += ["", "### 已固定的学习贡献诊断", "", "使用该模型已选alpha，保持初始动力学并仅在最终训练区间拟合读出；不是新增选优。", "",
                  "| 响应模型 | 固定初始动力学+最终读出Q | 最终Q | 非线性学习净改善pp | 所选非线性epoch |", "|---|---:|---:|---:|---:|"]
        for name, d in data["learning_diagnostics"].items():
            lines.append(f"| {name} | {d['fixed_dynamics_same_alpha_final_readout_Q']:.9f} | {d['fitted_Q']:.9f} | {d['nonlinear_fit_increment_RW_pp']:+.6f} | {data['selections'][name]['selected_epoch']} |")
        lines.append("")
    lines += ["## 核验范围与限制", "", "```json", json.dumps(result["counts"], ensure_ascii=False, indent=2), "```", "",
              "最大数值差异：", "", "```json", json.dumps(result["maximum_numerical_discrepancies"], ensure_ascii=False, indent=2), "```", "",
              "固定效应区间不能消除小样本、重叠H20标签和市场制度变化带来的限制。所有模型、原计划比较与两种块长均保留。区间跨零不等于架构永远无价值；超过RW也不自动证明条件时长优于静态或振幅机制。当前来源在这轮评分后已经暴露，后续修改不得继续称为同一未见确认。", "",
              f"计划SHA256：`{PLAN_SHA}`。完整逐任务审计记录、检查点与NPZ字节哈希、选择参数、数值残差见同名JSON；原生误差见 `NATIVE_METRICS.csv`。", ""]
    Path(path).write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--core-run", "--run", dest="core_run")
    p.add_argument("--neural-run")
    p.add_argument("--data-root", help="Optional directory containing the two canonical files under their frozen basenames")
    p.add_argument("--plan", default=str(STUDY/"audits/STATISTICAL_PLAN_BEFORE_FIT.json"))
    p.add_argument("--out", default=str(STUDY/"audits"))
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        r = self_test(); r["audit_script_sha256"] = sha(__file__)
        dump(Path(args.out)/"STATISTICS_SELF_TEST.json", r)
        print(json.dumps(r), flush=True)
        return
    if not args.core_run:
        p.error("--core-run is required for actual completed-run auditing")
    r = audit(args)
    print(json.dumps({"status": r["status"], "counts": r["counts"], "report": str(Path(args.out)/"INDEPENDENT_STATISTICAL_AUDIT_ZH.md")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
