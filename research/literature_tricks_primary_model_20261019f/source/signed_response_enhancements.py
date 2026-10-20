"""Stage-f preprocessing and model factory; the stage-e predictor is retained.

Raw price plus all three EWT analysis channels share the original two history
rates and original current-shock/volume gate. The shared readout has one drift
and eight signed response coefficients. Full/amplitude have15 fitted scalars,
static11 and shock-only13; the original raw model remains a reused9-scalar
reference. No component is deleted and no denoised-origin anchor is added.

ewtpy 0.2 returns analysis coefficients, not additive reconstructed modes.
The formal experiment directly uses analysis coefficients as features, as in
SWEWT-GRU; no inverse transform, subband deletion or price reconstruction is
performed. Current-snapshot origin availability is distinct from a verified
historical publication vintage.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from ewtpy import EWT1D

E_SOURCE = Path(__file__).resolve().parents[2] / "signed_response_search_20261019e/source"
if str(E_SOURCE) not in sys.path:
    sys.path.insert(0, str(E_SOURCE))
from signed_response_models import (DTYPE, HORIZONS, LOOKBACK, FitConfig,
                                    SignedResponse, ordered_rates, solve_readout)

RESPONSE_FAMILIES = ("signed_context_response", "static_response",
                     "amplitude_context", "signed_shock_only")
SETTINGS = ("raw", "window_past", "train_joint_test_window", "full_series_joint")
NEW_SETTINGS = SETTINGS[1:]
EWT_OPTIONS = {"N": 3, "log": 0, "detect": "locmax", "completion": 1,
               "reg": "average", "lengthFilter": 10, "sigmaFilter": 5}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(value):
    value = np.ascontiguousarray(value)
    header = f"{value.dtype.str}:{value.shape}:".encode()
    return hashlib.sha256(header + value.tobytes()).hexdigest()


def ewt_dependency():
    source = Path(inspect.getsourcefile(EWT1D)).resolve()
    return {"package": "ewtpy", "version": importlib.metadata.version("ewtpy"),
            "source_path": str(source), "source_sha256": digest(source)}


def window_changes(price, origin):
    origin = int(origin)
    if origin < LOOKBACK or origin >= len(price):
        raise ValueError("Origin lacks65 observed prices")
    return np.diff(ewt_analysis(np.asarray(price)[origin-LOOKBACK:origin+1]), axis=0)


def ewt_analysis(price):
    """All three analysis channels; these are not additive reconstructed modes."""
    p = np.asarray(price, dtype=np.float64)
    if p.ndim != 1 or len(p) < 65 or not np.isfinite(p).all():
        raise ValueError("A finite price segment of at least65 records is required")
    coefficients, _, _ = EWT1D(p, **EWT_OPTIONS)
    if coefficients.shape != (len(p), 3) or not np.isfinite(coefficients).all():
        raise FloatingPointError("Invalid EWT analysis coefficients")
    return coefficients


def transformed_inputs(d, setting, *, window_cache=None):
    """Return separately aligned train/val/final/test arrays and a public manifest.

    Input channels are raw change, volume surprise, valid, three analysis-channel
    changes. The three coefficient series are never summed into a price.
    The joint training and final arrays can differ for the same origin: their
    legal estimation cutoffs are a-1 and b-1 respectively. Labels, native-price
    scale and horizon variances are never transformed or recomputed here.
    """
    if setting not in SETTINGS:
        raise ValueError(setting)
    p, origins, masks = d["price"], d["origins"], d["masks"]
    _, a, b, n = d["manifest"]["row_boundaries"]
    filtered = {}
    if setting in ("window_past", "train_joint_test_window"):
        if window_cache is None:
            window_cache = np.stack([window_changes(p, int(t)) for t in origins])
        if window_cache.shape != (len(origins), LOOKBACK, 3):
            raise ValueError("Window cache shape mismatch")
    if setting == "full_series_joint":
        filtered["full"] = np.diff(ewt_analysis(p), axis=0)
    elif setting == "train_joint_test_window":
        filtered["train"] = np.diff(ewt_analysis(p[:a]), axis=0)
        filtered["final"] = np.diff(ewt_analysis(p[:b]), axis=0)
    arrays, subsets = {}, {}
    for subset in ("train", "val", "final", "test"):
        mask = masks[subset]
        selected_origins = origins[mask]
        raw = d["x"][mask]
        if setting == "raw":
            state_returns = np.zeros((*raw.shape[:2], 3), dtype=np.float64)
            boundary = "each origin t; raw returns only"
            scope = "origin_available"
        elif setting == "full_series_joint":
            delta = filtered["full"]
            state_returns = np.stack([delta[t-LOOKBACK:t] for t in selected_origins]) / d["scale"]
            boundary = n-1
            scope = "retrospective_full_sequence; includes prices after issuance"
        elif setting == "train_joint_test_window" and subset in ("train", "final"):
            delta = filtered[subset]
            state_returns = np.stack([delta[t-LOOKBACK:t] for t in selected_origins]) / d["scale"]
            boundary = (a if subset == "train" else b)-1
            scope = "joint_estimation_prefix; historical training representations use later prices inside that prefix"
        else:
            state_returns = window_cache[mask] / d["scale"]
            boundary = "each origin t, inclusive65-price window t-64:t"
            scope = "origin_available"
        packed = np.concatenate((raw, state_returns), axis=-1)
        if packed.shape != (int(mask.sum()), LOOKBACK, 6) or not np.isfinite(packed).all():
            raise FloatingPointError("Invalid transformed input")
        if not np.array_equal(packed[..., :3], raw):
            raise AssertionError("Raw shock or volume context was modified")
        arrays[subset] = packed
        subsets[subset] = {"count": len(selected_origins),
                           "origin_range": [int(selected_origins[0]), int(selected_origins[-1])],
                           "feature_price_endpoint": boundary, "information_scope": scope,
                           "input_sha256": array_digest(packed),
                           "native_target_sha256": array_digest(d["y"][mask])}
    manifest = {"setting": setting, "subsets": subsets,
                "input_columns": ["raw_change_over_fixed_train_rms", "volume_surprise",
                                  "volume_valid", "analysis_band0_change_over_fixed_train_rms",
                                  "analysis_band1_change_over_fixed_train_rms", "analysis_band2_change_over_fixed_train_rms"],
                "raw_plus_all_analysis_channels": setting != "raw",
                "raw_reference_uses_zero_padding_for_embedding_check_only": setting == "raw",
                "gate_uses_raw_current_shock_and_volume": True,
                "target": "unchanged published native-price cumulative changes at H1/H5/H20",
                "normalization": "unchanged original training-prefix RMS and target second moments",
                "ewt_options": EWT_OPTIONS, "analysis_bands": [0, 1, 2],
                "inverse_or_synthesis_used": False, "subband_deletion": False,
                "analysis_features_not_additive_price_reconstruction": True,
                "no_denoised_origin_anchor": True, "future_target_denoising": False,
                "train_final_inputs_may_differ": setting == "train_joint_test_window"}
    return arrays, manifest


class StateInputSignedResponse(SignedResponse):
    """Original conditional response shared by raw plus all3 analysis channels."""
    def __init__(self, family="signed_context_response", horizons=HORIZONS):
        if family not in RESPONSE_FAMILIES:
            raise ValueError("Only the four frozen stage-f response families are allowed")
        super().__init__(family=family, horizons=horizons)
        self.coefficients = torch.zeros(9, dtype=DTYPE)
        self.head_column_scale = torch.ones(9, dtype=DTYPE)

    def states_and_context(self, x):
        if x.ndim != 3 or x.shape[1:] != (LOOKBACK, 6):
            raise ValueError(f"Expected B x {LOOKBACK} x6; got {tuple(x.shape)}")
        rho = ordered_rates(self.rate_raw)
        lags = torch.arange(LOOKBACK-1, -1, -1, dtype=x.dtype, device=x.device)
        weights = (1-rho)[None] * rho[None]**lags[:, None]
        # Channel-major order: raw(faster/slower rate order), band0, band1, band2.
        # Each channel uses exactly the same ordered history rates.
        states = x[..., [0, 3, 4, 5]].transpose(1, 2) @ weights
        z = torch.stack((torch.log1p(torch.abs(x[:, -1, 0])),
                         x[:, -1, 1]*x[:, -1, 2]), dim=-1)
        return states, z, rho

    def components(self, x, horizons=None):
        hs = self.horizons if horizons is None else tuple(int(h) for h in horizons)
        states, z, rho = self.states_and_context(x)
        gate_input = z[:, :1] if self.family == "signed_shock_only" else z
        q = 2*torch.tanh(gate_input @ self.gate.T)
        if self.family in ("signed_context_response", "signed_shock_only"):
            future_rates = ordered_rates(self.rate_raw[None]+q)
        else:
            future_rates = rho[None].expand(len(x), -1)
        amplitude = torch.exp(q) if self.family == "amplitude_context" else torch.ones_like(q)
        steps = torch.arange(1, max(hs)+1, dtype=x.dtype, device=x.device)
        cumulative = torch.cumsum(future_rates[:, None, :]**steps[None, :, None], dim=1)
        index = torch.as_tensor(hs, device=x.device)-1
        response = (cumulative[:, index, None, :] *
                    (states*amplitude[:, None, :])[:, None, :, :]).reshape(len(x), len(hs), 8)
        h = torch.as_tensor(hs, dtype=x.dtype, device=x.device)
        base = torch.cat((h[None, :, None].expand(len(x), -1, -1), response), dim=-1)
        return base, {"states": states, "context": z, "historical_rates": rho,
                      "future_rates": future_rates, "amplitude": amplitude, "gate_shift": q}


def model_factory(family):
    return StateInputSignedResponse(family)


def fit_response_with_factory(family, x_train, y_train, variance, config, *,
                              x_val=None, y_val=None, fixed_epochs=None,
                              factory=model_factory):
    """Stage-e VP loop with an explicit factory; no other optimization change.

    The inherited basis, exact head solver, Adam, clipping, tolerance and
    selection/refit retention rules are unchanged. Only model construction is
    injectable. The caller supplies separate inputs for each estimation phase.
    """
    if (x_val is None) != (y_val is None):
        raise ValueError("Provide both validation inputs and labels")
    model = factory(family)
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=config.learning_rate) if params else None
    max_epochs = config.maximum_epochs if fixed_epochs is None else int(fixed_epochs)
    if not params:
        max_epochs = 0
    best_value, best_epoch, best_state = float("inf"), 0, None
    history, stale, positive_steps = [], 0, 0
    initial_parameters = {k: v.detach().clone() for k, v in model.named_parameters()}
    last_parameters = {k: v.detach().clone() for k, v in model.named_parameters()}
    for epoch in range(max_epochs+1):
        coefficients, objective, training_loss, residual, column_scale = solve_readout(
            model.basis(x_train), y_train, variance, config.alpha)
        model.set_coefficients(coefficients, column_scale)
        with torch.no_grad():
            selection = (torch.mean((model(x_val)-y_val)**2/variance)
                         if x_val is not None else training_loss)
        value = float(selection.detach())
        if not np.isfinite(value) or not bool(torch.isfinite(objective)):
            raise FloatingPointError("Nonfinite profiled objective or selection loss")
        record = {"epoch": epoch, "training_data_loss": float(training_loss.detach()),
                  "training_profiled_objective": float(objective.detach()),
                  "validation_loss": float(selection.detach()) if x_val is not None else None,
                  "linear_normal_equation_max_residual": float(residual.detach().abs().max()),
                  "historical_rates": ordered_rates(model.rate_raw).detach().tolist()}
        history.append(record)
        if value < best_value-config.selection_tolerance:
            best_value, best_epoch, best_state = value, epoch, {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch == max_epochs:
            break
        if fixed_epochs is None and x_val is not None and epoch >= config.minimum_epochs and stale >= config.patience:
            break
        if optimizer is None:
            break
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        gradients = [p.grad for p in params if p.grad is not None]
        if len(gradients) != len(params) or not all(bool(torch.isfinite(g).all()) for g in gradients):
            raise FloatingPointError("Missing or nonfinite nonlinear gradient")
        norm = float(torch.sqrt(sum(torch.sum(g*g) for g in gradients)))
        if norm > 0:
            positive_steps += 1
        record["gradient_norm_before_clip"] = norm
        torch.nn.utils.clip_grad_norm_(params, config.gradient_clip)
        optimizer.step()
        last_parameters = {k: v.detach().clone() for k, v in model.named_parameters()}
    if fixed_epochs is None:
        model.load_state_dict(best_state)
        selected_epoch = best_epoch
    else:
        selected_epoch = history[-1]["epoch"]
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    receipt = {"family": family, "model_factory": "StateInputSignedResponse",
               "config": asdict(config), "deterministic": True,
               "parameter_counts": model.parameter_counts(), "selected_epoch": selected_epoch,
               "epochs_executed": history[-1]["epoch"], "positive_gradient_steps": positive_steps,
               "validation_loss": best_value if x_val is not None else None,
               "selected_nonlinear_parameter_max_change": {
                   k: float(torch.max(torch.abs(v-initial_parameters[k])).detach()) for k, v in model.named_parameters()},
               "attempted_last_nonlinear_parameter_max_change": {
                   k: float(torch.max(torch.abs(last_parameters[k]-initial_parameters[k]))) for k in initial_parameters},
               "linear_readout_fitted_at_epoch0": True,
               "state_dict": {k: v.tolist() for k, v in state.items()}, "history": history}
    return model, receipt
