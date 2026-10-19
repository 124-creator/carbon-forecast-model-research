"""Implementation-only synthetic checks; no market files are read or scored."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parents[1]/"source"
sys.path.insert(0, str(SOURCE))
from signed_response_models import (DTYPE, FAMILIES, FitConfig, SignedResponse,
                                   fit_response, solve_readout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=Path(__file__).with_name("SIGNED_RESPONSE_SYNTHETIC_CHECK.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    rng = np.random.default_rng(20261019)
    x = rng.normal(size=(320, 64, 3))
    x[:, :, 0] *= .35
    x[:, :, 2] = 1.
    x = torch.tensor(x, dtype=DTYPE)
    truth = SignedResponse()
    with torch.no_grad():
        truth.rate_raw.add_(torch.tensor([-.3, .1], dtype=DTYPE))
        truth.gate.copy_(torch.tensor([[.65, -.35], [-.45, .6]], dtype=DTYPE))
    true_beta = torch.tensor([.012, 1.15, -.8], dtype=DTYPE)
    y = (truth.basis(x)@true_beta).detach() + torch.tensor(rng.normal(0, .01, size=(320, 3)), dtype=DTYPE)
    variance = torch.mean(y[:256]**2, dim=0)
    model = SignedResponse()
    with torch.no_grad():
        model.gate.copy_(torch.tensor([[.1, -.2], [.15, .1]], dtype=DTYPE))
    alpha = .001
    phi = model.basis(x[:256])
    beta, obj, data_loss, residual, scales = solve_readout(phi, y[:256], variance, alpha)
    gradients = torch.autograd.grad(obj, tuple(model.parameters()))
    analytic = torch.cat([g.reshape(-1) for g in gradients]).numpy()
    numerical = []
    epsilon = 1e-5
    for parameter in model.parameters():
        for index in range(parameter.numel()):
            with torch.no_grad():
                old = float(parameter.reshape(-1)[index])
                parameter.reshape(-1)[index] = old+epsilon
            plus = float(solve_readout(model.basis(x[:256]), y[:256], variance, alpha)[1].detach())
            with torch.no_grad():
                parameter.reshape(-1)[index] = old-epsilon
            minus = float(solve_readout(model.basis(x[:256]), y[:256], variance, alpha)[1].detach())
            with torch.no_grad():
                parameter.reshape(-1)[index] = old
            numerical.append((plus-minus)/(2*epsilon))
    numerical = np.asarray(numerical)
    gradient_gap = float(np.max(np.abs(analytic-numerical)))
    assert gradient_gap < 2e-7 and np.max(np.abs(analytic)) > 1e-5
    # Independent augmented least-squares solve; no normal-equation inversion.
    ph = phi.detach().numpy()
    yy, vv = y[:256].numpy(), variance.numpy()
    a = (ph/np.sqrt(vv)[None, :, None]).reshape(-1, ph.shape[-1])
    target = (yy/np.sqrt(vv)[None]).reshape(-1)
    ss = np.maximum(np.sqrt(np.mean(a*a, axis=0)), 1e-8)
    az = a/ss
    aug_a = np.vstack((az/np.sqrt(len(a)), np.sqrt(alpha)*np.eye(a.shape[-1])))
    aug_b = np.r_[target/np.sqrt(len(a)), np.zeros(a.shape[-1])]
    independent = np.linalg.lstsq(aug_a, aug_b, rcond=None)[0]/ss
    beta_gap = float(np.max(np.abs(independent-beta.detach().numpy())))
    primal_gradient = 2*a.T@(a@independent-target)/len(a)+2*alpha*ss**2*independent
    assert beta_gap < 1e-9 and np.max(np.abs(primal_gradient)) < 1e-8
    # W=0 shares basis and the fitted epoch0 function for shared-head arms.
    base = SignedResponse("signed_context_response").basis(x)
    zero_gate_gaps = {}
    for family in ("fixed_response", "static_response", "amplitude_context", "signed_shock_only"):
        gap = float(torch.max(torch.abs(base-SignedResponse(family).basis(x))).detach())
        assert gap == 0.
        zero_gate_gaps[family] = gap
    # Isolated +1 shock: future changes can continue, retract, then change sign.
    impulse = torch.zeros((1, 64, 3), dtype=DTYPE)
    impulse[0, -1, 0] = 1.
    impulse[:, :, 2] = 1.
    shape_model = SignedResponse("fixed_response")
    shapes = {}
    for name, values in {"continuation": [0., 1., 1.], "retraction": [0., -1., -1.],
                         "initial_continuation_then_retraction": [0., 1., -1.]}.items():
        shape_model.set_coefficients(torch.tensor(values, dtype=DTYPE))
        path = shape_model.path(impulse).detach().numpy()[0]
        increments = np.diff(np.r_[0., path])
        shapes[name] = {"cumulative_future_changes_h1_h5_h20": path[[0,4,19]].tolist(),
                        "future_increment_h1_h20": increments[[0,19]].tolist(),
                        "total_price_response_including_original_unit_shock_h20": float(1+path[-1])}
    assert all(v > 0 for v in shapes["continuation"]["cumulative_future_changes_h1_h5_h20"])
    assert all(v < 0 for v in shapes["retraction"]["cumulative_future_changes_h1_h5_h20"])
    mixed = shapes["initial_continuation_then_retraction"]
    assert mixed["future_increment_h1_h20"][0] > 0 > mixed["future_increment_h1_h20"][1]
    assert mixed["cumulative_future_changes_h1_h5_h20"][0] > 0 > mixed["cumulative_future_changes_h1_h5_h20"][-1]
    # A separate objective shape claim: amplitude cannot alter one isolated
    # mode's S5/S1 at fixed rho, while conditional future duration can.
    paired = impulse.repeat(2, 1, 1)
    paired[0, -1, 1], paired[1, -1, 1] = -1., 1.
    duration, amplitude = SignedResponse(), SignedResponse("amplitude_context")
    for candidate in (duration, amplitude):
        with torch.no_grad():
            candidate.gate[0,1] = .5
    dbase, abase = duration.basis(paired), amplitude.basis(paired)
    d_ratios = (dbase[:,1,1]/dbase[:,0,1]).detach().numpy()
    a_ratios = (abase[:,1,1]/abase[:,0,1]).detach().numpy()
    assert abs(np.diff(d_ratios)[0]) > .1 and abs(np.diff(a_ratios)[0]) < 1e-12
    # Nonlinear training is real: exact beta at every step, no market score.
    fit, receipt = fit_response("signed_context_response", x[:256], y[:256], variance,
                               FitConfig(alpha=.001, learning_rate=.01), fixed_epochs=20)
    first = receipt["history"][0]["training_profiled_objective"]
    last = receipt["history"][-1]["training_profiled_objective"]
    assert receipt["positive_gradient_steps"] == 20 and last < first
    # Reconstruct the saved state and forward exactly.
    restored = SignedResponse(receipt["family"])
    restored.load_state_dict({k: torch.tensor(v, dtype=DTYPE) for k,v in receipt["state_dict"].items()})
    assert torch.equal(restored(x), fit(x))
    output = {"status": "PASS_SYNTHETIC_ONLY", "market_files_read": 0,
              "market_fits": 0, "market_validation_or_test_scores": 0,
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [SOURCE/"signed_response_models.py", SOURCE/"run_signed_response_search.py"]},
              "parameter_counts": {family: SignedResponse(family).parameter_counts() for family in FAMILIES},
              "profiled_objective_finite_difference": {"step": epsilon, "analytic": analytic.tolist(),
                  "central_difference": numerical.tolist(), "maximum_absolute_gap": gradient_gap,
                  "training_column_rms_differentiated": True},
              "independent_augmented_lstsq_beta_max_gap": beta_gap,
              "independent_raw_coordinate_primal_gradient_max": float(np.max(np.abs(primal_gradient))),
              "internal_normal_equation_residual_max": float(residual.detach().abs().max()),
              "zero_gate_shared_basis_gaps": zero_gate_gaps,
              "isolated_positive_shock": shapes,
              "single_mode_shape_ratios_h5_over_h1": {"duration_context": d_ratios.tolist(), "amplitude_context": a_ratios.tolist()},
              "synthetic_gradient_training": {"epochs": 20, "positive_gradient_steps": receipt["positive_gradient_steps"],
                  "first_profiled_objective": first, "last_profiled_objective": last,
                  "parameter_changes": receipt["selected_nonlinear_parameter_max_change"]},
              "checkpoint_prediction_replay_gap": 0.,
              "mechanism_scope": "expressivity and numerical implementation only; synthetic truth is constructed in this model class, not independent market effectiveness"}
    report = args.report
    if report.exists():
        raise FileExistsError("Preserve an existing synthetic receipt")
    report.write_text(json.dumps(output, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"status": output["status"], "report": str(report),
                      "gradient_gap": gradient_gap, "beta_gap": beta_gap,
                      "synthetic_objective": [first,last]}))


if __name__ == "__main__":
    main()
