"""Exercise reviewed MIT RKN source; synthetic tensors only, no market data.

Pass a directory containing the five reviewed source/license files listed below.
No automatic download, dependency install, checkpoint load, or source patching.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

EXPECTED = {
    "rkn_cell/RKNCell.py": "b673ba2db446f0e7ab9005903b8fd1193d613fbd",
    "util/ConfigDict.py": "0d74b1269bcf962f98977e98d97f167bba641beb",
    "rkn_cell/__init__.py": "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391",
    "util/__init__.py": "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391",
    "LICENSE": "e183e0338004ef6e108a98be70c049f9bcc95ccb",
}


def run(source):
    for name, expected in EXPECTED.items():
        data = (source / name).read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if actual != expected:
            raise ValueError("Reviewed source mismatch: " + name)
    sys.path.insert(0, str(source))
    from rkn_cell.RKNCell import RKNCell

    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(17)
    config = RKNCell.get_default_config()
    config.num_basis = 3
    config.bandwidth = 1
    config.trans_net_hidden_units = [16]
    cell = RKNCell(8, config, dtype=torch.float64)
    observations = torch.randn(4, 12, 8)
    variances = torch.full_like(observations, 0.7, requires_grad=True)

    def rollout(obs, var):
        mean = torch.zeros(4, 16)
        cov = [torch.ones(4, 8), torch.ones(4, 8), torch.zeros(4, 8)]
        trace = []
        min_determinant = float("inf")
        for t in range(obs.shape[1]):
            post, post_cov, mean, cov = cell(mean, cov, obs[:, t], var[:, t])
            trace.append(post)
            for triple in (post_cov, cov):
                cu, cl, cs = triple
                determinant = cu * cl - cs.square()
                if not all(bool(torch.isfinite(v).all()) for v in triple):
                    raise ValueError("Non-finite covariance")
                if float(cu.detach().min()) < 0 or float(cl.detach().min()) < 0:
                    raise ValueError("Negative variance")
                min_determinant = min(min_determinant, float(determinant.detach().min()))
        if min_determinant < -1e-10:
            raise ValueError("Invalid covariance block")
        return torch.stack(trace, 1), min_determinant

    trace, determinant = rollout(observations, variances)
    loss = trace[:, -1].square().mean()
    loss.backward()
    if variances.grad is None or not bool(torch.isfinite(variances.grad).all()):
        raise ValueError("Observation variance gradient missing or invalid")
    variance_grad_norm = float(variances.grad.norm())
    transition_grad_norm = float(cell._tm_11_basis.grad.norm())
    if variance_grad_norm <= 0 or transition_grad_norm <= 0:
        raise ValueError("No effective predictive gradient")
    with torch.no_grad():
        changed = observations.clone()
        changed[:, 6:] += 100
        changed_trace, _ = rollout(changed, variances.detach())
        prefix_difference = float((changed_trace[:, :6] - trace.detach()[:, :6]).abs().max())
        if prefix_difference != 0:
            raise ValueError("Future inputs changed prefix states")
        config.never_invalid = False
        prior = torch.randn(4, 16)
        prior_cov = [torch.ones(4, 8), torch.ones(4, 8), torch.zeros(4, 8)]
        post, post_cov, _, _ = cell(prior, prior_cov, observations[:, 0],
                                   variances[:, 0].detach(), torch.zeros(4, dtype=torch.bool))
        if not torch.equal(post, prior) or any(not torch.equal(a, b) for a, b in zip(post_cov, prior_cov)):
            raise ValueError("Masked observations changed posterior")
    return dict(passed=True, scope="reviewed author-cell synthetic forward/backward only",
                author_commit="7cb29e1c57307a87729fd5e20ca60a22f5798da5",
                torch_version=torch.__version__, device="cpu", dtype="float64",
                batches=4, observations=12, latent_observation_dim=8, state_dim=16,
                parameters=sum(p.numel() for p in cell.parameters()),
                transition_bases=3, transition_bandwidth=1, coefficient_hidden=[16],
                loss=float(loss.detach()), minimum_covariance_block_determinant=determinant,
                observation_variance_gradient_norm=variance_grad_norm,
                transition_basis_gradient_norm=transition_grad_norm,
                future_perturbation_prefix_difference=prefix_difference,
                missing_observation_identity=True, optimizer_steps=0,
                market_fits=0, limitations=["Not full author-paper replication",
                    "No carbon forecast, predictive gain or algorithm novelty established",
                    "Cell hyperparameters chosen for smoke test, not market protocol"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.source), ensure_ascii=False))
