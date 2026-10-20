"""Independent coarse-only sampling review, with synthetic tensors only.

No fit_imputer, optimizer, market data, validation data or confirmation data.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
CONTROL = WORK / "scratch/innovation_review_20261017/mcssl_control.py"


class CoarseOnly(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reads = []

    def __getitem__(self, name):
        if name not in ("short", "short_mask", "sparse", "sparse_mask", "market"):
            raise AssertionError("Sampling read forbidden information: " + name)
        self.reads.append(name)
        return super().__getitem__(name)


def state_digest(net):
    digest = hashlib.sha256()
    for name, value in net.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().numpy().tobytes())
    return digest.hexdigest()


def main():
    torch.set_num_threads(1)
    torch.manual_seed(81829)
    torch.set_default_dtype(torch.float64)
    spec = importlib.util.spec_from_file_location("independent_mcssl_control", CONTROL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    rng = np.random.default_rng(81829)
    b = 8
    price = np.c_[np.zeros(b), rng.normal(size=(b,20)).cumsum(1)]
    day = np.c_[np.zeros(b), rng.integers(1,5,size=(b,20)).cumsum(1)]
    data = CoarseOnly(market=np.arange(b) % 4)
    retained = {"short":np.arange(10,21), "sparse":np.arange(0,21,2)}
    for view, ids in retained.items():
        x = np.zeros((b,20,3))
        x[:,:10,0] = np.diff(price[:,ids], axis=1)
        x[:,:10,1] = np.log1p(np.diff(ids))
        x[:,:10,2] = np.log1p(np.diff(day[:,ids], axis=1))
        data[view] = x
        data[view+"_mask"] = np.broadcast_to(np.arange(20) < 10, (b,20)).copy()
    q = module.PriceCompletionMLP(n_markets=4).double()
    q.train()
    before = state_digest(q)
    generator = torch.Generator().manual_seed(12345)
    rows, maximum_retained_error = [], 0.
    for view, ids in retained.items():
        one, receipt = module.sample_tokens(data, view, q, generator, return_receipt=True)
        two = module.sample_tokens(data, view, q, generator)
        assert not one.requires_grad and not two.requires_grad
        assert q.training, "sampling must restore caller train/eval state"
        assert one.shape == two.shape == (b,20,3)
        assert bool(torch.isfinite(one).all() and torch.isfinite(two).all())
        # Reconstruct relative price vertices from the completed increments.
        reconstructed = np.c_[np.zeros(b), one.numpy()[:,:,0].cumsum(1)]
        reconstructed -= reconstructed[:,-1:]
        expected = price[:,ids] - price[:,-1:]
        err = float(np.max(abs(reconstructed[:,ids] - expected)))
        maximum_retained_error = max(maximum_retained_error, err)
        assert err < 4e-14
        # Reaggregating a sampled fine path must give the original coarse path.
        if view == "short":
            aggregated_price = one.numpy()[:,10:,0]
            aggregated_days = np.expm1(one.numpy()[:,10:,2])
        else:
            aggregated_price = one.numpy()[:,:,0].reshape(b,10,2).sum(2)
            aggregated_days = np.expm1(one.numpy()[:,:,2]).reshape(b,10,2).sum(2)
        assert np.allclose(aggregated_price, data[view][:,:10,0], atol=4e-14, rtol=0)
        assert np.allclose(aggregated_days, np.expm1(data[view][:,:10,2]), atol=4e-14, rtol=0)
        difference = float((one-two).square().mean())
        assert difference > 0, "untrained diagnostic network should react to distinct noise draws"
        rows.append(dict(view=view, retained_price_max_error=err,
                         synthetic_untrained_two_completion_mse=difference,
                         sampled_paths_coarsen_back_to_input=True, receipt=receipt))
    assert state_digest(q) == before and all(p.grad is None for p in q.parameters())
    output = dict(status="PASS", scope="untrained synthetic sampling and information boundary only",
                  market_reads=0, market_fits=0, optimizer_steps=0,
                  source_sha256=hashlib.sha256(CONTROL.read_bytes()).hexdigest(),
                  numpy_version=np.__version__, torch_version=torch.__version__,
                  model_parameters=sum(p.numel() for p in q.parameters()), markets=4,
                  allowed_key_reads=sorted(set(data.reads)),
                  q_parameters_unchanged=True, q_gradients_absent=True,
                  maximum_retained_price_error=maximum_retained_error, checks=rows,
                  limitation="Positive diversity of an untrained q does not establish trained conditional-law quality; MSE training can suppress its noise dependence.")
    path = HERE / "MCSSL_INDEPENDENT_PREFLIGHT.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(status="PASS", max_error=maximum_retained_error,
                         q_parameters=output["model_parameters"], path=str(path))))


if __name__ == "__main__":
    main()
