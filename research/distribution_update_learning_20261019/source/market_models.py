"""Matched, ordinary probabilistic experts. No backbone is claimed as novel."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
import torch
from torch import nn

from scores import crps


def fit_global_mixture(y):
    relative = np.array([.5, 1., 2.])
    def unpack(theta):
        return softmax(np.r_[theta[:2], 0.]), relative * np.exp(theta[2])
    def objective(theta):
        w, s = unpack(theta)
        return float(crps(y, w, s).mean())
    result = minimize(objective, np.array([0., 1., 0.]), method="L-BFGS-B",
                      bounds=[(-6., 6.), (-6., 6.), (np.log(.25), np.log(2.5))],
                      options={"maxiter": 300, "ftol": 1e-12})
    w, s = unpack(result.x)
    return w, s, {"success": bool(result.success), "message": str(result.message),
                  "iterations": int(result.nit), "objective": float(result.fun),
                  "weights": w.tolist(), "scales": s.tolist()}


def torch_crps(y, weights, sigma0, factor):
    sigma = factor[:, None] * sigma0[None, :]
    z = y[:, None] / sigma
    first = sigma * np.sqrt(2 / np.pi) * torch.exp(-.5 * z.square()) + y[:, None] * torch.erf(z / np.sqrt(2))
    pair = np.sqrt(2 / np.pi) * torch.sqrt(sigma[:, :, None].square() + sigma[:, None, :].square())
    return (weights * first).sum(-1) - .5 * (weights[:, None] * weights[None, :] * pair).sum((-2, -1))


class ScaleProposal(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        if backbone == "MLP32_16":
            self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(60, 32), nn.Tanh(), nn.Linear(32, 16), nn.Tanh())
        elif backbone == "GRU16":
            self.encoder = nn.GRU(3, 16, batch_first=True)
        else:
            raise ValueError(backbone)
        self.head = nn.Linear(16, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        hidden = self.encoder(x)
        if self.backbone == "GRU16":
            hidden = hidden[1][-1]
        return .5 + torch.sigmoid(self.head(hidden)[:, 0])


def fit_proposal(xtrain, ytrain, xval, yval, weights, sigma0, backbone, seed, epochs=60):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = ScaleProposal(backbone)
    x, y = torch.tensor(xtrain, dtype=torch.float32), torch.tensor(ytrain, dtype=torch.float32)
    xv, yv = torch.tensor(xval, dtype=torch.float32), torch.tensor(yval, dtype=torch.float32)
    w, s = torch.tensor(weights, dtype=torch.float32), torch.tensor(sigma0, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.001)
    with torch.no_grad():
        best_score = float(torch_crps(yv, w, s, model(xv)).mean())
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_epoch, updates, history = 0, 0, [{"epoch": 0, "validation_CRPS": best_score}]
    generator = torch.Generator().manual_seed(seed + 909)
    for epoch in range(1, epochs + 1):
        model.train()
        for indices in torch.randperm(len(x), generator=generator).split(64):
            optimizer.zero_grad()
            loss = torch_crps(y[indices], w, s, model(x[indices])).mean()
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite neural training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            updates += 1
        model.eval()
        with torch.no_grad():
            score = float(torch_crps(yv, w, s, model(xv)).mean())
        history.append({"epoch": epoch, "validation_CRPS": score})
        if score < best_score - 1e-8:
            best_score, best_epoch = score, epoch
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    model.eval()
    return model, {"backbone": backbone, "seed": seed, "selected_epoch": best_epoch,
                   "actual_training_epochs": epochs, "gradient_optimizer_steps": updates,
                   "selected_validation_CRPS": best_score,
                   "parameters": sum(p.numel() for p in model.parameters()), "history": history}


def predict_factor(model, x):
    with torch.no_grad():
        return model(torch.tensor(x, dtype=torch.float32)).numpy().astype(float)


def fit_garch(log_returns, training_end):
    """Zero-mean Gaussian GARCH(1,1), QMLE on the training prefix only."""
    train = np.asarray(log_returns[1:training_end + 1])
    unit = max(np.sqrt(np.mean(train ** 2)), 1e-8)
    z = train / unit
    def unpack(theta):
        ab = .999 * softmax(np.r_[theta[:2], 0.])[:2]
        longrun = np.exp(theta[2])
        return longrun * (1 - ab.sum()), ab[0], ab[1], longrun
    def objective(theta):
        omega, alpha, beta, longrun = unpack(theta)
        v, value = longrun, 0.
        for item in z:
            value += .5 * (np.log(max(v, 1e-12)) + item * item / max(v, 1e-12))
            v = omega + alpha * item * item + beta * v
        return value / len(z)
    fitted = minimize(objective, [np.log(.08/.019), np.log(.90/.019), 0.], method="L-BFGS-B",
                      bounds=[(-8., 8.), (-8., 8.), (-4., 4.)], options={"maxiter": 200, "ftol": 1e-10})
    omega, alpha, beta, longrun = unpack(fitted.x)
    zall = np.asarray(log_returns) / unit
    # next_variance[i] is conditioned on observations through and including i.
    next_variance, v = np.zeros(len(zall)), longrun
    next_variance[0] = v
    for i in range(1, len(zall)):
        v = omega + alpha * zall[i] ** 2 + beta * v
        next_variance[i] = v * unit * unit
    return next_variance, longrun * unit * unit, alpha + beta, {
        "success": bool(fitted.success), "message": str(fitted.message), "iterations": int(fitted.nit),
        "omega": float(omega * unit * unit), "alpha": float(alpha), "beta": float(beta),
        "training_last_return_index": int(training_end), "zero_mean": True}
