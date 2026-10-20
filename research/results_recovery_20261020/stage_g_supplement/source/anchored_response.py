"""Stage-g training enhancements of the stage-e signed-duration response.

Same two exponential price states and three shared horizon coefficients.
No claim that warm starts or L2-SP are new. Official stage-f data are not
substituted: the runner uses explicitly identified recovered development data.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
DT = torch.float64
HORIZONS = (1, 5, 20)
LOOKBACK = 64
CAP = .98

def rates(raw: torch.Tensor) -> torch.Tensor:
    a = CAP * torch.sigmoid(raw[..., 0])
    b = a + (CAP-a) * torch.sigmoid(raw[..., 1])
    return torch.stack((a, b), -1)

def initial_rates() -> torch.Tensor:
    return torch.tensor([np.log(.5/.48), np.log(.4/.08)], dtype=DT)

class Response(nn.Module):
    def __init__(self, kind: str = 'duration', scale=None, start=None):
        super().__init__()
        if kind not in ('static', 'duration', 'amplitude', 'price_only'):
            raise ValueError(kind)
        self.kind = kind
        self.raw = nn.Parameter(initial_rates() if start is None else torch.tensor(start, dtype=DT))
        if kind != 'static':
            self.gate = nn.Parameter(torch.zeros((2, 1 if kind == 'price_only' else 2), dtype=DT))
        else:
            self.register_buffer('gate', torch.zeros((2, 2), dtype=DT))
        self.register_buffer('context_scale', torch.ones(2, dtype=DT) if scale is None else torch.tensor(scale, dtype=DT))
        self.register_buffer('beta', torch.zeros(3, dtype=DT))
    def basis(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape[1:]) != (64, 3):
            raise ValueError('Input must be B x 64 x 3')
        rho = rates(self.raw)
        lags = torch.arange(63, -1, -1, dtype=DT)
        states = x[..., 0] @ ((1-rho)[None] * rho[None]**lags[:, None])
        z = torch.stack((torch.log1p(x[:, -1, 0].abs()), x[:, -1, 1]*x[:, -1, 2]), -1) / self.context_scale
        if self.kind == 'price_only':
            z = z[:, :1]
        q = 2 * torch.tanh(z @ self.gate.T)
        rr = rates(self.raw[None] + q) if self.kind in ('duration', 'price_only') else rho[None].expand(len(x), -1)
        amp = torch.exp(q) if self.kind == 'amplitude' else torch.ones_like(q)
        steps = torch.arange(1, 21, dtype=DT)
        cum = (rr[:, None, :]**steps[None, :, None]).cumsum(1)[:, [0, 4, 19], :]
        drift = torch.tensor(HORIZONS, dtype=DT)[None, :, None].expand(len(x), -1, -1)
        return torch.cat((drift, cum*(states*amp)[:, None]), -1)
    def forward(self, x):
        return self.basis(x) @ self.beta
    def export(self):
        return {'kind': self.kind, 'state': {k:v.detach().tolist() for k,v in self.state_dict().items()}}
    @classmethod
    def restore(cls, saved):
        out = cls(saved['kind'])
        out.load_state_dict({k:torch.tensor(v, dtype=DT) for k,v in saved['state'].items()})
        return out

def solve(phi: torch.Tensor, y: torch.Tensor, variance: torch.Tensor, alpha: float):
    if alpha <= 0:
        raise ValueError('Positive ridge penalty required')
    a = (phi/torch.sqrt(variance)[None, :, None]).reshape(-1, 3)
    b = (y/torch.sqrt(variance)[None]).reshape(-1)
    scale = a.square().mean(0).sqrt().clamp_min(1e-8)
    az = a/scale
    system = az.T @ az / len(a) + alpha*torch.eye(3, dtype=DT)
    rhs = az.T @ b / len(a)
    w = torch.linalg.solve(system, rhs)
    beta = w/scale
    data = ((phi @ beta-y).square()/variance).mean()
    return beta, data+alpha*w.square().sum(), data

def context_rms(x):
    z = np.column_stack((np.log1p(np.abs(x[:, -1, 0])), x[:, -1, 1]*x[:, -1, 2]))
    return np.maximum(np.sqrt(np.mean(z*z, axis=0)), 1e-3)

@dataclass(frozen=True)
class Config:
    alpha: float
    lr: float
    gamma: float = 0.
    cap: int = 320
    patience: int = 12
    min_epochs: int = 15

def fit(kind, x, y, variance, config: Config, *, xv=None, yv=None, start=None, scaled=False, epochs=None):
    """Validation only selects a checkpoint; every head uses fitting labels only.

    Anchoring penalizes raw-rate distance to the fitted static start, and new
    gate weights to zero. Context RMS has no centering and is fitting-only.
    Fixed-epoch refits keep the final epoch, not the training-minimum epoch.
    """
    scale = context_rms(x.numpy()) if scaled else np.ones(2)
    model = Response(kind, scale, start)
    anchor = model.raw.detach().clone()
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    best, best_epoch, stale, steps = float('inf'), 0, 0, 0
    best_state = None
    history = []
    cap = config.cap if epochs is None else int(epochs)
    for ep in range(cap+1):
        beta, objective, data = solve(model.basis(x), y, variance, config.alpha)
        penalty = config.gamma*((model.raw-anchor).square().sum()+model.gate.square().sum())
        loss = objective+penalty
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite objective')
        with torch.no_grad():
            model.beta.copy_(beta.detach())
            val = float(((model(xv)-yv).square()/variance).mean()) if xv is not None else None
        history.append([ep, float(data.detach()), float(loss.detach()), val])
        if xv is not None:
            if val < best-1e-7:
                best, best_epoch, stale = val, ep, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
        if ep == cap or (epochs is None and ep >= config.min_epochs and stale >= config.patience):
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        if not torch.isfinite(grad):
            raise FloatingPointError('Nonfinite gradient')
        steps += int(float(grad) > 0)
        opt.step()
    selected = history[-1][0] if epochs is not None else best_epoch
    if epochs is None:
        model.load_state_dict(best_state)
    return model, {'config': config.__dict__, 'selected_epoch': selected, 'executed_epochs': history[-1][0],
                   'gradient_steps': steps, 'validation_loss': best if xv is not None else None,
                   'history_columns':['epoch','training_data_loss','training_total_loss','validation_loss'],
                   'history':history, 'anchor_rates_raw':anchor.tolist(), 'model':model.export()}
