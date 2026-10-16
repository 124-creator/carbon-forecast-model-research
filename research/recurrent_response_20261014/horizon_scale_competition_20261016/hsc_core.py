"""Separated recurrent scales and training-only relative-error competition."""
import numpy as np
import torch
from torch import nn

MODES = ('raw_gru', 'raw_lstm', 'scale_mean', 'gate_only',
         'shared_competition', 'horizon_competition')


def configure():
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.backends.mkldnn.enabled = False
    torch.use_deterministic_algorithms(True)


class ScaleNet(nn.Module):
    def __init__(self, mode, hidden):
        super().__init__()
        if mode not in MODES or hidden <= 0:
            raise ValueError('Unknown scale family or width')
        self.mode = mode
        self.scales = (1,) if mode.startswith('raw_') else (1, 5, 20)
        cell = nn.LSTM if mode == 'raw_lstm' else nn.GRU
        self.encoders = nn.ModuleList([cell(1, hidden, batch_first=True) for _ in self.scales])
        self.projections = nn.ModuleList([nn.Linear(hidden, 8, bias=False) for _ in self.scales])
        self.contexts = nn.ModuleList([nn.Linear(3, 8, bias=False) for _ in self.scales])
        self.biases = nn.Parameter(torch.zeros(len(self.scales), 8))
        self.readouts = nn.Parameter(torch.zeros(len(self.scales), 9, 20))
        if mode in ('gate_only', 'shared_competition', 'horizon_competition'):
            self.route_state = nn.Parameter(torch.zeros(len(self.scales), 9))
            if mode != 'shared_competition':
                self.route_horizon = nn.Parameter(torch.zeros(len(self.scales), 20))
                self.route_interaction = nn.Parameter(torch.zeros(len(self.scales), 9, 20))

    def features(self, price, context, origins, scale):
        ids = np.asarray(origins, dtype=np.int64)
        if not len(ids) or ids.min() < 1 or ids.max() >= len(price) or not np.isfinite(scale) or scale <= 0:
            raise ValueError('Invalid feature support')
        end = int(ids.max())
        current = np.arange(1, end + 1)
        q = np.array(context[ids], copy=True)
        q[:, 0] = np.tanh(q[:, 0] / scale)
        q = torch.as_tensor(q)
        features = []
        for k, width in enumerate(self.scales):
            delta = (price[current] - price[np.maximum(0, current - width)]) / (np.sqrt(width) * scale)
            x = torch.as_tensor(delta)[None, :, None]
            hidden = (self.encoders[k](x)[0][0] - self.encoders[k](torch.zeros_like(x))[0][0])[
                torch.as_tensor(ids - 1)]
            value = torch.tanh(self.projections[k](hidden) + self.contexts[k](q) + self.biases[k]) - torch.tanh(self.biases[k])
            features.append(torch.cat((value, torch.ones_like(value[:, :1])), dim=1))
        return torch.stack(features, dim=1)

    def combine(self, features):
        experts = torch.einsum('nkf,kfh->nkh', features, self.readouts)
        if hasattr(self, 'route_state'):
            logits = torch.einsum('nkf,kf->nk', features, self.route_state)[:, :, None].expand(-1, -1, 20)
            if hasattr(self, 'route_horizon'):
                logits = logits + self.route_horizon + torch.einsum('nkf,kfh->nkh', features, self.route_interaction)
            weights = torch.softmax(logits, dim=1)
        else:
            weights = torch.ones_like(experts) / len(self.scales)
        return dict(prediction=(weights * experts).sum(1), experts=experts, weights=weights)

    def forward(self, price, context, origins, scale):
        return self.combine(self.features(price, context, origins, scale))

    def initialize(self, features, residual):
        with torch.no_grad():
            for k in range(len(self.scales)):
                g = features[:, k]
                n, dim = g.shape
                head = torch.linalg.solve(2 * g.T @ g / n + .01 * torch.eye(dim),
                                          g.T @ residual / n)
                self.readouts[k].copy_(head)


def training_loss(model, output, residual):
    correction = output['prediction']
    main = ((correction - residual).square() + correction.square()).mean()
    penalty = .01 * model.readouts.square().sum() / (20 * len(model.scales))
    auxiliary = torch.zeros_like(main)
    competition = torch.zeros_like(main)
    if len(model.scales) > 1:
        error = (output['experts'] - residual[:, None]).square()
        auxiliary = (error + output['experts'].square()).mean()
        if model.mode in ('shared_competition', 'horizon_competition'):
            detached = error.detach()
            targets = torch.softmax(-detached / detached.mean(1, keepdim=True).clamp_min(1e-6), dim=1)
            weights = output['weights']
            if model.mode == 'shared_competition':
                targets = targets.mean(2, keepdim=True)
                weights = weights[:, :, :1]
            competition = -(targets * weights.clamp_min(1e-12).log()).sum(1).mean()
    loss = main + penalty + .1 * auxiliary + .05 * competition
    return loss, dict(main=float(main.detach()), auxiliary=float(auxiliary.detach()),
                      competition=float(competition.detach()))
