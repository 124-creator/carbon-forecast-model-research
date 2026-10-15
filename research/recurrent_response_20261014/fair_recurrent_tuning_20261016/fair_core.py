"""Matched full-prefix recurrent infrastructure; ordinary baselines only."""
import numpy as np
import torch
from torch import nn

MODES = ('gru_dense', 'gru_profile', 'gru_rank6', 'lstm_dense')

def configure():
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.backends.mkldnn.enabled = False
    torch.use_deterministic_algorithms(True)

def reference(price):
    bank = [(0., 1.)] + [(r, q) for r in (0., .25, .5, .75, .9, .97)
                         for q in (.02, .1, .25, .5, .8)]
    rho, permanent = np.array(bank).T
    transitory = 1 - permanent
    n = len(price)
    state = np.zeros(len(bank))
    variance = transitory / (1 - rho ** 2)
    states = np.zeros((n, len(bank)))
    errors = np.zeros_like(states)
    innovations = np.ones_like(states)
    for t in range(1, n):
        error = price[t] - price[t - 1] - (rho - 1) * state
        innovation = (rho - 1) ** 2 * variance + 1
        covariance = rho * (rho - 1) * variance + transitory
        state = rho * state + covariance * error / innovation
        variance = np.maximum(rho ** 2 * variance + transitory - covariance ** 2 / innovation, 0)
        states[t], errors[t], innovations[t] = state, error, innovation
    base = np.zeros((n, 20))
    context = np.zeros((n, 3))
    selected = 0
    for t in range(n):
        if t >= 60 and (t - 60) % 20 == 0:
            start = max(1, t - 240)
            scale = np.maximum(np.mean(errors[start:t] ** 2 / innovations[start:t], axis=0), 1e-12)
            criterion = np.log(innovations[start:t]).sum(0) + (t - start) * np.log(scale)
            selected = int(np.argmin(criterion))
        r, q = bank[selected]
        context[t] = [states[t, selected], r * (1 - q), 1 - q]
        base[t] = (r ** np.arange(1, 21) - 1) * states[t, selected]
    return base, context

class RecurrentNet(nn.Module):
    def __init__(self, mode, hidden):
        super().__init__()
        if mode not in MODES or hidden <= 0:
            raise ValueError('Unknown recurrent family or width')
        self.mode = mode
        self.encoder = (nn.LSTM if mode.startswith('lstm') else nn.GRU)(1, hidden, batch_first=True)
        self.U = nn.Linear(hidden, 8, bias=False)
        self.V = nn.Linear(3, 8, bias=False)
        self.bias = nn.Parameter(torch.zeros(8))
        if mode == 'gru_rank6':
            knots = np.array([0, 1, 2, 3, 6, 11, 20])
            basis = np.stack([np.interp(np.arange(1, 21), knots, np.r_[0., row])
                              for row in np.eye(6)], axis=1)
        else:
            basis = np.eye(20)
        self.register_buffer('basis', torch.as_tensor(basis))
        head = torch.zeros(9, basis.shape[1])
        if mode == 'gru_profile':
            self.register_buffer('readout', head)
        else:
            self.readout = nn.Parameter(head)

    def initialize(self, features, residual):
        with torch.no_grad():
            n, dim = features.shape
            dense = torch.linalg.solve(2 * features.T @ features / n + .01 * torch.eye(dim),
                                       features.T @ residual / n)
            right = dense @ self.basis
            coefficients = torch.linalg.solve(self.basis.T @ self.basis, right.T).T
            self.readout.copy_(coefficients)

    def correction(self, features):
        return features @ (self.readout @ self.basis.T)

    def features(self, price, context, origins, scale):
        origins = np.asarray(origins, dtype=np.int64)
        if len(origins) == 0 or origins.min() < 1 or origins.max() >= len(price) or scale <= 0:
            raise ValueError('Invalid feature support')
        delta = torch.as_tensor(np.diff(price[:int(origins.max()) + 1]) / scale)
        sequence = delta[None, :, None]
        observed = self.encoder(sequence)[0][0]
        neutral = self.encoder(torch.zeros_like(sequence))[0][0]
        hidden = (observed - neutral)[torch.as_tensor(origins - 1)]
        q = np.array(context[origins], copy=True)
        q[:, 0] = np.tanh(q[:, 0] / scale)
        g = torch.tanh(self.U(hidden) + self.V(torch.as_tensor(q)) + self.bias) - torch.tanh(self.bias)
        return torch.cat((g, torch.ones_like(g[:, :1])), dim=1)
