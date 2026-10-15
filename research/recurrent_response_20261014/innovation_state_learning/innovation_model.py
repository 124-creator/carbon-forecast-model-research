"""Own implementation; no third-party model code is imported."""
import torch
from torch import nn


def affine_scan(a, b):
    """All zero-initialized prefix states of x[t] = a[t] x[t-1] + b[t]."""
    offset = 1
    while offset < len(b):
        b = torch.cat((b[:offset], torch.bmm(a[offset:], b[:-offset, :, None]).squeeze(-1) + b[offset:]))
        a = torch.cat((a[:offset], torch.bmm(a[offset:], a[:-offset])))
        offset *= 2
    return b


def basis():
    locations = torch.arange(1, 21, dtype=torch.get_default_dtype())
    knots = torch.linspace(1, 20, 6)
    interpolation = (1 - (locations[:, None] - knots).abs() / (knots[1] - knots[0])).clamp_min(0)
    return torch.linalg.qr(interpolation, mode='reduced').Q


class InnovationNet(nn.Module):
    def __init__(self, mode):
        super().__init__()
        if mode not in ('innovation_shift', 'raw_shift', 'innovation_diagonal', 'constant_shift'):
            raise ValueError(mode)
        self.mode = mode
        self.register_buffer('Q', basis())
        self.register_buffer('decay', .9 ** torch.arange(20, dtype=torch.get_default_dtype()))
        if mode != 'constant_shift':
            self.encoder = nn.GRU(1, 16, batch_first=True)
            self.U = nn.Linear(16, 8, bias=False)
            self.V = nn.Linear(3, 8, bias=False)
            self.c = nn.Parameter(torch.zeros(8))
        self.B = nn.Parameter(torch.zeros(1 if mode == 'constant_shift' else 9, 6))

    def gains(self, delta, context):
        if self.mode == 'constant_shift':
            features = torch.ones((len(delta), 1), dtype=delta.dtype, device=delta.device)
        else:
            observed = self.encoder(delta[None, :, None])[0][0]
            neutral = self.encoder(torch.zeros_like(delta)[None, :, None])[0][0]
            previous = torch.cat((torch.zeros_like(observed[:1]), observed[:-1] - neutral[:-1]))
            features = torch.tanh(self.U(previous) + self.V(context[:-1]) + self.c) - torch.tanh(self.c)
            features = torch.cat((features, torch.ones_like(features[:, :1])), dim=1)
        raw = (features @ self.B) @ self.Q.T
        return .09 * self.decay * raw / torch.sqrt(1 + raw.square().sum(-1, keepdim=True))

    def forward(self, delta, context, base):
        gain = self.gains(delta, context)
        shift = torch.diag(torch.ones(19, dtype=delta.dtype, device=delta.device), diagonal=1)
        if self.mode == 'innovation_diagonal':
            shift = .9 * torch.eye(20, dtype=delta.dtype, device=delta.device)
        a = shift.expand(len(delta), -1, -1)
        if self.mode != 'raw_shift':
            first = torch.zeros(20, dtype=delta.dtype, device=delta.device)
            first[0] = 1
            a = a - gain[:, :, None] * first[None, None, :]
        observation = delta - base[:-1, 0]
        states = affine_scan(a, gain * observation[:, None])
        corrections = torch.cat((torch.zeros_like(base[:1]), states.cumsum(-1)))
        return base + corrections
