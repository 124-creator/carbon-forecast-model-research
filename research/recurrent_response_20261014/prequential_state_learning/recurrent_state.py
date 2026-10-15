"""Own recurrent-state learning implementation, not an ALPaCA reproduction."""
import numpy as np
import torch
from torch import nn


def profile_head(g, e):
    n, dim = g.shape
    return torch.linalg.solve(2 * g.T @ g / n + .01 * torch.eye(dim, dtype=g.dtype, device=g.device), g.T @ e / n)


def mature_heads(g, e, origins, query_origins, complete=False, cap=240):
    horizons = torch.arange(1, e.shape[1] + 1, device=g.device)
    delays = torch.full_like(horizons, e.shape[1]) if complete else horizons
    available = origins[None, None, :] + delays[None, :, None] < query_origins[:, None, None]
    recency_rank = torch.flip(torch.cumsum(torch.flip(available.long(), [-1]), -1), [-1])
    mask = (available & (recency_rank <= cap)).to(g.dtype)
    counts = mask.sum(-1)
    denominator = counts.clamp_min(1)
    gram = torch.einsum('qhn,nd,ne->qhde', mask, g, g) / denominator[:, :, None, None]
    rhs = torch.einsum('qhn,nd,nh->qhd', mask, g, e) / denominator[:, :, None]
    regularizer = .01 * torch.eye(g.shape[1], dtype=g.dtype, device=g.device)
    return torch.linalg.solve(2 * gram + regularizer, rhs.unsqueeze(-1)).squeeze(-1), counts


def prequential_loss(g, e, origins, queries, complete=False):
    heads, counts = mature_heads(g, e, origins, origins[queries], complete=complete)
    prediction = (g[queries, None, :] * heads).sum(-1)
    valid = counts >= 40
    if not bool(valid.any()):
        raise ValueError('No historical query has the required mature support')
    return ((prediction - e[queries]).square() + prediction.square())[valid].mean()


class RecurrentPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.GRU(1, 16, batch_first=True)
        self.U = nn.Linear(16, 8, bias=False)
        self.V = nn.Linear(3, 8, bias=False)
        self.c = nn.Parameter(torch.zeros(8))
        self.B = nn.Parameter(torch.zeros(8, 20))

    def features(self, delta, context, origins, history):
        if history == 'full':
            sequence = delta[:int(origins.max())][None, :, None]
            hidden = self.encoder(sequence)[0][0, origins - 1]
        elif history == 'window':
            sequence = torch.stack([delta[int(i) - 60:int(i)] for i in origins])[:, :, None]
            hidden = self.encoder(sequence)[0][:, -1]
        else:
            raise ValueError('Unknown history encoding')
        return torch.tanh(self.U(hidden) + self.V(context) + self.c) - torch.tanh(self.c)
