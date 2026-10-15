"""Ordinary full-history GRU with explicitly matched source-message ablations."""
import torch
from torch import nn

MODES = ('price', 'clock', 'raw_hold', 'raw_event', 'hold', 'event')


class EventNet(nn.Module):
    def __init__(self, mode):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode
        channels = dict(price=1, clock=5, raw_hold=11, raw_event=11, hold=13, event=13)
        self.encoder = nn.GRU(channels[mode], 16, batch_first=True)
        self.U = nn.Linear(16, 8, bias=False)
        self.V = nn.Linear(3, 8, bias=False)
        self.c = nn.Parameter(torch.zeros(8))
        self.register_buffer('B', torch.zeros(9, 20))
        if mode in ('event', 'hold'):
            self.source_encoder = nn.GRU(7, 8, batch_first=True)

    def messages(self, data):
        if self.mode in ('event', 'hold'):
            observed = self.source_encoder(data['source'][None])[0][0]
            neutral = self.source_encoder(data['neutral_source'][None])[0][0]
            state = observed - neutral
        elif self.mode in ('raw_hold', 'raw_event'):
            state = data['source'][:, :6] - data['neutral_source'][:, :6]
        else:
            raise ValueError('Price and clock controls have no source messages')
        padded = torch.cat((torch.zeros_like(state[:1]), state))
        held = padded[data['source_index']]
        return held if self.mode in ('hold', 'raw_hold') else held - padded[data['previous_index']]

    def features(self, data, context, origins):
        parts = [data['delta'][:, None]]
        if self.mode in ('raw_hold', 'raw_event', 'hold', 'event'):
            parts.append(self.messages(data))
        if self.mode != 'price':
            parts.append(data['clock'])
        inputs = torch.cat(parts, dim=1)
        observed = self.encoder(inputs[None])[0][0]
        neutral = self.encoder(torch.zeros_like(inputs)[None])[0][0]
        hidden = torch.cat((torch.zeros_like(observed[:1]), observed - neutral))[origins]
        features = torch.tanh(self.U(hidden) + self.V(context[origins]) + self.c) - torch.tanh(self.c)
        return torch.cat((features, torch.ones_like(features[:, :1])), dim=1)

    def forward(self, data, context, base, origins):
        return base[origins] + self.features(data, context, origins) @ self.B
