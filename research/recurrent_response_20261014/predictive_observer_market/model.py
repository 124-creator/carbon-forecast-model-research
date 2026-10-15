"""Predictive wrapper around pinned MIT RKNCell; not a new algorithm."""
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

R = Path(__file__).resolve().parent
SOURCE = R / 'third_party/rkn_share'
for name, expected in json.loads((R / 'AUTHOR_SOURCE.json').read_text())['source_objects'].items():
    # Git can normalize LF/CRLF in working trees; reviewed source text is LF.
    data = (SOURCE / name).read_bytes().replace(b'\r\n', b'\n')
    actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    if actual != expected:
        raise ValueError('Author source mismatch: ' + name)
sys.path.insert(0, str(SOURCE))
from rkn_cell.RKNCell import RKNCell


class Model(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        if kind == 'GRUpatch':
            self.encoder = nn.GRU(5, 16, batch_first=True)
            self.head = nn.Linear(16, 20)
            return
        if kind not in ('RKNconditional', 'RKNglobal'):
            raise ValueError(kind)
        self.embedding = nn.Sequential(nn.Linear(5, 16), nn.Tanh())
        self.mean = nn.Linear(16, 8)
        self.variance = nn.Linear(16, 8)
        config = RKNCell.get_default_config()
        config.num_basis = 3
        config.bandwidth = 1
        config.trans_net_hidden_units = [16]
        self.cell = RKNCell(8, config)
        self.head = nn.Linear(16, 20)

    def forward(self, x):
        if self.kind == 'GRUpatch':
            return self.head(self.encoder(x.reshape(-1, 12, 5))[0][:, -1])
        h = self.embedding(x.reshape(-1, 12, 5))
        observation = F.normalize(self.mean(h), dim=-1, eps=1e-8)
        variance_input = h if self.kind == 'RKNconditional' else h * 0
        variance = F.softplus(self.variance(variance_input)) + 1e-4
        mean = x.new_zeros((len(x), 16))
        cov = [x.new_ones((len(x), 8)), x.new_ones((len(x), 8)), x.new_zeros((len(x), 8))]
        post = mean
        for t in range(12):
            post, _, mean, cov = self.cell(mean, cov, observation[:, t], variance[:, t])
        return self.head(post)
