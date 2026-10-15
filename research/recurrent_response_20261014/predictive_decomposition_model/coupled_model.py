"""Own experimental architecture; no author code or weights are imported."""
import torch
from torch import nn


def ladder_step(x, previous, rates):
    states = (1-rates)*previous + rates*x[:, None]
    bands = torch.stack([states[:, 0], states[:, 1]-states[:, 0],
                         states[:, 2]-states[:, 1], x-states[:, 2]], dim=-1)
    return states, bands


class ForecastModel(nn.Module):
    def __init__(self, kind):
        super().__init__()
        if kind not in ['plain', 'fixed', 'global', 'coupled']:
            raise ValueError('Unknown model kind')
        self.kind = kind
        if kind == 'plain':
            self.encoder = nn.GRU(1, 16, batch_first=True)
            self.head = nn.Linear(16, 20)
            return
        self.encoder = nn.GRUCell(4, 16)
        self.head = nn.Linear(16, 20)
        target = torch.tensor([1/20, 1/5, 1/2])
        stick = torch.cat([target[:1], (target[1:]-target[:-1])/(1-target[:-1])])
        self.register_buffer('fixed_logits', torch.logit(stick))
        if kind == 'global':
            self.raw_rates = nn.Parameter(self.fixed_logits.clone())
        if kind == 'coupled':
            self.feedback = nn.Linear(16, 3)
            nn.init.zeros_(self.feedback.weight)
            with torch.no_grad():
                self.feedback.bias.copy_(self.fixed_logits)

    def forward(self, x, return_trace=False):
        if self.kind == 'plain':
            hidden = self.encoder(x[:, :, None])[0]
            prediction = self.head(hidden[:, -1])
            return (prediction, {'hidden': hidden}) if return_trace else prediction
        hidden = x.new_zeros(x.shape[0], 16)
        states = x.new_zeros(x.shape[0], 3)
        traces = {'components': [], 'rates': [], 'hidden': []}
        for observed in x.unbind(1):
            logits = self.raw_rates if self.kind == 'global' else self.fixed_logits
            if self.kind == 'coupled':
                logits = self.feedback(hidden)
            stick = logits.sigmoid().expand(x.shape[0], -1)
            rates = 1-torch.cumprod(1-stick, dim=-1)
            states, bands = ladder_step(observed, states, rates)
            hidden = self.encoder(bands, hidden)
            if return_trace:
                for key, value in [('components', bands), ('rates', rates), ('hidden', hidden)]:
                    traces[key].append(value)
        prediction = self.head(hidden)
        if return_trace:
            return prediction, {key: torch.stack(value, dim=1) for key, value in traces.items()}
        return prediction
