"""Own stopped-model audit: coordinate equivalence, not a market experiment."""
import copy
import json
import torch
from coupled_model import ForecastModel, ladder_step

torch.set_num_threads(1)
torch.manual_seed(37)
model = ForecastModel('coupled').double()
with torch.no_grad():
    model.feedback.weight.normal_(0, .2)
inputs = torch.randn(3, 60, dtype=torch.float64)
basis = torch.tensor([[0., 1., 0., 0.], [0., -1., 1., 0.],
                      [0., 0., -1., 1.], [1., 0., 0., -1.]], dtype=torch.float64)
alternative = copy.deepcopy(model.encoder)
with torch.no_grad():
    alternative.weight_ih.copy_(model.encoder.weight_ih @ basis)
hidden = inputs.new_zeros(3, 16)
states = inputs.new_zeros(3, 3)
for observed in inputs.unbind(1):
    rates = 1-torch.cumprod(1-model.feedback(hidden).sigmoid(), dim=-1)
    states, _ = ladder_step(observed, states, rates)
    hidden = alternative(torch.cat([observed[:, None], states], dim=1), hidden)
error = float((model(inputs)-model.head(hidden)).abs().max())
if error > 1e-12:
    raise ValueError('Coordinate-equivalence check failed')
print(json.dumps(dict(passed=True, optimized=not __debug__, max_forecast_error=error,
    basis_determinant=float(torch.linalg.det(basis)),
    scope='Nonzero-feedback full-sequence coordinate equivalence; not market performance')))
