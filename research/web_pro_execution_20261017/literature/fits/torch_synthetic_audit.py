"""Audit original FITS and an isolated output-length fix using synthetic inputs only."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import importlib.util
import json
import platform
import sys
import warnings

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'official'))
from models.FITS import Model as OriginalFITS

spec = importlib.util.spec_from_file_location('fits_lengthfix', ROOT / 'patched' / 'FITS_lengthfix.py')
patched_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patched_module)
PatchedFITS = patched_module.Model
torch.set_num_threads(1)
torch.manual_seed(17)

def config(L, H, K=4):
    return SimpleNamespace(seq_len=L, pred_len=H, individual=False, enc_in=1, cut_freq=K)

def pair(L, H, K=4):
    original = OriginalFITS(config(L, H, K))
    patched = PatchedFITS(config(L, H, K))
    patched.load_state_dict(original.state_dict())
    return original, patched

shapes = []
for L in [20, 21, 60]:
    for H in [1, 5, 20]:
        original, patched = pair(L, H)
        x = torch.arange(L, dtype=torch.float32).reshape(1, L, 1)
        y0, _ = original(x)
        y1, _ = patched(x)
        M = L + H
        assert y0.shape[1] == 2 * (M // 2)
        assert y1.shape[1] == M
        difference = None
        if M % 2 == 0:
            difference = float((y0 - y1).abs().max())
            assert difference == 0.0
        shapes.append({
            'L': L, 'H': H, 'original_length': y0.shape[1], 'patched_length': y1.shape[1],
            'expected_length': M, 'even_length_max_abs_difference': difference,
        })

L, H = 20, 5
original, patched = pair(L, H)
x = torch.arange(L, dtype=torch.float32).reshape(1, L, 1)
target = torch.arange(L + H, dtype=torch.float32).reshape(1, L + H, 1)
loss_shapes = {}
for name, model in [('original', original), ('patched', patched)]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            loss = F.mse_loss(model(x)[0], target)
        loss_shapes[name] = {'status': 'finite', 'loss': float(loss)}
    except RuntimeError as exc:
        loss_shapes[name] = {'status': 'RuntimeError', 'reason': str(exc)}
assert loss_shapes['original']['status'] == 'RuntimeError'
assert loss_shapes['patched']['status'] == 'finite'

# Opposite discarded Nyquist modes have equal mean, variance and retained spectrum.
sign = (-1.0) ** torch.arange(L)
xp = (30 + 2 * sign).reshape(1, L, 1)
xm = (30 - 2 * sign).reshape(1, L, 1)
collisions = []
for seed in [0, 17, 42]:
    torch.manual_seed(seed)
    for name, cls in [('original', OriginalFITS), ('patched', PatchedFITS)]:
        model = cls(config(L, H))
        # Exercise arbitrary complex weights, including imaginary parts, without training.
        with torch.no_grad():
            for p in model.parameters():
                p.copy_(torch.complex(torch.randn_like(p.real), torch.randn_like(p.real)))
        yp, _ = model(xp)
        ym, _ = model(xm)
        diff = float((yp[:, -H:] - ym[:, -H:]).abs().max())
        collisions.append({'model': name, 'seed': seed, 'forecast_max_abs_difference': diff})
        assert diff < 1e-5
assert float(abs(xp[0, -1, 0] - xm[0, -1, 0])) == 4.0

# Independently expand the normalization cancellation: mu + A(x-mu) + s*c_bias.
model = PatchedFITS(config(20, 5))
torch.manual_seed(99)
x = 30 + 3 * torch.randn(3, 20, 1)
y, _ = model(x)
mu = x.mean(dim=1, keepdim=True)
centered = x - mu
s = (centered.var(dim=1, keepdim=True) + 1e-5).sqrt()
retained = torch.fft.rfft(centered, dim=1)[:, :model.dominance_freq, :].permute(0, 2, 1)
linear = F.linear(retained, model.freq_upsampler.weight, bias=None).permute(0, 2, 1)
def invert(low):
    padded = torch.zeros(low.shape[0], 13, 1, dtype=low.dtype)
    padded[:, :low.shape[1], :] = low
    return torch.fft.irfft(padded, n=25, dim=1) * model.length_ratio
bias = model.freq_upsampler.bias.reshape(1, -1, 1)
expanded = mu + invert(linear) + s * invert(bias)
equivalence = float((y - expanded).abs().max())
assert equivalence < 1e-5

# The author's default harmonic rule can be invalid for a very short window.
default_K = (20 // 24 + 1) * 2 + 10
try:
    OriginalFITS(config(20, 5, default_K))(xp)
    default_cutoff = {'status': 'unexpected_success'}
except RuntimeError as exc:
    default_cutoff = {'status': 'RuntimeError', 'reason': str(exc)}
assert default_cutoff['status'] == 'RuntimeError'

files = sorted((ROOT / 'official').rglob('*')) + [ROOT / 'patched' / 'FITS_lengthfix.py']
receipt = {
    'scope': 'synthetic forward/algebra checks only; no training or market observations',
    'source_repo': 'https://github.com/VEWOXIC/FITS',
    'source_commit': 'd040bb015b6299da26d879b90dd19c80fb72c160',
    'source_commit_date': '2025-06-04T13:16:38Z',
    'source_license': 'Apache-2.0',
    'source_requirements_torch': '2.7.0',
    'actual_torch': torch.__version__, 'python': platform.python_version(),
    'dependency_note': 'Actual runtime differs from current author pin; sufficient for the stated shape/algebra checks, not a reproduction result.',
    'patch': 'Only functional change: irfft(..., n=self.seq_len+self.pred_len, dim=1). Original source untouched.',
    'shape_cases': shapes,
    'backcast_plus_forecast_loss_shape': loss_shapes,
    'high_frequency_collision': {
        'definition': 'L=20, K=4, x_plus[t]=30+2*(-1)^t, x_minus[t]=30-2*(-1)^t',
        'current_price_difference': 4.0, 'cases': collisions,
    },
    'normalization_equivalence_max_abs_error': equivalence,
    'invalid_default_cutoff': {'L': 20, 'H': 5, 'K': default_K, **default_cutoff},
    'sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file() and '__pycache__' not in str(p)},
    'status': 'PASS: all asserted synthetic properties verified',
}
(ROOT / 'torch_synthetic_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({k: receipt[k] for k in ['status', 'actual_torch', 'shape_cases', 'backcast_plus_forecast_loss_shape', 'high_frequency_collision', 'normalization_equivalence_max_abs_error', 'invalid_default_cutoff']}, indent=2))
