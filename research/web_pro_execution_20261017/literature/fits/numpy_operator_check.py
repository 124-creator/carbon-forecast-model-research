"""Source-equivalent algebra checks; no training, market data, or new model."""
from pathlib import Path
import json
import numpy as np

rng = np.random.default_rng(17)
L, H, K = 20, 5, 4
M, eta = L + H, (L + H) / L
Ko = int(K * eta)
W = rng.normal(size=(Ko, K)) + 1j * rng.normal(size=(Ko, K))
b = rng.normal(size=Ko) + 1j * rng.normal(size=Ko)

def operator(x, bias=True, corrected_n=True):
    mu = x.mean()
    s = np.sqrt(np.var(x, ddof=1) + 1e-5)
    z = (x - mu) / s
    spec = np.fft.rfft(z)[:K]
    low = W @ spec + (b if bias else 0)
    padded = np.zeros(M // 2 + 1, dtype=complex)
    padded[:Ko] = low
    inverse = np.fft.irfft(padded, n=M) if corrected_n else np.fft.irfft(padded)
    return mu + s * eta * inverse, spec, mu, s

x_plus = 30 + 2 * (-1.0) ** np.arange(L)
x_minus = 30 - 2 * (-1.0) ** np.arange(L)
y_plus, spec_plus, mu_plus, s_plus = operator(x_plus)
y_minus, spec_minus, mu_minus, s_minus = operator(x_minus)
row = {
    'check_type': 'NumPy algebra from author model, NOT author PyTorch execution or market training',
    'L': L, 'H': H, 'K': K, 'K_out': Ko,
    'source_default_irfft_output_length': len(operator(x_plus, corrected_n=False)[0]),
    'corrected_irfft_output_length': len(y_plus),
    'retained_spectrum_max_difference': float(np.max(abs(spec_plus - spec_minus))),
    'mean_difference': float(abs(mu_plus - mu_minus)),
    'std_difference': float(abs(s_plus - s_minus)),
    'forecast_max_difference': float(np.max(abs(y_plus[-H:] - y_minus[-H:]))),
    'current_price_difference': float(abs(x_plus[-1] - x_minus[-1])),
}
print(json.dumps(row, indent=2))
Path(__file__).with_name('numpy_operator_check.json').write_text(json.dumps(row, indent=2) + '\n')
