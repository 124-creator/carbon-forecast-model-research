"""Independent NumPy/math evaluation of the exact small frozen PatchTST.

Implements saved-state inference only: 21 levels, 5 patches of length/stride 5,
two heads, one post-BatchNorm encoder, GELU, learned position, flatten head,
RevIN (no affine). No training framework or author forward is imported.
"""
from __future__ import annotations
import math
import numpy as np


def _linear(x, state, stem):
    return x @ state[stem + ".weight"].T + state[stem + ".bias"]


def _batchnorm(x, state, stem):
    return ((x - state[stem + ".running_mean"]) / np.sqrt(state[stem + ".running_var"] + 1e-5)
            * state[stem + ".weight"] + state[stem + ".bias"])


def predict_raw(state, levels, metadata):
    c = metadata["configuration"]
    fixed = dict(enc_in=1, seq_len=21, pred_len=20, e_layers=1, n_heads=2,
                 d_ff=64, dropout=0.0, fc_dropout=0.0, head_dropout=0.0,
                 individual=False, patch_len=5, stride=5, padding_patch="end",
                 revin=True, affine=False, subtract_last=False, decomposition=False)
    for key, value in fixed.items():
        assert c[key] == value, (key, c[key], value)
    price = np.asarray(levels, dtype=np.float64)
    assert price.ndim == 2 and price.shape[1] == 21
    n, width, head_width = len(price), int(c["d_model"]), int(c["d_model"]) // 2
    mean = price.mean(axis=1, keepdims=True)
    sd = np.sqrt(np.mean((price - mean) ** 2, axis=1, keepdims=True) + 1e-5)
    normalized = (price - mean) / sd
    # ReplicationPad1d((0,5)) then unfold(size=5,step=5): starts 0,5,10,15,20.
    padded = np.concatenate([normalized, np.repeat(normalized[:, -1:], 5, axis=1)], axis=1)
    patches = np.stack([padded[:, start:start + 5] for start in (0, 5, 10, 15, 20)], axis=1)
    base = "core.model.backbone"
    x = _linear(patches, state, base + ".W_P") + state[base + ".W_pos"]
    block = base + ".encoder.layers.0"
    q = _linear(x, state, block + ".self_attn.W_Q").reshape(n, 5, 2, head_width).transpose(0, 2, 1, 3)
    k = _linear(x, state, block + ".self_attn.W_K").reshape(n, 5, 2, head_width).transpose(0, 2, 3, 1)
    v = _linear(x, state, block + ".self_attn.W_V").reshape(n, 5, 2, head_width).transpose(0, 2, 1, 3)
    score = (q @ k) * state[block + ".self_attn.sdp_attn.scale"]
    weight = np.exp(score - score.max(axis=-1, keepdims=True))
    weight /= weight.sum(axis=-1, keepdims=True)
    attn = (weight @ v).transpose(0, 2, 1, 3).reshape(n, 5, width)
    attn = _linear(attn, state, block + ".self_attn.to_out.0")
    x = _batchnorm(x + attn, state, block + ".norm_attn.1")
    ff = _linear(x, state, block + ".ff.0")
    # Exact GELU used by nn.GELU(approximate='none'). Python's math.erf is
    # applied elementwise; no SciPy or PyTorch dependency is needed.
    erf = np.fromiter((math.erf(float(a)) for a in (ff / math.sqrt(2)).ravel()),
                      dtype=np.float64, count=ff.size).reshape(ff.shape)
    ff = 0.5 * ff * (1.0 + erf)
    ff = _linear(ff, state, block + ".ff.3")
    x = _batchnorm(x + ff, state, block + ".norm_ffn.1")
    flat = x.transpose(0, 2, 1).reshape(n, width * 5)
    forecast_normalized_level = _linear(flat, state, "core.model.head.linear")
    forecast_level = forecast_normalized_level * sd + mean
    return forecast_level - price[:, -1:]
