"""Unmodified author PatchTST prediction core under this project's protocol.

Author repository: yuqinie98/PatchTST, Apache-2.0.
Fixed commit: 204c21efe0b39603ad6e2ca640ef5896646ab1a9.
This wrapper changes configuration/input-output shape only; it is NOT an
author-benchmark reproduction. Each instance is trained on one market's history.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import torch
from torch import Tensor, nn


COMMIT = "204c21efe0b39603ad6e2ca640ef5896646ab1a9"
VENDOR = Path(__file__).resolve().parent / "vendor" / "PatchTST_supervised"
_AUTHOR_MODEL = None


def _author_model():
    """Load exact upstream bytes without polluting an existing `layers` package.

    Upstream uses absolute `layers.*` imports. Temporary import aliases satisfy
    those imports; all pre-existing generic module bindings are restored. No
    source replacement, forward patch, or third-party reimplementation occurs.
    """
    global _AUTHOR_MODEL
    if _AUTHOR_MODEL is not None:
        return _AUTHOR_MODEL
    generic = ["layers", "layers.PatchTST_layers", "layers.RevIN", "layers.PatchTST_backbone"]
    previous = {name: sys.modules.get(name) for name in generic}
    namespace = "_carbon_patchtst_author_204c21ef"
    package = ModuleType(namespace)
    package.__path__ = [str(VENDOR)]
    sys.modules.setdefault(namespace, package)
    def load(suffix, relative):
        name = namespace + "." + suffix
        if name in sys.modules:
            return sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, VENDOR / relative)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    try:
        temp = ModuleType("layers")
        temp.__path__ = [str(VENDOR / "layers")]
        sys.modules["layers"] = temp
        for leaf in ("PatchTST_layers", "RevIN", "PatchTST_backbone"):
            module = load(leaf, "layers/" + leaf + ".py")
            sys.modules["layers." + leaf] = module
        _AUTHOR_MODEL = load("PatchTST", "models/PatchTST.py").Model
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
    return _AUTHOR_MODEL


class LocalPatchTST(nn.Module):
    """Single-market level forecaster: [B,21] or [B,21,1] -> [B,20].

    Default 21 price vertices matches the information in the shared GRU's
    20 price increments. Context 20 is retained as an explicit supported config.
    """

    def __init__(self, d_model: int = 16, affine: bool = False,
                 context_length: int = 21) -> None:
        super().__init__()
        if d_model not in (16, 32):
            raise ValueError("Bounded protocol permits d_model 16 or 32.")
        if context_length not in (20, 21):
            raise ValueError("Bounded protocol permits context_length 20 or 21.")
        self.config = SimpleNamespace(
            enc_in=1, seq_len=context_length, pred_len=20,
            e_layers=1, n_heads=2, d_model=d_model, d_ff=64,
            dropout=0.0, fc_dropout=0.0, head_dropout=0.0,
            individual=False, patch_len=5, stride=5, padding_patch="end",
            revin=True, affine=bool(affine), subtract_last=False,
            decomposition=False, kernel_size=25,
        )
        # Author defaults retained: BatchNorm, residual attention, post-norm,
        # learned positional encoding, original flatten head and RevIN.
        self.core = _author_model()(self.config)

    def forward(self, levels: Tensor) -> Tensor:
        if levels.ndim == 2:
            levels = levels.unsqueeze(-1)
        if levels.ndim != 3 or levels.shape[1:] != (self.config.seq_len, 1):
            raise ValueError(f"Provide {self.config.seq_len} past PRICE LEVELS for ONE market, shape [B,L,1].")
        return self.core(levels).squeeze(-1)

    def normalized_delta(self, levels: Tensor, training_scale: Tensor | float = 1.0) -> Tensor:
        """External task adapter: subtract current price, divide train scale sqrt(h).

        If levels were already divided by the training market scale, pass 1.0.
        This operation does not replace any author normalization or model head.
        """
        prediction = self(levels)
        current = levels[:, -1] if levels.ndim == 2 else levels[:, -1, 0]
        scale = torch.as_tensor(training_scale, device=levels.device, dtype=levels.dtype)
        if scale.ndim > 0:
            scale = scale.reshape(-1, 1)
        if bool((scale <= 0).any()):
            raise ValueError("training_scale must be strictly positive.")
        horizon = torch.arange(1, 21, device=levels.device, dtype=levels.dtype).sqrt()
        return (prediction - current[:, None]) / scale / horizon

    def metadata(self) -> dict:
        return {"author_repository": "yuqinie98/PatchTST", "author_commit": COMMIT,
                "license": "Apache-2.0", "configuration": vars(self.config),
                "patch_count": self.core.model.backbone.patch_num,
                "parameters": sum(p.numel() for p in self.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "author_core_modified": False, "author_benchmark_reproduced": False,
                "local_single_market_model": True}
