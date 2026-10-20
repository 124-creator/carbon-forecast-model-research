"""Matched point baselines for the SMP experiment; no new-method claims."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from smp_model import SMPModel, MIN_MARK_SCALE


class DirectRecurrent(nn.Module):
    def __init__(self, input_dim, cell="gru", hidden_size=32):
        super().__init__()
        self.encoder = {"gru": nn.GRU, "lstm": nn.LSTM}[cell](input_dim, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 20)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, state=None, return_aux=False):
        values, _ = self.encoder(x)
        m = self.head(values[:, -1])
        return (m, None) if return_aux else m


class DirectGRUAux(nn.Module):
    """Direct GRU mean plus exactly the same current-state hurdle supervision.

    This prevents interpreting generic auxiliary likelihood regularization as
    an advantage of future-state propagation. The point head is unconstrained.
    """
    def __init__(self, input_dim, hidden_size=32):
        super().__init__()
        template = SMPModel(input_dim, hidden_size, 20, "full")
        self.encoder, self.q_head, self.b_head = template.encoder, template.q_head, template.b_head
        self.head = nn.Linear(hidden_size, 20)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def initialize_head(self, q0, mean_abs_nonzero):
        # Reuse the exact initialization contract without adding a second model.
        return SMPModel.initialize_head(self, q0, mean_abs_nonzero)

    n_q_groups = n_b_groups = 9

    def forward(self, x, state, return_aux=False):
        _, hidden = self.encoder(x)
        z = hidden[-1]
        m = self.head(z)
        if not return_aux:
            return m
        logq = F.log_softmax(self.q_head(z).reshape(-1, 9, 3), dim=-1)
        b = F.softplus(self.b_head(z).reshape(-1, 9, 2)) + MIN_MARK_SCALE
        rows = torch.arange(len(x), device=x.device)
        aux = dict(q=logq.exp(), b=b, log_q=logq,
                   q_current=logq[rows, state].exp(), b_current=b[rows, state],
                   log_q_current=logq[rows, state])
        return m, aux


_AUTHOR_MODEL = None


def author_patchtst_class():
    """Import fixed upstream bytes through temporary package aliases."""
    global _AUTHOR_MODEL
    if _AUTHOR_MODEL is not None:
        return _AUTHOR_MODEL
    vendor = Path(__file__).resolve().parent / "vendor" / "PatchTST_supervised"
    generic = ["layers", "layers.PatchTST_layers", "layers.RevIN", "layers.PatchTST_backbone"]
    previous = {name: sys.modules.get(name) for name in generic}
    namespace = "_smp_author_patchtst_204c21ef"
    package = ModuleType(namespace)
    package.__path__ = [str(vendor)]
    sys.modules.setdefault(namespace, package)

    def load(leaf, relative):
        name = namespace + "." + leaf
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, vendor / relative)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        return sys.modules[name]

    try:
        temp = ModuleType("layers")
        temp.__path__ = [str(vendor / "layers")]
        sys.modules["layers"] = temp
        for leaf in ("PatchTST_layers", "RevIN", "PatchTST_backbone"):
            sys.modules["layers." + leaf] = load(leaf, "layers/" + leaf + ".py")
        _AUTHOR_MODEL = load("PatchTST", "models/PatchTST.py").Model
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
    return _AUTHOR_MODEL


class PatchTSTAdapter(nn.Module):
    """Author core with a documented same-information supervised task adapter.

    All shared input channels enter the unmodified channel-independent upstream
    core. A shared linear channel readout maps [B,H,C] to the single price-change
    target. This readout and RevIN=False adapt to already-normalized heterogeneous
    price/calendar/state/auction tokens. It is not the author's benchmark setup.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.config = SimpleNamespace(
            enc_in=input_dim, seq_len=20, pred_len=20, e_layers=1, n_heads=2,
            d_model=16, d_ff=64, dropout=0., fc_dropout=0., head_dropout=0.,
            individual=False, patch_len=5, stride=5, padding_patch="end",
            revin=False, affine=False, subtract_last=False, decomposition=False, kernel_size=25)
        self.core = author_patchtst_class()(self.config)
        self.readout = nn.Linear(input_dim, 1)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, x, state=None, return_aux=False):
        m = self.readout(self.core(x)).squeeze(-1)
        return (m, None) if return_aux else m


FAMILIES = ("smp_full", "smp_no_clock", "smp_no_sign", "smp_pooled_mark",
            "smp_iid_state", "smp_one_step", "smp_no_aux", "gru_native",
            "gru_normalized", "gru_aux", "lstm_native", "patchtst_native")


def make_model(family, input_dim, q0=.1, mean_abs_nonzero=1.):
    if family not in FAMILIES:
        raise ValueError("Unknown model family: " + family)
    if family.startswith("smp_"):
        variant = family[4:]
        net = SMPModel(input_dim, variant="full" if variant == "no_aux" else variant)
    elif family == "gru_aux":
        net = DirectGRUAux(input_dim)
    elif family == "patchtst_native":
        net = PatchTSTAdapter(input_dim)
    else:
        net = DirectRecurrent(input_dim, cell="lstm" if family == "lstm_native" else "gru")
    if hasattr(net, "initialize_head"):
        net.initialize_head(q0, mean_abs_nonzero)
    return net


def auxiliary_weight(family):
    return .05 if (family.startswith("smp_") and family != "smp_no_aux") or family == "gru_aux" else 0.
