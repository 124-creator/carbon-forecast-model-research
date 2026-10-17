"""Execute selected unchanged DSOF author methods through a causal thin adapter.

Own integration code. Author source is NOT distributed with this module. Pass a
private reference checkout at commit 11cebd980a1f77ef0e1db9ac313dbfe6d37984e9.
This is an execution anchor, not the authors' complete benchmark reproduction.
The author's data/driver imports are replaced by inert interface declarations;
the original model forward, residual forward, ER, TD, losses, and step methods
are imported unchanged. Causal buffers are reconstructed from observed prices.
Only single-channel price input is supported. Scaling belongs to the caller.
"""
from __future__ import annotations

import hashlib
import importlib.util
import itertools
from pathlib import Path
import sys
import types
import warnings

import numpy as np
import torch
from torch import nn


AUTHOR_COMMIT = "11cebd980a1f77ef0e1db9ac313dbfe6d37984e9"
EXPECTED_BLOBS = {
    "src/models/blocks/trendDecomposition.py": "7330109b4cac6933b222d4243b59ba2ae02dba33",
    "src/models/DLinear.py": "a15a1ceba0893d1d3b5bc8d7201e02bd1afe5040",
    "src/models/MLP.py": "6d2ade34407fd231f415c06d3274e704f1e626cf",
    "src/trainers/trainerBaseERTD.py": "4652b39c136bc7ee6c3f2e672439fc2b3ae7b8dd",
    "src/trainers/w_student/trainerBaseFramework.py": "578d8310d244cfba33da0f6395b77ebad7453293",
    "src/trainers/w_student/trainerFramework_ERTDRes.py": "02d91c07bfdf8bd1b57d4b601b0b084d49318010",
}


def _load_author_classes(author_root):
    root = Path(author_root).resolve()
    hashes = {}
    for rel, expected in EXPECTED_BLOBS.items():
        raw = (root / rel).read_bytes()
        got = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if got != expected:
            raise ValueError(f"Author source mismatch: {rel}: {got} != {expected}")
        hashes[rel] = hashlib.sha256(raw).hexdigest()

    # Imported trainer files refer to the original experiment driver and dataset
    # only through inheritance/type declarations here. No dataset is instantiated.
    names = ["models", "models.blocks", "trainers", "trainers.w_student",
             "trainers.trainerBase", "data", "data.data_loader", "utils", "utils.tools"]
    file_modules = [
        ("models.blocks.trendDecomposition", "src/models/blocks/trendDecomposition.py"),
        ("models.DLinear", "src/models/DLinear.py"),
        ("models.MLP", "src/models/MLP.py"),
        ("trainers.trainerBaseERTD", "src/trainers/trainerBaseERTD.py"),
        ("trainers.w_student.trainerBaseFramework", "src/trainers/w_student/trainerBaseFramework.py"),
        ("trainers.w_student.trainerFramework_ERTDRes", "src/trainers/w_student/trainerFramework_ERTDRes.py"),
    ]
    all_names = names + [n for n, _ in file_modules]
    previous = {n: sys.modules.get(n) for n in all_names}
    loaded = {}
    try:
        for name in names:
            mod = types.ModuleType(name)
            mod.__path__ = []
            sys.modules[name] = mod
        sys.modules["trainers.trainerBase"].TrainerBase = type("UnusedDriverBase", (), {})
        sys.modules["data.data_loader"].DatasetCustom = type("UnusedDatasetType", (), {})
        sys.modules["utils.tools"].Struct = type("UnusedConfigType", (), {})
        with warnings.catch_warnings():
            for name, rel in file_modules:
                spec = importlib.util.spec_from_file_location(name, root / rel)
                mod = importlib.util.module_from_spec(spec)
                sys.modules[name] = mod
                spec.loader.exec_module(mod)
                loaded[name] = mod
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
    return (loaded["models.DLinear"].Model,
            loaded["models.MLP"].Model,
            loaded["trainers.w_student.trainerFramework_ERTDRes"].Exp_TS2VecSupervised,
            hashes)


class _TrackedOptimizer:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.calls = 0
        self.nonempty_gradient_steps = 0

    def step(self):
        self.calls += 1
        if any(p.grad is not None for g in self.optimizer.param_groups for p in g["params"]):
            self.nonempty_gradient_steps += 1
        return self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)


class _ForwardInterface:
    """Own dtype-preserving tensor bridge; source models execute their forward."""
    def __init__(self, owner, model, teacher):
        self.owner, self.model, self.teacher = owner, model, teacher

    def ffn(self, dataset, x, y, x_mark, y_mark, mode="train"):
        out = self.model(x.to(dtype=self.owner.dtype, device="cpu"))
        out["true"] = y.to(dtype=self.owner.dtype, device="cpu")
        return out

    def get_future(self, x):
        return x[:, -self.owner.args.pred_len:]

    def gt4update(self, out):
        return out["true"]

    def td4update_pseudo(self, curr, k):
        tail = curr["predT"][:, :-k]
        if self.owner.pseudo_hook is None:
            return tail
        t, o = self.owner.current_origin, self.owner.current_origin - k
        observed = self.owner.current_history[o + 1:t + 1].unsqueeze(0)
        issued = self.owner.forecast_records[o]["teacher"]
        info = dict(current_origin=t, old_origin=o, age=k,
                    teacher_tail=tail.detach().clone(),
                    current_teacher=curr["predT"].detach().clone(),
                    old_issued_teacher=issued.detach().clone(),
                    observed_prefix=observed.detach().clone(),
                    prefix_error=observed - issued[:, :k])
        replacement = self.owner.pseudo_hook(info)
        if replacement.shape != tail.shape or not torch.isfinite(replacement).all():
            raise ValueError("Pseudo hook must return finite [1,H-k,1] targets")
        return replacement.detach().to(tail)


class _AdapterMixin:
    def __init__(self, author_classes, *, context=21, horizon=20, seed=17,
                 dtype=torch.float64, replay_size=300, replay_batch=32,
                 gamma=0.9, td_k=(1,), er=True, td=True,
                 optimizer_config=None, pseudo_hook=None):
        DLinear, MLP, _, source_hashes = author_classes
        if context < 1 or horizon < 1 or any(k < 1 or k > horizon for k in td_k):
            raise ValueError("Invalid context, horizon or TD ages")
        if len(td_k) == 0 or replay_batch < 1 or replay_size < replay_batch:
            raise ValueError("Empty TD age set or invalid replay capacity")
        self.dtype, self.device = dtype, torch.device("cpu")
        torch.manual_seed(seed)
        self.numpy_rng_state = np.random.RandomState(seed).get_state()
        self.args = types.SimpleNamespace(seq_len=context, pred_len=horizon,
                                         er=er, td=td)
        cfg = types.SimpleNamespace(pred_len=horizon, individual=False, enc_in=1,
                                    mlp_width=16, mlp_depth=3, mlp_dropout=0.0,
                                    act_str="Identity")
        self.main_model = DLinear(cfg, context).to(dtype=dtype)
        self.student_model = MLP(cfg, context + horizon).to(dtype=dtype)
        self.mainFFN = _ForwardInterface(self, self.main_model, True)
        self.studentFFN = _ForwardInterface(self, self.student_model, False)
        self.criterion = nn.MSELoss()
        self.MSEReductNone = nn.MSELoss(reduction="none")
        self.laststepA, self.laststepB = [None] * 4, [None] * 4
        self.replayBuffer = [None] * 4
        self.replayBufferSize, self.batchReplayBufferSize = replay_size, replay_batch
        self.num_ERepochs, self.freq_ERupdate, self.count = 1, 1, 0
        self.indices = list(td_k)
        self.discountedW = torch.stack([
            torch.tensor([1.0] * (k - 1) + [gamma ** j for j in range(horizon - k + 1)],
                         dtype=dtype) for k in self.indices
        ]).unsqueeze(-1)
        # Defaults are the author's Exchange/DLinear_MLP optimizer YAML. The
        # caller may supply a fully specified matched application protocol.
        if optimizer_config is None:
            optimizer_config = {
                "teacher_train": dict(opt="adam", lr=0.0024324647254947433, amsgrad=True),
                "student_train": dict(opt="adamw", lr=0.0011932670228165677, amsgrad=True),
                "teacher_slow": dict(opt="adamw", lr=0.00010112274178326875, amsgrad=True),
                "student_slow": dict(opt="adamw", lr=0.000783882718834249, amsgrad=True),
                "teacher_fast": dict(opt="adam", lr=0.0001, amsgrad=True),
                "student_fast": dict(opt="adamw", lr=0.0008776466424276737, amsgrad=True),
            }
        self.optimizer_config = optimizer_config
        self.optimizers = {}
        for key, spec in optimizer_config.items():
            kw = dict(spec)
            kind = kw.pop("opt")
            params = self.main_model.parameters() if key.startswith("teacher") else self.student_model.parameters()
            cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}[kind]
            self.optimizers[key] = _TrackedOptimizer(cls(params, **kw))
        self.opt = {"train": [self.optimizers[k] for k in ("teacher_train", "student_train")],
                    "test": {"batch": [self.optimizers[k] for k in ("teacher_slow", "student_slow")],
                             "online": [self.optimizers[k] for k in ("teacher_fast", "student_fast")]}}
        self.source_hashes, self.pseudo_hook = source_hashes, pseudo_hook
        self.start_origin = self.current_origin = None
        self.forecast_records = {}
        self.pretrain_batches = self.prediction_calls = self.online_events = 0

    def clean(self, outputs):
        return {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                for k, v in outputs.items()}

    def store_grad(self):
        pass  # DLinear and MLP have no gradient-driven memory hook.

    def update_ocp(self, prev, curr):
        pass  # DLinear and MLP have no change-point hook.

    def train(self, mode=True):
        self.main_model.train(mode)
        self.student_model.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def parameters(self):
        return itertools.chain(self.main_model.parameters(), self.student_model.parameters())

    def state_dict(self):
        return {**{"teacher." + k: v for k, v in self.main_model.state_dict().items()},
                **{"student." + k: v for k, v in self.student_model.state_dict().items()}}

    def load_state_dict(self, state):
        self.main_model.load_state_dict({k[8:]: v for k, v in state.items() if k.startswith("teacher.")})
        self.student_model.load_state_dict({k[8:]: v for k, v in state.items() if k.startswith("student.")})

    def _tensor(self, x):
        x = torch.as_tensor(x, dtype=self.dtype, device="cpu")
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        if x.ndim != 3 or x.shape[-1] != 1:
            raise ValueError("Expected [batch,length,1] price tensors")
        return x

    def pretrain_batch(self, x, y):
        x, y = self._tensor(x), self._tensor(y)
        if x.shape[1] != self.args.seq_len or y.shape[1] != self.args.pred_len:
            raise ValueError("Context/horizon mismatch")
        self.train()
        for opt in self.opt["train"]:
            opt.zero_grad()
        # This is the original author's train_step, not a copied implementation.
        result = self.train_step(None, x, y, torch.zeros_like(x), torch.zeros_like(y))
        self.pretrain_batches += 1
        return result

    def predict(self, x):
        x = self._tensor(x)
        if x.shape[1] != self.args.seq_len:
            raise ValueError("Context mismatch")
        self.eval()
        y = torch.zeros((x.shape[0], self.args.pred_len, 1), dtype=self.dtype)
        with torch.no_grad():
            result = self._process_one_batch(None, x, y, torch.zeros_like(x),
                                            torch.zeros_like(y), mode="test")["pred"].detach().clone()
        self.prediction_calls += 1
        return result

    def online_event(self, observed_history, *, origin=None):
        """Input exactly the observed prefix p[0:t+1]; return new t-origin forecast.

        Reconstruct the source methods' buffer views just before they are read.
        No unknown future label is accepted. Stored issued forecasts are never
        recomputed/replaced. Calls must advance by exactly one observation.
        """
        history = torch.as_tensor(observed_history, dtype=self.dtype, device="cpu")
        if history.ndim == 1:
            history = history[:, None]
        if history.ndim != 2 or history.shape[1] != 1 or not torch.isfinite(history).all():
            raise ValueError("Observed history must be finite [t+1,1]")
        t, L, H = len(history) - 1, self.args.seq_len, self.args.pred_len
        if origin is not None and origin != t:
            raise ValueError("History length must equal origin+1; future suffix forbidden")
        if t < L - 1:
            raise ValueError("Insufficient observed context")
        if self.current_origin is not None and t != self.current_origin + 1:
            raise ValueError("Online events must be consecutive observations")
        if self.start_origin is None:
            self.start_origin = t
        self.current_origin, self.current_history = t, history
        x = history[t - L + 1:t + 1].unsqueeze(0)
        y = torch.zeros((1, H, 1), dtype=self.dtype)
        self.laststepA = [None] * 4
        if t - H >= self.start_origin:
            past_x = history[t - H - L + 1:t].unsqueeze(0)
            mature_y = history[t - H + 1:t + 1].unsqueeze(0)
            self.laststepA = [past_x, mature_y, torch.zeros_like(past_x), torch.zeros_like(mature_y)]
        self.laststepB = [None] * 4
        K = max(self.indices)
        if t - K >= self.start_origin:
            past_x = history[t - K - L + 1:t].unsqueeze(0)
            mature_prefix = history[t - K + 1:t + 1].unsqueeze(0)
            self.laststepB = [past_x, mature_prefix, torch.zeros_like(past_x), torch.zeros_like(mature_prefix)]
        self.eval()
        old_rng = np.random.get_state()
        np.random.set_state(self.numpy_rng_state)
        try:
            # Original test_step executes original ER, TD, then the new forecast.
            out = self.test_step(None, x, y, torch.zeros_like(x), torch.zeros_like(y))
            self.numpy_rng_state = np.random.get_state()
        finally:
            np.random.set_state(old_rng)
        self.forecast_records[t] = {"teacher": out["predT"].detach().clone(),
                                    "combined": out["pred"].detach().clone()}
        self.online_events += 1
        return out["pred"].detach().clone()

    def execution_receipt(self):
        methods = set()
        if self.pretrain_batches or self.prediction_calls or self.online_events:
            methods.update(["_process_one_batch", "get_final_output", "DLinear.forward", "MLP.forward"])
        if self.pretrain_batches:
            methods.update(["train_step", "backward_lossT_train", "backward_lossFT"])
        if self.online_events:
            methods.add("test_step")
            if self.args.er:
                methods.add("er")
            if self.args.td:
                methods.add("td")
        if self.optimizers["teacher_slow"].nonempty_gradient_steps:
            methods.update(["backward_lossT_test", "backward_lossFT"])
        if self.optimizers["student_fast"].nonempty_gradient_steps:
            methods.add("backward_tdlossFT")
        return {"author_commit": AUTHOR_COMMIT, "author_file_sha256": self.source_hashes,
                "source_classes": ["models.DLinear.Model", "models.MLP.Model",
                                   "trainers.w_student.trainerFramework_ERTDRes.Exp_TS2VecSupervised"],
                "original_methods_executed": sorted(methods),
                "method_count_basis": "entry calls and actual optimizer gradient steps; source control flow inspected",
                "execution_scope": "unchanged author core through own causal interface; not complete benchmark driver",
                "optimizer_calls": {k: {"calls": v.calls, "nonempty_gradient_steps": v.nonempty_gradient_steps}
                                    for k, v in self.optimizers.items()},
                "parameters": sum(p.numel() for p in self.parameters()),
                "dtype": str(self.dtype), "start_origin": self.start_origin,
                "last_origin": self.current_origin, "issued_forecasts": len(self.forecast_records),
                "pretrain_batches": self.pretrain_batches, "prediction_calls": self.prediction_calls,
                "online_events": self.online_events,
                "pseudo_hook_installed": self.pseudo_hook is not None}


def make_author_dsof(author_root, **kwargs):
    """Return an adapter that inherits the imported, unchanged author trainer."""
    classes = _load_author_classes(author_root)
    cls = type("AuthorDSOFExecutionAdapter", (_AdapterMixin, classes[2]), {})
    return cls(classes, **kwargs)
