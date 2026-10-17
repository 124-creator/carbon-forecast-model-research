"""Run the unchanged SOLID author retrieval/head-update routine causally.

Integration code for the event-feedback study, not an author benchmark
reproduction. The MIT-licensed Exp_Main.select_with_distance function is loaded
from a fixed, hash-verified author source without changing its AST. Only its data,
checkpoint, forward, diagnostic and result-I/O interfaces are supplied here.

Each original GRU input window is flattened into one packet row. Author seq_len=1
then retrieves whole windows by their exact flattened Euclidean distance. Origin
indices live only in x_mark; retrieval and the model do not see those indices.

The non-AMP author update is a SUM of neighbor gradients at the initial head,
followed by ONE head update per query, not mean-gradient SGD. Every query
restarts from the same offline checkpoint. Query future labels are never passed
to the routine; author future-truth diagnostics and scoring are disabled.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn, optim


AUTHOR_COMMIT = "3c74e5f8b3c921a2e7cda30f36117c295bf8c112"
AUTHOR_REPOSITORY = "https://github.com/HALF111/calibration_CDS"
EXPECTED_BLOBS = {
    "exp/exp_main.py": "e021dfbe79885e379018168b475915f17cba2883",
    "LICENSE": "974cdf19be0c4c70cb33715c0c9dfa5ed80c2609",
}


def default_author_root():
    return Path(__file__).resolve().parents[1] / "backup_review" / "author_solid"


def array_hash(values):
    h = hashlib.sha256()
    for name, value in sorted(values.items()):
        a = np.ascontiguousarray(value)
        h.update(name.encode())
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def model_arrays(model):
    return {k: v.detach().cpu().numpy().copy()
            for k, v in model.state_dict().items()}


def verify_author(author_root=None):
    root = default_author_root() if author_root is None else Path(author_root)
    files = {}
    for relative, expected in EXPECTED_BLOBS.items():
        raw = (root / relative).read_bytes()
        blob = hashlib.sha1(
            b"blob " + str(len(raw)).encode() + b"\0" + raw
        ).hexdigest()
        if blob != expected:
            raise ValueError(f"Author source mismatch: {relative}: {blob}")
        files[relative] = dict(git_blob_sha=blob,
                              sha256=hashlib.sha256(raw).hexdigest(),
                              bytes=len(raw))
    source = (root / "exp/exp_main.py").read_text()
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "Exp_Main")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
              and n.name == "select_with_distance")
    return dict(repository=AUTHOR_REPOSITORY, commit=AUTHOR_COMMIT,
                license="MIT", license_copyright="2021 DAMO Academy @ Alibaba",
                files=files,
                original_function="Exp_Main.select_with_distance",
                original_function_first_line=fn.lineno,
                original_function_last_line=fn.end_lineno,
                original_function_ast_sha256=hashlib.sha256(
                    ast.dump(fn, include_attributes=False).encode()).hexdigest())


class _TorchCheckpointBridge:
    def __init__(self, adapter):
        self.adapter = adapter

    def __getattr__(self, name):
        return getattr(torch, name)

    def load(self, path, *args, **kwargs):
        # The author routine unconditionally loads a CUDA checkpoint. Supply
        # the offline checkpoint in memory, with no checkpoint file I/O.
        expected = os.path.join("./checkpoints/SOLID_CAUSAL_BRIDGE",
                                "checkpoint.pth")
        if str(path) != expected:
            raise ValueError("Unexpected checkpoint request")
        self.adapter.checkpoint_requests += 1
        return copy.deepcopy(self.adapter.base.state_dict())


class _PathBridge:
    join = staticmethod(os.path.join)

    @staticmethod
    def exists(path):
        if str(path).startswith("./results/"):
            return True
        raise ValueError("Author routine attempted unexpected path access")


class _OSResultBridge:
    path = _PathBridge()

    @staticmethod
    def makedirs(path, *args, **kwargs):
        raise ValueError("Author result writing is delegated to the caller")


def _load_original_routine(adapter, author_root):
    tree = ast.parse((author_root / "exp/exp_main.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "Exp_Main")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
              and n.name == "select_with_distance")
    original_dump = ast.dump(fn, include_attributes=False)
    # The original node is unchanged; extracting it avoids unused benchmark
    # imports. All executed algorithm statements retain their original AST.
    module = ast.Module(body=[copy.deepcopy(fn)], type_ignores=[])
    assert ast.dump(module.body[0], include_attributes=False) == original_dump

    def no_query_score(*args, **kwargs):
        adapter.suppressed_metric_calls += 1
        return (float("nan"),) * 5

    namespace = dict(torch=_TorchCheckpointBridge(adapter), nn=nn,
                     optim=optim, np=np, os=_OSResultBridge(), time=time,
                     warnings=__import__("warnings"), copy=copy, math=math,
                     metric=no_query_score)
    exec(compile(module, str(author_root / "exp/exp_main.py"), "exec"),
         namespace)
    return namespace["select_with_distance"]


class SolidAuthorAdapter:
    """Fixed offline GRU/LSTM backbone and actual author local-head routine.

    data implements the study CausalData interface:
        data.at(origin) -> (x[L,C], scale)
        data.batch([origin], available_through=t, labels=True/False)
    Queries must be at/after data.start. The caller supplies the selected OFFLINE
    teacher, never the final online teacher; state hashes identify this input.
    artifact_sink(record) optionally receives per-query arrays for private
    storage. This constructor loads no market data and performs no model fit.
    """
    def __init__(self, data, pretrained_teacher, *, author_root=None,
                 context_samples=128, neighbors=10, learning_rate=.001,
                 remove_distance=False, cache_latents=True, deadline=None,
                 artifact_sink=None, earliest_feature_origin=60):
        if context_samples < 1 or not 1 <= neighbors <= context_samples:
            raise ValueError("Require 1 <= neighbors <= context_samples")
        if learning_rate < 0 or not math.isfinite(learning_rate):
            raise ValueError("Finite nonnegative adaptation learning rate required")
        if not isinstance(pretrained_teacher.encoder, (nn.GRU, nn.LSTM)):
            raise TypeError("Only the study GRU/LSTM encoder is supported")
        if not isinstance(pretrained_teacher.head, nn.Linear):
            raise TypeError("A single nn.Linear head is required")
        if pretrained_teacher.head.bias is None:
            raise TypeError("The original routine requires a weight and bias")
        if any(p.dtype != torch.float64 for p in pretrained_teacher.parameters()):
            raise TypeError("This adapter requires the frozen float64 protocol")
        if any(p.device.type != "cpu" for p in pretrained_teacher.parameters()):
            raise TypeError("CPU-only adapter")
        self.author_root = (default_author_root() if author_root is None
                            else Path(author_root).resolve())
        self.author_identity = verify_author(self.author_root)
        self.data = data
        self.base = copy.deepcopy(pretrained_teacher).eval()
        self.model = self.base
        self.device = torch.device("cpu")
        self.initial_arrays = model_arrays(self.base)
        self.initial_hash = array_hash(self.initial_arrays)
        self.horizon = int(self.base.head.out_features)
        self.sqrt_h = np.sqrt(np.arange(1, self.horizon + 1, dtype=float))
        self.context_samples = int(context_samples)
        self.neighbors = int(neighbors)
        self.learning_rate = float(learning_rate)
        self.remove_distance = bool(remove_distance)
        self.cache_latents = bool(cache_latents)
        self.deadline = deadline
        self.artifact_sink = artifact_sink
        self.earliest_feature_origin = int(earliest_feature_origin)
        self.features = {}
        self.latents = {}
        self.records = []
        self.normalized_predictions = []
        self.raw_predictions = []
        self.checkpoint_requests = 0
        self.suppressed_metric_calls = 0
        self.encoder_calls = 0
        self.gradient_calls = 0
        self.forward_equivalence_checked = False
        self._routine = _load_original_routine(self, self.author_root)
        self._has_run = False
        self.selected_channels = []
        self.period = 1
        self.linear_map = {"StudyGRU": ["head"]}
        self.args = SimpleNamespace(
            show_loss_details=False, seq_len=1, label_len=0,
            pred_len=self.horizon, adapt_start_pos=1, use_amp=False,
            draw_adapt_figure=False, batch_size=1, model="StudyGRU",
            learning_rate=self.learning_rate, adapted_lr_times=1.0,
            adapt_part_channels=False, remove_distance=self.remove_distance,
            remove_cycle=False, remove_nearest=False,
            use_nearest_data=False, use_further_data=False,
            c_out=1, data_path="study_packets", lambda_period=.1,
            test_train_num=self.context_samples,
            selected_data_num=self.neighbors)
        self.test_train_num = self.context_samples

    def _check_deadline(self):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError("SOLID frozen deadline reached; partial stream ineligible")

    def _get_feature(self, origin):
        if origin > self.current_query:
            raise ValueError("Attempted to access future input")
        if origin not in self.features:
            x, scale = self.data.at(origin)
            x = np.asarray(x, dtype=np.float64)
            if x.ndim != 2 or not np.isfinite(x).all():
                raise ValueError("Invalid original model input")
            self.features[origin] = (x.copy(), float(scale))
        return self.features[origin]

    def _latent(self, origin, x):
        if origin not in self.latents:
            with torch.no_grad():
                sequence, _ = self.base.encoder(x)
                value = sequence[:, -1].detach().clone()
                if not self.forward_equivalence_checked:
                    direct = self.base(x)
                    cached = self.base.head(value)
                    if not torch.allclose(direct, cached, rtol=0., atol=1e-12):
                        raise ValueError("Teacher.forward is not head(last encoder state)")
                    self.forward_equivalence_checked = True
            self.latents[origin] = value
            self.encoder_calls += 1
        return self.latents[origin]

    def _get_data_at_test_time(self, flag):
        if flag != "test":
            raise ValueError("Only causal query streams are supported")
        return range(len(self.query_origins)), self._packet_stream()

    def _packet_stream(self):
        for query in self.query_origins:
            self._check_deadline()
            self.current_query = int(query)
            available = (self.current_query - self.horizon
                         - self.earliest_feature_origin + 1)
            count = min(self.context_samples, available)
            if count < self.neighbors:
                raise ValueError("Insufficient fully mature candidate origins")
            self.test_train_num = self.args.test_train_num = count
            begin = self.current_query - self.horizon - count + 1
            ids = np.arange(begin, self.current_query + 1)
            xs = [self._get_feature(int(o))[0] for o in ids]
            self.current_shape = xs[-1].shape
            if any(x.shape != self.current_shape for x in xs):
                raise ValueError("Original model windows must share shape")
            flat = np.stack([x.reshape(-1) for x in xs])
            # Metadata stays outside both distance and backbone input.
            marks = ids.astype(float).reshape(1, -1, 1)
            self.current_record = dict(
                origin=self.current_query, selected_origins=[],
                individual_weight_gradients=[], individual_bias_gradients=[],
                candidate_first=int(begin),
                candidate_last=int(self.current_query-self.horizon),
                candidate_count=int(count), maximum_label_index=-1)
            self.current_cur_model = None
            self.current_hook_handles = []
            yield (torch.tensor(flat[None]),
                   torch.zeros((1, self.horizon, flat.shape[1]), dtype=torch.float64),
                   torch.tensor(marks),
                   torch.zeros((1, self.horizon, 1), dtype=torch.float64))

    def _attach_gradient_recording(self, cur_model):
        if self.current_cur_model is cur_model:
            return
        if self.current_cur_model is not None:
            raise ValueError("Unexpected second model within one query")
        self.current_cur_model = cur_model
        record = self.current_record

        def weight_hook(gradient):
            record["individual_weight_gradients"].append(
                gradient.detach().cpu().numpy().copy())
            self.gradient_calls += 1

        def bias_hook(gradient):
            record["individual_bias_gradients"].append(
                gradient.detach().cpu().numpy().copy())

        self.current_hook_handles = [
            cur_model.head.weight.register_hook(weight_hook),
            cur_model.head.bias.register_hook(bias_hook)]

    def _process_one_batch_with_model(self, cur_model, dataset_object,
                                     batch_x, batch_y, batch_x_mark,
                                     batch_y_mark, return_mid_embedding=False):
        self._check_deadline()
        if return_mid_embedding:
            raise ValueError("Unrequested author branch")
        origin = int(batch_x_mark[0, -1, 0].item())
        if origin > self.current_query:
            raise ValueError("Attempted future-origin forward")
        original_x, scale = self._get_feature(origin)
        packet = batch_x.detach().cpu().numpy().reshape(-1)
        if not np.array_equal(packet, original_x.reshape(-1)):
            raise ValueError("Packet differs from original model input")
        x = torch.tensor(original_x[None], dtype=torch.float64)
        is_query = origin == self.current_query
        if not is_query:
            if origin + self.horizon > self.current_query:
                raise ValueError("Attempted immature adaptation label")
            self._attach_gradient_recording(cur_model)
            labeled = self.data.batch([origin], self.current_query, labels=True)
            target = torch.tensor(labeled["y"], dtype=torch.float64).unsqueeze(-1)
            self.current_record["selected_origins"].append(origin)
            self.current_record["maximum_label_index"] = max(
                self.current_record["maximum_label_index"], origin+self.horizon)
        else:
            target = torch.zeros((1, self.horizon, 1), dtype=torch.float64)
        if self.cache_latents:
            pred = cur_model.head(self._latent(origin, x)).unsqueeze(-1)
        else:
            pred = cur_model(x).unsqueeze(-1)
            self.encoder_calls += 1
        if not torch.isfinite(pred).all():
            raise FloatingPointError("Nonfinite SOLID prediction")
        if is_query:
            self._finish_query(cur_model, pred.detach().numpy()[0, :, 0], scale)
        return pred, target

    def _finish_query(self, cur_model, normalized, scale):
        record = self.current_record
        if len(record["selected_origins"]) != self.neighbors:
            raise ValueError("Author did not use the fixed neighbor count")
        wg = np.stack(record.pop("individual_weight_gradients"))
        bg = np.stack(record.pop("individual_bias_gradients"))
        if len(wg) != self.neighbors or len(bg) != self.neighbors:
            raise ValueError("Unexpected actual backward count")
        # Preserve the original NumPy left-to-right summation.
        summed_w, summed_b = np.zeros_like(wg[0]), np.zeros_like(bg[0])
        for a, b in zip(wg, bg):
            summed_w = summed_w + a
            summed_b = summed_b + b
        final_w = cur_model.head.weight.detach().numpy().copy()
        final_b = cur_model.head.bias.detach().numpy().copy()
        expected_w = self.initial_arrays["head.weight"] - self.learning_rate*summed_w
        expected_b = self.initial_arrays["head.bias"] - self.learning_rate*summed_b
        update_error = max(float(np.max(np.abs(final_w-expected_w))),
                           float(np.max(np.abs(final_b-expected_b))))
        if update_error > 1e-12:
            raise AssertionError("Observed head update differs from author gradient sum")
        for name, value in model_arrays(cur_model).items():
            if not name.startswith("head.") and not np.array_equal(
                    value, self.initial_arrays[name]):
                raise AssertionError("Author adaptation changed frozen encoder state")
        current = self.data.batch([self.current_query], self.current_query, labels=False)
        raw = float(current["base"][0]) + float(scale)*self.sqrt_h*normalized
        record.update(
            actual_selected_count=len(record["selected_origins"]),
            head_updates=1, neighbor_backward_calls=len(wg),
            sum_weight_gradient_norm=float(np.linalg.norm(summed_w)),
            sum_bias_gradient_norm=float(np.linalg.norm(summed_b)),
            summed_gradient_sha256=array_hash({"weight":summed_w,"bias":summed_b}),
            adapted_head_sha256=array_hash({"weight":final_w,"bias":final_b}),
            update_identity_max_abs_error=update_error,
            encoder_unchanged=True, query_future_label_supplied=False)
        artifact = dict(origin=self.current_query,
                        selected_origins=np.array(record["selected_origins"], dtype=int),
                        normalized_prediction=normalized.copy(), raw_prediction=raw.copy(),
                        summed_weight_gradient=summed_w, summed_bias_gradient=summed_b,
                        adapted_head_weight=final_w, adapted_head_bias=final_b)
        if self.artifact_sink is not None:
            self.artifact_sink(artifact)
        self.records.append(record.copy())
        self.normalized_predictions.append(normalized.copy())
        self.raw_predictions.append(raw.copy())
        for handle in self.current_hook_handles:
            handle.remove()
        self.current_hook_handles = []

    def run(self, query_origins):
        if self._has_run:
            raise RuntimeError("One adapter instance permits one recorded stream")
        self._has_run = True
        ids = np.asarray(query_origins, dtype=int)
        if ids.ndim != 1 or len(ids) == 0 or np.any(np.diff(ids) <= 0):
            raise ValueError("Require nonempty, strictly increasing query origins")
        if ids[0] < int(self.data.start):
            raise ValueError("Queries must follow offline training/validation cutoff")
        self.query_origins = ids.copy()
        started = time.perf_counter()
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            self._routine(self, "SOLID_CAUSAL_BRIDGE", test=0,
                          is_training_part_params=True, use_adapted_model=True,
                          test_train_epochs=1, weights_given=None)
        if len(self.records) != len(ids):
            raise AssertionError("Incomplete query stream")
        if array_hash(model_arrays(self.base)) != self.initial_hash:
            raise AssertionError("Offline base checkpoint changed")
        pred = np.stack(self.raw_predictions)
        receipt = dict(
            status="COMPLETE", eligible=True, author=self.author_identity,
            queries=len(ids), query_origins=ids.tolist(),
            context_samples=self.context_samples, neighbors=self.neighbors,
            learning_rate=self.learning_rate, period=1, remove_cycle=False,
            remove_distance=self.remove_distance,
            original_test_train_epochs=1, use_amp=False,
            head_updates=len(self.records), neighbor_backward_calls=self.gradient_calls,
            cache_latents=self.cache_latents, encoder_calls=self.encoder_calls,
            cached_encoder_origins=len(self.latents),
            forward_cache_equivalence_checked=self.forward_equivalence_checked,
            checkpoint_requests=self.checkpoint_requests,
            suppressed_author_metric_calls=self.suppressed_metric_calls,
            author_query_scoring_disabled=True, query_future_labels_supplied=False,
            initial_offline_state_sha256=self.initial_hash,
            final_offline_state_sha256=array_hash(model_arrays(self.base)),
            raw_prediction_sha256=array_hash({"predictions":pred}),
            seconds=time.perf_counter()-started,
            input_layout="One full model input flattened per packet; origin only in marks",
            retrieval=("Author recency ordering (remove_distance=true)" if self.remove_distance
                       else "Author Euclidean top-k over flattened original inputs"),
            update="Unchanged author sum of neighbor gradients; one head step, reset per query",
            execution_scope="Actual fixed author select_with_distance with own causal data/checkpoint/forward/diagnostic-I/O bridge; not full benchmark reproduction",
            diagnostics=self.records)
        return pred, receipt


def run_solid_author(data, pretrained_teacher, query_origins, **kwargs):
    return SolidAuthorAdapter(data, pretrained_teacher, **kwargs).run(query_origins)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-author", action="store_true", required=True)
    parser.add_argument("--author-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_author(args.author_root), indent=2))


if __name__ == "__main__":
    main()
