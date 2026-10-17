"""Revision-residual calibration for a DSOF-style GRU forecasting experiment.

Original study implementation. This is an application adaptation of the DSOF
learning paths, not a redistribution of its authors' implementation. All model
inputs use observed prices; the streaming learner receives one event at a time.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import copy
import csv
import hashlib
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import torch
from torch import nn

H = 20
L = 20
K = (1, 2, 4, 8, 16)
SQRT_H = np.sqrt(np.arange(1, H + 1, dtype=float))
SCORE_H = np.array([0, 4, 19])


def configure():
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False


def digest_arrays(arrays):
    h = hashlib.sha256()
    for key, arr in sorted(arrays.items()):
        a = np.asarray(arr)
        h.update(key.encode()); h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode()); h.update(a.tobytes())
    return h.hexdigest()


def arrays(net):
    return {k: v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()}


@dataclass
class Market:
    name: str
    dates: np.ndarray
    days: np.ndarray
    price: np.ndarray
    covariates: np.ndarray
    covariate_names: tuple
    source_sha256: str


def load_market(path, name, with_auction=False, end_exclusive="2024-01-01"):
    raw = Path(path).read_bytes()
    rows = list(csv.DictReader(raw.decode("utf-8-sig").splitlines()))
    all_dates = np.array([r["date"][:10] for r in rows])
    if len(rows) == 0 or np.any(all_dates[1:] <= all_dates[:-1]):
        raise ValueError("Unique, strictly increasing observed dates required")
    rows = [r for r in rows if r["date"][:10] < end_exclusive]
    dates = np.array([r["date"][:10] for r in rows])
    p = np.array([float(r["price"]) for r in rows])
    names = ("auction_volume", "cover_ratio", "bidder_count") if with_auction else ()
    cov = np.array([[float(r[k]) for k in names] for r in rows]).reshape(len(rows), len(names))
    if not np.isfinite(p).all() or np.any(p <= 0):
        raise ValueError("Invalid raw prices; no imputation")
    if names and (not np.isfinite(cov).all() or np.any(cov <= 0)):
        raise ValueError("Missing/nonpositive auction covariate; stop instead of filling")
    days = np.array([date.fromisoformat(d).toordinal() for d in dates])
    return Market(name, dates, days, p, cov, names, hashlib.sha256(raw).hexdigest())


class CausalData:
    def __init__(self, market, start_date, validation_size=80):
        self.market = market
        self.start = int(np.searchsorted(market.dates, start_date))
        mature = np.arange(60, self.start - H, dtype=int)
        if len(mature) < validation_size + H + 80:
            raise ValueError("Insufficient pre-online training/validation history")
        self.validation = mature[-validation_size:]
        self.train = mature[mature + H < self.validation[0]]
        self.floor_end = int(self.validation[0])
        self.floor = max(float(np.sqrt(np.mean(np.diff(market.price[:self.floor_end]) ** 2))) * .01, 1e-8)
        if len(market.covariate_names):
            log_cov = np.log(market.covariates[:self.floor_end])
            self.cov_mean = log_cov.mean(0)
            self.cov_std = np.maximum(log_cov.std(0), 1e-6)
        else:
            self.cov_mean = self.cov_std = np.empty(0)
        self._features = {}

    def at(self, t):
        """Pure prefix feature map: no fitted suffix statistic or future input."""
        if t < 60 or t >= len(self.market.price):
            raise ValueError("Invalid feature origin")
        if t not in self._features:
            p, day = self.market.price, self.market.days
            s = max(float(np.sqrt(np.mean(np.diff(p[t-60:t+1]) ** 2))), self.floor)
            dp = np.diff(p[t-L:t+1]) / s
            gap = np.log1p(np.diff(day[t-L:t+1])) / np.log(8.)
            x = np.stack([dp, gap], axis=1)
            if len(self.market.covariate_names):
                c = (np.log(self.market.covariates[t-L+1:t+1]) - self.cov_mean) / self.cov_std
                x = np.concatenate([x, c], axis=1)
            # Fixed feature clipping, not quote cleaning; recorded for every model.
            self._features[t] = (np.clip(x, -10., 10.), s)
        x, s = self._features[t]
        return x.copy(), s

    def batch(self, origins, available_through, labels=True):
        ids = np.asarray(origins, dtype=int)
        if ids.size == 0 or ids.max() > available_through:
            raise ValueError("Requested unknown input")
        xs, ss = zip(*(self.at(int(t)) for t in ids))
        out = dict(origin=ids.copy(), x=np.stack(xs), scale=np.array(ss), base=self.market.price[ids].copy())
        if labels:
            if ids.max() + H > available_through:
                raise ValueError("Requested immature labels")
            future = self.market.price[ids[:, None] + np.arange(1, H+1)]
            out["y"] = (future - out["base"][:, None]) / out["scale"][:, None] / SQRT_H
        return out

    def support_receipt(self):
        return dict(market=self.market.name, source_sha256=self.market.source_sha256,
                    online_start_index=self.start, online_start_date=str(self.market.dates[self.start]),
                    train_origins=self.train.tolist(), validation_origins=self.validation.tolist(),
                    max_train_label=int(self.train[-1]+H), first_validation_origin=int(self.validation[0]),
                    max_validation_label=int(self.validation[-1]+H), scale_floor=self.floor,
                    scale_floor_and_covariate_stats_end_exclusive=self.floor_end,
                    covariates=list(self.market.covariate_names),
                    cov_mean=self.cov_mean.tolist(), cov_std=self.cov_std.tolist(),
                    scale="Each origin's observed past 60 increments RMS, with training-only 1% RMS floor",
                    input="20 normalized observed increments, log(1+observed calendar gap)/log(8), optional historical auction covariates",
                    feature_clip=[-10,10], quote_imputation=False)


class Teacher(nn.Module):
    def __init__(self, channels, hidden=32, cell="gru"):
        super().__init__()
        cls = {"gru": nn.GRU, "lstm": nn.LSTM}[cell]
        self.encoder = cls(channels, hidden, batch_first=True)
        self.head = nn.Linear(hidden, H)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, x):
        z, _ = self.encoder(x)
        return self.head(z[:, -1])


class Student(nn.Module):
    """The DSOF default is a factorized affine MLP, Identity activation."""
    def __init__(self, channels, width=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(L*channels + H, width), nn.Identity(),
                                 nn.Linear(width, width), nn.Identity(), nn.Linear(width, H))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, teacher):
        return self.net(torch.cat([x.flatten(1), teacher.detach()], dim=1))


def score_norm(pred, batch):
    error = (pred - batch["y"]) * batch["scale"][:, None] * SQRT_H
    rw = batch["y"] * batch["scale"][:, None] * SQRT_H
    return float(np.mean(np.mean(error[:, SCORE_H]**2, 0) / np.mean(rw[:, SCORE_H]**2, 0)))


def _update(loss, optimizer, parameters):
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite objective; no automatic retry")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
    optimizer.step()
    return grad


def pretrain(data, seed, shrink, max_epochs=40, patience=8, deadline=None, cell="gru"):
    configure(); torch.manual_seed(seed)
    started = time.perf_counter()
    train = data.batch(data.train, data.start-1)
    val = data.batch(data.validation, data.start-1)
    x, y = torch.tensor(train["x"]), torch.tensor(train["y"])
    vx = torch.tensor(val["x"])
    teacher = Teacher(x.shape[-1], cell=cell)
    student = Student(x.shape[-1])
    optim_t = torch.optim.AdamW(teacher.parameters(), lr=.003, weight_decay=.001)
    optim_s = torch.optim.AdamW(student.parameters(), lr=.003, weight_decay=.001)
    saved_t, saved_s = copy.deepcopy(teacher.state_dict()), copy.deepcopy(student.state_dict())
    best, selected, stale, trace = 1., 0, 0, []
    initial_hash = digest_arrays({**{"T."+k:v for k,v in arrays(teacher).items()},
                                  **{"S."+k:v for k,v in arrays(student).items()}})
    for epoch in range(1, max_epochs+1):
        if deadline and time.monotonic() >= deadline:
            raise TimeoutError("Frozen deadline")
        pt = teacher(x)
        ps = student(x, pt)
        ft = pt.detach() + ps
        loss_t = (pt-y).square().mean() + shrink*pt.square().mean()
        loss_s = (ft-y).square().mean() + shrink*ft.square().mean()
        gt = _update(loss_t, optim_t, teacher.parameters())
        gs = _update(loss_s, optim_s, student.parameters())
        with torch.no_grad():
            vt = teacher(vx)
            vf = vt + student(vx, vt)
            vscore = score_norm(vf.numpy(), val)
        trace.append(dict(epoch=epoch, teacher_objective=float(loss_t.detach()),
                          student_objective=float(loss_s.detach()), teacher_gradient=gt,
                          student_gradient=gs, validation_score=vscore))
        if vscore < best - 1e-6:
            best, selected, stale = vscore, epoch, 0
            saved_t, saved_s = copy.deepcopy(teacher.state_dict()), copy.deepcopy(student.state_dict())
        else:
            stale += 1
        if stale >= patience:
            break
    teacher.load_state_dict(saved_t); student.load_state_dict(saved_s)
    state = {**{"T."+k:v for k,v in arrays(teacher).items()}, **{"S."+k:v for k,v in arrays(student).items()}}
    receipt = dict(seed=seed, shrink=shrink, cell=cell, initial_state_sha256=initial_hash,
                   final_state_sha256=digest_arrays(state), selected_epoch=selected,
                   epochs=len(trace), validation_score=best, trace=trace,
                   teacher_parameters=sum(p.numel() for p in teacher.parameters()),
                   student_parameters=sum(p.numel() for p in student.parameters()),
                   seconds=time.perf_counter()-started, support=data.support_receipt())
    return teacher, student, receipt


def teacher_stream(data, initial_teacher, *, seed, shrink, stop_index=None,
                   teacher_lr=.0001, replay_size=128, replay_batch=16, deadline=None):
    """Cache the independent teacher path, shared exactly by student interventions.

    In DSOF the teacher has its own supervised loss; its outputs are detached in
    the student's loss. With a common replay schedule its path is independent of
    the student. ER outputs are recorded BEFORE the teacher's optimizer step.
    """
    started = time.perf_counter()
    teacher = copy.deepcopy(initial_teacher)
    opt = torch.optim.AdamW(teacher.parameters(), lr=teacher_lr, weight_decay=.001)
    rng = np.random.default_rng(seed + 20261017)
    end = len(data.market.price) if stop_index is None else int(stop_index)
    events, replay, updates = [], [], 0
    for t in range(data.start, end):
        if deadline and time.monotonic() >= deadline:
            raise TimeoutError("Frozen deadline")
        mature = t-H
        if mature >= data.start:
            replay.append(mature); replay = replay[-replay_size:]
        event = dict(origin=t, er_origins=np.empty(0, dtype=int), er_teacher=np.empty((0,H)))
        if len(replay) >= replay_batch:
            ids = np.array(replay)[rng.choice(len(replay), replay_batch, replace=False)]
            b = data.batch(ids, t)
            pred = teacher(torch.tensor(b["x"]))
            event["er_origins"], event["er_teacher"] = ids.copy(), pred.detach().numpy().copy()
            loss = (pred-torch.tensor(b["y"])).square().mean()+shrink*pred.square().mean()
            grad = _update(loss, opt, teacher.parameters())
            event["teacher_objective"], event["teacher_gradient"] = float(loss.detach()), grad
            updates += 1
        x, scale = data.at(t)
        with torch.no_grad():
            current = teacher(torch.tensor(x[None])).numpy()[0].copy()
            static = initial_teacher(torch.tensor(x[None])).numpy()[0].copy()
            if t - max(K) >= data.start:
                old_ids = np.array([t-k for k in K])
                b_old = data.batch(old_ids, t, labels=False)
                old = teacher(torch.tensor(b_old["x"])).numpy().copy()
            else:
                old_ids, old = np.empty(0,dtype=int), np.empty((0,H))
        event.update(x=x, scale=scale, base=float(data.market.price[t]), current=current,
                     static=static, fast_origins=old_ids, fast_teacher=old)
        events.append(event)
    receipt = dict(events=len(events), slow_teacher_updates=updates, replay_size=replay_size,
                   replay_batch=replay_batch, seed=seed, lr=teacher_lr, shrink=shrink,
                   seconds=time.perf_counter()-started, initial_state_sha256=digest_arrays(arrays(initial_teacher)),
                   final_state_sha256=digest_arrays(arrays(teacher)),
                   teacher_path_independent_of_student=True, teacher_fast_updates=0)
    return events, teacher, receipt


def calibration_features(data, event, old_events, mode="full"):
    t, scale = event["origin"], event["scale"]
    dp10 = np.diff(data.market.price[t-10:t+1])
    rms10 = max(float(np.sqrt(np.mean(dp10**2))), data.floor)
    v = np.array([*event["current"][SCORE_H], dp10.mean()/scale,
                  np.log(rms10/scale), np.log1p(data.market.days[t]-data.market.days[t-1])/np.log(8.)])
    v = np.tanh(v)
    result, old_tails = [], []
    for a in K:
        old = old_events[t-a]
        old_price = old["base"]+old["scale"]*SQRT_H*old["current"]
        e = data.market.price[t-a+1:t+1]-old_price[:a]
        normalized = e/scale/np.sqrt(np.arange(1,a+1))
        u = np.array([np.tanh(normalized[-1]), np.tanh(normalized.mean()),
                      np.tanh((e[-1]-e[0])/scale/np.sqrt(a)), a/H])
        if mode == "no_prefix":
            phi = np.r_[1., v]
        elif mode == "linear":
            phi = np.r_[1., u, v]
        elif mode == "intercept":
            phi = np.ones(1)
        else:
            phi = np.r_[1., u, v, np.outer(u,v).ravel()]
        result.append(phi)
        row = np.full(H, np.nan)
        row[:H-a] = old_price[a:]
        old_tails.append(row)
    return np.stack(result), np.stack(old_tails)


def calibration_stream(data, events, *, alpha=.1, mode="full", old_target=False,
                       event_window=128, min_events=32, refit_every=5, deadline=None):
    """Fit only archived current-teacher errors with fully mature true H20.

    A teacher issuance contributes K task views, each with weight 1/K. Ridge
    strength is expressed per UNIQUE issuance, not per expanded row. This is a
    prediction algorithm, not a claim that overlapping targets are independent.
    """
    started = time.perf_counter()
    by_t = {e["origin"]:e for e in events}
    pending, mature, results, fits, beta = {}, [], {}, 0, None
    total_matured = 0
    changes = []
    for event in events:
        t = event["origin"]
        if deadline and time.monotonic() >= deadline:
            raise TimeoutError("Frozen deadline")
        q = t-H
        if q in pending:
            saved = pending[q]
            issued = by_t[q]
            future = data.market.price[q+np.arange(1,H+1)]
            assert q+H <= t
            price_pred = issued["base"]+issued["scale"]*SQRT_H*issued["current"]
            target = np.broadcast_to((future-price_pred)/issued["scale"]/SQRT_H, (len(K),H)).copy()
            if old_target:
                target = (future[None]-saved["old_tails"])/issued["scale"]/SQRT_H
            mature.append(dict(origin=q, phi=saved["phi"], y=target))
            mature = mature[-event_window:]
            total_matured += 1
            if len(mature) >= min_events and (total_matured-min_events)%refit_every == 0:
                phi = np.stack([r["phi"] for r in mature])
                y = np.stack([r["y"] for r in mature])
                n, views, d = phi.shape
                z = phi.reshape(n*views,d)
                yy = y.reshape(n*views,H)
                if old_target:
                    b = []
                    for j in range(H):
                        valid = np.isfinite(yy[:,j])
                        if not valid.any():
                            b.append(np.zeros(d)); continue
                        # Every issuance receives equal total weight at each
                        # horizon among its eligible age views.
                        vcount = valid.reshape(n,views).sum(1)
                        w = np.repeat(1/np.maximum(vcount,1),views)[valid]/n
                        zv = z[valid]
                        b.append(np.linalg.solve(zv.T@(zv*w[:,None])+alpha*np.eye(d),
                                                 zv.T@(yy[valid,j]*w)))
                    beta = np.stack(b,axis=1)
                else:
                    # Sum views / K / unique events = mean over n*K rows.
                    beta = np.linalg.solve(z.T@z/(n*views)+alpha*np.eye(d), z.T@yy/(n*views))
                fits += 1
                changes.append(dict(event=t, latest_label=t, earliest_issue=mature[0]["origin"],
                                    latest_issue=q, unique_issuances=n, task_views=n*views,
                                    beta_norm=float(np.linalg.norm(beta))))
        if len(event["fast_origins"]):
            phi, old_tails = calibration_features(data,event,by_t,mode)
            pending[t] = dict(phi=phi.copy(), old_tails=old_tails.copy())
            correction = np.zeros((len(K),H)) if beta is None else phi@beta
            if not np.isfinite(correction).all():
                raise FloatingPointError("Nonfinite calibrator")
            results[t] = correction.copy()
    receipt = dict(alpha=alpha, mode=mode, old_target=old_target, event_window=event_window,
                   min_events=min_events, refit_every=refit_every, ridge_refits=fits,
                   feature_dimension=0 if beta is None else int(beta.shape[0]),
                   issued_calibrations=len(results), seconds=time.perf_counter()-started,
                   revision_residual_target=not old_target, full_H20_maturity_required=True,
                   updates=changes, final_beta_sha256=None if beta is None else digest_arrays({"beta":beta}),
                   expanded_task_views_are_not_independent_samples=True)
    return results, beta, receipt


def student_stream(data, events, initial_student, *, variant, shrink, lr=.0003,
                   corrections=None, gamma=.9, deadline=None):
    started = time.perf_counter()
    student = copy.deepcopy(initial_student)
    slow = torch.optim.AdamW(student.parameters(),lr=lr,weight_decay=.001)
    fast = torch.optim.AdamW(student.parameters(),lr=lr,weight_decay=.001)
    preds, trace, slow_steps, fast_steps = [], [], 0, 0
    discounts = torch.tensor([[1.]*(k-1)+[gamma**j for j in range(H-k+1)] for k in K])
    for event in events:
        t = event["origin"]
        if deadline and time.monotonic() >= deadline:
            raise TimeoutError("Frozen deadline")
        loss_s = loss_f = grad_f = 0.
        if len(event["er_origins"]):
            b = data.batch(event["er_origins"],t)
            pt = torch.tensor(event["er_teacher"])
            forecast = pt+student(torch.tensor(b["x"]),pt)
            loss = (forecast-torch.tensor(b["y"])).square().mean()+shrink*forecast.square().mean()
            _update(loss,slow,student.parameters()); loss_s=float(loss.detach()); slow_steps+=1
        if variant != "er" and len(event["fast_origins"]):
            old = data.batch(event["fast_origins"],t,labels=False)
            pt_old = torch.tensor(event["fast_teacher"])
            pred_old = pt_old+student(torch.tensor(old["x"]),pt_old)
            curr_price = event["base"]+event["scale"]*SQRT_H*event["current"]
            if variant == "self_td":
                with torch.no_grad():
                    ts = student(torch.tensor(event["x"][None]),torch.tensor(event["current"][None])).numpy()[0]
                curr_price = curr_price+event["scale"]*SQRT_H*ts
            pseudo, mask = [], []
            for i,k in enumerate(K):
                o = t-k
                prefix = data.market.price[o+1:t+1].copy()
                tail = curr_price[:H-k].copy()
                if corrections is not None:
                    tail += event["scale"]*SQRT_H[:H-k]*corrections[t][i,:H-k]
                target_price = np.r_[prefix,tail]
                pseudo.append((target_price-old["base"][i])/old["scale"][i]/SQRT_H)
                mask.append(np.r_[np.ones(k),np.zeros(H-k)] if variant == "masked" else np.ones(H))
            target = torch.tensor(np.stack(pseudo))
            weights = discounts*torch.tensor(np.stack(mask))
            objective = ((pred_old-target).square()*weights).mean()
            grad_f = _update(objective,fast,student.parameters())
            loss_f=float(objective.detach()); fast_steps+=1
        with torch.no_grad():
            current = torch.tensor(event["current"][None])
            norm = current+student(torch.tensor(event["x"][None]),current)
            pred = event["base"]+event["scale"]*SQRT_H*norm.numpy()[0]
        if not np.isfinite(pred).all():
            raise FloatingPointError("Nonfinite issued forecast")
        preds.append(pred.copy())
        trace.append(dict(origin=t,slow_objective=loss_s,fast_objective=loss_f,fast_gradient=grad_f))
    pred_array=np.stack(preds)
    receipt=dict(variant=variant,lr=lr,shrink=shrink,gamma=gamma,td_k=list(K),
                 slow_steps=slow_steps,fast_steps=fast_steps,seconds=time.perf_counter()-started,
                 initial_state_sha256=digest_arrays(arrays(initial_student)),
                 final_state_sha256=digest_arrays(arrays(student)),
                 prediction_sha256=digest_arrays({"predictions":pred_array}),trace=trace,
                 correction_enters_training_targets_only=corrections is not None,
                 old_forecasts_rewritten=False)
    return pred_array,student,receipt


def metric_cells(data, events, predictions, years):
    origins=np.array([e["origin"] for e in events])
    rows=[]
    for year in years:
        ids=np.flatnonzero((data.market.dates[origins]>=f"{year}-01-01") &
                           (data.market.dates[origins]<f"{year+1}-01-01") &
                           (origins+H<len(data.market.price)))
        ids=ids[data.market.dates[origins[ids]+H]<f"{year+1}-01-01"]
        if not len(ids):
            raise ValueError(f"No complete H20 origins for {year}")
        oo=origins[ids]
        for j in SCORE_H:
            y=data.market.price[oo+j+1]; pred=predictions[ids,j]
            rw=data.market.price[oo]
            mse=float(np.mean((pred-y)**2)); rw_mse=float(np.mean((rw-y)**2))
            rows.append(dict(market=data.market.name,year=year,horizon=int(j+1),origins=len(ids),
                             mse=mse,rw_mse=rw_mse,relative_mse=mse/rw_mse,
                             mae=float(np.mean(np.abs(pred-y))),
                             directional_accuracy=float(np.mean(np.sign(pred-rw)==np.sign(y-rw)))))
    return rows
