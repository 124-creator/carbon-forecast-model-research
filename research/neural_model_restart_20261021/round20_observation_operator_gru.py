"""R20: activity-clock and effective-sampling observation operators over warm GRU.

Code/purechecks and zero-operator replay delivery. New-variant fitting requires
explicit --run-development and is NOT part of the delivery execution.
A changes range amplitude, not normalized shape; M changes geometry likelihood
and observation moments. These are predictive working mechanisms, not identified
true activity, trade count, measurement bias or integrated variance. Existing
GRU outperformed old mean fields; this is a new, controlled hypothesis.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import time
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import round18_direct_controls as C18
import gru_warmstart_replay as REPLAY
import sampling_density_prior_probability as SDP

S18, S17, B = C18.S18, C18.S17, C18.B
KP, DTYPE, ASSETS = S18.KP, C18.DTYPE, C18.ASSETS
HERE = Path(__file__).resolve().parent
VARIANTS = ('continued', 'clock_only', 'density_only', 'joint')
SEEDS = C18.SEEDS
MAX_EPOCHS, PATIENCE, MIN_EPOCHS = 200, 20, 30
OLD_LR, NEW_LR, FIELD_LAMBDA = .001, .01, .05
BETA0_INIT, BETA1_INIT = 1., math.log(math.expm1(1.))
GEOMETRY_TOL = 1e-10
EXPECTED_ADAPTER_SHA256 = 'bb1799ad67e1901d3209e3a01fad96515b82fb72bdcdc8e5b58fe7cd350a14f7'
ZERO_GATE_PATH = HERE / 'round20_observation_operator_gru_zero_gate.json'


def geometry31(F17, observed):
    """Observable a,z only; never infer order of extrema from daily OHLC."""
    a = ((F17[:, 0] + 1) / 2).clamp(0, 1)
    z = ((F17[:, 1] + 1) / 2).clamp(0, 1)
    edges = torch.linspace(0, 1, 6, dtype=F17.dtype, device=F17.device)
    ia = (torch.bucketize(a.contiguous(), edges, right=True) - 1).clamp(0, 4)
    iz = (torch.bucketize(z.contiguous(), edges, right=True) - 1).clamp(0, 4)
    atom = ((a - 1).abs() <= GEOMETRY_TOL) & (z.abs() <= GEOMETRY_TOL)
    edge = ((z - 1).abs() <= GEOMETRY_TOL) & ~atom
    cell = torch.where(atom, torch.zeros_like(ia), torch.where(edge, 1 + ia, 6 + 5 * ia + iz))
    return torch.where(observed, cell, torch.full_like(cell, -1))


def fit_volume_ar(datasets):
    """Intercept AR1, consecutive pre2025 volumeZ, including fixed imputed zeros.

    OLS slope is clipped to [-.95,.95], intercept refitted conditional on that
    slope. Residual std uses ddof2. No postcut observation enters any estimate.
    """
    out = {}
    for a in ASSETS:
        D = datasets[a]; v = np.asarray(D['F17'][:D['cut'], 2], dtype=float)
        assert len(v) >= 4 and np.isfinite(v).all()
        x, y = v[:-1], v[1:]; denominator = float(((x - x.mean()) ** 2).sum())
        raw = float(((x - x.mean()) * (y - y.mean())).sum() / denominator) if denominator > 0 else 0.
        rho = float(np.clip(raw, -.95, .95)); intercept = float(y.mean() - rho * x.mean())
        residual = y - intercept - rho * x
        std = float(np.sqrt(np.sum(residual ** 2) / max(len(residual) - 2, 1)))
        out[a] = dict(n_rows=len(v), n_pairs=len(x), first_row=0, last_row=D['cut'] - 1,
            raw_rho=raw, rho=rho, intercept=intercept, mu=intercept / (1 - rho),
            residual_std=std, residual_variance_ddof=2,
            data='all consecutive t<cut F17 volumeZ; fixed S17 normalizer/imputation',
            slope_clip=[-.95, .95], intercept_rule='refit at clipped slope')
    return out


class ObservationOperator(nn.Module):
    def __init__(self, decoder, variant):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant; self.decoder = copy.deepcopy(decoder)
        clock, density = variant in ('clock_only', 'joint'), variant in ('density_only', 'joint')
        self.lambda_A = nn.Parameter(torch.tensor(0., dtype=DTYPE), requires_grad=clock)
        self.alpha_N = nn.Parameter(torch.tensor(0., dtype=DTYPE), requires_grad=density)
        self.beta0 = nn.Parameter(torch.tensor(BETA0_INIT if density else 0., dtype=DTYPE), requires_grad=density)
        self.beta1 = nn.Parameter(torch.tensor(BETA1_INIT if density else 0., dtype=DTYPE), requires_grad=density)
        assert not any(True for _ in self.decoder.parameters())

    def prior_activity(self, volume):
        v = volume.clamp(-5, 5)
        logA = -self.lambda_A * F.softplus(-(v + 1))
        gamma = self.beta0 + F.softplus(self.beta1) * v  # gamma may be NEGATIVE
        pi = torch.softmax(gamma[..., None] * self.decoder.log_M.to(v), dim=-1)
        dense = torch.zeros_like(pi); dense[..., -1] = 1
        q = (1 - self.alpha_N) * dense + self.alpha_N * pi
        return torch.exp(logA), logA, q, gamma

    def effective_c(self, volume):
        A, _, q, _ = self.prior_activity(volume)
        return A * (q * self.decoder.c_M_relative_to_dense.to(q)).sum(dim=-1)

    def current(self, volume, cell, observed, base_R):
        A, logA, q, gamma = self.prior_activity(volume)
        dense = torch.zeros_like(q); dense[..., -1] = 1
        q = torch.where(observed[:, None], q, dense)
        # Zero-range rows are forced dense and then skipped by KP, not fitted.
        safe_cell = torch.where(observed, cell, torch.zeros_like(cell))
        safe_R = torch.where(observed, base_R, torch.ones_like(base_R))
        moments = self.decoder(safe_cell, q, safe_R)
        return dict(A=A, logA=logA, q=q, gamma=gamma, **moments)

    def penalty(self):
        value = self.lambda_A.new_zeros(())
        if self.lambda_A.requires_grad:
            value = value + .005 * self.lambda_A.square()
        if self.alpha_N.requires_grad:
            value = value + .001 * ((self.beta0 - BETA0_INIT).square() + (self.beta1 - BETA1_INIT).square())
            value = value + .005 * self.alpha_N.square()
        return value

    @torch.no_grad()
    def project(self):
        self.lambda_A.clamp_(0, 1); self.alpha_N.clamp_(0, 1)

    def summary(self):
        return dict(lambda_A=float(self.lambda_A.detach()), alpha_N=float(self.alpha_N.detach()),
            beta0=float(self.beta0.detach()), beta1_raw=float(self.beta1.detach()),
            beta1_softplus=float(F.softplus(self.beta1).detach()),
            beta_init=[BETA0_INIT, BETA1_INIT], gamma_can_be_negative=True)


class OperatorGRU(C18.DirectControls):
    def __init__(self, datasets, parameters, baseline, decoder, arfit, variant, seed, warm_state):
        # Load complete original GRU/head, process/bank/cache state before adding modules.
        super().__init__(datasets, parameters, baseline, 'shared_GRU', seed)
        self.load_state_dict(warm_state, strict=True)
        for a in ASSETS:
            np.testing.assert_array_equal(getattr(self, f'bank_{a}').numpy(), datasets[a]['bank_index'])
            for k in ('m0', 'P0', 'K0'):
                torch.testing.assert_close(getattr(self, f'{k}_{a}'), baseline[a][k], rtol=0, atol=0)
            for k in ('mu', 'phi', 'Q', 's_R'):
                assert float(getattr(self.processes[a], k)) == float(parameters[a][k])
        self.variant = variant
        self.operator = ObservationOperator(decoder, variant)
        self.bucket_head = nn.Linear(8, 3).to(dtype=DTYPE)
        nn.init.zeros_(self.bucket_head.weight); nn.init.zeros_(self.bucket_head.bias)
        nodes, weights = np.polynomial.hermite.hermgauss(16)
        weights = weights / weights.sum()
        self.register_buffer('gh_nodes', torch.tensor(nodes, dtype=DTYPE))
        self.register_buffer('gh_weights', torch.tensor(weights, dtype=DTYPE))
        for a in ASSETS:
            observed = torch.tensor(datasets[a]['observed'], dtype=torch.bool)
            self.register_buffer(f'observed_{a}', observed)
            self.register_buffer(f'cells_{a}', geometry31(torch.tensor(datasets[a]['F17'], dtype=DTYPE), observed))
            self.register_buffer(f'ar_{a}', torch.tensor([arfit[a][k] for k in ('mu', 'rho', 'residual_std')], dtype=DTYPE))

    def cbar(self, a, T):
        bank = getattr(self, f'bank_{a}')
        return self.operator.effective_c(T['F17'][bank, 2]).mean()

    def filtered_fields(self, a, D, T, stop, detach_every=128):
        assert D['cut'] <= stop <= D['n']
        process = self.processes[a]; observed = getattr(self, f'observed_{a}')[:stop]
        base_R = T['cell_R'][:stop] * torch.exp(process.s_R)
        fields = self.operator.current(T['F17'][:stop, 2], getattr(self, f'cells_{a}')[:stop], observed, base_R)
        cbar = self.cbar(a, T)
        # cell_b already includes the OLD c512 proxy frame; no extra log(c512).
        x = T['log_proxy'][:stop] - T['cell_b'][:stop] - fields['logA'] - fields['Delta_b'] + torch.log(cbar)
        mu, phi, Q, _ = process.parameters_bounded()
        m, P, K = KP.kalman_prefix(x, fields['Rmix'], observed, phi, Q, mu, *process.initial_state(), validate=False)
        f5 = torch.cat([T['F17'][:stop], K[:, None], (m - mu)[:, None]], 1)
        hidden = f5.new_zeros((1, 1, 8)); scalar, bucket = [], []
        width = detach_every if detach_every else stop
        for lo in range(0, stop, width):
            out, hidden = self.gru(f5[None, lo:lo + width], hidden)
            scalar.append(.5 * torch.tanh(self.head(out[0]).squeeze(-1)))
            bucket.append(.5 * torch.tanh(self.bucket_head(out[0])))
            if detach_every:
                hidden = hidden.detach()
        raw_scalar, raw_bucket = torch.cat(scalar), torch.cat(bucket)
        bank = getattr(self, f'bank_{a}')
        center_scalar = raw_scalar[bank].mean(); center_bucket = raw_bucket[bank].mean(dim=0)
        ds, db = raw_scalar - center_scalar, raw_bucket - center_bucket
        reference = torch.cat([(raw_scalar[bank] - center_scalar)[:, None], raw_bucket[bank] - center_bucket], 1)
        return dict(m=m, P=P, K=K, m_pred=m + ds, delta_scalar=ds, delta_bucket=db,
            reference=reference, center_scalar=center_scalar, center_bucket=center_bucket,
            cbar=cbar, x=x, base_R=base_R, **fields)

    def future_factor(self, a, current_volume, cbar):
        mu, rho, sigma = getattr(self, f'ar_{a}')
        k = torch.arange(1, 23, dtype=DTYPE, device=current_volume.device)
        mean = mu + rho.pow(k) * (current_volume[:, None] - mu)
        variance = sigma.square() * (1 - rho.pow(2 * k)) / (1 - rho.square())
        v = mean[..., None] + torch.sqrt(2 * variance)[..., None] * self.gh_nodes
        ceff = self.operator.effective_c(v)
        # This centered quadrature form makes a constant-one integrand EXACT1
        # while preserving alpha/lambda directional derivatives at their zero gate.
        expected = 1 + ((ceff - 1) * self.gh_weights).sum(dim=-1)
        return expected / cbar

    def forecast(self, a, T, q, fields):
        dailybase = self.processes[a].future_daily_means(fields['m_pred'][q], fields['P'][q])
        factor = self.future_factor(a, T['F17'][q, 2], fields['cbar'])
        daily = dailybase * factor
        basebucket = torch.stack([daily[:, 0], daily[:, 1:5].sum(1), daily[:, 5:22].sum(1)], 1)
        positive = basebucket * torch.exp(fields['delta_bucket'][q])
        horizons = positive.cumsum(dim=1) / positive.new_tensor([1., 5., 22.])
        return horizons, dict(positive_buckets=positive, future_factor=factor, daily_before_buckets=daily)


def make_optimizer(model):
    old = list(model.gru.parameters()) + list(model.head.parameters())
    new = list(model.bucket_head.parameters()) + [p for p in model.operator.parameters() if p.requires_grad]
    assert len({id(p) for p in old + new}) == len(old + new)
    assert {id(p) for p in model.parameters() if p.requires_grad} == {id(p) for p in old + new}
    return torch.optim.Adam([dict(params=old, lr=OLD_LR), dict(params=new, lr=NEW_LR)])


def objective(model, datasets, T):
    losses, refs, centers = {}, {}, {}
    for a in ASSETS:
        D = datasets[a]; ids = D['tr']; q = D['origin'][ids]
        assert len(q) and (q + 22 < D['cut']).all()
        fields = model.filtered_fields(a, D, T[a], D['cut'], 128)
        value, _ = model.forecast(a, T[a], q, fields)
        losses[a] = B.qlike(value, T[a]['y'][ids], T[a]['ewma'][ids]).mean()
        refs[a] = fields['reference'].square().mean(dim=0).sum()
        centers[a] = dict(scalar=float(fields['center_scalar'].detach()), bucket=fields['center_bucket'].detach().tolist())
    data = sum(losses.values()) / len(ASSETS)
    field_penalty = FIELD_LAMBDA * sum(refs.values()) / len(ASSETS)
    operator_penalty = model.operator.penalty()
    return data + field_penalty + operator_penalty, dict(data=data, losses=losses,
        field_penalty=field_penalty, operator_penalty=operator_penalty, centers=centers)


def train_epoch(model, datasets, T, optimizer):
    model.train(); optimizer.zero_grad(set_to_none=True)
    loss, info = objective(model, datasets, T)
    assert torch.isfinite(loss)
    loss.backward()
    norm = float(nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 2))
    assert np.isfinite(norm)
    optimizer.step(); model.operator.project()
    return dict(train_joint_QLIKE=float(info['data'].detach()),
        train_asset_QLIKE={a: float(v.detach()) for a, v in info['losses'].items()},
        field_penalty=float(info['field_penalty'].detach()), operator_penalty=float(info['operator_penalty'].detach()),
        bankcenter=info['centers'], gradient_norm_before_clip=norm,
        optimizer_steps=1, backward_calls=1, operator=model.operator.summary())


@torch.no_grad()
def evaluate(model, datasets, T):
    model.eval(); predictions, scores, aux, diagnostics = {}, {}, {}, {}
    for a in ASSETS:
        D = datasets[a]; q = D['origin'][D['va']]
        fields = model.filtered_fields(a, D, T[a], D['n'], 0)
        value, future = model.forecast(a, T[a], q, fields)
        assert torch.isfinite(value).all() and (value > 0).all()
        predictions[a] = value.numpy()
        scores[a] = float(B.qlike(value, T[a]['y'][D['va']], T[a]['ewma'][D['va']]).mean())
        aux[a] = {name: fields[name][q].numpy() for name in ('m', 'P', 'K', 'm_pred', 'delta_scalar', 'delta_bucket',
            'A', 'q', 'w', 'Delta_b', 'Rmix', 'between_bias_variance', 'x')}
        aux[a].update({name: v.numpy() for name, v in future.items()})
        diagnostics[a] = dict(cbar=float(fields['cbar']), scalar_center=float(fields['center_scalar']),
            bucket_center=fields['center_bucket'].tolist(),
            bank_delta={str(j): S17.distribution(fields['reference'][:, j].numpy()) for j in range(4)},
            validation_fields={name: S17.distribution(fields[name][q].numpy()) for name in
                               ('A', 'gamma', 'Delta_b', 'Rmix', 'between_bias_variance', 'delta_scalar')},
            futurefactor=S17.distribution(future['future_factor'].numpy().ravel()))
    return predictions, sum(scores.values()) / len(ASSETS), scores, aux, diagnostics


def train_one(datasets, parameters, baseline, decoder, arfit, variant, seed, warm_state):
    T = S17.make_tensors(datasets)
    model = OperatorGRU(datasets, parameters, baseline, decoder, arfit, variant, seed, warm_state)
    immutable = {k: v.clone() for k, v in model.named_buffers()}
    pred, best, scores, aux, fields = evaluate(model, datasets, T)
    history = [dict(epoch=0, validation_joint_QLIKE=best, validation_asset_QLIKE=scores)]
    best_state = copy.deepcopy(model.state_dict()); best_epoch = bad = 0
    optimizer = make_optimizer(model)
    for epoch in range(1, MAX_EPOCHS + 1):
        info = train_epoch(model, datasets, T, optimizer)
        _, score, scores, _, _ = evaluate(model, datasets, T)
        history.append(dict(epoch=epoch, validation_joint_QLIKE=score, validation_asset_QLIKE=scores, **info))
        if score < best - 1e-5:
            best, best_epoch, bad = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if bad >= PATIENCE and epoch >= MIN_EPOCHS:
            break
    model.load_state_dict(best_state)
    pred, score, scores, aux, fields = evaluate(model, datasets, T)
    for name, value in model.named_buffers():
        assert torch.equal(value, immutable[name]), f'Frozen buffer changed: {name}'
    info = dict(variant=variant, seed=seed, best_joint_epoch=best_epoch, epochs_run=len(history) - 1,
        best_joint_validation_QLIKE=best, returned_joint_validation_QLIKE=score,
        selected_epoch_asset_scores=scores, trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        old_GRU_head_parameters=369, bucket_parameters=27, operator=model.operator.summary(), fields=fields,
        history=history, selected_old_epoch=REPLAY.SELECTED[seed], optimizer_reset=True,
        seed_note='initialization replicate on same dataset, not an independent dataset')
    return model, pred, aux, info


def synthetic_decoder():
    support = SDP.FROZEN.structural_support()
    p = support.astype(float); p /= p.sum(axis=1, keepdims=True)
    b = np.arange(7)[:, None] * .21 + np.arange(31)[None, :] * .003
    R = .1 + np.arange(7)[:, None] * .02 + np.arange(31)[None, :] * .001
    calibration = dict(M=list(SDP.FROZEN.MS), p_cell_given_M=p.tolist(), b=b.tolist(), R=R.tolist(), supported=support.tolist())
    return SDP.ProbabilityDensityMomentDecoder(calibration, [.25, .4, .55, .7, .8, .9, 1.])


def _synthetic_checks():
    datasets, raw = S17.synthetic_pair(); parameters = S18._synthetic_parameters()
    T = S17.make_tensors(datasets); baseline = C18.precompute_baseline(datasets, parameters, T)
    arfit = fit_volume_ar(datasets); decoder = synthetic_decoder()
    warm = C18._nonzero(C18.DirectControls(datasets, parameters, baseline, 'shared_GRU', 11))
    expected, _, _, _ , _ = C18.evaluate(warm, datasets, T)
    models = {v: OperatorGRU(datasets, parameters, baseline, decoder, arfit, v, 11, warm.state_dict()) for v in VARIANTS}
    max_zero_error = 0.
    for model in models.values():
        pred, _, _, _, _ = evaluate(model, datasets, T)
        for a in ASSETS:
            np.testing.assert_allclose(pred[a], expected[a], rtol=REPLAY.RTOL, atol=REPLAY.ATOL)
            max_zero_error = max(max_zero_error, float(np.abs(pred[a] - expected[a]).max()))
    # Geometry atom/edge/interior and skipped zero range, including tolerance.
    geometry = torch.tensor([[1., -1., 0.], [-.2, 1., 0.], [0., 0., 0.], [1., -1., 0.]], dtype=DTYPE)
    torch.testing.assert_close(geometry31(geometry, torch.tensor([True, True, True, False])),
                               torch.tensor([0, 3, 18, -1]), rtol=0, atol=0)
    operator = models['joint'].operator
    alpha = operator.alpha_N
    v = torch.tensor([-3., -.3, 2.], dtype=DTYPE); cells = torch.tensor([0, 1, 6]); observed = torch.ones(3, dtype=torch.bool)
    def moments(gate):
        _, _, q0, _ = operator.prior_activity(v)
        pi = torch.softmax((operator.beta0 + F.softplus(operator.beta1) * v)[:, None] * decoder.log_M, dim=-1)
        dense = torch.zeros_like(pi); dense[:, -1] = 1
        out = decoder(cells, (1 - gate) * dense + gate * pi, torch.full((3,), .3, dtype=DTYPE))
        return torch.cat([out['Delta_b'], out['Rmix']])
    derivative = torch.autograd.functional.jacobian(moments, alpha)
    assert torch.isfinite(derivative).all() and derivative.abs().max() > 1e-8
    right = (moments(alpha + 1e-8) - moments(alpha)) / 1e-8
    torch.testing.assert_close(derivative, right, rtol=5e-5, atol=2e-5)
    # Full law of total variance, not a mixture of precisions.
    q = torch.full((3, 7), 1 / 7, dtype=DTYPE)
    mixed = decoder(cells, q, decoder.variance[-1, cells])
    biases = decoder.bias[:, cells].T; variances = decoder.variance[:, cells].T
    mean = (mixed['w'] * biases).sum(1)
    total = (mixed['w'] * variances).sum(1) + (mixed['w'] * (biases - mean[:, None]).square()).sum(1)
    torch.testing.assert_close(mixed['Rmix'], total)
    assert operator.prior_activity(torch.tensor([-5.], dtype=DTYPE))[3].item() < 0
    # Constant A/M raw-P frame: A cancels after its same bank normalization.
    # Exact prescribed formula, plus old/new dense-reference alignment residual.
    old_b512, new_b512, old_c512, new_c512 = -.2, -.18, .94, .95
    A, cM, bM, logP = .6, .4, -.8, -8.
    base_b = old_b512 - math.log(old_c512)
    cbar = A * cM / new_c512
    x = logP - base_b - math.log(A) - (bM - new_b512) + math.log(cbar)
    alignment = new_b512 - old_b512 + math.log(old_c512 / new_c512)
    assert abs(x - (logP - bM + math.log(cM) + alignment)) < 1e-14
    # Prior to bank re-scaling, subtracting logA produces the sigma-risk frame.
    unscaled = x - math.log(cbar)
    assert abs(unscaled - (logP - math.log(A) - bM + math.log(old_c512) + new_b512 - old_b512)) < 1e-14
    # Exercise the implemented operator/filter for a constant A and point-M prior.
    # M16 has support in all cells; zero-range rows are still dense and skipped.
    point_M, constant_A = 4, .6
    def constant_prior(volume):
        q = volume.new_zeros((*volume.shape, 7)); q[..., point_M] = 1
        return (torch.full_like(volume, constant_A), torch.full_like(volume, math.log(constant_A)),
                q, torch.zeros_like(volume))
    with patch.object(operator, 'prior_activity', side_effect=constant_prior):
        with torch.no_grad():
            for a in ASSETS:
                D = datasets[a]; state = models['joint'].filtered_fields(a, D, T[a], D['n'], 0)
                ratio = decoder.c_M_relative_to_dense[point_M]
                torch.testing.assert_close(state['cbar'], ratio * constant_A)
                observed_rows = getattr(models['joint'], f'observed_{a}')
                cells_a = getattr(models['joint'], f'cells_{a}')[observed_rows]
                delta_bias = decoder.bias[point_M, cells_a] - decoder.bias[-1, cells_a]
                expected_x = (T[a]['log_proxy'] - T[a]['cell_b'])[observed_rows] - delta_bias + torch.log(ratio)
                torch.testing.assert_close(state['x'][observed_rows], expected_x, rtol=1e-12, atol=1e-12)
                q_a = D['origin'][D['va']]
                factors = models['joint'].future_factor(a, T[a]['F17'][q_a, 2], state['cbar'])
                torch.testing.assert_close(factors, torch.ones_like(factors), rtol=1e-12, atol=1e-12)
    changed_data = copy.deepcopy(datasets)
    for a in ASSETS:
        changed_data[a]['F17'][datasets[a]['cut']:, 2] += 1e6
    assert fit_volume_ar(changed_data) == arfit
    for a in ASSETS:
        changed_T = {k: v.clone() for k, v in T[a].items()}
        changed_T['F17'][datasets[a]['cut']:, 2] += 99
        torch.testing.assert_close(models['joint'].cbar(a, changed_T), models['joint'].cbar(a, T[a]), rtol=0, atol=0)
    for variant, model in models.items():
        # Exercise nonzero mechanisms/buckets in synthetic fixtures only.
        with torch.no_grad():
            if model.operator.lambda_A.requires_grad:
                model.operator.lambda_A.fill_(.2)
            if model.operator.alpha_N.requires_grad:
                model.operator.alpha_N.fill_(.3)
            model.bucket_head.weight.fill_(.015)
        buffers = {k: v.clone() for k, v in model.named_buffers()}
        gradT = {a: {**T[a], 'F17': T[a]['F17'].clone().requires_grad_(), 'y': T[a]['y'].clone().requires_grad_()} for a in ASSETS}
        loss, info = objective(model, datasets, gradT)
        loss.backward()
        for a in ASSETS:
            D = datasets[a]
            assert (gradT[a]['F17'].grad[D['cut']:] == 0).all()
            assert (gradT[a]['y'].grad[D['va']] == 0).all()
            # Predict an origin using no future volume; bank stays pre25.
            start = D['cut'] + 2
            changed = {k: v.clone() for k, v in T[a].items()}
            changed['F17'][start + 1:, 2] += 20
            with torch.no_grad():
                first = model.filtered_fields(a, D, T[a], D['n'], 0)
                second = model.filtered_fields(a, D, changed, D['n'], 0)
                p, details = model.forecast(a, T[a], np.array([start]), first)
                pp, _ = model.forecast(a, changed, np.array([start]), second)
                torch.testing.assert_close(p, pp, rtol=0, atol=0)
                assert (details['positive_buckets'] > 0).all() and (p > 0).all()
                np.testing.assert_allclose(p.numpy() * [1., 5., 22.], details['positive_buckets'].cumsum(1).numpy())
                skip = ~getattr(model, f'observed_{a}')
                assert (first['K'][skip] == 0).all()
                assert (first['q'][skip, -1] == 1).all() and (first['q'][skip, :-1] == 0).all()
        optimizer = make_optimizer(model)
        assert [g['lr'] for g in optimizer.param_groups] == [OLD_LR, NEW_LR]
        with patch('torch.autograd.backward', wraps=torch.autograd.backward) as backward:
            with patch.object(optimizer, 'step', wraps=optimizer.step) as step:
                train_epoch(model, datasets, T, optimizer)
        assert backward.call_count == step.call_count == 1
        for name, value in model.named_buffers():
            assert torch.equal(value, buffers[name]), name
    # projection boundaries; disabled parameters remain exactly zero and frozen.
    with torch.no_grad():
        operator.lambda_A.fill_(2); operator.alpha_N.fill_(-1)
    operator.project(); assert operator.lambda_A == 1 and operator.alpha_N == 0
    assert all(not p.requires_grad and p == 0 for p in models['continued'].operator.parameters())
    bad = copy.deepcopy(datasets); bad[ASSETS[0]]['origin'][bad[ASSETS[0]]['tr'][-1]] = bad[ASSETS[0]]['cut'] - 22
    try:
        objective(models['continued'], bad, T)
    except AssertionError:
        pass
    else:
        raise AssertionError('Nonmature common-h22 origin accepted')
    return dict(zero_operator_warm_GRU_all_variants=True, max_zero_prediction_absolute_error=max_zero_error,
        alpha0_gradient_finite_nonzero_and_right_difference=True, probability_adapter_total_variance=True,
        gamma_negative_allowed=True, observed_geometry31_and_zero_skip=True,
        constant_AM_raw_proxy_frame_and_dense_alignment_verified=True,
        constant_A_cancels_after_same_bank_rescaling=True, cbar_and_volume_AR_trainonly=True,
        no_future_volume_in_predictor=True, positive_unbudgeted_buckets_and_h22_maturity=True,
        train_no_validation_feature_target_gradient=True, one_backward_and_step_per_epoch=True,
        frozen_theta_decoder_cache_bank_AR_buffers_unchanged=True, distinct_optimizer_groups_correct=True,
        projection_and_disabled_parameters_correct=True, real_variant_fitting_executed=False)


def synthetic_checks():
    with ExitStack() as stack:
        for target in ('pandas.read_csv', 'numpy.load', 'pathlib.Path.read_text',
                       'round17_shared_precision.prepare_pair', 'round18_frozen_moments.load_stage16_parameters',
                       'sampling_density_prior_probability.load_probability_decoder'):
            stack.enter_context(patch(target, side_effect=AssertionError('real reader forbidden in purechecks')))
        checks = _synthetic_checks()
    checks.update(real_input_readers_guarded=True, market_files_opened=False)
    return checks


def source_hashes():
    out = REPLAY.source_hashes(); out['warmstart_exporter'] = out.pop('runner')
    paths, manifest = REPLAY.output_paths()
    sources = dict(runner=Path(__file__), moments=Path(SDP.FROZEN.__file__), probability_adapter=Path(SDP.__file__),
        calibration_JSON=SDP.CALIBRATION, calibration_NPZ=SDP.CALIBRATION.with_suffix('.npz'),
        warmstart_manifest=manifest, **{f'warmstart_seed{s}': p for s, p in paths.items()})
    out.update({name: dict(path=str(Path(p).resolve()), sha256=REPLAY.sha256(p)) for name, p in sources.items()})
    return out


def prepare_development():
    hashes = source_hashes()
    assert hashes['probability_adapter']['sha256'] == EXPECTED_ADAPTER_SHA256, 'Frozen probability adapter changed'
    parameters, provenance = S18.load_stage16_parameters(); datasets, audit = S17.prepare_pair()
    for a in ASSETS:
        assert (datasets[a]['date'] < B.RESERVED).all()
        assert abs(provenance[a]['reference_c512'] - audit['proxy_c512']) < 1e-14
    frozen, baseline_gate = S18.verify_real_baseline(datasets, parameters)
    T = S17.make_tensors(datasets); baseline = C18.precompute_baseline(datasets, parameters, T)
    decoder = SDP.load_probability_decoder(); arfit = fit_volume_ar(datasets)
    paths, manifest_path = REPLAY.output_paths()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')); warm = {}
    for seed, path in paths.items():
        assert REPLAY.sha256(path) == manifest['results'][str(seed)]['checkpoint_sha256']
        own = torch.load(path, weights_only=True, map_location='cpu')
        assert own['metadata']['seed'] == seed and own['metadata']['epoch'] == REPLAY.SELECTED[seed]
        assert own['metadata']['frozen_parameters'] == parameters
        for name, old in own['metadata']['source_hashes'].items():
            assert REPLAY.sha256(old['path']) == old['sha256'], f'Warmstart dependency changed: {name}'
        warm[seed] = own['state_dict']
    return dict(datasets=datasets, parameters=parameters, baseline=baseline, decoder=decoder,
        arfit=arfit, warm=warm, T=T, audit=audit, provenance=provenance, hashes=hashes, baseline_gate=baseline_gate)


def zero_operator_gate(context):
    results = {}
    with np.load(REPLAY.ARCHIVE_PREDICTIONS, allow_pickle=False) as saved:
        for seed in SEEDS:
            results[seed] = {}
            for variant in VARIANTS:
                model = OperatorGRU(context['datasets'], context['parameters'], context['baseline'], context['decoder'],
                                    context['arfit'], variant, seed, context['warm'][seed])
                predictions, joint, scores, _, fields = evaluate(model, context['datasets'], context['T'])
                errors = {a: REPLAY.prediction_gate(context['datasets'][a], saved, predictions[a], a, seed) for a in ASSETS}
                results[seed][variant] = dict(errors=errors, joint_QLIKE=joint, asset_scores=scores,
                    cbar={a: fields[a]['cbar'] for a in ASSETS}, operator=model.operator.summary())
    assert source_hashes() == context['hashes']
    return dict(status='ZERO OPERATOR/BUCKET ONLY; NO VARIANT FITTING', source_hashes=context['hashes'],
        results=results, baseline_gate=context['baseline_gate'], volume_AR=context['arfit'],
        train_c_M=context['decoder'].c_M.tolist(), train_c_ratio=context['decoder'].c_M_relative_to_dense.tolist(),
        old_c512=context['audit']['proxy_c512'],
        c_reference_note='ratio uses same-new-reference train cM/c512; dense exactly1; old cell_b frame unchanged',
        new_old_dense_c_ratio=float(context['decoder'].c_M[-1]) / context['audit']['proxy_c512'],
        conditional_independence='future volume innovations independent of latent-risk innovations as working hypothesis',
        constant_activity_note='constant A cancels in current same-bank rescaling and future factor; not identified',
        real_variant_fitting_executed=False, holdout_features_targets_used=False)


def run_zero_gate():
    REPLAY.assert_new_outputs([ZERO_GATE_PATH])
    context = prepare_development(); result = zero_operator_gate(context)
    with ZERO_GATE_PATH.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def run_development(tag):
    """Future main-review entry. This function is not run during code delivery."""
    assert re.fullmatch(r'round20_observation_operator_gru_[A-Za-z0-9_-]+', tag)
    paths = dict(results=HERE / f'{tag}_results.json', predictions=HERE / f'{tag}_validation_predictions.npz')
    checkpoints = {(v, s): HERE / f'{tag}_{v}_seed{s}.pt' for v in VARIANTS for s in SEEDS}
    REPLAY.assert_new_outputs([*paths.values(), *checkpoints.values()])
    started = time.time(); context = prepare_development(); gate = zero_operator_gate(context)
    D, T = context['datasets'], context['T']
    pred = {a: {} for a in ASSETS}; arrays = dict(horizons=np.array(B.HORIZONS))
    meta = dict(config=dict(variants=VARIANTS, seeds=SEEDS, old_lr=OLD_LR, new_lr=NEW_LR,
        field_penalty=FIELD_LAMBDA, clock_penalty=.005, density_alpha_penalty=.005,
        density_beta_penalty=.001, max_epochs=MAX_EPOCHS, patience=PATIENCE, min_epochs=MIN_EPOCHS),
        source_hashes=context['hashes'], zero_operator_gate=gate, normalizer=context['audit']['normalization'],
        audit=context['audit'], volume_AR=context['arfit'], frozen_parameters=context['parameters'],
        status='REUSED 2025 DEVELOPMENT; no true A/N/IV identification; seeds not independent datasets',
        mechanisms='A changes amplitude but not normalized shape; M changes shape likelihood and b/R',
        future='GH16 conditional volume AR1 expectation; working conditional independence of volume/risk innovations',
        bucket='all variants identical scalar+3bucket heads; no fixed total budget; raw Parkinson mean targets',
        prior='volume only, clipped+-5 inside operator; current geometry only in likelihood',
        selection='one equal-asset/horizon joint epoch including0; old selected epochs never changed',
        training='GRU hidden detach128; operator KF full prefix; all outputs one data+penalty backward; Adam reset',
        fits={}, scores={}, joint_scores={}, checkpoints={})
    for a in ASSETS:
        q = D[a]['origin'][D[a]['va']]
        arrays.update({f'{a}_date': D[a]['date'][q], f'{a}_origin': q,
            f'{a}_label_end': D[a]['date'][q + 22], f'{a}_target': D[a]['y'][D[a]['va']],
            f'{a}_bank_date': D[a]['date'][D[a]['bank_index']]})
    for variant in VARIANTS:
        meta['fits'][variant] = []; values = {a: [] for a in ASSETS}
        for seed in SEEDS:
            model, p, aux, info = train_one(D, context['parameters'], context['baseline'], context['decoder'],
                context['arfit'], variant, seed, context['warm'][seed])
            meta['fits'][variant].append(info)
            for a in ASSETS:
                key = f'{variant}_seed{seed}'; pred[a][key] = p[a]; values[a].append(p[a])
                for name, value in aux[a].items():
                    arrays[f'{a}_{key}_{name}'] = value
            ckpt = checkpoints[(variant, seed)]
            payload = REPLAY.payload(model, dict(seed=seed, variant=variant, selected_epoch=info['best_joint_epoch'],
                source_hashes=context['hashes'], volume_AR=context['arfit'], normalizer=context['audit']['normalization']))
            with ckpt.open('xb') as handle:
                torch.save(payload, handle)
            torch.load(ckpt, weights_only=True, map_location='cpu')
            meta['checkpoints'][f'{variant}_seed{seed}'] = dict(path=str(ckpt), sha256=REPLAY.sha256(ckpt))
            print(variant, seed, 'joint_epoch', info['best_joint_epoch'], flush=True)
        for a in ASSETS:
            pred[a][f'{variant}_ensemble'] = np.mean(values[a], axis=0)
    for a in ASSETS:
        meta['scores'][a] = S17.metrics(pred[a], D[a]); arrays.update({f'{a}_{k}': v for k, v in pred[a].items()})
    for name in pred[ASSETS[0]]:
        perasset, perh = {}, {}
        for a in ASSETS:
            y = D[a]['y'][D[a]['va']]; v = pred[a][name]; scale = D[a]['ewma'][D[a]['va']]
            perasset[a] = dict(known_scale_QLIKE=float((np.log(v) + y / v - np.log(scale)[:, None]).mean()),
                              MSE=float(((v - y) ** 2).mean()))
        for h in B.HORIZONS:
            perh[str(h)] = {metric: sum(meta['scores'][a][name][str(h)][metric] for a in ASSETS) / 2
                for metric in ('mean_QLIKE', 'loss_difference_vs_EWMA22', 'MSE')}
        meta['joint_scores'][name] = dict(asset_scores=perasset, per_horizon=perh,
            mean_known_scale_QLIKE=sum(x['known_scale_QLIKE'] for x in perasset.values()) / 2,
            MSE=sum(x['MSE'] for x in perasset.values()) / 2)
    assert source_hashes() == context['hashes']
    meta['elapsed_seconds'] = time.time() - started
    with paths['results'].open('x', encoding='utf-8') as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    with paths['predictions'].open('xb') as handle:
        np.savez_compressed(handle, **arrays)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    gate = ap.add_mutually_exclusive_group()
    gate.add_argument('--check-only', action='store_true'); gate.add_argument('--zero-operator-gate', action='store_true')
    gate.add_argument('--run-development', action='store_true')
    ap.add_argument('--tag', default='round20_observation_operator_gru_development')
    args = ap.parse_args(); torch.set_num_threads(1); torch.set_num_interop_threads(1)
    if args.run_development:
        run_development(args.tag)
    elif args.zero_operator_gate:
        run_zero_gate()
    else:
        print(json.dumps(dict(checks=synthetic_checks(), source_sha256=REPLAY.sha256(__file__)), indent=2), flush=True)


if __name__ == '__main__':
    main()
