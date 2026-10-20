"""Signed Markov Propagation: a conditional structural forecasting candidate.

The Markov reward mean recursion is standard mathematics, not a new theorem.
This module holds the GRU context fixed within each issued forecast while the
observable future sign/zero-age state evolves. The GRU is trainable offline.

Units: outputs and nonzero mark scales are cumulative price changes divided by
the caller's known origin scale s_t; there is no sqrt(h) factor here. The caller
owns raw-price restoration, risk weights, causal samples, and training support.

State index = (last_nonzero_sign + 1) * 3 + min(zero_age, 2).
Event order = zero, negative, positive. Mark order = negative, positive.
The one_step variant is exactly the full model: selecting the h=1 training
objective is exclusively the caller's responsibility.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


VARIANTS = ('full', 'no_clock', 'no_sign', 'pooled_mark', 'iid_state', 'one_step')
MIN_MARK_SCALE = 1e-4
NEXT_ZERO = (1, 2, 2, 4, 5, 5, 7, 8, 8)
NEXT_NEGATIVE = 0
NEXT_POSITIVE = 6


def _state_tensor(observed_state: Tensor, batch: int, device: torch.device) -> Tensor:
    state = torch.as_tensor(observed_state, device=device)
    if state.ndim == 2 and state.shape[1] == 1:
        state = state[:, 0]
    if state.ndim != 1 or state.shape[0] != batch:
        raise ValueError('observed_state must have shape [B] or [B,1].')
    if state.dtype == torch.bool or state.dtype.is_floating_point or state.dtype.is_complex:
        raise TypeError('observed_state must use an integer dtype.')
    state = state.to(dtype=torch.long)
    if bool(((state < 0) | (state > 8)).any()):
        raise ValueError('observed_state must lie in {0,...,8}.')
    return state


def propagate_mean(q: Tensor, b: Tensor, observed_state: Tensor, horizon: int = 20) -> Tensor:
    """Exact finite-state mean recursion, differentiable in q and b.

    q: [B,9,3] row probabilities in zero/negative/positive order.
    b: [B,9,2] strictly positive negative/positive magnitude means.
    Returns [B,H] cumulative-change means from each observed initial state.

    Callers supplying q/b directly are responsible for valid probabilities and
    positive scales. This function checks structural shapes and state indices;
    it does not detach or silently renormalize parameters.
    """
    if q.ndim != 3 or q.shape[1:] != (9, 3):
        raise ValueError('q must have shape [B,9,3].')
    if b.ndim != 3 or b.shape != (q.shape[0], 9, 2):
        raise ValueError('b must have shape [B,9,2] with the same B as q.')
    if q.device != b.device or q.dtype != b.dtype:
        raise ValueError('q and b must have matching device and dtype.')
    if not q.dtype.is_floating_point:
        raise TypeError('q and b must use floating point tensors.')
    if isinstance(horizon, bool) or int(horizon) != horizon or int(horizon) < 1:
        raise ValueError('horizon must be a positive integer.')
    state = _state_tensor(observed_state, q.shape[0], q.device)
    next_zero = torch.tensor(NEXT_ZERO, dtype=torch.long, device=q.device)
    # F_0 is zero. The immediate expected reward is zero at symmetric init.
    reward = q[:, :, 2] * b[:, :, 1] - q[:, :, 1] * b[:, :, 0]
    future = torch.zeros_like(reward)
    forecasts = []
    for _ in range(int(horizon)):
        future = (
            reward
            + q[:, :, 0] * future.index_select(1, next_zero)
            + q[:, :, 1] * future[:, NEXT_NEGATIVE:NEXT_NEGATIVE + 1]
            + q[:, :, 2] * future[:, NEXT_POSITIVE:NEXT_POSITIVE + 1]
        )
        forecasts.append(future.gather(1, state[:, None])[:, 0])
    return torch.stack(forecasts, dim=1)


def _group_indices(variant: str):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown variant {variant!r}; expected one of {VARIANTS}.')
    if variant == 'no_clock':
        groups = [s // 3 for s in range(9)]
        return groups, groups
    if variant == 'no_sign':
        groups = [s % 3 for s in range(9)]
        return groups, groups
    if variant == 'pooled_mark':
        return list(range(9)), [0] * 9
    if variant == 'iid_state':
        return [0] * 9, [0] * 9
    return list(range(9)), list(range(9))


class SMPModel(nn.Module):
    """Shared current-history GRU plus genuinely tied source-state heads.

    no_clock: q and b each have three free groups, one per last sign.
    no_sign: q and b each have three free groups, one per zero-age bucket.
    pooled_mark: q has nine free groups; b has one free group.
    iid_state: q and b each have one free group.
    full / one_step: q and b each have nine free groups.

    All variants receive identical x and observed_state. Head tying constrains
    the future state mechanism, not the information available at the origin.
    """
    def __init__(self, input_dim: int, hidden_size: int = 32, horizon: int = 20,
                 variant: str = 'full'):
        super().__init__()
        if input_dim < 1 or hidden_size < 1 or horizon < 1:
            raise ValueError('input_dim, hidden_size, and horizon must be positive.')
        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden_size)
        self.horizon = int(horizon)
        self.variant = variant
        q_groups, b_groups = _group_indices(variant)
        self.n_q_groups = max(q_groups) + 1
        self.n_b_groups = max(b_groups) + 1
        self.register_buffer('q_group_index', torch.tensor(q_groups, dtype=torch.long))
        self.register_buffer('b_group_index', torch.tensor(b_groups, dtype=torch.long))
        self.encoder = nn.GRU(self.input_dim, self.hidden_size, num_layers=1,
                              batch_first=True, dropout=0.0)
        self.q_head = nn.Linear(self.hidden_size, 3 * self.n_q_groups)
        self.b_head = nn.Linear(self.hidden_size, 2 * self.n_b_groups)
        self.initialize_head(q0=0.5, mean_abs_nonzero=1.0)

    @torch.no_grad()
    def initialize_head(self, q0: float, mean_abs_nonzero: float) -> dict[str, float]:
        """Zero head weights, symmetric signs, common b, and exact RW means.

        q0 is clipped to [.001,.95]. A requested magnitude at or below the
        softplus floor is represented by floor+1e-8; this adjustment is returned
        explicitly. This routine never changes the GRU initialization.
        """
        if not math.isfinite(float(q0)) or not math.isfinite(float(mean_abs_nonzero)):
            raise ValueError('Initialization statistics must be finite.')
        if mean_abs_nonzero < 0:
            raise ValueError('mean_abs_nonzero must be nonnegative.')
        clipped_q0 = min(0.95, max(0.001, float(q0)))
        magnitude = max(float(mean_abs_nonzero), MIN_MARK_SCALE + 1e-8)
        logits = self.q_head.bias.new_tensor([
            math.log(clipped_q0),
            math.log((1.0 - clipped_q0) / 2.0),
            math.log((1.0 - clipped_q0) / 2.0),
        ])
        shifted = magnitude - MIN_MARK_SCALE
        inverse_softplus = shifted + math.log(-math.expm1(-shifted))
        self.q_head.weight.zero_()
        self.b_head.weight.zero_()
        self.q_head.bias.copy_(logits.repeat(self.n_q_groups))
        self.b_head.bias.fill_(inverse_softplus)
        return {
            'requested_q0': float(q0), 'applied_q0': clipped_q0,
            'requested_mean_abs_nonzero': float(mean_abs_nonzero),
            'applied_mean_abs_nonzero': magnitude,
        }

    def forward(self, x: Tensor, observed_state: Tensor, return_aux: bool = False):
        if x.ndim != 3 or x.shape[2] != self.input_dim or x.shape[1] < 1:
            raise ValueError('x must have shape [B,L,input_dim], with L >= 1.')
        state = _state_tensor(observed_state, x.shape[0], x.device)
        _, hidden = self.encoder(x)
        context = hidden[-1]
        logits = self.q_head(context).reshape(-1, self.n_q_groups, 3)
        log_q_free = F.log_softmax(logits, dim=-1)
        q_free = log_q_free.exp()
        b_free = F.softplus(self.b_head(context).reshape(-1, self.n_b_groups, 2)) + MIN_MARK_SCALE
        q = q_free.index_select(1, self.q_group_index)
        log_q = log_q_free.index_select(1, self.q_group_index)
        b = b_free.index_select(1, self.b_group_index)
        prediction = propagate_mean(q, b, state, self.horizon)
        if not return_aux:
            return prediction
        rows = torch.arange(x.shape[0], device=x.device)
        aux = {
            'q': q, 'b': b,
            'q_current': q[rows, state],
            'b_current': b[rows, state],
            'log_q': log_q,
            'log_q_current': log_q[rows, state],
        }
        return prediction, aux


def hurdle_nll(aux: dict[str, Any], next_delta_over_s: Tensor,
               reduction: str = 'none', event: Tensor | None = None) -> Tensor:
    """One-step signed exponential hurdle negative log likelihood.

    zero: -log(q0)
    negative/positive: -log(q_sign) + log(b_sign) + abs(delta/s)/b_sign

    The caller should supply event computed from the original price delta using
    its fixed 1e-10 rule. If omitted, exact nonzero is used in these normalized
    units; no raw-price threshold is silently applied after normalization.
    """
    q = aux['q_current']
    b = aux['b_current']
    if q.ndim != 2 or q.shape[1] != 3 or b.shape != (q.shape[0], 2):
        raise ValueError('aux current probabilities/scales must be [B,3]/[B,2].')
    delta = torch.as_tensor(next_delta_over_s, dtype=q.dtype, device=q.device)
    if delta.ndim == 2 and delta.shape[1] == 1:
        delta = delta[:, 0]
    if delta.ndim != 1 or delta.shape[0] != q.shape[0]:
        raise ValueError('next_delta_over_s must have shape [B] or [B,1].')
    if event is None:
        event_mask = delta != 0
    else:
        event_mask = torch.as_tensor(event, dtype=torch.bool, device=q.device)
        if event_mask.ndim == 2 and event_mask.shape[1] == 1:
            event_mask = event_mask[:, 0]
        if event_mask.shape != delta.shape:
            raise ValueError('event must have shape [B] or [B,1].')
        if bool((event_mask & (delta == 0)).any()):
            raise ValueError('An event with exactly zero normalized delta has no sign.')
    mark_index = (delta > 0).to(torch.long)
    event_index = mark_index + 1
    log_q = aux.get('log_q_current')
    if log_q is None:
        log_q = q.clamp_min(torch.finfo(q.dtype).tiny).log()
    if log_q.shape != q.shape:
        raise ValueError('log_q_current must have the same shape as q_current.')
    chosen_log_q = log_q.gather(1, event_index[:, None])[:, 0]
    chosen_b = b.gather(1, mark_index[:, None])[:, 0]
    nonzero_loss = -chosen_log_q + chosen_b.log() + delta.abs() / chosen_b
    loss = torch.where(event_mask, nonzero_loss, -log_q[:, 0])
    if reduction == 'none':
        return loss
    if reduction == 'mean':
        return loss.mean()
    if reduction == 'sum':
        return loss.sum()
    raise ValueError("reduction must be 'none', 'mean', or 'sum'.")
