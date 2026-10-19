"""Exact finite-lattice Gaussian variance-mixture propagation.

Independent mathematical implementation; no author code is copied.
The reset Markov kernel itself is established, not an innovation claim.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

Q = np.array([.25, 1., 4.])
PI = np.array([.45, .40, .15])
HORIZONS = (1, 5, 20)
SHIFT = (0, 1, 5)
C = np.sqrt(2 / np.pi)


def propagate(rho, horizons=HORIZONS, pi=PI):
    """Return exact endpoint weights and unit-one-step-variance sigmas.

    State variance Q[k]/E_pi[Q]; accumulated lattice index is
    n_middle + 5*n_high. All horizons share one initial pi and one kernel.
    """
    rho = np.asarray(rho, dtype=float).reshape(-1)
    pi = np.asarray(pi, dtype=float)
    if not np.all(np.isfinite(rho)) or not np.all(np.isfinite(pi)) or np.any((rho < 0) | (rho > 1)) or np.any(pi < 0) or abs(pi.sum()-1) > 1e-12:
        raise ValueError("Invalid probability")
    v = np.zeros((len(rho), 3, 6))
    for k, offset in enumerate(SHIFT):
        v[:, k, offset] = pi[k]
    result = {}
    for h in range(1, max(horizons)+1):
        if h in horizons:
            m = np.arange(5*h+1)
            result[h] = (v.sum(axis=1), np.sqrt((h*.25+.75*m)/(h*np.dot(pi, Q))))
        if h == max(horizons):
            break
        total = v.sum(axis=1)
        nxt = np.zeros((len(rho), 3, v.shape[-1]+5))
        for k, offset in enumerate(SHIFT):
            incoming = rho[:, None]*v[:, k, :] + (1-rho[:, None])*pi[k]*total
            nxt[:, k, offset:offset+v.shape[-1]] = incoming
        v = nxt
    return result


def abs_normal(y, sigma):
    y = np.asarray(y, dtype=float)[..., None]
    sigma = np.asarray(sigma, dtype=float)
    z = y/sigma
    return sigma*C*np.exp(-.5*z*z)+y*(2*ndtr(z)-1)


def pair_matrix(sigma):
    s = np.asarray(sigma)
    return C*np.sqrt(s[:, None]**2+s[None, :]**2)


def crps(y, weights, sigma):
    w = np.asarray(weights)
    return np.sum(w*abs_normal(y, sigma), axis=-1)-.5*np.einsum("...i,ij,...j->...", w, pair_matrix(sigma), w)


def population_risk(rho, truth_rho, horizons=HORIZONS, pi=PI):
    """Exact expected CRPS under this model, per condition and horizon."""
    pred, truth = propagate(rho, horizons, pi), propagate(truth_rho, horizons, pi)
    result = []
    for h in horizons:
        w, s = pred[h]
        v = truth[h][0]
        b = pair_matrix(s)
        result.append(np.einsum("bi,ij,bj->b", w, b, v)-.5*np.einsum("bi,ij,bj->b", w, b, w))
    return np.stack(result, axis=-1)


def sample_paths(rng, rho, h=20, pi=PI, leverage=0.):
    """Independent forecast-origin paths; optional deliberate misspecification.

    leverage>0 changes step j's scale according to step j-1's sign.
    It is outside the frozen constant-emission forecast family.
    """
    rho = np.asarray(rho).reshape(-1)
    n = len(rho)
    states = rng.choice(3, size=n, p=pi)
    values = np.empty((n, h))
    sigmas = np.sqrt(Q/np.dot(pi, Q))
    for j in range(h):
        if j:
            reset = rng.random(n) >= rho
            states[reset] = rng.choice(3, size=int(reset.sum()), p=pi)
        scale = sigmas[states].copy()
        if leverage and j:
            scale *= np.exp(-leverage*np.sign(values[:, j-1]))/np.sqrt(np.cosh(2*leverage))
        values[:, j] = scale*rng.normal(size=n)
    return values


def path_emissions(path, pi=PI, stable=False):
    s = np.sqrt(Q/np.dot(pi, Q))
    z = np.asarray(path)[:, :, None]/s
    log_density = -.5*z*z-np.log(np.sqrt(2*np.pi)*s)
    if not np.all(np.isfinite(log_density)):
        raise ValueError("Nonfinite path log emission")
    if stable:
        offset = log_density.max(axis=-1, keepdims=True)
        # Fourth channel stores the removed log density. At least one of the
        # three shifted emission densities is exactly one at every step.
        return np.concatenate([np.exp(log_density-offset), offset], axis=-1)
    return np.exp(log_density)


def path_nll_from_emissions(emissions, rho, pi=PI):
    """Forward HMM likelihood, average negative log density per time step."""
    n, h, channels = emissions.shape
    if channels not in (3, 4):
        raise ValueError("Three raw emissions or three shifted emissions plus log offset required")
    if not (0 <= rho <= 1) or not np.all(np.isfinite(emissions)):
        raise ValueError("Invalid emissions or persistence")
    predicted = np.broadcast_to(pi, (n, 3)).copy()
    nll = np.zeros(n)
    for j in range(h):
        joint = predicted*emissions[:, j, :3]
        denom = joint.sum(axis=1)
        if np.any(denom <= 0):
            raise FloatingPointError("HMM likelihood underflow; no clipping to a false density is permitted")
        nll -= np.log(denom)
        if channels == 4:
            nll -= emissions[:, j, 3]
        posterior = joint/denom[:, None]
        predicted = rho*posterior+(1-rho)*pi
    return nll/h


def torch_propagate(rho, horizons=HORIZONS, pi=PI):
    """Differentiable exact version used for the gradient identity audit."""
    import torch
    import torch.nn.functional as F
    rho = rho.reshape(-1)
    p = torch.as_tensor(pi, dtype=rho.dtype, device=rho.device)
    v = torch.stack([F.pad(torch.ones_like(rho)[:, None]*p[k], (offset, 5-offset))
                     for k, offset in enumerate(SHIFT)], dim=1)
    result = {}
    for h in range(1, max(horizons)+1):
        if h in horizons:
            m = torch.arange(5*h+1, dtype=rho.dtype, device=rho.device)
            s = torch.sqrt((h*.25+.75*m)/(h*torch.dot(p, torch.as_tensor(Q, dtype=rho.dtype, device=rho.device))))
            # Explicit zero term preserves an exactly-zero differentiable h=1 route.
            result[h] = (v.sum(dim=1)+rho[:, None]*0., s)
        if h == max(horizons):
            break
        total = v.sum(dim=1)
        v = torch.stack([F.pad(rho[:, None]*v[:, k, :]+(1-rho[:, None])*p[k]*total,
                              (offset, 5-offset)) for k, offset in enumerate(SHIFT)], dim=1)
    return result
