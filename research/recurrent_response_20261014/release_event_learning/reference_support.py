"""Own audited reference/support math, unchanged from prequential_state_learning/train_state.py."""
import numpy as np
import torch


def configure():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    torch.use_deterministic_algorithms(True)


def reference(price):
    bank = [(0., 1.)] + [(r, q) for r in (0., .25, .5, .75, .9, .97) for q in (.02, .1, .25, .5, .8)]
    rho, permanent = np.array(bank).T
    transitory = 1 - permanent
    n = len(price)
    state = np.zeros(len(bank))
    variance = transitory / (1 - rho ** 2)
    states = np.zeros((n, len(bank)))
    errors = np.zeros_like(states)
    innovations = np.ones_like(states)
    for t in range(1, n):
        error = price[t] - price[t - 1] - (rho - 1) * state
        innovation = (rho - 1) ** 2 * variance + 1
        covariance = rho * (rho - 1) * variance + transitory
        state = rho * state + covariance * error / innovation
        variance = np.maximum(rho ** 2 * variance + transitory - covariance ** 2 / innovation, 0)
        states[t], errors[t], innovations[t] = state, error, innovation
    base = np.zeros((n, 20))
    context = np.zeros((n, 3))
    selected = 0
    for t in range(n):
        if t >= 60 and (t - 60) % 20 == 0:
            start = max(1, t - 240)
            scale = np.maximum(np.mean(errors[start:t] ** 2 / innovations[start:t], axis=0), 1e-12)
            criterion = np.log(innovations[start:t]).sum(0) + (t - start) * np.log(scale)
            selected = int(np.argmin(criterion))
        r, q = bank[selected]
        context[t] = [states[t, selected], r * (1 - q), 1 - q]
        base[t] = (r ** np.arange(1, 21) - 1) * states[t, selected]
    return base, context


def support(price, end):
    origins = np.arange(max(60, end - 20 - 480), end - 20)
    boundary = len(origins) - 60
    train = np.arange(boundary - 20)
    validation = np.arange(boundary, len(origins))
    if len(train) < 60 or origins[train[-1]] + 20 >= origins[validation[0]] or origins[-1] + 20 >= end:
        raise ValueError('Maturity or purging violation')
    windows = np.stack([np.diff(price[t - 60:t + 1]) for t in origins])
    scale = max(float(windows[train].std()), 1e-6)
    target = np.array([price[t + 1:t + 21] - price[t] for t in origins])
    return origins, train, validation, scale, target
