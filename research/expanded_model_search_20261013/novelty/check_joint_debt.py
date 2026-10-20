"""Algebra-only verification: no model fit, no prices, no market IO."""
from itertools import product
from pathlib import Path
import json
import numpy as np

def phi(records, bounds):
    groups = {}
    for tau, w, a, b in records:
        c, k = groups.get(tau, (0.0, 0.0))
        groups[tau] = (c + w * (a*a - b*b), k + w * (a-b))
    return sum(c - 2*k*(bounds[tau][0] if k >= 0 else bounds[tau][1])
               for tau, (c, k) in groups.items())

def psi(records, bounds):
    # Discard guaranteed future credits: never borrow against pending gains.
    endpoints = set(r[0] for r in records)
    return sum(max(0.0, phi([r for r in records if r[0] == tau], bounds))
               for tau in endpoints)

def debt(records, outcomes):
    return sum(w*((a-outcomes[tau])**2 - (b-outcomes[tau])**2)
               for tau, w, a, b in records)

def verify():
    rng = np.random.default_rng(20261013)
    max_error = 0.0
    strict = 0
    for _ in range(200):
        bounds = {i: (-1.0, 1.0) for i in range(3)}
        records = [(int(rng.integers(3)), float(rng.uniform(.1, 1)),
                    float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1)))
                   for _ in range(8)]
        worst = max(debt(records, dict(enumerate(corner)))
                    for corner in product([-1.0, 1.0], repeat=3))
        joint = phi(records, bounds)
        max_error = max(max_error, abs(worst-joint))
        if abs(worst-joint) > 1e-10:
            raise RuntimeError('Joint support formula disagrees with corners')
        separate = sum(phi([r], bounds) for r in records)
        if joint > separate + 1e-10:
            raise RuntimeError('Joint potential exceeds independent relaxation')
        strict += int(joint < separate - 1e-10)
        # When one endpoint becomes known, realized debt + remaining potential
        # cannot exceed the original worst-case pending debt.
        value = float(rng.uniform(-1, 1))
        realized = debt([r for r in records if r[0] == 0], {0: value})
        remainder = [r for r in records if r[0] != 0]
        if realized + phi(remainder, bounds) > joint + 1e-10:
            raise RuntimeError('Maturation increased certified budget')
        if realized + psi(remainder, bounds) > psi(records, bounds) + 1e-10:
            raise RuntimeError('Maturation increased positive debt budget')
        if psi(records, bounds) > sum(max(0, phi([r], bounds)) for r in records) + 1e-10:
            raise RuntimeError('Grouped positive debt exceeds independent debt')
    records = [(0, .5, .1, 0), (0, .5, -.1, 0)]
    result = {'scope': 'algebra identities only; no training, no prices',
              'cases': 200, 'corners_per_case': 8,
              'max_corner_formula_error': max_error,
              'strict_joint_relaxation_cases': strict,
              'example_joint': phi(records, {0: (-1, 1)}),
              'example_separate': sum(phi([r], {0: (-1, 1)}) for r in records),
              'all_checks_passed': True}
    Path(__file__).with_name('ALGEBRA_CHECK.json').write_text(
        json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    verify()
