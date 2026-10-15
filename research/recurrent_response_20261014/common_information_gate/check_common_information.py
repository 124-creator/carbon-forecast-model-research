"""Exact finite-population counterexample; not a carbon forecast experiment."""
import itertools
import json
import numpy as np


def check() -> dict:
    values = np.array(list(itertools.product([-1., 1.], repeat=4)))
    a, b, c, noise = values.T
    target = 100 + a + b + c + noise
    projections = {}
    for name, information in [('past', values[:, :2]), ('current', values[:, 1:3]), ('common', values[:, 1:2])]:
        _, groups = np.unique(information, axis=0, return_inverse=True)
        count = np.bincount(groups)
        projections[name] = (groups[:, None] == groups[None, :]) / count[groups][:, None]
    old = projections['past'] @ target
    new = projections['current'] @ target
    revision = new - old
    common = float(np.mean((projections['common'] @ revision) ** 2))
    past = float(np.mean((projections['past'] @ revision) ** 2))
    path = float(np.mean(revision ** 2))
    risk = float(np.mean(np.concatenate([(old - target) ** 2, (new - target) ** 2])))
    if common != 0 or past != 1 or path != 2 or risk != 2:
        raise ValueError('Finite-population oracle check failed')
    n = len(values)
    previous_features = np.column_stack([a, b, a * 0, a * 0])
    current_features = np.column_stack([a * 0, a * 0, b, c])
    design = np.vstack([previous_features, current_features])
    anchors = np.concatenate([100 + b, 100 + c])
    response = np.tile(target, 2) - anchors
    difference = current_features - previous_features
    offset = c - b
    gram = design.T @ design / (2 * n)
    rhs = design.T @ response / (2 * n)
    cases = {}
    for name, projection in [('path', np.eye(n)), ('past', projections['past']), ('common', projections['common'])]:
        # Fixed penalty coefficient 1; include the observed-price anchor offset.
        dp, op = projection @ difference, projection @ offset
        coefficients = np.linalg.solve(gram + dp.T @ dp / n, rhs - dp.T @ op / n)
        predictions = anchors + design @ coefficients
        cases[name] = dict(coefficients=coefficients.tolist(),
            prediction_mse=float(np.mean((predictions - np.tile(target, 2)) ** 2)),
            conditional_revision_squared=float(np.mean((dp @ coefficients + op) ** 2)))
    common_design = projections['common'] @ difference
    common_offset = projections['common'] @ offset
    augmented = np.vstack([design / np.sqrt(2*n), common_design / np.sqrt(n)])
    augmented_response = np.concatenate([response / np.sqrt(2*n), -common_offset / np.sqrt(n)])
    augmented_coefficients = np.linalg.lstsq(augmented, augmented_response, rcond=None)[0]
    equivalence_error = float(np.max(np.abs(augmented_coefficients - cases['common']['coefficients'])))
    omitted = np.linalg.solve(gram + common_design.T @ common_design / n, rhs)
    omitted_risk = float(np.mean((anchors + design @ omitted - np.tile(target, 2)) ** 2))
    if equivalence_error > 1e-12 or not np.isclose(cases['common']['prediction_mse'], risk) or omitted_risk <= risk:
        raise ValueError('Affine penalty equivalence or offset negative control failed')
    constant_coefficients = np.array([0., -1., 0., -1.])
    constant_forecasts = anchors + design @ constant_coefficients
    constant_penalty = float(np.mean((common_design @ constant_coefficients + common_offset) ** 2))
    constant_risk = float(np.mean((constant_forecasts - np.tile(target, 2)) ** 2))
    if constant_penalty != 0 or constant_risk != 4:
        raise ValueError('Moment-only identification negative control failed')
    return dict(passed=True, scope='exact 16-point population, not sampled market data',
                oracle_common_conditional_squared=common,
                oracle_full_past_conditional_squared=past,
                oracle_pathwise_revision_squared=path,
                oracle_prediction_mse=risk, penalty_cases=cases,
                augmented_coefficient_max_error=equivalence_error,
                omitted_anchor_prediction_mse=omitted_risk,
                constant_predictor_common_penalty=constant_penalty,
                constant_predictor_mse=constant_risk, market_fits=0,
                novelty='Not established; fixed-feature objective is affine quadratic regularization',
                limitations=['Oracle finite-window example, not fitted GRU or full-history sufficiency',
                             'No guarantee of market improvement, conditional-moment estimation, or independence'])


if __name__ == '__main__':
    print(json.dumps(check(), ensure_ascii=False))
