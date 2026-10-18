"""Independent finite-matrix checks; no market data, neural model or author import.

Run: PYTHONPATH=carbon_deps python -B \
    scratch/new_round_20261018/transport_theory/check_transported_prior.py
The only input file read is this script's own bytes for its SHA-256 receipt.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def gram(x):
    return x.T @ x / len(x)


def transport(old_x, new_x, old_precision, strength):
    mapping = np.linalg.lstsq(old_x, new_x, rcond=None)[0]
    residual = new_x - old_x @ mapping
    precision = mapping.T @ old_precision @ mapping + strength * gram(residual)
    return mapping, residual, (precision + precision.T) / 2


def ridge(x, y, precision):
    return np.linalg.solve(gram(x) + precision, x.T @ y / len(x))


def invertible(rng, d):
    left, _ = np.linalg.qr(rng.normal(size=(d, d)))
    right, _ = np.linalg.qr(rng.normal(size=(d, d)))
    return left @ np.diag(np.linspace(0.4, 2.1, d)) @ right.T


class Checks:
    def __init__(self):
        self.rows = []
        self.numeric_comparisons = 0
        self.max_abs_error = 0.0

    def close(self, name, actual, expected, atol=5e-10, rtol=5e-10):
        a, b = np.asarray(actual), np.asarray(expected)
        assert a.shape == b.shape, (name, a.shape, b.shape)
        err = float(np.max(np.abs(a - b))) if a.size else 0.0
        self.numeric_comparisons += int(a.size)
        self.max_abs_error = max(self.max_abs_error, err)
        assert np.allclose(a, b, atol=atol, rtol=rtol), (name, err)
        self.rows.append({"name": name, "passed": True, "elements": int(a.size), "max_abs_error": err})

    def expect(self, name, condition, **facts):
        assert bool(condition), (name, facts)
        self.rows.append({"name": name, "passed": True, **facts})


def main():
    rng = np.random.default_rng(20261018)
    check = Checks()
    lam = 0.3
    n, d, nq, h = 72, 6, 13, 20

    for trial in range(8):
        x = rng.normal(size=(n, d)) @ np.diag(np.linspace(0.8, 2.0, d))
        q = rng.normal(size=(nq, d))
        y = rng.normal(size=(n, h))
        z = rng.normal(size=(d, d))
        precision = z.T @ z / d + 0.2 * np.eye(d)
        change = invertible(rng, d)

        a, e, transported = transport(x, x @ change, precision, lam)
        check.close(f"pure_gl_map_{trial}", a, change)
        check.close(f"pure_gl_residual_{trial}", e, np.zeros_like(e))
        check.close(f"pure_gl_prior_{trial}", transported, change.T @ precision @ change)
        old_head = ridge(x, y, precision)
        new_head = ridge(x @ change, y, transported)
        check.close(f"pure_gl_head_{trial}", new_head, np.linalg.solve(change, old_head))
        check.close(f"pure_gl_query_prediction_{trial}", q @ old_head, (q @ change) @ new_head)

        new_x = x @ invertible(rng, d) + 0.6 * rng.normal(size=(n, d))
        a, e, transported = transport(x, new_x, precision, lam)
        check.close(f"least_squares_orthogonality_{trial}", x.T @ e / n, np.zeros((d, d)))
        check.close(f"covariance_decomposition_{trial}", gram(new_x), a.T @ gram(x) @ a + gram(e))
        min_eig = float(np.linalg.eigvalsh(transported).min())
        check.expect(f"full_rank_spd_{trial}", min_eig > 1e-10, minimum_eigenvalue=min_eig)

        b, c = invertible(rng, d), invertible(rng, d)
        a2, e2, l2 = transport(x @ b, new_x @ c, b.T @ precision @ b, lam)
        check.close(f"two_frame_map_{trial}", a2, np.linalg.solve(b, a @ c))
        check.close(f"two_frame_residual_{trial}", e2, e @ c)
        check.close(f"two_frame_precision_{trial}", l2, c.T @ transported @ c)

    # Singular A is permitted when the novel residual spans all new columns.
    orthogonal, _ = np.linalg.qr(rng.normal(size=(n, 2 * d)))
    old_x = orthogonal[:, :d] * np.sqrt(n)
    new_x = orthogonal[:, d:] * np.sqrt(n)
    a, e, completed = transport(old_x, new_x, lam * np.eye(d), lam)
    check.close("orthogonal_subspace_mapping_zero", a, np.zeros_like(a))
    check.close("orthogonal_subspace_residual", e, new_x)
    check.close("orthogonal_subspace_completion", completed, lam * np.eye(d))
    check.expect("singular_mapping_completed_spd", np.linalg.eigvalsh(completed).min() > 0.29)
    uncompleted = a.T @ (lam * np.eye(d)) @ a
    check.expect("missing_residual_term_removes_prior_counterexample", np.linalg.norm(uncompleted) < 1e-25,
                 uncompleted_precision_norm=float(np.linalg.norm(uncompleted)))

    # The SPD guarantee explicitly excludes a rank-deficient new representation.
    rank_bad = new_x.copy()
    rank_bad[:, -1] = rank_bad[:, 0]
    _, _, bad_precision = transport(old_x, rank_bad, lam * np.eye(d), lam)
    smallest = float(np.linalg.eigvalsh(bad_precision).min())
    check.expect("rank_loss_not_claimed_spd", np.linalg.matrix_rank(rank_bad) == d - 1 and abs(smallest) < 1e-12,
                 minimum_eigenvalue=smallest, rank=int(np.linalg.matrix_rank(rank_bad)))

    # Generalized covariance penalty is exactly ordinary ridge in whitened coordinates.
    x0 = rng.normal(size=(n, d)) @ np.diag(np.linspace(0.7, 2.5, d))
    y = rng.normal(size=(n, h))
    q = rng.normal(size=(nq, d))
    eigenvalues, eigenvectors = np.linalg.eigh(gram(x0))
    white = eigenvectors @ np.diag(eigenvalues ** -0.5) @ eigenvectors.T
    covariance_head = ridge(x0, y, lam * gram(x0))
    whitened_head = ridge(x0 @ white, y, lam * np.eye(d))
    check.close("covariance_is_whitened_ridge", q @ covariance_head, q @ white @ whitened_head)
    check.close("covariance_is_scaled_ols", covariance_head,
                np.linalg.solve(gram(x0), x0.T @ y / n) / (1 + lam))

    # Full whitening plus Lambda_0=lambda I collapses to current covariance forever.
    current = x0 @ white
    precision = lam * np.eye(d)
    check.close("initial_full_whitening", gram(current), np.eye(d))
    for step in range(4):
        new = current @ invertible(rng, d) + 0.5 * rng.normal(size=(n, d))
        _, _, precision = transport(current, new, precision, lam)
        check.close(f"whitened_initialization_collapse_{step}", precision, lam * gram(new))
        current = new

    # Historical difference is a transported initial difference, on fixed support.
    current = x0.copy()
    initial_precision = lam * np.eye(d)
    precision = initial_precision.copy()
    cumulative = np.eye(d)
    initial_difference = initial_precision - lam * gram(x0)
    for step in range(4):
        new = current @ invertible(rng, d) + 0.35 * rng.normal(size=(n, d))
        previous_difference = precision - lam * gram(current)
        a, _, precision = transport(current, new, precision, lam)
        cumulative = cumulative @ a
        check.close(f"difference_recursion_{step}", precision - lam * gram(new), a.T @ previous_difference @ a)
        check.close(f"initial_difference_closed_form_{step}", precision,
                    lam * gram(new) + cumulative.T @ initial_difference @ cumulative)
        current = new
    difference_norm = float(np.linalg.norm(precision - lam * gram(current)))
    check.expect("nonwhitened_history_not_current_covariance", difference_norm > 1e-3,
                 difference_norm=difference_norm)

    # Different histories can change the metric with the same final representation.
    middle = x0 + 0.9 * rng.normal(size=x0.shape)
    _, _, middle_precision = transport(x0, middle, initial_precision, lam)
    _, _, return_precision = transport(middle, x0, middle_precision, lam)
    return_gap = float(np.linalg.norm(return_precision - initial_precision))
    check.expect("nonlinear_round_trip_changes_prior", return_gap > 1e-3, precision_gap=return_gap)
    prediction_gap = float(np.linalg.norm(q @ ridge(x0, y, return_precision) - q @ ridge(x0, y, initial_precision)))
    check.expect("same_final_features_can_change_predictions", prediction_gap > 1e-4,
                 prediction_gap=prediction_gap)

    # Pure coordinate round trips, unlike subspace round trips, are invariant.
    change = invertible(rng, d)
    _, _, midway = transport(x0, x0 @ change, initial_precision, lam)
    _, _, back = transport(x0 @ change, x0, midway, lam)
    check.close("pure_gl_round_trip_prior", back, initial_precision)

    # Scalar false gain from a fixed identity penalty is removed by transport.
    xs = rng.normal(size=(n, 1))
    xs /= np.sqrt(gram(xs)[0, 0])
    ys = xs.copy()
    pold = xs @ ridge(xs, ys, np.array([[lam]]))
    fixed_new = (2 * xs) @ ridge(2 * xs, ys, np.array([[lam]]))
    _, _, scalar_prior = transport(xs, 2 * xs, np.array([[lam]]), lam)
    moved_new = (2 * xs) @ ridge(2 * xs, ys, scalar_prior)
    fixed_gain = float(np.mean((ys - pold) ** 2) - np.mean((ys - fixed_new) ** 2))
    moved_gain = float(np.mean((ys - pold) ** 2) - np.mean((ys - moved_new) ** 2))
    check.expect("fixed_identity_false_scale_gain", fixed_gain > 0.01, synthetic_gain=fixed_gain)
    check.close("transport_eliminates_pure_scale_gain", np.array(moved_gain), np.array(0.0))

    # Support E=0 is not a promise that a nonlinear encoder's query values match.
    a, e, same_precision = transport(x0, x0.copy(), initial_precision, lam)
    check.close("support_same_prior", same_precision, initial_precision)
    new_q = q + 0.3
    query_gap = float(np.linalg.norm(new_q @ ridge(x0, y, same_precision) - q @ ridge(x0, y, initial_precision)))
    check.expect("support_zero_residual_does_not_imply_query_invariance", query_gap > 1e-3,
                 query_prediction_gap=query_gap)

    receipt = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "seed": 20261018,
        "checks": len(check.rows),
        "numeric_comparisons": check.numeric_comparisons,
        "maximum_absolute_comparison_error": check.max_abs_error,
        "scope": "Only generated finite matrices, analytic identities and synthetic linear solves",
        "market_files_read": 0,
        "confirmation_files_read": 0,
        "market_fits": 0,
        "neural_models_executed": 0,
        "author_core_executed": False,
        "findings": {
            "spd_requires_positive_previous_metric_positive_strength_and_full_new_column_rank": True,
            "pure_gl_invariance_requires_support_and_query_transform_together": True,
            "full_whitened_initialization_collapses_to_covariance_penalty": True,
            "history_difference_is_transport_of_initial_difference_on_fixed_support": True,
            "same_final_features_can_have_different_prior_due_to_history": True,
            "proof_is_not_novelty_or_market_effectiveness_certification": True,
        },
        "details": check.rows,
    }
    output = Path(__file__).with_name("TRANSPORT_SYNTHETIC_RECEIPT.json")
    if output.exists():
        raise FileExistsError("Preserve previous receipt; choose a new audit location for another execution")
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "details"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
