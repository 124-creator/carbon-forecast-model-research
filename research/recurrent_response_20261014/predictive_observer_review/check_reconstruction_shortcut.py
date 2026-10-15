"""Analytical diagnostic, not a carbon experiment or author-code execution."""
import json
from pathlib import Path


def check():
    a, innovation_variance = 0.5, 1.0
    stationary_variance = innovation_variance / (1.0 - a * a)
    cases = []
    for b in (0.0, 0.5, 1.0):
        # Prior=b*y_previous, gain=1; posterior=y_current for every b.
        reconstruction_mse = 0.0
        forecast_mse = innovation_variance + (b - a) ** 2 * stationary_variance
        cases.append(dict(transition=b, reconstruction_mse=reconstruction_mse,
                          one_step_forecast_mse=forecast_mse))
    if len({row["reconstruction_mse"] for row in cases}) != 1:
        raise ValueError("Reconstruction invariance failed")
    if not cases[1]["one_step_forecast_mse"] < cases[0]["one_step_forecast_mse"]:
        raise ValueError("Forecast distinction failed")
    return dict(passed=True, scope="population AR(1) analytical counterexample",
                a=a, innovation_variance=innovation_variance, cases=cases,
                claim="Same-time perfect reconstruction does not identify predictive dynamics",
                not_claim="Does not invalidate KalmanNet, ECNN, RKN, or predictive training")


if __name__ == "__main__":
    result = check()
    print(json.dumps(result, ensure_ascii=False))
