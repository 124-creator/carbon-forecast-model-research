"""Once-only fixed synthetic timing: not a market fit or evidence of accuracy."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from fit import fit_neural


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    output = args.root / "results" / "TIMING.json"
    scope = args.root / "provenance" / "TIMING_PROTOCOL.json"
    if not scope.exists() or output.exists():
        raise RuntimeError("Require frozen timing protocol and refuse repeated timing")
    protocol = json.loads(scope.read_text())
    for rel, expected in protocol["source_sha256"].items():
        if hashlib.sha256((args.root / rel).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Timing source changed: " + rel)
    rng = np.random.default_rng(8241)

    def make(n):
        x = rng.normal(size=(n, 20, 8))
        x[:, :, 1] = np.log(2.) / np.log(8.)
        scales = np.exp(rng.normal(0, .25, n))
        increments = rng.normal(size=(n, 20)) + .15 * x[:, -1, 0, None]
        increments[rng.uniform(size=increments.shape) < .08] = 0.
        y = np.cumsum(increments, axis=1)
        delta = y * scales[:, None]
        return dict(x=x, y=y, delta=delta, scale=scales, state=rng.integers(0, 9, n),
                    event=np.where(delta[:, 0] == 0, 0, np.where(delta[:, 0] < 0, 1, 2)),
                    origin=np.arange(n))

    train, val = make(1600), make(100)
    deadline = time.monotonic() + 600
    rows = []
    for family in protocol["families"]:
        _, receipt = fit_neural(train, val, family, .1, 17, max_epochs=3, deadline=deadline)
        rows.append(receipt)
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  protocol_sha256=hashlib.sha256(scope.read_bytes()).hexdigest(),
                  scope="Three fixed synthetic timing fits; no market data or selection",
                  market_fits=0, synthetic_fits=len(rows), total_optimizer_steps=sum(r["optimizer_steps"] for r in rows),
                  rows=rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"synthetic_fits": len(rows), "market_fits": 0,
                      "timings": [{k:r[k] for k in ("family", "wall_seconds", "optimizer_steps", "parameters")} for r in rows]}))


if __name__ == "__main__":
    main()
