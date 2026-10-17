#!/usr/bin/env python3
"""Check the recorded Linux CPU runtime without installing packages or reading data."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional new JSON receipt; an existing file is refused")
    args = parser.parse_args()
    if args.output is not None and (args.output.exists() or args.output.is_symlink()):
        raise FileExistsError("Environment receipt already exists")
    root = Path(__file__).resolve().parent
    lock = json.loads((root / "ENVIRONMENT_LOCK.json").read_text(encoding="utf-8"))
    observed = {"python": platform.python_version(), "system": platform.system(),
                "machine": platform.machine(), "packages": {}}
    failures = []
    for field in ("python", "system", "machine"):
        if observed[field] != lock[field]:
            failures.append(f"{field}: expected {lock[field]}, observed {observed[field]}")
    for name, expected in lock["packages"].items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        observed["packages"][name] = actual
        if actual != expected:
            failures.append(f"{name}: expected {expected}, observed {actual}")
    if not failures:
        # Disable model-hub communication before importing the author dependency.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[name] = "1"
        import numpy as np
        import torch
        import transformers
        observed["imported_versions"] = {"numpy": np.__version__, "torch": torch.__version__,
                                         "transformers": transformers.__version__}
        for name, actual in observed["imported_versions"].items():
            if actual != lock["packages"][name]:
                failures.append(f"Actual import for {name} differs from the recorded version")
        if torch.version.cuda is not None:
            failures.append("Expected the recorded CPU torch build")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        values = np.arange(4, dtype=np.float64)
        tensor = torch.from_numpy(values)
        observed["numpy_torch_float64_bridge"] = bool(np.array_equal(tensor.numpy(), values))
        if not observed["numpy_torch_float64_bridge"]:
            failures.append("NumPy/PyTorch float64 bridge failed")
        observed["torch_math_threads"] = torch.get_num_threads()
    result = {"passed": not failures, "observed": observed, "failures": failures,
              "package_installations": 0, "network_requests": 0, "market_fits": 0,
              "scope": "Recorded package versions and synthetic tensor bridge; no data or model fitting"}
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        sys.exit(2)


if __name__ == "__main__":
    main()
