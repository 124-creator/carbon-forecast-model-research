"""Source identity, scalar walk-forward and future perturbation, no market data."""
from pathlib import Path
from types import SimpleNamespace
import ast
import hashlib
import json

import numpy as np

from statistical_reference import (reference, reference_at, anchor_arrays,
                                   SOURCE_GIT_BLOB, SOURCE_FILE_SHA256, SOURCE_COMMIT)

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
SOURCE = WORK / "carbon_workspace/research/recurrent_response_20261014/fair_recurrent_tuning_20261016/fair_core.py"
COPY = HERE / "statistical_reference.py"


def function_text(path):
    text = path.read_text()
    tree = ast.parse(text)
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "reference")
    return "".join(text.splitlines(keepends=True)[node.lineno-1:node.end_lineno])


class GuardedPrices:
    """Refuses any numeric read beyond the declared observed prefix."""
    def __init__(self, values, last_allowed):
        self.values, self.last_allowed, self.read_stops = values, last_allowed, []

    def __len__(self):
        return len(self.values)

    def __getitem__(self, key):
        assert isinstance(key, slice) and key.start is None and key.step is None
        assert key.stop is not None and key.stop <= self.last_allowed + 1
        self.read_stops.append(int(key.stop))
        return self.values[key]


def main():
    source = SOURCE.read_bytes()
    source_sha = hashlib.sha256(source).hexdigest()
    blob_sha = hashlib.sha1(b"blob " + str(len(source)).encode() + b"\0" + source).hexdigest()
    assert source_sha == SOURCE_FILE_SHA256 and blob_sha == SOURCE_GIT_BLOB
    original_function = function_text(SOURCE)
    assert original_function == function_text(COPY), "function source changed"
    namespace = {"np": np}
    exec(compile(original_function, str(SOURCE), "exec"), namespace)
    original_reference = namespace["reference"]
    rng = np.random.default_rng(9187)
    price = 100 + rng.normal(scale=.5, size=620).cumsum()
    price[45] = price[44]
    base, context = reference(price)
    checks, max_error = [], 0.
    for n in (1, 7, 60, 61, 81, 85, 241, 302, 620):
        a, c = original_reference(price[:n])
        b, d = reference(price[:n])
        error = max(float(np.max(abs(a-b))), float(np.max(abs(c-d))))
        assert error == 0.
        max_error = max(max_error, error)
        checks.append(dict(kind="original_function_equivalence", prefix=n, maximum_error=error))
    for origin in (0, 59, 60, 79, 80, 239, 240, 299, 500):
        guard = GuardedPrices(price, origin)
        a, c = reference_at(guard, origin)
        assert np.array_equal(a, base[origin]) and np.array_equal(c, context[origin])
        assert guard.read_stops == [origin+1]
        altered = price.copy()
        altered[origin+1:] += 10000 + np.arange(len(price)-origin-1) ** 2
        altered_base, altered_context = reference(altered)
        assert np.array_equal(altered_base[:origin+1], base[:origin+1])
        assert np.array_equal(altered_context[:origin+1], context[:origin+1])
        checks.append(dict(kind="scalar_and_future_perturbation", origin=origin,
                           only_prefix_requested=True, maximum_error=0.))
    price_b = 20 + rng.normal(scale=.1, size=500).cumsum()
    market_a = SimpleNamespace(name="a", price=GuardedPrices(price, 310))
    market_b = SimpleNamespace(name="b", price=GuardedPrices(price_b, 260))
    data = dict(market_names=["b", "a"], market=np.array([1,0,1,0,1]),
                origin=np.array([80,260,310,60,80]))
    anchors = anchor_arrays(data, [market_a, market_b])
    expected = np.stack([reference_at(price if mid == 1 else price_b, int(t))[0]
                         for mid,t in zip(data["market"], data["origin"])])
    assert np.array_equal(anchors, expected)
    assert market_a.price.read_stops == [311] and market_b.price.read_stops == [261]
    receipt = dict(status="PASS", market_reads=0, market_fits=0, optimizer_steps=0,
                   source_commit=SOURCE_COMMIT, source_git_blob=blob_sha, source_sha256=source_sha,
                   copied_function_sha256=hashlib.sha256(original_function.encode()).hexdigest(),
                   module_sha256=hashlib.sha256(COPY.read_bytes()).hexdigest(),
                   source_function_exact_text_equal=True, numpy_version=np.__version__,
                   maximum_absolute_error=max_error, prefix_and_scalar_checks=checks,
                   mixed_market_row_order_and_duplicate_origins=True,
                   guarded_anchor_prefix_stops={"a":311,"b":261},
                   output_units="raw cumulative price changes, shape N x 20",
                   note="Existing project reference; no algorithmic or market-fit changes.")
    path = HERE / "STATISTICAL_REFERENCE_CHECK.json"
    path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(dict(status="PASS", module_sha256=receipt["module_sha256"],
                         source_exact=True, max_error=max_error, receipt=str(path))))


if __name__ == "__main__":
    main()
