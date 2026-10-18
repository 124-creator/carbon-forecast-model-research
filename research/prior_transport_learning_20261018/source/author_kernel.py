"""Execute unchanged R2D2 ridge-method AST with a modern CPU solve shim.

This is a numerical-kernel task adaptation, not execution of the original image
benchmark, CNN, class-label loader, calibration layer or CUDA training wrapper.
The original source file is distributed unchanged under its MIT license.
"""
from __future__ import annotations
import ast
import hashlib
from pathlib import Path
import numpy as np
import torch

SOURCE = Path(__file__).resolve().parents[1]/'vendor/r2d2/fewshots/models/r2d2.py'
EXPECTED_SHA256 = '332ce2d9ad06bde4831b3099ec240ffd89a6378ac59d59fe28f313f2cfef5277'
_methods = None
_calls = 0


def _load():
    global _methods
    if _methods is not None:
        return _methods
    raw = SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_SHA256:
        raise ValueError('Unchanged fixed author source identity required')
    root = ast.parse(raw.decode())
    cls = next(n for n in root.body if isinstance(n, ast.ClassDef) and n.name == 'RRNet')
    wanted = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('rr_standard', 'rr_woodbury')]
    # Nodes, argument lists and function bodies are not changed. Only the
    # enclosing legacy training package is omitted from execution.
    module = ast.Module(body=wanted, type_ignores=[])
    def legacy_gesv(rhs, matrix):
        return torch.linalg.solve(matrix, rhs), None  # unused LU return
    scope = {'np': np, 'torch': torch, 'mm': torch.mm, 'inv': torch.inverse,
             't': torch.transpose, 't_': lambda x: torch.transpose(x, 0, 1),
             'gesv': legacy_gesv}
    exec(compile(module, str(SOURCE), 'exec'), scope)
    _methods = {name: scope[name] for name in ('rr_standard', 'rr_woodbury')}
    return _methods


def author_ridge(x, y, lambda0=.1, dual=False):
    global _calls
    methods = _load()
    class Context:
        n_augment = 1
        @staticmethod
        def lambda_rr(identity):
            return lambda0 * identity
    n, d = x.shape
    identity = torch.eye(n if dual else d, dtype=x.dtype, device=x.device)
    # The author's kernel scales x in place; clone preserves the caller's graph
    # and its source features. y has the same sqrt(n) normalization as author.
    result = methods['rr_woodbury' if dual else 'rr_standard'](
        Context(), x.clone(), 1, n, identity, y / np.sqrt(n), True)
    _calls += 1
    return result


def call_count():
    return _calls


def synthetic_check():
    torch.set_num_threads(1)
    gen = torch.Generator().manual_seed(18018)
    x = torch.randn(71, 8, generator=gen, dtype=torch.float64, requires_grad=True)
    y = torch.randn(71, 20, generator=gen, dtype=torch.float64)
    a = author_ridge(x, y)
    b = torch.linalg.solve(x.T@x/len(x)+.1*torch.eye(8,dtype=x.dtype), x.T@y/len(x))
    c = author_ridge(x, y, dual=True)
    ga = torch.autograd.grad(a.square().sum(), x, retain_graph=True)[0]
    gb = torch.autograd.grad(b.square().sum(), x)[0]
    checks = {'primal_error': float((a-b).abs().max().detach()),
              'dual_error': float((a-c).abs().max().detach()),
              'gradient_error': float((ga-gb).abs().max().detach()),
              'author_calls': call_count(), 'source_sha256': EXPECTED_SHA256,
              'scope': 'unchanged author rr_standard/rr_woodbury numerical kernels, forward and backward; no original benchmark executed'}
    checks['pass'] = max(checks[k] for k in ('primal_error','dual_error','gradient_error')) < 1e-10
    if not checks['pass']:
        raise AssertionError(checks)
    return checks


if __name__ == '__main__':
    import json
    print(json.dumps(synthetic_check(), indent=2))
