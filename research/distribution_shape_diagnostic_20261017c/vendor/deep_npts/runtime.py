"""Load fixed, unmodified GluonTS mathematical cores without its estimator.

The local adapter replaces only GluonTS constructor metadata validation and
import plumbing. It does not replace the author network or distribution math.
Upstream bytes are verified against SOURCE_MANIFEST before execution. No package
is installed, and no global gluonts module or import hook is registered.
"""
from __future__ import annotations

import builtins
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace


def _validated(*args, **kwargs):
    """Configuration/serialization-only adapter; explicit constructors retained."""
    def decorate(function):
        return function
    return decorate


@lru_cache(maxsize=1)
def load_author():
    here = Path(__file__).resolve().parent
    manifest = json.loads((here / "SOURCE_MANIFEST.json").read_text())
    for item in manifest["files"]:
        data = (here / item["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise RuntimeError("Frozen author file changed: " + item["path"])

    def simple_module(name, filename):
        spec = importlib.util.spec_from_file_location(name, here / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    distribution = simple_module("_shape_author_distribution", "discrete_distribution.py")
    scaling = simple_module("_shape_author_scaling", "scaling.py")

    def local_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name == "gluonts.core.component":
            return SimpleNamespace(validated=_validated)
        if level == 0 and name == "gluonts.torch.distributions":
            return SimpleNamespace(DiscreteDistribution=distribution.DiscreteDistribution)
        if level == 1 and name == "scaling":
            return scaling
        return builtins.__import__(name, globals, locals, fromlist, level)

    network = ModuleType("_shape_author_network")
    network.__file__ = str(here / "_network.py")
    namespace_builtins = dict(vars(builtins))
    namespace_builtins["__import__"] = local_import
    network.__dict__["__builtins__"] = namespace_builtins
    exec(compile((here / "_network.py").read_bytes(), network.__file__, "exec"), network.__dict__)
    return SimpleNamespace(network=network, distribution=distribution, scaling=scaling)
