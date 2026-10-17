# PROCEED source and license provenance

The 16 upstream files listed in `SOURCE_MANIFEST.json` are byte-identical copies from
`SJTU-DMTai/OnlineTSF`, commit `0a8646199f6dcbf5346f933508b33c5e35b5708b`, retrieved
2026-10-17. Their SHA-256 and Git blob SHA-1 identifiers are retained.

The author's `pyproject.toml` declares `license = "Apache-2.0"`. No standalone
`LICENSE` file was present in the repository tree inspected at that commit.
This notice accurately records that declaration; it is not an invented author
copyright statement or a claim that an upstream license file was retrieved.
The standard Apache License, Version 2.0 text is available at
https://www.apache.org/licenses/LICENSE-2.0 . All source attribution is retained.

Associated paper: Lifan Zhao and Yanyan Shen, *Proactive Model Adaptation Against
Concept Drift for Online Time Series Forecasting*, KDD 2025, pages 2020–2031,
DOI: https://doi.org/10.1145/3690624.3709210 . The author's full arXiv v5 was read:
https://arxiv.org/html/2412.08435v5 . ACM full-text access was not established.

Runtime entry points are the untouched `models/NLinear.py` and
`adapter/proceed.py`. Their import closure is `adapter/module/{base,generator,
ssf,up,down_up}.py`, `layers/RevIN.py`, PyTorch, NumPy and Transformers.
The remaining vendored files are provenance and audit material. The complete
upstream application and its dependencies have not been vendored, and `run.py`
is not executed. Its training call is commented out at this commit.

Our experiment entry is `../../stage_b.py`. Its `NoDriftProceed` subclass
replaces the concept difference supplied to the unchanged generator with a
detached zero tensor. Both concept encoders and the complete generator remain
instantiated, and the control is independently trained. Upstream files are not
patched. The local trainer changes data support, scaling, validation, optimizer
schedule and evaluation to fit the project's disclosed development protocol;
it does not reproduce the paper's benchmark experiment.
