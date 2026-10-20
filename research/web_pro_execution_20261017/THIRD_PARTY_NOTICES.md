# Third-party source notices

This research directory includes a small, attributed selection of author source
files. It does not claim ownership of the published models or reproduce their
complete benchmark applications. Original source comments and notices are kept.

| Source | Fixed revision | Local location | License evidence |
|---|---|---|---|
| [cure-lab/LTSF-Linear](https://github.com/cure-lab/LTSF-Linear) | `0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6` | `vendor/ltsf_linear` | Upstream Apache-2.0 LICENSE retained in that directory, including DLinear authors' notice |
| [SJTU-DMTai/OnlineTSF](https://github.com/SJTU-DMTai/OnlineTSF) | `0a8646199f6dcbf5346f933508b33c5e35b5708b` | `vendor/proceed` | Upstream pyproject declares Apache-2.0; see LICENSE_NOTICE.md for the exact scope and absence of a separate upstream LICENSE |
| [VEWOXIC/FITS](https://github.com/VEWOXIC/FITS) | `d040bb015b6299da26d879b90dd19c80fb72c160` | `literature/fits/official` | Upstream Apache-2.0 LICENSE retained |

The full Apache License, Version 2.0 terms are included in the retained LICENSE
files and are also published at <https://www.apache.org/licenses/LICENSE-2.0>.
Copyright notices attached to one upstream project are not asserted for another.

The LTSF-Linear model copies normalize line endings only; original Git blob
identifiers and local SHA-256 values are both retained. The PROCEED source copies
are byte-identical to their recorded upstream Git blobs. The isolated FITS
`patched/FITS_lengthfix.py` declares its one functional change: specify
`n=seq_len+pred_len` for irFFT. It is a compatibility fix, not a new method.

The new Stage A/Stage B training and audit scripts adapt data support, validation,
randomization, and evaluation to the project protocol. The no-drift subclass is
an explicit ablation and does not alter the vendored author files. No model
weights, author benchmark datasets, or paper PDF files are distributed here.
