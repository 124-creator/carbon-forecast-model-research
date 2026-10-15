# Data attribution and restricted working prefix

The existing project records attribute the author-provided price snapshot to **Yingying Fan (2025)**, Figshare dataset version **10.6084/m9.figshare.28682348.v1**.

Source: https://doi.org/10.6084/m9.figshare.28682348.v1
Recorded license: **Creative Commons Attribution 4.0 International (CC BY 4.0)**, https://creativecommons.org/licenses/by/4.0/ .

Stage N reuses only the previously approved Guangzhou CSV prefix: header plus 950 records through 2021-12-31, 29204 bytes. Prefix SHA256: 1366ca4814dcdb8aa43296eb64768d64768d1336d6f0fc9088648d6d7ccdbe43318070.

The authoritative machine-checked prefix SHA256 is specified in core.py and DATA_RECEIPT.json. No price is altered, smoothed, winsorized, filled or dropped. A prefix subset is the only data extraction; source-row and Excel-format columns are retained for provenance but not used by the models. All later records and other markets are excluded from this experiment.

This attribution follows the project's existing source/license audit. It does not claim a new independent verification of authorship, exchange-certified settlement/close semantics or historical first-availability timestamps. The model outputs and uncertainty calculations must not be described as an official or live market feed.
