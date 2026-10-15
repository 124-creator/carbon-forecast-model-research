# Data attribution and restricted working prefix

The existing project records attribute the author-provided price snapshot to **Yingying Fan (2025)**, Figshare dataset version **10.6084/m9.figshare.28682348.v1**.

Source: https://doi.org/10.6084/m9.figshare.28682348.v1
Recorded license: **Creative Commons Attribution 4.0 International (CC BY 4.0)**, https://creativecommons.org/licenses/by/4.0/ .

Stage N reuses only the previously approved Guangzhou CSV prefix: header plus 950 records through 2021-12-31, 29204 bytes. Prefix SHA256: 1366ca4814dcdb8aa43296eb64768d1336d6f0fc9088648d6d7ccdbe43318070.

No price is altered, smoothed, winsorized, filled or dropped. A prefix subset is the only data extraction; source-row and Excel-format columns are retained for provenance but not used by the models. All later records and other markets are excluded from this experiment.

This attribution follows the project's existing source/license audit. It does not claim a new independent verification of authorship, exchange-certified settlement/close semantics or historical first-availability timestamps. The model outputs and uncertainty calculations must not be described as an official or live market feed.

Publication note: the first version of this attribution-only document duplicated six characters while transcribing the checksum. This document was corrected after results; core.py, the protocol, numerical data receipts, training and scores always used the correct checksum above and were not changed.
