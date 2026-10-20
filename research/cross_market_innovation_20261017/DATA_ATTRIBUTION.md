# Main CSV attribution, license and transformations

These five CSVs are the exact inputs used in this study: 5,864 development observations and 261 Shanghai confirmation observations. The dataset publishers identify their deposits as **Creative Commons Attribution 4.0 International (CC BY 4.0)**. See [license text and terms](https://creativecommons.org/licenses/by/4.0/). Attribution does not imply endorsement of this analysis by the data authors. Repository code and vendored code retain their respective licensing; this notice covers the derived data files.

## Guangzhou, Hubei and EU_REPORTED

**Source:** Yingying Fan (2025), Figshare dataset version 1, DOI [10.6084/m9.figshare.28682348.v1](https://doi.org/10.6084/m9.figshare.28682348.v1), published 2025-03-28; the fixed author-data exports and original provenance used here are preserved in [the base repository](https://github.com/124-creator/carbon-forecast-model-research/tree/1c67e562e721dd82fff4e4143b97a798a1cf44fa/data/parsed/figshare_28682348). The current Figshare landing page was not available through this session's read channel; depositor licensing and identity follow the repository's previously recorded source evidence, rather than a new claim of complete landing-page access.

Transformations from the fixed repository CSVs: retain dates before 2024-01-01; rename `author_reported_price` to `price`; preserve price strings and provenance columns; serialize with LF line endings. Source/provenance columns are not model features. No interpolation, smoothing, winsorization, outlier deletion or unit conversion was performed. The 178 later observations present in the three original files are excluded from this study, not withheld as a new independent confirmation cohort.

| Published derivative | Rows | SHA-256 |
|---|---:|---|
| `data/GUANGZHOU_development.csv` | 1,434 | `8d78a4f13c5a799519546fd3bfa0e0ac5eaa52b0441a983a95ef06f5e5e34051` |
| `data/HUBEI_development.csv` | 1,408 | `04655d59c1b585f6c35c02a43469a965fe3589eb613d7d8d8e46996751b7d797` |
| `data/EU_REPORTED_development.csv` | 1,515 | `ca7c192380897076ef57516a11ea2ccefda6ba92ab9823d9aa8f2bb281cefedc` |

The EU series is intentionally named EU_REPORTED because instrument, venue, currency and roll convention have not been fully authenticated. Hubei's reported price is not certified as the exchange's close. A historical Guangzhou comparison is documented in the base repository; this study does not claim a fresh successful download from that official endpoint.

## Shanghai

**Source:** Xiao, Min; sun, zhen chen; Zhang, Diao; Liu, Mingyue; Zhao, Yingxiu; Liu, Yu (2026). *Datasets for Carbon Pricing and Energy Markets in Sustainable Production (2003–2025)*, version 1. Mendeley Data. DOI [10.17632/d4bv52fgmd.1](https://doi.org/10.17632/d4bv52fgmd.1). [Publisher landing page](https://data.mendeley.com/datasets/d4bv52fgmd/1), published 2026-02-23. CC BY 4.0 is recorded on the deposit and in its README.

The fixed ZIP contains `data/碳价格与配额.xlsx`, `Sheet1`, columns `TradingDate`, `CityName`, `收盘价`, `碳配额`. All selected rows identify Shanghai. The dataset's title spans other datasets from 2003; the Shanghai series itself begins 2013-12-19.

Transformations: select date and the author-reported close, preserve source row number, convert dates to ISO strings, sort ascending, and partition at 2024-01-01. No imputation, calendar expansion, smoothing, clipping or outlier deletion was performed. A faithful export does not establish that the original author performed no preprocessing, nor certify units or a complete exchange trading calendar. `source_row` is audit metadata, not an input feature.

| Published derivative | Rows | SHA-256 |
|---|---:|---|
| `data/SHANGHAI_D4_development.csv` | 1,507 | `425424353bb3ac09e3d96b291680b4179e86f180da626a5d19e5c1e58ed960a1` |
| `data/SHANGHAI_D4_confirmation.csv` | 261 | `5ef3c7664023027f844c6476b40a24c2906d100a35a2415a3d1b4b27ae57ccfc` |

ZIP SHA-256: `b4ddba5c3609dfa3c13ed4a0c5998c952225c69dc1bc6c8a8f4c73a5fdfd2862`. Selected workbook SHA-256: `41592d944375b3109010c151d9b1c103fb906d608afcbfd529d7381f18896697`.

The confirmation CSV is now public because the fixed experiment is complete. The historical provenance directory's references to an unopened confirmation partition describe its earlier acquisition-stage state. Replication of this public package must not be described as another blind confirmation.

## Other acquired datasets

Mingzhu Liao's national log-price deposit [10.17632/5skvz8sgrz.1](https://doi.org/10.17632/5skvz8sgrz.1) and Zhang Qianyun et al.'s seven-pilot daily transaction-mean dataset [10.11922/sciencedb.819](https://doi.org/10.11922/sciencedb.819) are separately documented, with authors, licenses, source URLs, fixed hashes, download and export scripts in `data_provenance/`. Their daily values are not part of this main experiment package's five input CSVs or claimed predictor fits. They are not silently concatenated with reported close prices.
