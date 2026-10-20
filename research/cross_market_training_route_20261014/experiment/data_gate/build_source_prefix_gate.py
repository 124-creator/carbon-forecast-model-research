"""Metadata-only gate: never opens price tables or hashes originals."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
AUDIT = '数据真实性核对与选题调研_20261007/06_核心核查结果.json'
PUB = 'GitHub发布_20261007/-123/metadata/audit_results.json'
LICENSE = 'GitHub发布_20261007/-123/DATA_LICENSES.md'

def main():
    core = json.loads((ROOT / AUDIT).read_text(encoding='utf-8'))
    public = json.loads((ROOT / PUB).read_text(encoding='utf-8'))
    markets = {}
    for market in ('GDEA', 'HBEA', 'EUA'):
        raw = core[market]['raw']
        markets[market] = {
            'role': 'target_Guangzhou' if market == 'GDEA' else 'auxiliary_historical_training_only',
            'source_audit_path': AUDIT,
            'audit_locator': '/' + market + '/raw',
            'source_path_as_recorded': raw['path'],
            'full_snapshot_metadata_only': {k: raw[k] for k in ('rows', 'date_min', 'date_max')},
            'prefixes': {c: {'exact_rows': None, 'actual_last_observation_date': None, 'status': 'UNKNOWN_NO_EXACT_PREFIX_RECEIPT'} for c in ('2019-12-31', '2021-12-31')},
            'requested_950_rows': {'rows_claimed_in_task': 950, 'verified_rows': None, 'actual_cutoff': None, 'status': 'UNVERIFIED_IN_INSPECTED_METADATA'} if market == 'GDEA' else None,
            'training_years_already_used': None,
            'validation_years_already_used': None,
            'test_years_already_used': None,
            'prior_use_status': 'UNKNOWN_NO_SELECTED_ROUTE_SPLIT_AUDIT_FOUND; do not infer from full snapshot range',
            'license_manifest_path': LICENSE,
            'license_status': 'NO_SOURCE_SPECIFIC_TRAINING_LICENSE_VERIFIED; exchange bulk redistribution unresolved',
            'pool_admission': 'BLOCKED_PENDING_EXACT_PREFIX_SCOPE_AND_USE_PERMISSION',
            'gaps': ['exact prefix row count', 'actual final observed date at each cutoff', 'isolated local prefix file scope receipt', 'selected route train/validation/test years', 'source-specific permitted research use']
        }
    wang = next(x for x in public['published_datasets'] if x['dataset'] == 'author_wang_eu_futures')
    markets['EUA_author_Wang_alternative'] = {
        'source_audit_path': PUB, 'audit_locator': '/published_datasets/dataset=author_wang_eu_futures',
        'path': wang['path'], 'rows': wang['rows'], 'date_min': wang['date_min'], 'actual_cutoff': wang['date_max'],
        'prefixes': {c: {'exact_rows': wang['rows'], 'actual_last_observation_date': wang['date_max'], 'basis': 'entire audited snapshot ends before cutoff, not a recomputed partial prefix'} for c in ('2019-12-31', '2021-12-31')},
        'license': wang['license'], 'license_manifest_path': LICENSE,
        'source_grade': wang['source_grade'], 'contract_roll_currency_unit_verified': False,
        'training_years_already_used': None, 'test_years_already_used': None,
        'pool_admission': 'CONDITIONAL_OFFLINE_NONCOMMERCIAL_ALTERNATIVE_NOT_EQUIVALENT_TO_EXISTING_EUA_SERIES',
        'conditions': ['attribution and noncommercial restriction', 'freeze exact identity and do not substitute silently', 'no independently authenticated ICE price claim', 'verify target-compatible definition or explicitly use heterogeneous auxiliary task']
    }
    result = {
        'schema_version': 1, 'task_date': '2026-10-14', 'status': 'BLOCKED_FOR_THREE_MARKET_POOL_EXACT_PREFIX_GAPS',
        'repo_relative_to_workspace': 'GitHub发布_20261007/-123',
        'scope': 'existing metadata/audit and license only; no online acquisition',
        'price_tables_read': False, '2022_prices_read': False, 'boundary_price_rows_read': False,
        'original_full_file_hash_computed': False, 'models_deserialized': False, 'training_ran': False,
        'markets': markets,
        'shared_historical_training_time_boundary_rules': [
            'Freeze a common calendar upper bound C before training; 2019-12-31 and 2021-12-31 are candidate policy bounds, not verified observed cutoff dates.',
            'For every market require an exact isolated-prefix receipt with count, date_min, actual date_max <= C and source identity before admitting any prices.',
            'Every feature, target endpoint and normalization/calibration/selection input must lie within its allowed training boundary; discard boundary-crossing labels.',
            'For simulated forecast origin t, auxiliary supervision endpoints must not exceed min(C,t); parameters/scalers cannot use subsequent records.',
            'Respect market-specific chronological sequences; do not synchronize misaligned quotes, intersect trading dates, forward-fill one market into another, or claim economic linkage.',
            'Target is improvement of Guangzhou prediction only; cross-market training is statistical representation transfer, not causal/economic connectedness evidence.',
            'No 2022 prices or original boundary/full-file hash reads are authorized by this gate. Existing full-snapshot aggregates cannot determine earlier prefix counts.',
            'Document already-used train/validation/test years before selection; previously inspected tests are not untouched holdouts.',
            'A lawfully available public frozen snapshot may support explicitly historical offline, non-real-time-certified experiments; observation dates do not prove first availability.',
            'Never modify original prices; maintain provenance and permission separately from source authenticity.'
        ],
        'evidence_limits': ['No precise GDEA 950 receipt discovered in inspected metadata.', 'No HBEA/existing EUA counts through 2019 or 2021 established.', 'Full source date ranges are metadata only and are not allowed training windows.', 'Legal public access is not itself blanket training or redistribution permission.'],
        'evidence_paths': [AUDIT, PUB, LICENSE, '数据真实性核对与选题调研_20261007/07_GDEA官方核对摘要.json', '数据真实性核对与选题调研_20261007/07_HBEA官方核对摘要.json'],
    }
    target = OUT / 'SOURCE_PREFIX_GATE.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    parsed = json.loads(target.read_text(encoding='utf-8'))
    if parsed['markets']['GDEA']['requested_950_rows']['verified_rows'] is not None:
        raise RuntimeError('Unsupported 950 certification')
    for m in ('GDEA', 'HBEA', 'EUA'):
        if any(x['exact_rows'] is not None for x in parsed['markets'][m]['prefixes'].values()):
            raise RuntimeError('Invented partial-prefix count')
    if parsed['markets']['EUA_author_Wang_alternative']['rows'] != 2059:
        raise RuntimeError('Unexpected audited alternative snapshot count')
    print(json.dumps({'output': str(target.relative_to(ROOT)), 'status': parsed['status'], 'metadata_only_checks': 'PASS', 'prices_read': False}, ensure_ascii=False))

if __name__ == '__main__':
    main()
