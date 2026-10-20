"""Package completed private inputs and run evidence; never train or fetch data.

Repository-backed code and public reports remain in Git. This archive contains
raw inputs, prepared arrays, weights, per-origin forecasts and private audit
evidence. Use --mapping to relocate sources in a separately reproduced run.
No Library identifiers, credentials, or unrelated source directories are read.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

STUDY = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def default_mapping():
    files = [
        {'source': f'/tmp/carbon-distribution-update-data/development/{m}_development.csv',
         'archive_path': f'inputs/development/{m}_development.csv'}
        for m in ('EUA_LONG', 'GUANGZHOU', 'HUBEI', 'SHANGHAI')]
    files += [
        {'source': '/tmp/carbon-conditional-persistence-runs/sources/plos_0296105_s001.xls',
         'archive_path': 'inputs/plos_shenzhen/plos_0296105_s001.xls'},
        {'source': '/tmp/carbon-checkpoint-replay-partial.json',
         'archive_path': 'audits/checkpoint_replay_partial.json'},
        {'source': '/tmp/carbon-followup-data-diagnostic-20261019b.json',
         'archive_path': 'diagnostics/prior_f1/full_diagnostic.json'},
        {'source': '/tmp/carbon-followup-diagnostic-20261019b.py',
         'archive_path': 'diagnostics/prior_f1/replay_selected_diagnostic.py'},
    ]
    directories = [
        {'source': '/tmp/carbon-conditional-persistence-runs', 'archive_prefix': 'runs/base',
         'exclude': ['sources/plos_0296105_s001.xls']},
        {'source': '/tmp/carbon-persistence-market-runs', 'archive_prefix': 'runs/persistence'},
        {'source': '/tmp/carbon-gas-normalization-run', 'archive_prefix': 'runs/gas'},
        {'source': '/tmp/carbon-persistence-screen-runs', 'archive_prefix': 'runs/synthetic'},
        {'source': '/tmp/carbon-gas-pipeline-synthetic-t7ppvkav', 'archive_prefix': 'smoke/gas_pipeline'},
        {'source': '/tmp/carbon-checkpoint-replay-cache', 'archive_prefix': 'audits/checkpoint_replay_cache'},
        {'source': '/tmp/carbon-score-replay-20261019b', 'archive_prefix': 'audits/original_score_replay_cache'},
        {'source': '/tmp/carbon-score-replay-qualified-20261019b', 'archive_prefix': 'audits/qualified_score_replay_cache'},
        {'source': '/tmp/carbon-price-crps-cancellation-20261019b', 'archive_prefix': 'audits/price_cancellation'},
        {'source': '/tmp/carbon-stable-price-crps-20261019b', 'archive_prefix': 'audits/stable_price_crps'},
    ]
    return {'files': files, 'directories': directories}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--mapping', type=Path)
    p.add_argument('--receipt', type=Path, default=STUDY/'provenance/PRIVATE_ARCHIVE_RECEIPT.json')
    args = p.parse_args()
    assert not args.archive.exists(), 'Refuse to overwrite an existing archive'
    mapping = json.loads(args.mapping.read_text()) if args.mapping else default_mapping()
    pairs = [(Path(f['source']), f['archive_path']) for f in mapping['files']]
    for row in mapping['directories']:
        root = Path(row['source'])
        assert root.is_dir(), f'Missing private run directory: {root}'
        for f in sorted(root.rglob('*')):
            assert not f.is_symlink(), f'Symlink excluded by contract: {f}'
            if not f.is_file():
                continue
            relative = f.relative_to(root).as_posix()
            if relative in row.get('exclude', []) or '__pycache__' in f.parts:
                continue
            assert not f.name.endswith('.writing'), f'An artifact is still being written: {f}'
            pairs.append((f, row['archive_prefix'].rstrip('/')+'/'+relative))
    assert len({name for _, name in pairs}) == len(pairs), 'Duplicate archive destinations'
    assert len({str(f.resolve()) for f, _ in pairs}) == len(pairs), 'Duplicate private sources'
    records = []
    for source, target in pairs:
        assert source.is_file() and not source.is_symlink()
        assert not target.startswith('/') and '..' not in Path(target).parts
        records.append({'archive_path': target, 'original_path': str(source),
                        'bytes': source.stat().st_size, 'sha256': digest(source)})
    created = datetime.now(timezone.utc).isoformat()
    manifest = {'created_utc': created, 'study': STUDY.name, 'files': records,
                'mapping': mapping, 'payload_files': len(records),
                'payload_uncompressed_bytes': sum(r['bytes'] for r in records),
                'repository_code_and_public_reports_duplicated': False}
    readme = '''# Private carbon conditional-persistence experiment archive

This file contains private input tables, prepared arrays, all completed market
checkpoints and origin-level predictions, synthetic paths, implementation-smoke
evidence, logs, and independent audit caches. Public code, reports, fit receipts,
aggregate results and figures are versioned separately in the GitHub study
research/conditional_persistence_20261019b.

MANIFEST.json maps every archived file to its original local path, exact size
and SHA256. The Shenzhen XLS occurs only once under inputs/plos_shenzhen.
Prepared shared panels are in runs/base/prepared. Candidate and GAS outputs
are in runs/persistence and runs/gas. For full reproduction see REPRODUCE_ZH.md
in the public study. Cache paths recorded in receipts identify original runs;
do not reuse a cache with changed audit code or changed model artifacts.

The original frozen price-CRPS scores are preserved. Some Shenzhen h20 price
CRPS values suffer floating-point cancellation; qualified score replay and
stable-price companion arrays are archived separately. No forecast weights,
training results or primary normalized-CRPS decision were changed.

The archive is private; retaining a publicly accessible source for research
does not establish permission to republish every raw input in a public repo.
No reserved 2024+ market-price inputs are included.
'''
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        z.writestr('README.md', readme)
        z.writestr('MANIFEST.json', json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
        for (source, target), record in zip(pairs, records):
            z.write(source, target)
            assert source.stat().st_size == record['bytes'] and digest(source) == record['sha256'], \
                f'Private source changed while packaging: {source}'
    with zipfile.ZipFile(args.archive) as z:
        assert z.testzip() is None
        assert len(z.infolist()) == len(records)+2
    receipt = {'created_utc': created, 'archive_filename': args.archive.name,
               'archive_bytes': args.archive.stat().st_size, 'archive_sha256': digest(args.archive),
               'payload_files': len(records), 'payload_uncompressed_bytes': manifest['payload_uncompressed_bytes'],
               'zip_members': len(records)+2, 'integrity': 'All sources hashed before/after copy; ZIP CRC check passed',
               'scope': 'Private input/weights/forecasts/audit caches only; repository-backed artifacts stay in Git',
               'raw_publication': False, 'reserved_2024plus_prices_included': False,
               'packager_sha256': digest(__file__)}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
