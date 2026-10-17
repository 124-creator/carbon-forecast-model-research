"""Prepare an isolated replay from exact frozen SMP source and local data.

check-sources reads code/protocol only. prepare additionally hashes exactly the
five explicitly mapped development CSVs; it never scans adjacent directories.
No network request, data parser, model import, fit, or result audit is run here.
The original protocol bytes are retained. Only --data-map relocates data paths.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import sys


EXPECTED_PROTOCOL_SHA256 = 'b10409de072efaba5fce1d570c15e6e0b4128f5234f037c25afeecfd90efc503'
EXPECTED_AUDIT_SHA256 = '669a7dc6fb74d5724ce194d6424a76758b9aa4fb84f0719fedd99bdf2311fb52'
LICENSE_SHA256 = 'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4'
PROTOCOL_RELATIVE = 'provenance/PROTOCOL.json'
AUDIT_RELATIVE = 'source/independent_results.py'
SERIES = {'EEX_EUA_T3PA', 'GUANGZHOU', 'HUBEI', 'SHANGHAI', 'EU_REPORTED'}


def sha256_bytes(body):
    return hashlib.sha256(body).hexdigest()


def json_no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_bytes(), object_pairs_hook=json_no_duplicates)


def safe_source_relative(name):
    if not isinstance(name, str):
        raise ValueError('Source path must be a string.')
    path = PurePosixPath(name)
    if (path.is_absolute() or '..' in path.parts or '\\' in name
            or name != path.as_posix() or not path.parts
            or path.parts[0] in {'results', 'local_only', 'data'}
            or path.suffix.lower() in {'.csv', '.tsv', '.npz', '.npy', '.pt', '.pth', '.zip'}):
        raise ValueError('Unsafe or non-source copy path: ' + name)
    return path


def source_bytes(root, relative, expected):
    safe_source_relative(relative)
    file = root / relative
    if file.is_symlink() or not file.is_file() or not file.resolve().is_relative_to(root):
        raise ValueError('Missing or unsafe source file: ' + relative)
    body = file.read_bytes()
    if sha256_bytes(body) != expected:
        raise ValueError('Source identity mismatch: ' + relative)
    return body


def inspect_source_bundle(source_root):
    root = Path(source_root).expanduser().resolve()
    protocol_bytes = source_bytes(root, PROTOCOL_RELATIVE, EXPECTED_PROTOCOL_SHA256)
    protocol = json.loads(protocol_bytes, object_pairs_hook=json_no_duplicates)
    markets = protocol['markets']
    if len(markets) != 5 or {m['name'] for m in markets} != SERIES:
        raise ValueError('Protocol does not contain the exact five frozen series.')
    if protocol['data_end_exclusive'] != '2024-01-01':
        raise ValueError('Development data scope changed.')
    declared = protocol['source_sha256']
    if len(declared) != 12:
        raise ValueError('Frozen training source inventory changed.')
    files = {relative: source_bytes(root, relative, expected)
             for relative, expected in declared.items()}
    if EXPECTED_AUDIT_SHA256 is None:
        raise RuntimeError('Independent audit delivery has not yet been pinned.')
    auxiliary = {
        'source/vendor/LICENSE': source_bytes(root, 'source/vendor/LICENSE', LICENSE_SHA256),
        AUDIT_RELATIVE: source_bytes(root, AUDIT_RELATIVE, EXPECTED_AUDIT_SHA256),
    }
    if set(files) & set(auxiliary):
        raise ValueError('An auxiliary source unexpectedly overlaps the training manifest.')
    return root, protocol, protocol_bytes, files, auxiliary


def new_destination(path):
    candidate = Path(path).expanduser().absolute()
    if candidate.exists() or candidate.is_symlink():
        raise FileExistsError('Work root must not exist; no overwrite or implicit restart.')
    resolved = candidate.resolve()
    if resolved.exists() or resolved.is_symlink():
        raise FileExistsError('Resolved work root already exists.')
    return resolved


def hash_local_file(path):
    digest = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        for body in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(body)
            size += len(body)
    return digest.hexdigest(), size


def verify_data_mapping(mapping_path, markets):
    """Hash only the explicitly named local files; never discover other data."""
    map_file = Path(mapping_path).expanduser().resolve()
    map_body = map_file.read_bytes()
    mapping = json.loads(map_body, object_pairs_hook=json_no_duplicates)
    expected = {m['name']: m for m in markets}
    if not isinstance(mapping, dict) or set(mapping) != set(expected):
        raise ValueError('Data map must contain exactly the five frozen names.')
    resolved, receipt = {}, []
    for name in sorted(expected):
        value = mapping[name]
        if not isinstance(value, str) or not value:
            raise ValueError('Each data-map value must be a local CSV path string.')
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = map_file.parent / path
        path = path.resolve()
        if not path.is_file():
            raise ValueError('Mapped development file is missing: ' + name)
        if 'sealed_confirmation' in path.parts:
            raise ValueError('A confirmation directory cannot be mapped as development data.')
        actual, size = hash_local_file(path)
        if actual != expected[name]['sha256']:
            raise ValueError('Mapped file differs from the frozen development SHA256: ' + name)
        resolved[name] = str(path)
        receipt.append(dict(name=name, path=str(path), sha256=actual, bytes=size))
    return resolved, receipt, sha256_bytes(map_body)


def command_entry(name, argv, condition=None):
    return dict(name=name, argv=[str(x) for x in argv],
                shell=shlex.join([str(x) for x in argv]), condition=condition)


def replay_commands(work_root):
    root = Path(work_root)
    python = sys.executable
    protocol = root / PROTOCOL_RELATIVE
    data_map = root / 'data_map.json'
    runner = root / 'source/run_development.py'
    selector = root / 'source/selection.py'
    auditor = root / AUDIT_RELATIVE
    d1_selection = root / 'results/D1_SELECTION.json'
    d1_audit = root / 'audits/D1_independent'
    d2_audit = root / 'audits/D2_independent'
    gate = 'Only after complete D1, independent D1 audit PASS, matching selections, and advance_to_D2=true. No confirmation access.'
    return [
        command_entry('D1_run', [python, '-B', runner, '--protocol', protocol, '--phase', 'D1', '--data-map', data_map]),
        command_entry('D1_selection', [python, '-B', selector, '--protocol', protocol, '--phase', 'D1'], 'After D1 completes without failed jobs.'),
        command_entry('D1_independent_results', [python, '-B', auditor, '--protocol', protocol, '--phase', 'D1', '--output', d1_audit], 'After D1 selection; output directory must not exist.'),
        command_entry('D2_run', [python, '-B', runner, '--protocol', protocol, '--phase', 'D2', '--selection', d1_selection, '--data-map', data_map], gate),
        command_entry('D2_assessment', [python, '-B', selector, '--protocol', protocol, '--phase', 'D2', '--selection', d1_selection], 'Only after eligible D2 completes without failed jobs.'),
        command_entry('D2_independent_results', [python, '-B', auditor, '--protocol', protocol, '--phase', 'D2', '--d1-selection', d1_audit / 'SELECTION.json', '--output', d2_audit], 'Only after eligible D2; output directory must not exist. This does not authorize confirmation.'),
    ]


def write_exclusive_bytes(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(body)


def write_exclusive_json(path, value):
    write_exclusive_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode())


def prepare(args):
    # Existing destination refusal occurs before reading any mapped price file.
    destination = new_destination(args.work_root)
    source, protocol, protocol_bytes, files, auxiliary = inspect_source_bundle(args.source_root)
    mapping, data_receipt, input_map_sha = verify_data_mapping(args.data_map, protocol['markets'])
    commands = replay_commands(destination)
    plan = dict(protocol_sha256=sha256_bytes(protocol_bytes), source_root=str(source), work_root=str(destination),
                frozen_source_files=len(files), auxiliary_files=len(auxiliary), verified_development_files=data_receipt,
                data_map_input_sha256=input_map_sha, commands=commands,
                network_requests=0, market_fits=0, results_copied=0, model_weights_copied=0,
                confirmation_directories_scanned=0,
                interpretation='A new replay under retained historical protocol bytes; not external preregistration or inherited result/confirmation evidence.')
    if args.dry_run:
        print(json.dumps(dict(dry_run=True, output_created=False, mapped_files_hashed=5, **plan), ensure_ascii=False, indent=2))
        return
    # All source and data identities have been verified before directory creation.
    # If an I/O failure interrupts writing, retain the partial directory and
    # refuse reusing it; do not erase or silently continue a failed preparation.
    destination.mkdir(parents=True, exist_ok=False)
    write_exclusive_bytes(destination / PROTOCOL_RELATIVE, protocol_bytes)
    for relative, body in {**files, **auxiliary}.items():
        write_exclusive_bytes(destination / relative, body)
    write_exclusive_json(destination / 'data_map.json', mapping)
    write_exclusive_json(destination / 'REPRODUCTION_COMMANDS.json', dict(commands=commands, auto_execute=False))
    write_exclusive_bytes(destination / '.gitignore', b'local_only/\n__pycache__/\n*.py[cod]\ndata_map.json\n')
    receipt = dict(created_utc=datetime.now(timezone.utc).isoformat(), status='PREPARED_NO_FITS',
                   helper_sha256=sha256_bytes(Path(__file__).read_bytes()),
                   protocol_bytes_unchanged=True,
                   frozen_source_sha256={name: sha256_bytes(body) for name, body in files.items()},
                   auxiliary_sha256={name: sha256_bytes(body) for name, body in auxiliary.items()},
                   output_data_map_sha256=sha256_bytes((destination / 'data_map.json').read_bytes()),
                   independent_metric_audit_performed=False, **plan)
    write_exclusive_json(destination / 'provenance/PREPARATION_RECEIPT.json', receipt)
    if sha256_bytes((destination / PROTOCOL_RELATIVE).read_bytes()) != EXPECTED_PROTOCOL_SHA256:
        raise AssertionError('Copied protocol identity changed.')
    print(json.dumps(dict(status='PREPARED_NO_FITS', **plan), ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    check = sub.add_parser('check-sources', help='Verify frozen code/protocol and audit auxiliaries only; no data-map or market file is opened.')
    check.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[1])
    prep = sub.add_parser('prepare', help='Hash the five mapped development CSVs and prepare a new isolated work root; never execute fits.')
    prep.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[1])
    prep.add_argument('--work-root', type=Path, required=True)
    prep.add_argument('--data-map', type=Path, required=True)
    prep.add_argument('--dry-run', action='store_true', help='Still hash the five mapped CSVs, but create no output and perform no fit.')
    args = parser.parse_args()
    if args.action == 'check-sources':
        root, protocol, body, files, auxiliary = inspect_source_bundle(args.source_root)
        print(json.dumps(dict(status='SOURCE_IDENTITY_PASS', source_root=str(root), protocol_sha256=sha256_bytes(body),
                              frozen_source_files=len(files), auxiliary_files=len(auxiliary),
                              expected_development_identities=[dict(name=m['name'], sha256=m['sha256']) for m in protocol['markets']],
                              market_files_opened=0, confirmation_files_opened=0, network_requests=0, fits=0), ensure_ascii=False, indent=2))
    else:
        prepare(args)


if __name__ == '__main__':
    main()
