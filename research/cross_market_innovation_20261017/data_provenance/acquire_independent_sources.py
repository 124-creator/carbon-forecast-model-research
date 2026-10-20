"""Reproduce the three independent public datasets in a NEW directory.

No network retries, authentication, mirrors, price output, or market fits.
The already-run acquisitions used the same published URLs via curl; this
convenience wrapper is supplied for future reproduction and checked by dry-run.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import urllib.parse
import urllib.request
import zipfile

SOURCE = Path(__file__).resolve().parent
DOWNLOADS = [
    ('mendeley_d4_data.zip', 'https://data.mendeley.com/public-files/datasets/d4bv52fgmd/files/4935504f-cf2d-45c0-add4-d6c20d667a45/file_downloaded', 508904, 'b4ddba5c3609dfa3c13ed4a0c5998c952225c69dc1bc6c8a8f4c73a5fdfd2862'),
    ('mendeley_5s_processed.xlsx', 'https://data.mendeley.com/public-files/datasets/5skvz8sgrz/files/5cb129aa-e29b-4b8f-a3bb-dc3fafb3ffa8/file_downloaded', 109902, 'f6d9be7c2e4bb4463cd123936882bbca79626312a7f4a718804cd217816b0c06'),
    ('sciencedb_819_v6.xls', 'https://china.scidb.cn/download?fileId=5eeeecf15e972ddb1b4881d7', 374272, '5d362db0f8e0a7609d35a04afb25a8bcbbc67167b56c41c3800febf2364cf223'),
    ('sciencedb_819_original.xls', 'https://china.scidb.cn/download?fileId=5eeeecf15e972ddb1b4881d8', 1529344, '8f8ee080cb9afa9420f04311a461c784a215cc164f0a0f35d83844410b2bc95a'),
]
COPY_FILES = ['INDEPENDENT_SOURCE_SCOPE.json', 'prepare_independent_data.py',
              'prepare_sciencedb_data.py', 'verify_data_exports.py']


class PublicRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urllib.parse.urlsplit(newurl)
        if target.scheme != 'https' or target.hostname not in {'data.mendeley.com', 'china.scidb.cn'}:
            raise RuntimeError('Unapproved redirect; preserve receipt and stop')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError('Output exists; refusing to overwrite prior data or results')
    for name in COPY_FILES:
        if not (SOURCE / name).is_file():
            raise FileNotFoundError(name)
    plan = {'output': str(output), 'datasets': 3, 'download_files': len(DOWNLOADS),
            'download_bytes': sum(item[2] for item in DOWNLOADS),
            'urls': [item[1] for item in DOWNLOADS], 'network_retries': 0,
            'copy_files': COPY_FILES, 'dry_run': args.dry_run}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    output.mkdir(parents=True, exist_ok=False)
    raw = output / 'local_only' / 'raw'
    raw.mkdir(parents=True)
    (output / 'provenance').mkdir()
    (output / '.gitignore').write_text('local_only/\n__pycache__/\n*.pyc\n')
    for name in COPY_FILES:
        shutil.copy2(SOURCE / name, output / name)
    receipt = dict(plan, started_utc=dt.datetime.now(dt.timezone.utc).isoformat(), records=[])
    opener = urllib.request.build_opener(PublicRedirect())
    try:
        for name, url, expected_size, expected_sha in DOWNLOADS:
            request = urllib.request.Request(url, headers={'User-Agent': 'CarbonForecastResearch/1.0'})
            with opener.open(request, timeout=35) as response:
                body = response.read(expected_size + 1)
                status = response.status
                final_url = response.url
            actual_sha = hashlib.sha256(body).hexdigest()
            if len(body) != expected_size or actual_sha != expected_sha:
                raise ValueError('Published source byte/hash mismatch: ' + name)
            with (raw / name).open('xb') as stream:
                stream.write(body)
            receipt['records'].append({'name': name, 'url': url, 'final_url': final_url,
                                       'status': status, 'bytes': len(body), 'sha256': actual_sha})
        with zipfile.ZipFile(raw / 'mendeley_d4_data.zip') as archive:
            members = archive.infolist()
            if len({item.filename for item in members}) != len(members):
                raise ValueError('Duplicate ZIP member')
            if sum(item.file_size for item in members) > 40_000_000:
                raise ValueError('ZIP expansion limit')
            for item in members:
                path = PurePosixPath(item.filename)
                if path.is_absolute() or '..' in path.parts or '\\' in item.filename or stat.S_ISLNK(item.external_attr >> 16):
                    raise ValueError('Unsafe ZIP member path/type')
            body = archive.read('data/碳价格与配额.xlsx')
        if len(body) != 66625 or hashlib.sha256(body).hexdigest() != '41592d944375b3109010c151d9b1c103fb906d608afcbfd529d7381f18896697':
            raise ValueError('Extracted target workbook identity mismatch')
        with (raw / 'mendeley_d4_carbon.xlsx').open('xb') as stream:
            stream.write(body)
        receipt['status'] = 'PASS'
    except Exception as error:
        receipt.update(status='FAILED_STOPPED', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        receipt['finished_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        (output / 'provenance' / 'REPRODUCTION_ACQUISITION.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'output': str(output),
                      'next_commands': ['python prepare_independent_data.py',
                                        'python prepare_sciencedb_data.py',
                                        'python verify_data_exports.py']}, indent=2))


if __name__ == '__main__':
    main()
