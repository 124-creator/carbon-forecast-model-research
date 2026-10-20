"""Extract author-supplied mean prices and preserve a separate benchmark cohort.

No closing-price relabeling, calendar filling, price repair, return calculation,
model fitting or score computation. The two XLS variants count only once.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path

import xlrd

ROOT = Path(__file__).resolve().parent
MARKETS = {'上海': 'SHANGHAI', '北京': 'BEIJING', '广东': 'GUANGDONG',
           '深圳': 'SHENZHEN', '湖北': 'HUBEI', '天津': 'TIANJIN', '重庆': 'CHONGQING'}
FILES = {
    'sciencedb_819_v6.xls': (374272, '59f01f50f6e9a8f1b063b9adf2c3e9c1',
                           '5d362db0f8e0a7609d35a04afb25a8bcbbc67167b56c41c3800febf2364cf223'),
    'sciencedb_819_original.xls': (1529344, '507c5fd1d5ffc31c54987e07e3e0dcd0',
                                 '8f8ee080cb9afa9420f04311a461c784a215cc164f0a0f35d83844410b2bc95a'),
}


def source_date(cell, datemode: int) -> str:
    value = cell.value
    # Larger workbook Guangdong dates use YYYYMMDD numeric values. The v6
    # workbook uses Excel dates. This converts the date representation only.
    if cell.ctype in (xlrd.XL_CELL_NUMBER, xlrd.XL_CELL_DATE):
        if 19000101 <= value <= 21001231 and value == int(value):
            return dt.datetime.strptime(str(int(value)), '%Y%m%d').date().isoformat()
        converted = xlrd.xldate_as_datetime(value, datemode)
        if converted.time() != dt.time(0):
            raise ValueError('Unexpected non-midnight date')
        return converted.date().isoformat()
    if cell.ctype == xlrd.XL_CELL_TEXT:
        return dt.date.fromisoformat(value).isoformat()
    raise ValueError('Unsupported source date cell type')


def read_book(filename: str) -> tuple[dict, dict]:
    path = ROOT / 'local_only' / 'raw' / filename
    data = path.read_bytes()
    expected = FILES[filename]
    if (len(data), hashlib.md5(data).hexdigest(), hashlib.sha256(data).hexdigest()) != expected:
        raise ValueError('Workbook hash or size mismatch')
    book = xlrd.open_workbook(path, on_demand=True)
    if book.sheet_names() != list(MARKETS):
        raise ValueError('Unexpected market sheet list')
    result, formats = {}, {}
    for name in MARKETS:
        sheet = book.sheet_by_name(name)
        if sheet.row_values(0)[:4] != ['日期', '配额日成交量（吨）', '配额日成交额（元）', '配额日成交均价（元/吨）']:
            raise ValueError('Unexpected target header')
        rows, packed_dates = [], 0
        for number in range(1, sheet.nrows):
            cell = sheet.cell(number, 0)
            if isinstance(cell.value, (float, int)) and 19000101 <= cell.value <= 21001231:
                packed_dates += 1
            date = source_date(cell, book.datemode)
            target = sheet.cell_value(number, 3)
            if not isinstance(target, (int, float)) or not math.isfinite(target) or target <= 0:
                raise ValueError('Invalid mean-price target; no deletion or imputation')
            rows.append((date, target, number + 1))
        rows.sort(key=lambda row: row[0])
        if len(rows) != len({row[0] for row in rows}):
            raise ValueError('Duplicate market/date identity')
        result[name] = rows
        formats[name] = {'date_numeric_YYYYMMDD_records': packed_dates,
                         'observed_records': len(rows), 'invalid_target_records': 0,
                         'duplicate_dates': 0}
    book.release_resources()
    return result, formats


def main() -> None:
    output = ROOT / 'SCIENCEDB_DATA_MANIFEST.json'
    if output.exists():
        raise FileExistsError('Existing cohort manifest; refusing overwrite')
    scope = ROOT / 'INDEPENDENT_SOURCE_SCOPE.json'
    if not scope.is_file():
        raise RuntimeError('Missing pre-download split protocol')
    small, small_format = read_book('sciencedb_819_v6.xls')
    large, large_format = read_book('sciencedb_819_original.xls')
    for name in MARKETS:
        if [(d, p) for d, p, _ in small[name]] != [(d, p) for d, p, _ in large[name]]:
            raise ValueError('Source variants disagree: manual source review required')
    manifest = {
        'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'doi': '10.11922/sciencedb.819', 'dataset_version': 'V1',
        'landing_url': 'https://www.scidb.cn/detail?dataSetId=633694461347495939',
        'license': 'CC BY 4.0; public landing metadata',
        'target_field': '配额日成交均价（元/吨）', 'csv_target_column': 'mean_price',
        'unit': 'CNY per tonne, as explicitly labeled by author workbook',
        'target_kind': 'author-curated daily transaction mean price; not closing price',
        'source_side_processing': 'Depositor says non-trading dates were removed. Data-paper search extraction reports monthly weighted-mean outlier screening and replacing verified abnormalities with preceding-day transaction data. Full original paper download was blocked; replacement dates and raw originals are not supplied by this dataset.',
        'source_side_processing_evidence': {
            'article_doi': '10.11922/csdata.2019.0030.zh',
            'primary_search_url': 'https://www.sciengine.com/doi/pdf/943AE063EA1F4AD9AC55194E6AFCAE72',
            'web_refs': ['turn72search0', 'turn83search0'],
            'full_paper_read': False,
        },
        'benchmark_status': 'separate author-cleaned mean-price sensitivity cohort; not automatically included in original close/Latest primary pool',
        'scope_sha256': hashlib.sha256(scope.read_bytes()).hexdigest(),
        'parser_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'files': [{'filename': name, 'bytes': item[0], 'md5': item[1], 'sha256': item[2]} for name, item in FILES.items()],
        'source_variants_identical_for_date_and_mean_price': True,
        'source_variant_correction_note': 'Initial exploratory comparison treated larger Guangdong YYYYMMDD numeric dates as Excel serials; this parser recognizes the explicit date format and all 732 dates then match v6. No prices altered.',
        'format_audit': {'v6': small_format, 'larger_variant': large_format},
        'counted_variants': 1, 'total_records': sum(len(rows) for rows in small.values()),
        'model_fit': False, 'returns_or_price_statistics': False,
        'price_values_printed': False, 'extra_price_transformations': [],
        'splits': 'First floor(0.60*n) chronological records train, next floor(0.20*n) validation, rest confirmation, fixed before acquisition because all series end before 2024.',
        'source_row_is_provenance_only_not_a_feature': True,
        'markets': [],
    }
    plans = []
    for name, code in MARKETS.items():
        rows = small[name]
        train_n, validation_n = len(rows) * 3 // 5, len(rows) // 5
        groups = {'train': rows[:train_n], 'validation': rows[train_n:train_n + validation_n],
                  'confirmation_sealed': rows[train_n + validation_n:]}
        metadata = {'market': name, 'series_id': 'SCIENCEDB_' + code + '_MEAN',
                    'records': len(rows), 'date_min': rows[0][0], 'date_max': rows[-1][0],
                    'previously_new_market_relative_to_figshare_and_shanghai_d4': name in {'北京', '深圳', '天津', '重庆'},
                    'partitions': {}}
        for partition, group in groups.items():
            path = ROOT / 'local_only' / 'sciencedb_v6' / partition / (code + '_MEAN.csv')
            if path.exists():
                raise FileExistsError('Existing cohort CSV; refusing overwrite')
            stream = io.StringIO(newline='')
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerow(['date', 'mean_price', 'source_row'])
            for date, value, source_row in group:
                writer.writerow([date, repr(value), source_row])
            data = stream.getvalue().encode('utf-8')
            plans.append((path, data))
            metadata['partitions'][partition] = {
                'path': str(path.relative_to(ROOT)), 'records': len(group),
                'date_min': group[0][0] if group else None,
                'date_max': group[-1][0] if group else None,
                'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
            }
        manifest['markets'].append(metadata)
    for path, data in plans:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(data)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'total_records': manifest['total_records'],
                      'source_variants_identical': True,
                      'markets': [{'market': row['market'], 'records': row['records'],
                                   'date_min': row['date_min'], 'date_max': row['date_max'],
                                   'partition_counts': {key: value['records'] for key, value in row['partitions'].items()}}
                                  for row in manifest['markets']],
                      'manifest': str(output)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
