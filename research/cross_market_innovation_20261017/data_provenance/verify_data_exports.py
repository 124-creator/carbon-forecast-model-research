"""Independent source-to-CSV audit; never imports either preparation script."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
from pathlib import Path

import openpyxl
import xlrd

BASE = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_csv(meta: dict, column: str) -> list:
    path = BASE / meta['path']
    require(path.stat().st_size == meta['bytes'], 'CSV byte size differs')
    require(sha(path) == meta['sha256'], 'CSV hash differs')
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == ['date', column, 'source_row'], 'CSV schema differs')
        rows = [(r['date'], float(r[column]), int(r['source_row'])) for r in reader]
    require(len(rows) == meta['records'], 'CSV row count differs')
    require([r[0] for r in rows] == sorted(r[0] for r in rows), 'CSV is not date sorted')
    require(len({r[0] for r in rows}) == len(rows), 'CSV duplicates date')
    require(rows[0][0] == meta['date_min'] and rows[-1][0] == meta['date_max'], 'CSV date bounds differ')
    return rows


def main() -> None:
    output = BASE / 'DATA_EXPORT_AUDIT.json'
    if output.exists():
        raise FileExistsError('Existing audit receipt; refusing overwrite')
    independent_path = BASE / 'INDEPENDENT_DATA_MANIFEST.json'
    independent = json.loads(independent_path.read_text())
    audit, comparisons = [], 0
    for market in independent['series']:
        source = BASE / 'local_only' / 'raw' / market['filename']
        require(source.stat().st_size == market['bytes'] and sha(source) == market['sha256'], 'Workbook identity differs')
        wb = openpyxl.load_workbook(source, read_only=True, data_only=False)
        ws = wb['Sheet1']
        source_map = {}
        for number, values in enumerate(ws.iter_rows(min_row=2, values_only=True), 2):
            if values[0] is None:
                require(all(v is None for v in values), 'Partially blank source row')
                continue
            source_date = values[0]
            if isinstance(source_date, str):
                date = dt.date.fromisoformat(source_date).isoformat()
            elif isinstance(source_date, dt.datetime):
                date = source_date.date().isoformat()
            else:
                date = source_date.isoformat()
            column = 2 if market['series_id'] == 'SHANGHAI_D4' else 1
            source_map[number] = (date, float(values[column]), number)
        wb.close()
        used = set()
        for partition, meta in market['partitions'].items():
            rows = read_csv(meta, market['output_target'])
            expected = []
            for row in source_map.values():
                if partition == 'development' and row[0] <= '2023-12-31':
                    expected.append(row)
                if partition == 'confirmation_sealed' and '2024-01-01' <= row[0] <= '2025-12-31':
                    expected.append(row)
            expected.sort(key=lambda row: row[0])
            require(rows == expected, 'CSV differs from source numeric value/date/row identity')
            for row in rows:
                require(row[2] not in used, 'Partition source-row overlap')
                used.add(row[2])
            comparisons += len(rows)
        require(len(used) == len(source_map), 'Unexpected unused source record')
        audit.append({'series_id': market['series_id'], 'source_record_comparisons': len(used),
                      'exact_source_value_preservation': True, 'disjoint_partitions': True})
    science_path = BASE / 'SCIENCEDB_DATA_MANIFEST.json'
    science = json.loads(science_path.read_text())
    wb = xlrd.open_workbook(BASE / 'local_only' / 'raw' / 'sciencedb_819_v6.xls')
    for market in science['markets']:
        ws = wb.sheet_by_name(market['market'])
        raw = []
        for number in range(1, ws.nrows):
            date = xlrd.xldate_as_datetime(ws.cell_value(number, 0), wb.datemode).date().isoformat()
            raw.append((date, float(ws.cell_value(number, 3)), number + 1))
        raw.sort(key=lambda row: row[0])
        train = len(raw) * 60 // 100
        valid = len(raw) * 20 // 100
        expected = {'train': raw[:train], 'validation': raw[train:train + valid],
                    'confirmation_sealed': raw[train + valid:]}
        combined = []
        for partition in ['train', 'validation', 'confirmation_sealed']:
            rows = read_csv(market['partitions'][partition], 'mean_price')
            require(rows == expected[partition], 'ScienceDB split or source values differ')
            combined.extend(rows)
            comparisons += len(rows)
        require(combined == raw, 'ScienceDB coverage or ordering differs')
        audit.append({'series_id': market['series_id'], 'source_record_comparisons': len(raw),
                      'exact_source_value_preservation': True, 'disjoint_partitions': True})
    require(comparisons == 6724, 'Total source record count differs')
    receipt = {
        'status': 'PASS', 'audited_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'source_record_comparisons': comparisons, 'csv_files': 25,
        'price_values_printed': False, 'returns_or_price_statistics': False,
        'model_fit': False, 'preparation_modules_imported': False,
        'parser_independence': 'Audit directly rereads source workbooks and source-row identities; no preparation-code import.',
        'independent_manifest_sha256': sha(independent_path),
        'sciencedb_manifest_sha256': sha(science_path),
        'audit_code_sha256': sha(Path(__file__)), 'series': audit,
        'limits': 'Source-to-export fidelity and split audit only. Does not certify historical quote provenance, source-side cleaning, vintage availability, economic target identity, or model test independence.',
    }
    with output.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
