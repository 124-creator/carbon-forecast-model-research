#!/usr/bin/env python3
"""Read-only metadata/overlap audit; no fitting, predictions, or raw-value export.

Inputs are public author files downloaded outside Git. Numerical diagnostics are
restricted to dates before 2024-01-01; workbook decoders may decode later cells.
No claim of never-decoded later data or exchange authentication is made.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import openpyxl
import xlrd


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def date_value(x):
    if isinstance(x, dt.datetime):
        return x.date().isoformat()
    if isinstance(x, dt.date):
        return x.isoformat()
    # The Guangdong Date column mixes styled Excel dates and numeric YYYYMMDD.
    # Decode the latter explicitly; these 8-digit values are not Excel serials.
    if isinstance(x, (int, float)) and math.isfinite(x) and int(x) == x:
        s = str(int(x))
        if len(s) == 8 and s[:2] in {'19', '20'}:
            try:
                return dt.datetime.strptime(s, '%Y%m%d').date().isoformat()
            except ValueError:
                return None
    if isinstance(x, str):
        for fmt in ('%Y-%m-%d', '%m/%d/%Y', '%Y/%m/%d', '%Y%m%d'):
            try:
                return dt.datetime.strptime(x.strip(), fmt).date().isoformat()
            except ValueError:
                pass
    return None


def number(x):
    try:
        v = float(str(x).strip().replace(',', ''))
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def table(path: Path):
    if path.suffix == '.xlsx':
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        sheets = []
        for ws in wb:
            sheets.append((ws.title, [list(r) for r in ws.iter_rows(values_only=True)]))
        wb.close()
        return sheets
    if path.suffix == '.xls':
        wb = xlrd.open_workbook(path, on_demand=True)
        sheets = []
        for ws in wb.sheets():
            rows = []
            for i in range(ws.nrows):
                row = []
                for c in ws.row(i):
                    row.append(xlrd.xldate_as_datetime(c.value, wb.datemode)
                               if c.ctype == xlrd.XL_CELL_DATE else c.value)
                rows.append(row)
            sheets.append((ws.name, rows))
        wb.release_resources()
        return sheets
    with path.open(encoding='utf-8-sig', newline='') as f:
        return [('CSV', list(csv.reader(f)))]


def audit_table(path, date_name, target_name, cutoff):
    sheets = table(path)
    assert len(sheets) == 1, 'Explicit selection required for a multi-sheet source'
    sheet_name, rows = sheets[0]
    header = [str(x).strip() for x in rows[0]]
    date_col = header.index(date_name)
    target_col = header.index(target_name) if target_name is not None else None
    dated, admitted, invalid_date, excluded_dates = [], [], 0, []
    for row_no, row in enumerate(rows[1:], 2):
        day = date_value(row[date_col])
        if day is None:
            invalid_date += 1
            continue
        dated.append(day)
        if day >= cutoff:
            excluded_dates.append(day)
            continue
        vals = {h: number(row[i]) if i < len(row) else None
                for i, h in enumerate(header) if i != date_col}
        admitted.append({'date': day, 'source_row': row_no, 'values': vals})
    dates = [r['date'] for r in admitted]
    ordered_dates = sorted(set(dates))
    columns = {}
    for h in header:
        if h == date_name:
            continue
        vals = [r['values'][h] for r in admitted]
        valid = [x for x in vals if x is not None]
        columns[h] = {'numeric_finite': len(valid), 'missing_or_nonnumeric': len(vals)-len(valid),
                      'zeros': sum(x == 0 for x in valid), 'negative': sum(x < 0 for x in valid)}
    result = {
        'path': str(path), 'sha256': digest(path), 'bytes': path.stat().st_size,
        'sheet': sheet_name, 'headers': header, 'data_rows': len(rows)-1,
        'all_dated_rows': len(dated), 'all_date_min': min(dated) if dated else None,
        'all_date_max': max(dated) if dated else None,
        'invalid_date_rows': invalid_date, 'all_year_counts': dict(sorted(Counter(x[:4] for x in dated).items())),
        'numeric_audit_cutoff_exclusive': cutoff, 'precutoff_rows': len(admitted),
        'later_rows_numerically_excluded': len(excluded_dates),
        'precutoff_date_min': min(dates) if dates else None,
        'precutoff_date_max': max(dates) if dates else None,
        'precutoff_duplicate_dates': len(dates)-len(set(dates)),
        'precutoff_weekend_dates': sum(dt.date.fromisoformat(x).weekday() >= 5 for x in dates),
        'original_date_order': 'ascending' if dates == sorted(dates) else 'descending' if dates == sorted(dates, reverse=True) else 'unsorted',
        'largest_gap_days': max((dt.date.fromisoformat(b)-dt.date.fromisoformat(a)).days
                                for a, b in zip(ordered_dates, ordered_dates[1:])) if len(ordered_dates)>1 else None,
        'precutoff_numeric_columns': columns,
        'target_column': target_name,
        'raw_values_exported': False,
    }
    if target_col is not None:
        target = [r['values'][target_name] for r in admitted]
        result['valid_positive_target_rows'] = sum(x is not None and x > 0 for x in target)
        result['nonpositive_target_rows'] = sum(x is not None and x <= 0 for x in target)
        result['missing_target_rows'] = sum(x is None for x in target)
    if target_name and {'Max', 'Min'} <= set(header):
        triples = [(r['values'][target_name],r['values']['Min'],r['values']['Max']) for r in admitted]
        valid = [(p,l,h) for p,l,h in triples if p is not None and l is not None and h is not None]
        result['range_check'] = {'valid_triplets': len(valid),'max_below_min': sum(h<l for p,l,h in valid),
                                 'target_outside_reported_range': sum(p<l or p>h for p,l,h in valid)}
    return result, admitted


def compare_source(rows, target, reference_path, reference_target):
    with reference_path.open(encoding='utf-8-sig', newline='') as f:
        ref = {r['date']: number(r[reference_target]) for r in csv.DictReader(f)
               if r['date'] < '2024-01-01'}
    pairs = [(r['values'][target], ref[r['date']]) for r in rows
             if r['date'] in ref and r['values'][target] is not None and ref[r['date']] is not None]
    return {'reference_path': str(reference_path), 'reference_sha256': digest(reference_path),
            'common_valid_dates': len(pairs),'exact_float_matches': sum(a == b for a,b in pairs),
            'within_absolute_1e_10': sum(abs(a-b) <= 1e-10 for a,b in pairs),
            'max_absolute_difference': max((abs(a-b) for a,b in pairs),default=None),
            'interpretation': 'Matching reported values establishes overlap, not an independent source or official provenance.'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-root', type=Path, required=True)
    ap.add_argument('--repo-root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError('Existing receipt will not be overwritten')
    cutoff = '2024-01-01'
    specs = [
        ('TGSINFO_GUANGZHOU','Data_GuangZhou.xlsx','交易日期','收盘价','GUANGZHOU_development.csv'),
        ('ZZY_GUANGDONG','zzy/GuangdongCarbonPrice.xls','Date','Closing Price','GUANGZHOU_development.csv'),
        ('ZZY_HUBEI','zzy/HubeiCarbonPrice.xlsx','Date','Latest','HUBEI_development.csv'),
        ('ZZY_SHANGHAI','zzy/ShanghaiCarbonPrice.csv','Date','close','SHANGHAI_D4_development.csv'),
        ('ZZY_EXTERNAL','zzy/AssociatedExternalFactors.csv','Date',None,None),
    ]
    result = {'schema_version':1,'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
              'audit_script_sha256':digest(Path(__file__)), 'fits':0,'predictions_or_metrics_read':False,
              'source_workbooks_modified':False, 'sources':{},
              'scope':'Metadata for all dates; numerical fields and overlaps only before 2024. Entire worksheet decoding may include later cells. No claim of untouched confirmation.',
              'license':'Both public repositories returned license=null and had no LICENSE file in the complete tree. Raw data retained outside Git; no license or source rights inferred from public availability.'}
    private_rows = {}
    for sid, rel, date_name, target, reference in specs:
        summary, rows = audit_table(args.raw_root/rel,date_name,target,cutoff)
        if reference:
            summary['legacy_overlap'] = compare_source(rows,target,args.repo_root/'research/cross_market_innovation_20261017/data'/reference,'price')
        result['sources'][sid] = summary
        private_rows[sid] = rows
    # MATLAB xlsread excludes the all-text date column. The source removes rows
    # containing NaN from its numeric matrix, flips, then takes 1522 values.
    bridge = args.raw_root/'TGSINFO-VMD-GZ.csv'
    if bridge.exists():
        with bridge.open(encoding='utf-8-sig',newline='') as f:
            reader = csv.DictReader(f); fields = reader.fieldnames; rr = list(reader)
        raw_tgs_rows = table(args.raw_root/'Data_GuangZhou.xlsx')[0][1][1:]
        assert all(isinstance(r[0],str) for r in raw_tgs_rows)
        assert all(r[1] is None or isinstance(r[1],(int,float)) for r in raw_tgs_rows)
        source_rows = list(reversed([r for r in private_rows['TGSINFO_GUANGZHOU']
                                    if r['values']['收盘价'] is not None]))[:1522]
        totals = [sum(float(r[c]) for c in fields) for r in rr]
        prices = [r['values']['收盘价'] for r in source_rows]
        assert len(totals) == len(prices)
        result['decomposition_bridge'] = {'path':str(bridge),'sha256':digest(bridge), 'rows':len(rr),'columns':fields,
            'source_rule':'numeric price column of text-date sheet; remove NaN rows as MATLAB script; flip; take first 1522',
            'matlab_executed':False, 'matlab_source_sha256':digest(args.raw_root/'TGSINFO_VMD_GZ.m'),
            'source_date_min':min(r['date'] for r in source_rows),'source_date_max':max(r['date'] for r in source_rows),
            'max_absolute_sum_minus_source_price':max(abs(a-b) for a,b in zip(totals,prices)),
            'warning':'Algebraic reconstruction does not establish causal decomposition; future-inclusive VMD was performed before the notebook split.'}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'out':str(args.out),'sources':{s:{k:v.get(k) for k in ['data_rows','precutoff_rows','precutoff_date_min','precutoff_date_max','nonpositive_target_rows','missing_target_rows','legacy_overlap']} for s,v in result['sources'].items()},'bridge':result.get('decomposition_bridge')},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
