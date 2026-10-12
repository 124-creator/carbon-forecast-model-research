"""Fresh local data checks and explicitly synthetic objective diagnostics."""
import datetime
import hashlib
import json
from collections import Counter
from decimal import Decimal
from typing import Any

def matured(rows, observed_index):
    """Call after observing current price, before issuing new forecasts."""
    return [row for row in rows if row['target_index'] <= observed_index]

def audit(directory, sources) -> dict[str, Any]:
    rows, totals = [], []
    for source in sources:
        name = source['filename']
        if '/' in name or '\\' in name or name in ('.', '..'):
            raise ValueError('Invalid snapshot filename')
        body = (directory / name).read_bytes()
        if len(body) != source['bytes'] or hashlib.sha256(body).hexdigest() != source['sha256']:
            raise ValueError('Snapshot bytes differ from recorded receipt')
        data = json.loads(body)['data']
        totals.append(int(data['total']))
        rows.extend(data['rows'])
    if not rows or len(set(totals)) != 1 or len(rows) != totals[0]:
        raise ValueError('Pagination incomplete')
    if len({r['currentDay'] for r in rows}) != len(rows):
        raise ValueError('Duplicate dates')
    for row in rows:
        datetime.date.fromisoformat(row['currentDay'])
        if row['productCode'] != '000001' or row['productName'] != 'GDEA':
            raise ValueError('Unexpected product')
    rows.sort(key=lambda r: r['currentDay'])
    years = {}
    previous = None
    for row in rows:
        bucket = years.setdefault(row['currentDay'][:4], dict(rows=0, missing_quantity=0,
            positive_quantity=0, zero_quantity=0, message_quantity_conflicts=0,
            missing_or_nonpositive_end_price=0, unchanged_end_price_from_previous_record=0))
        bucket['rows'] += 1
        q = row.get('exchangeRate')
        if q in (None, ''):
            bucket['missing_quantity'] += 1
        else:
            q = Decimal(str(q))
            if not q.is_finite() or q < 0:
                raise ValueError('Invalid quantity')
            bucket['positive_quantity'] += int(q > 0)
            bucket['zero_quantity'] += int(q == 0)
            bucket['message_quantity_conflicts'] += int(q > 0 and row.get('message') == '当日无成交')
        value = row.get('endPrice')
        price = None if value in (None, '') else Decimal(str(value))
        if price is not None and not price.is_finite():
            raise ValueError('Nonfinite price')
        bucket['missing_or_nonpositive_end_price'] += int(price is None or price <= 0)
        bucket['unchanged_end_price_from_previous_record'] += int(price is not None and price == previous)
        previous = price
    totals = Counter()
    for bucket in years.values():
        totals.update(bucket)
    return dict(totals, date_min=rows[0]['currentDay'], date_max=rows[-1]['currentDay'],
        by_year=years, training_allowed=False, old_model_outputs_read=False,
        source_integrity='receipt_hashes_match_not_external_authentication',
        price_semantics='endPrice is reported field; independently certified close definition unresolved',
        redistribution_permission='unconfirmed', raw_daily_data_uploaded=False)

def objective_probe():
    grid = [i / 10000 for i in range(10001)]
    # These are constructed risk curves, not fitted or sampled market risks.
    def excess(a):
        return max(a*a - 2*a, a*a)
    def absolute(a):
        return max(100 + a*a - 2*a, 1 + a*a)
    return dict(absolute_worst_risk_optimum=min(grid, key=absolute),
                excess_worst_risk_optimum=min(grid, key=excess),
                grid_points=len(grid), market_data_used=False,
                empirical_prediction_gain_tested=False,
                scope='Constructed two-scenario counterexample only; not MJERS implementation')

if __name__ == '__main__':
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-root', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text(encoding='utf-8'))
    sources = [r for r in receipt['sources'] if r['filename'] in
               ('gdea_api_page1.json', 'gdea_api_page2.json', 'gdea_api_page3.json')]
    if len(sources) != 3:
        raise ValueError('Three GDEA receipts required')
    result = audit(args.local_root, sources)
    result['sources'] = sources
    result['executed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    result['scope'] = 'Fresh parse of existing snapshots; no network refetch or model training'
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'gdea_data_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.out / 'synthetic_objective_probe.json').write_text(json.dumps(objective_probe(), indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('sources', 'by_year')}, ensure_ascii=False))
