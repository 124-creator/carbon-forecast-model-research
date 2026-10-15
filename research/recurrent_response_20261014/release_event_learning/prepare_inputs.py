"""Prepare a fresh local-only bundle; never modify an existing run."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from event_data import parse_article
from experiment_io import HERE, load_prefix, write_json


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-map', required=True)
    parser.add_argument('--source-root', action='append', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError('Output must be a fresh directory; preserve previous artifacts')
    roots = [Path(path).resolve() for path in args.source_root]
    supplied = json.loads(Path(args.source_map).read_text(encoding='utf-8'))
    expected = {r['url']: r for r in json.loads((HERE / 'SOURCE_INPUT_RECEIPTS.json').read_text(encoding='utf-8'))}
    if len(supplied) != len(expected) or {r['url'] for r in supplied} != set(expected):
        raise ValueError('Raw map is not the frozen source corpus')
    events, excluded, verified = [], [], []
    for item in supplied:
        path = Path(item['file']).resolve()
        if path.suffix.lower() != '.html' or not any(path.is_relative_to(root) for root in roots):
            raise ValueError('Raw HTML path outside explicit source roots')
        if not item['url'].startswith('https://www.stats.gov.cn/'):
            raise ValueError('Nonofficial source URL')
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        known = expected[item['url']]
        if len(content) != known['bytes'] or digest != known['sha256']:
            raise ValueError('Source body differs from frozen retrieved snapshot')
        record = parse_article(content.decode('utf-8'))
        verified.append(dict(url=item['url'], bytes=len(content), sha256=digest))
        if record is None:
            excluded.append(dict(url=item['url'], reason='No supported literal publication header'))
        else:
            events.append(dict(record, url=item['url'], source_sha256=digest))
    events.sort(key=lambda r: r['available_date'])
    if len({r['available_date'] for r in events}) != len(events):
        raise ValueError('Duplicate source available dates')
    price, date_text, receipt = load_prefix()
    dates = np.array(date_text, dtype='datetime64[D]').astype('int64')
    source_dates = np.array([r['available_date'] for r in events], dtype='datetime64[D]').astype('int64')
    source_values = np.array([r['values'] for r in events], dtype=np.float64)
    local = out / 'local_only'
    local.mkdir(parents=True)
    write_json(local / 'source_events.json', events)
    np.savez_compressed(local / 'INPUTS.npz', price=price, dates=dates,
                        source_dates=source_dates, source_values=source_values)
    audit = dict(raw_bodies_verified=len(verified), independently_parsed_events=len(events),
                 selected_thermal_rows=int(source_values.size), excluded_no_supported_publication=excluded,
                 historical_first_version_certified=False, per_page_network_refetch_this_round=False,
                 source_available_rule='literal displayed publication + 1 calendar day',
                 raw_and_prices_kept_local=True, carbon_prefix=receipt,
                 source_event_sha256=hashlib.sha256((local / 'source_events.json').read_bytes()).hexdigest(),
                 local_bundle_sha256=hashlib.sha256((local / 'INPUTS.npz').read_bytes()).hexdigest(),
                 source_events_available_by_carbon_prefix_end=int(np.sum(source_dates <= dates[-1])))
    write_json(out / 'SOURCE_INPUT_RECEIPTS.json', verified)
    write_json(out / 'DATA_AUDIT.json', audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
