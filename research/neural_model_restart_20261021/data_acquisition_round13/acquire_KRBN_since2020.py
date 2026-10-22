from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from collections import Counter
import urllib.request, urllib.error, json, csv, hashlib, math

OUT = Path(__file__).resolve().parent
NY = ZoneInfo('America/New_York')
START = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp())
END = int(datetime(2026, 10, 21, tzinfo=timezone.utc).timestamp())
URL = f'https://query1.finance.yahoo.com/v8/finance/chart/KRBN?period1={START}&period2={END}&interval=1d&events=div%2Csplits%2CcapitalGains'
OLD = OUT / 'KRBN_ETF_USD_daily_completed_20240102_20261020.csv'
OLD_SHA = hashlib.sha256(OLD.read_bytes()).hexdigest()
EXPECTED_OLD_SHA = 'd1ff17cfb541809385d3ba6e918c61ce147682a91ea13bea93a426c33474d42a'
assert OLD_SHA == EXPECTED_OLD_SHA, 'Existing 703-row snapshot differs from delivered hash; do not continue silently.'

def save_json(name, obj):
    with (OUT / name).open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

stamp = datetime.now(timezone.utc)
request = {'url':URL, 'retrieved_utc':stamp.isoformat(), 'requested_start':'2020-01-01', 'requested_last_completed_date':'2026-10-20', 'source_group':'Yahoo Finance'}
try:
    req = urllib.request.Request(URL, headers={'User-Agent':'Mozilla/5.0 (compatible; public carbon-data research)', 'Accept':'application/json'})
    with urllib.request.urlopen(req, timeout=30) as r:
        body = r.read()
        request.update(http_status=r.status, final_url=r.geturl(), headers=dict(r.headers.items()))
except urllib.error.HTTPError as e:
    request.update(http_status=e.code, error=str(e))
    with (OUT / 'yahoo_KRBN_since2020_error.txt').open('xb') as f: f.write(e.read())
    save_json('KRBN_since2020_acquisition_log.json', request)
    raise

raw_name = 'yahoo_KRBN_since2020_raw.json'
with (OUT / raw_name).open('xb') as f: f.write(body)
request.update(raw_file=raw_name, raw_bytes=len(body), raw_sha256=hashlib.sha256(body).hexdigest())
payload = json.loads(body)
assert payload.get('chart',{}).get('result'), payload
q = payload['chart']['result'][0]
meta = q['meta']
assert meta['symbol']=='KRBN' and meta['instrumentType']=='ETF' and meta['currency']=='USD', meta
request['instrument_metadata'] = meta
quote = q['indicators']['quote'][0]
adj = q['indicators'].get('adjclose',[{}])[0].get('adjclose',[])
rows = []
for i, ts in enumerate(q.get('timestamp', [])):
    row = {'date':datetime.fromtimestamp(ts,NY).date().isoformat(), 'timestamp_utc':ts}
    for c in ('open','high','low','close','volume'):
        values = quote.get(c,[])
        row[c] = values[i] if i < len(values) else None
    row['adj_close'] = adj[i] if i < len(adj) else None
    rows.append(row)
exchange_today = stamp.astimezone(NY).date().isoformat()
completed = [r for r in rows if r['date'] <= '2026-10-20' and r['date'] < exchange_today]
excluded = [r for r in rows if r not in completed]
assert completed, 'No completed daily rows'
filename = 'KRBN_ETF_USD_daily_completed_' + completed[0]['date'].replace('-','') + '_20261020.csv'
with (OUT / filename).open('x', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(completed[0])); w.writeheader(); w.writerows(completed)
sha = hashlib.sha256((OUT / filename).read_bytes()).hexdigest()
actions = []
for event_type, records in q.get('events',{}).items():
    for ev in records.values():
        actions.append({'event_type':event_type, **ev, 'event_date':datetime.fromtimestamp(ev['date'],NY).date().isoformat()})
actions.sort(key=lambda x:(x['event_date'], x['event_type']))
save_json('KRBN_since2020_corporate_actions.json', actions)

def valid(r,c):
    return r[c] is not None and math.isfinite(float(r[c]))

bad_ohlc = [r['date'] for r in completed if all(valid(r,c) for c in ('open','high','low','close')) and not float(r['low']) <= min(float(r['open']),float(r['close'])) <= max(float(r['open']),float(r['close'])) <= float(r['high'])]
with OLD.open(encoding='utf-8',newline='') as f: old = {r['date']:r for r in csv.DictReader(f)}
new = {r['date']:r for r in completed}
comparison = {}
for c in ('open','high','low','close','volume','adj_close'):
    diffs = [abs(float(old[d][c])-float(new[d][c])) for d in old if d in new and old[d][c] and new[d][c] is not None]
    comparison[c] = {'compared_rows':len(diffs), 'different_rows_gt_1e_5':sum(x>1e-5 for x in diffs), 'max_absolute_difference':max(diffs,default=None)}
assert hashlib.sha256(OLD.read_bytes()).hexdigest() == OLD_SHA, 'Original snapshot unexpectedly changed.'
quality = {
    'created_utc':datetime.now(timezone.utc).isoformat(), 'data_file':filename, 'csv_sha256':sha,
    'rows':len(completed), 'first_date':completed[0]['date'], 'last_date':completed[-1]['date'],
    'rows_per_year':dict(Counter(r['date'][:4] for r in completed)),
    'retrieved_utc':stamp.isoformat(), 'source_url':URL,
    'raw_rows':len(rows), 'excluded_rows':excluded,
    'completion_rule':'Trading date in America/New_York <=2026-10-20 and strictly earlier than retrieval exchange-local date; API period2=2026-10-21T00:00:00Z is exclusive.',
    'duplicate_dates':len(completed)-len(new),
    'strictly_increasing_dates':all(completed[i]['date']<completed[i+1]['date'] for i in range(len(completed)-1)),
    'weekend_rows':sum(datetime.fromisoformat(r['date']).weekday()>=5 for r in completed),
    'missing_or_nonfinite_by_column':{c:sum(not valid(r,c) for r in completed) for c in ('open','high','low','close','volume','adj_close')},
    'ohlc_order_violation_count':len(bad_ohlc), 'ohlc_order_violation_dates':bad_ohlc,
    'nonpositive_price_rows':sum(any(valid(r,c) and float(r[c])<=0 for c in ('open','high','low','close','adj_close')) for r in completed),
    'nonpositive_volume_rows':sum(valid(r,'volume') and r['volume']<=0 for r in completed),
    'zero_high_low_range_rows':sum(r['high']==r['low'] for r in completed),
    'volume_min':min(r['volume'] for r in completed if valid(r,'volume')), 'volume_max':max(r['volume'] for r in completed if valid(r,'volume')),
    'corporate_action_event_counts':dict(Counter(x['event_type'] for x in actions)),
    'adjusted_close_differs_from_close_rows':sum(valid(r,'adj_close') and valid(r,'close') and abs(r['adj_close']-r['close'])>1e-5 for r in completed),
    'vendor_first_trade_date':datetime.fromtimestamp(meta['firstTradeDate'],NY).isoformat(),
    'first_row_matches_vendor_first_trade_date':completed[0]['date']==datetime.fromtimestamp(meta['firstTradeDate'],NY).date().isoformat(),
    'original_703_snapshot_file':OLD.name, 'original_703_snapshot_sha256_before':OLD_SHA, 'original_703_snapshot_sha256_after':hashlib.sha256(OLD.read_bytes()).hexdigest(),
    'original_703_date_overlap':sum(d in new for d in old), 'overlap_field_comparison':comparison,
    'calendar_validation':'No exchange-calendar package or second vendor used; missing sessions not independently assessed.'
}
save_json('KRBN_since2020_quality_report.json', quality)
initial_log = json.loads((OUT / 'acquisition_log.json').read_text(encoding='utf-8'))
issuer = next(r for r in initial_log['requests'] if r['name']=='kraneshares_KRBN_identity')
manifest = {
    'instrument':{'symbol':'KRBN','name':'KraneShares Global Carbon Strategy ETF','type':'ETF','exchange':'NYSEArca','currency':'USD','price_unit':'USD per ETF share','volume_unit':'ETF shares','timezone':'America/New_York','exposure':'Multi-market carbon allowance futures strategy; issuer benchmark S&P Global Carbon Credit Index. Not EUA physical allowances, EUA spot, or a single EUA futures contract.'},
    'dataset':{'file':filename,'rows':len(completed),'first_date':completed[0]['date'],'last_date':completed[-1]['date'],'sha256':sha,'source_url':URL,'retrieved_utc':stamp.isoformat(),'ohlc_adjustment':'Yahoo quote open/high/low/close as returned, without manual reconstruction; adj_close separate vendor adjustment field, not to mix with raw OHLC.','raw_file':raw_name},
    'issuer_identity_reused':{'url':issuer['url'],'retrieved_utc':issuer['retrieved_utc'],'local_snapshot':issuer['raw_file'],'sha256':issuer['sha256'],'snapshot_hash_verified':hashlib.sha256((OUT/issuer['raw_file']).read_bytes()).hexdigest()==issuer['sha256']},
    'scope':'Same Yahoo data source and existing KraneShares issuer snapshot only; all files new in approved directory; no models/test results read; original 703-row snapshot preserved.'
}
save_json('KRBN_since2020_instrument_manifest.json', manifest)
request.update(daily_file=filename, daily_rows=len(completed), daily_sha256=sha, actual_first_date=completed[0]['date'], actual_last_date=completed[-1]['date'])
save_json('KRBN_since2020_acquisition_log.json', request)
print(json.dumps(quality, ensure_ascii=False, indent=2))
