from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from collections import Counter
import csv, json, math, hashlib

OUT = Path(__file__).resolve().parent
log = json.loads((OUT / 'acquisition_log.json').read_text(encoding='utf-8'))
request = next(x for x in log['requests'] if x['name'] == 'yahoo_KRBN')
payload = json.loads((OUT / request['raw_file']).read_text(encoding='utf-8'))['chart']['result'][0]
retrieved = datetime.fromisoformat(request['retrieved_utc'])
exchange_today = retrieved.astimezone(ZoneInfo('America/New_York')).date().isoformat()
with (OUT / 'yahoo_KRBN_daily.csv').open(encoding='utf-8', newline='') as f:
    reader = csv.DictReader(f)
    columns = reader.fieldnames
    all_rows = list(reader)
rows = [r for r in all_rows if r['date'] < exchange_today]
excluded = [r for r in all_rows if r['date'] >= exchange_today]
output = OUT / 'KRBN_ETF_USD_daily_completed_20240102_20261020.csv'
with output.open('w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=columns); w.writeheader(); w.writerows(rows)

events = []
for event_type, items in payload.get('events', {}).items():
    for event in items.values():
        events.append({'event_type':event_type, **event, 'event_date':datetime.fromtimestamp(event['date'], ZoneInfo('America/New_York')).date().isoformat()})
(OUT / 'KRBN_corporate_actions.json').write_text(json.dumps(events, indent=2, ensure_ascii=False), encoding='utf-8')

cols = ('open','high','low','close','volume','adj_close')
invalid = {c:sum(not r[c] or not math.isfinite(float(r[c])) for r in rows) for c in cols}
violation_dates = [r['date'] for r in rows if not (float(r['low']) <= min(float(r['open']),float(r['close'])) <= max(float(r['open']),float(r['close'])) <= float(r['high']))]
quality = {
    'checked_utc':datetime.now(timezone.utc).isoformat(),
    'data_file':output.name,
    'source_raw_file':request['raw_file'],
    'source_url':request['url'],
    'retrieved_utc':request['retrieved_utc'],
    'retrieved_exchange_local':retrieved.astimezone(ZoneInfo('America/New_York')).isoformat(),
    'original_rows':len(all_rows), 'completed_rows':len(rows),
    'start_date':rows[0]['date'], 'end_date':rows[-1]['date'],
    'excluded_current_or_future_date_rows':excluded,
    'completion_rule':'Retain only exchange-local dates earlier than retrieval exchange-local date. Current NYSEArca session was still open at retrieval (14:11 UTC < regular-session end 20:00 UTC).',
    'rows_per_year':dict(Counter(r['date'][:4] for r in rows)),
    'duplicate_dates':len(rows)-len({r['date'] for r in rows}),
    'strictly_increasing_dates':all(rows[i]['date']<rows[i+1]['date'] for i in range(len(rows)-1)),
    'missing_or_nonfinite_by_column':invalid,
    'ohlc_order_violation_count':len(violation_dates), 'ohlc_order_violation_dates':violation_dates,
    'nonpositive_price_rows':sum(any(float(r[c])<=0 for c in ('open','high','low','close','adj_close')) for r in rows),
    'nonpositive_volume_rows':sum(float(r['volume'])<=0 for r in rows),
    'noninteger_volume_rows':sum(float(r['volume']) != int(float(r['volume'])) for r in rows),
    'zero_high_low_range_rows':sum(float(r['high'])==float(r['low']) for r in rows),
    'ohlcv_complete_and_valid_rows':sum(all(r[c] and math.isfinite(float(r[c])) for c in ('open','high','low','close','volume')) and r['date'] not in violation_dates for r in rows),
    'volume_min':min(int(r['volume']) for r in rows),
    'volume_max':max(int(r['volume']) for r in rows),
    'adjusted_close_differs_from_close_rows':sum(abs(float(r['adj_close'])-float(r['close']))>1e-5 for r in rows),
    'corporate_action_event_counts':dict(Counter(x['event_type'] for x in events)),
    'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
    'calendar_validation':'No exchange-calendar package or independent vendor reconciliation used. Weekends and date monotonicity checked; no missing-session assertion is made.',
    'weekend_rows':sum(datetime.fromisoformat(r['date']).weekday()>=5 for r in rows)
}
(OUT / 'KRBN_quality_report.json').write_text(json.dumps(quality, indent=2, ensure_ascii=False), encoding='utf-8')
manifest = {
    'created_utc':datetime.now(timezone.utc).isoformat(),
    'instrument':{
        'symbol':'KRBN', 'name':'KraneShares Global Carbon Strategy ETF', 'instrument_type':'ETF',
        'venue':'NYSEArca', 'currency':'USD', 'price_unit':'USD per ETF share', 'volume_unit':'ETF shares',
        'exchange_timezone':'America/New_York',
        'exposure':'Portfolio of carbon-allowance futures across multiple cap-and-trade markets; issuer benchmarks to S&P Global Carbon Credit Index.',
        'not_equivalent_to':'EUA spot, EUA physical allowance price, or a single EUA futures contract',
        'identity_source_urls':['https://kraneshares.com/etf/krbn/','https://kraneshares.com/etf/keua/'],
        'vendor_identity':{k:payload['meta'].get(k) for k in ('symbol','longName','shortName','instrumentType','currency','fullExchangeName','exchangeTimezoneName')}
    },
    'dataset':{
        'file':output.name, 'rows':len(rows), 'start':rows[0]['date'], 'end':rows[-1]['date'],
        'frequency':'Trading-day daily', 'retrieved_utc':request['retrieved_utc'], 'url':request['url'],
        'ohlc_adjustment':'Provider quote open/high/low/close as returned; no manual adjustment or synthetic reconstruction. Separate adj_close is Yahoo dividend/split-adjusted field and must not be mixed with unadjusted OHLC for candlestick arithmetic.',
        'current_session_excluded':True, 'raw_snapshot_retained':request['raw_file'],
        'recommended_use':'Independent instrument OHLCV robustness/identification experiment. Report KRBN separately from EUA; avoid treating it as an extension or replacement of existing EUA close series.'
    },
    'attempts':[
        {'source':'Yahoo Finance','symbols':['CFI=F','KEUA','KRBN'],'result':'KRBN OHLCV acquired; CFI=F and KEUA returned HTTP404 No data found, symbol may be delisted. CFI=F carbon identity not established and not used.'},
        {'source':'Stooq','symbols':['CK.F'],'identity_evidence_url':'https://stooq.com/q/?s=ck.f','result':'Public search listing identifies ICE CO2 EU Allowances. Direct page and CSV returned HTTP200 HTML JavaScript proof-of-work browser-verification pages, not quotes. No browser-check bypass or data fabricated.'},
        {'source':'KraneShares issuer','symbols':['KRBN','KEUA'],'result':'Issuer identity HTML acquired. KEUA issuer closure announcement: regular trading ceased 2026-03-13; liquidation date 2026-03-20. KRBN distinct global portfolio ETF confirmed.'}
    ],
    'scope':'Three independent public source groups maximum. No packages installed; no login, payment, messages to third parties, or legacy modifications.'
}
(OUT / 'instrument_and_availability_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(quality, indent=2, ensure_ascii=False))
