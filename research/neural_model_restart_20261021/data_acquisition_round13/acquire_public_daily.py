from pathlib import Path
from datetime import datetime, timezone
import urllib.request, urllib.error, urllib.parse, json, csv, io, hashlib, concurrent.futures

OUT = Path(__file__).resolve().parent
START = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp())
END = int(datetime(2026, 10, 22, tzinfo=timezone.utc).timestamp())
HEADERS = {'User-Agent': 'Mozilla/5.0 (compatible; public carbon-data research)', 'Accept': '*/*'}

def fetch(item):
    name, url = item
    stamp = datetime.now(timezone.utc).isoformat()
    record = {'name': name, 'url': url, 'retrieved_utc': stamp}
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=25) as r:
            body = r.read()
            record.update(status=r.status, final_url=r.geturl(), headers=dict(r.headers.items()))
        suffix = '.json' if body.lstrip().startswith((b'{', b'[')) else ('.csv' if body.lstrip().startswith(b'Date,') else '.txt')
        raw = OUT / (name + '_raw' + suffix)
        raw.write_bytes(body)
        record.update(raw_file=raw.name, bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
        if suffix == '.json':
            payload = json.loads(body)
            result = ((payload.get('chart') or {}).get('result') or [])
            if result:
                q = result[0]
                meta = q.get('meta', {})
                record['instrument_metadata'] = meta
                ts = q.get('timestamp') or []
                quote = (q.get('indicators', {}).get('quote') or [{}])[0]
                adj = (q.get('indicators', {}).get('adjclose') or [{}])[0].get('adjclose', [])
                rows = []
                for i, t in enumerate(ts):
                    row = {'date': datetime.fromtimestamp(t, timezone.utc).strftime('%Y-%m-%d'), 'timestamp_utc': t}
                    for k in ('open', 'high', 'low', 'close', 'volume'):
                        values = quote.get(k) or []
                        row[k] = values[i] if i < len(values) else None
                    row['adj_close'] = adj[i] if i < len(adj) else None
                    rows.append(row)
                if rows:
                    p = OUT / (name + '_daily.csv')
                    with p.open('w', newline='', encoding='utf-8') as f:
                        w = csv.DictWriter(f, fieldnames=list(rows[0]))
                        w.writeheader(); w.writerows(rows)
                    record.update(daily_file=p.name, n_rows=len(rows), first_date=rows[0]['date'], last_date=rows[-1]['date'], non_null={k:sum(x[k] is not None for x in rows) for k in ('open','high','low','close','volume')}, nonzero_volume_rows=sum((x['volume'] or 0)>0 for x in rows))
            else:
                record['provider_error'] = (payload.get('chart') or {}).get('error')
        elif suffix == '.csv':
            rows = list(csv.DictReader(io.StringIO(body.decode('utf-8-sig'))))
            record.update(n_rows=len(rows), columns=list(rows[0]) if rows else [], first_date=rows[0].get('Date') if rows else None, last_date=rows[-1].get('Date') if rows else None)
    except urllib.error.HTTPError as e:
        body = e.read()
        p = OUT / (name + '_error.txt'); p.write_bytes(body)
        record.update(status=e.code, error=str(e), raw_file=p.name, bytes=len(body))
    except Exception as e:
        record.update(error=type(e).__name__ + ': ' + str(e))
    return record

urls = []
for symbol in ('CFI=F', 'KEUA', 'KRBN'):
    encoded = urllib.parse.quote(symbol, safe='')
    urls.append(('yahoo_' + symbol.replace('=','_'), f'https://query1.finance.yahoo.com/v8/finance/chart/{encoded}?period1={START}&period2={END}&interval=1d&events=div%2Csplits'))
urls.append(('stooq_CK_F', 'https://stooq.com/q/d/l/?s=ck.f&d1=20240101&d2=20261021&i=d'))
urls.append(('stooq_CK_F_identity', 'https://stooq.com/q/?s=ck.f'))
urls.append(('kraneshares_KEUA_identity', 'https://kraneshares.com/etf/keua/'))
urls.append(('kraneshares_KRBN_identity', 'https://kraneshares.com/etf/krbn/'))
with concurrent.futures.ThreadPoolExecutor(max_workers=7) as ex:
    records = list(ex.map(fetch, urls))
(OUT / 'acquisition_log.json').write_text(json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(), 'source_groups':['Yahoo Finance', 'Stooq', 'KraneShares (issuer identity pages)'], 'requests':records}, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps([{k:v for k,v in r.items() if k not in ('headers','instrument_metadata')} for r in records], indent=2, ensure_ascii=False))
