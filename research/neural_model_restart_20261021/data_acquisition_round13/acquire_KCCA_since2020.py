from pathlib import Path
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
from collections import Counter
import urllib.request,urllib.error,json,csv,hashlib,math,concurrent.futures
OUT=Path(__file__).resolve().parent
START=int(datetime(2020,1,1,tzinfo=timezone.utc).timestamp())
END=int(datetime(2026,10,21,tzinfo=timezone.utc).timestamp())
URL=f'https://query1.finance.yahoo.com/v8/finance/chart/KCCA?period1={START}&period2={END}&interval=1d&events=div%2Csplits%2CcapitalGains'
ISSUER='https://kraneshares.com/etf/kcca/'

def save(name,data):
    with (OUT/name).open('x',encoding='utf-8') as f: json.dump(data,f,indent=2,ensure_ascii=False)

def fetch(item):
    name,url=item
    rec={'name':name,'url':url,'retrieved_utc':datetime.now(timezone.utc).isoformat()}
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 (compatible; public carbon-data research)','Accept':'*/*'})
        with urllib.request.urlopen(req,timeout=30) as r:
            body=r.read();rec.update(http_status=r.status,final_url=r.geturl(),headers=dict(r.headers.items()))
        filename=name+'_raw'+('.json' if body.lstrip().startswith(b'{') else '.html')
        with (OUT/filename).open('xb') as f:f.write(body)
        rec.update(raw_file=filename,bytes=len(body),sha256=hashlib.sha256(body).hexdigest())
    except urllib.error.HTTPError as e:
        filename=name+'_error.txt'
        with (OUT/filename).open('xb') as f:f.write(e.read())
        rec.update(http_status=e.code,error=str(e),raw_file=filename)
    except Exception as e:rec['error']=type(e).__name__+': '+str(e)
    return rec

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
    requests=list(ex.map(fetch,[('yahoo_KCCA_since2020',URL),('kraneshares_KCCA_identity',ISSUER)]))
request=requests[0]
if request.get('http_status')!=200:
    save('KCCA_acquisition_log.json',{'requests':requests});raise RuntimeError('Yahoo KCCA retrieval did not return200')
q=json.loads((OUT/request['raw_file']).read_text(encoding='utf-8'))['chart']['result'][0]
meta=q['meta'];request['instrument_metadata']=meta
assert meta['symbol']=='KCCA',meta
stamp=datetime.fromisoformat(request['retrieved_utc']);tz=ZoneInfo(meta.get('exchangeTimezoneName','America/New_York'))
quote=q['indicators']['quote'][0];adj=q['indicators'].get('adjclose',[{}])[0].get('adjclose',[])
rows=[]
for i,ts in enumerate(q.get('timestamp',[])):
    row={'date':datetime.fromtimestamp(ts,tz).date().isoformat(),'timestamp_utc':ts}
    for c in ('open','high','low','close','volume'):
        vals=quote.get(c,[]);row[c]=vals[i] if i<len(vals) else None
    row['adj_close']=adj[i] if i<len(adj) else None
    rows.append(row)
completed=[r for r in rows if r['date']<stamp.astimezone(tz).date().isoformat() and r['date']<='2026-10-20']
assert completed,'No completed daily rows'
filename='KCCA_ETF_'+meta['currency']+'_daily_completed_'+completed[0]['date'].replace('-','')+'_20261020.csv'
with (OUT/filename).open('x',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(completed[0]));w.writeheader();w.writerows(completed)
sha=hashlib.sha256((OUT/filename).read_bytes()).hexdigest()
actions=[]
for typ,records in q.get('events',{}).items():
    for ev in records.values():actions.append({'event_type':typ,**ev,'event_date':datetime.fromtimestamp(ev['date'],tz).date().isoformat()})
actions.sort(key=lambda x:(x['event_date'],x['event_type']))
save('KCCA_corporate_actions.json',actions)
def valid(r,c):return r[c] is not None and math.isfinite(float(r[c]))
def quality(rs):
    complete=[r for r in rs if all(valid(r,c) for c in ('open','high','low','close','volume','adj_close'))]
    bad=[r['date'] for r in complete if not r['low']<=min(r['open'],r['close'])<=max(r['open'],r['close'])<=r['high']]
    return {'rows':len(rs),'complete_rows':len(complete),'missing_or_nonfinite_by_column':{c:sum(not valid(r,c) for r in rs) for c in ('open','high','low','close','volume','adj_close')},
        'ohlc_order_violation_count':len(bad),'ohlc_order_violation_dates':bad,
        'nonpositive_price_rows':sum(any(r[c]<=0 for c in ('open','high','low','close','adj_close')) for r in complete),
        'zero_volume_rows':sum(r['volume']==0 for r in complete),'negative_volume_rows':sum(r['volume']<0 for r in complete),
        'zero_high_low_range_rows':sum(r['high']==r['low'] for r in complete),
        'zero_high_low_range_dates':[r['date'] for r in complete if r['high']==r['low']],
        'volume_min':min((r['volume'] for r in complete),default=None),'volume_max':max((r['volume'] for r in complete),default=None)}
report={'created_utc':datetime.now(timezone.utc).isoformat(),'data_file':filename,'sha256':sha,'source_url':URL,'retrieved_utc':request['retrieved_utc'],
    'instrument_metadata':meta,'rows':len(completed),'first_date':completed[0]['date'],'last_date':completed[-1]['date'],'raw_rows':len(rows),
    'excluded_rows':[r for r in rows if r not in completed],
    'completion_rule':'Exclusive API period2 2026-10-21T00:00Z plus exchange-local dates <=2026-10-20 and strictly before retrieval exchange-local date.',
    'rows_per_year':dict(Counter(r['date'][:4] for r in completed)),
    'duplicate_dates':len(completed)-len({r['date'] for r in completed}),
    'strictly_increasing_dates':all(completed[i]['date']<completed[i+1]['date'] for i in range(len(completed)-1)),
    'weekend_rows':sum(datetime.fromisoformat(r['date']).weekday()>=5 for r in completed),
    'full_sample_quality':quality(completed),'since_2024_quality':quality([r for r in completed if r['date']>='2024-01-01']),
    'corporate_action_event_counts':dict(Counter(a['event_type'] for a in actions)),
    'adjusted_close_differs_from_close_rows':sum(valid(r,'close') and valid(r,'adj_close') and abs(r['close']-r['adj_close'])>1e-5 for r in completed),
    'vendor_first_trade_date':datetime.fromtimestamp(meta['firstTradeDate'],tz).isoformat(),
    'first_row_matches_vendor_first_trade_date':completed[0]['date']==datetime.fromtimestamp(meta['firstTradeDate'],tz).date().isoformat(),
    'first_row':completed[0],'last_row':completed[-1],
    'limitations':'No independent exchange-calendar completeness or second-vendor reconciliation; no repair or model targets/scores computed.'}
save('KCCA_quality_report.json',report)
request.update(daily_file=filename,daily_rows=len(completed),daily_first_date=completed[0]['date'],daily_last_date=completed[-1]['date'],daily_sha256=sha)
save('KCCA_acquisition_log.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'ticker_scope':['KCCA'],'requests':requests})
print(json.dumps(report,indent=2,ensure_ascii=False))
print(json.dumps({'issuer_request':{k:v for k,v in requests[1].items() if k!='headers'}},indent=2,ensure_ascii=False))
