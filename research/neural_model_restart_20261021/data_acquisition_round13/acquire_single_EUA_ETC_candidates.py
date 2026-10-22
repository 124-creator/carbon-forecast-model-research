from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from collections import Counter
import urllib.request, urllib.error, json, csv, hashlib, math, concurrent.futures

OUT=Path(__file__).resolve().parent
START=int(datetime(2020,1,1,tzinfo=timezone.utc).timestamp())
END=int(datetime(2026,10,21,tzinfo=timezone.utc).timestamp())
ITEMS=[]
for symbol in ('CO2.L','CO2P.L','CARB.L'):
    ITEMS.append(('yahoo_'+symbol.replace('.','_')+'_etc',f'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={START}&period2={END}&interval=1d&events=div%2Csplits%2CcapitalGains'))
ITEMS += [
    ('hanetf_CO2_issuer','https://hanetf.com/fund/co2-sparkchange-physical-carbon-eua-etc/'),
    ('wisdomtree_CARB_issuer','https://www.wisdomtree.eu/en-gb/products/ucits-etfs-unleveraged-etps/commodities/wisdomtree-carbon')
]

def save(name,obj):
    with (OUT/name).open('x',encoding='utf-8') as f: json.dump(obj,f,indent=2,ensure_ascii=False)

def fetch(item):
    name,url=item
    stamp=datetime.now(timezone.utc)
    rec={'name':name,'url':url,'retrieved_utc':stamp.isoformat()}
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 (compatible; public carbon-data research)','Accept':'*/*'})
        with urllib.request.urlopen(req,timeout=30) as r:
            body=r.read(); rec.update(http_status=r.status,final_url=r.geturl(),headers=dict(r.headers.items()))
        rawname=name+'_raw'+('.json' if body.lstrip().startswith(b'{') else '.html')
        with (OUT/rawname).open('xb') as f: f.write(body)
        rec.update(raw_file=rawname,bytes=len(body),sha256=hashlib.sha256(body).hexdigest())
        if not name.startswith('yahoo_'): return rec
        payload=json.loads(body)
        if not payload.get('chart',{}).get('result'):
            rec['provider_error']=payload; return rec
        q=payload['chart']['result'][0]; meta=q['meta']; rec['instrument_metadata']=meta
        tz=ZoneInfo(meta.get('exchangeTimezoneName','Europe/London'))
        quote=q['indicators']['quote'][0]; adj=q['indicators'].get('adjclose',[{}])[0].get('adjclose',[])
        rows=[]
        for i,ts in enumerate(q.get('timestamp',[])):
            row={'date':datetime.fromtimestamp(ts,tz).date().isoformat(),'timestamp_utc':ts}
            for c in ('open','high','low','close','volume'):
                vals=quote.get(c,[]); row[c]=vals[i] if i<len(vals) else None
            row['adj_close']=adj[i] if i<len(adj) else None
            rows.append(row)
        completed=[r for r in rows if r['date']<stamp.astimezone(tz).date().isoformat() and r['date']<='2026-10-20']
        if not completed: return rec
        symbol=meta['symbol']; key=symbol.replace('.','_')
        filename=f'{key}_ETC_daily_completed_'+completed[0]['date'].replace('-','')+'_20261020.csv'
        with (OUT/filename).open('x',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(completed[0])); w.writeheader(); w.writerows(completed)
        rec.update(daily_file=filename,daily_rows=len(completed),daily_first_date=completed[0]['date'],daily_last_date=completed[-1]['date'],daily_sha256=hashlib.sha256((OUT/filename).read_bytes()).hexdigest())
        actions=[]
        for typ,entries in q.get('events',{}).items():
            for e in entries.values(): actions.append({'event_type':typ,**e,'event_date':datetime.fromtimestamp(e['date'],tz).date().isoformat()})
        save(key+'_ETC_corporate_actions.json',actions)
        def valid(r,c): return r[c] is not None and math.isfinite(float(r[c]))
        bad=[r['date'] for r in completed if all(valid(r,c) for c in ('open','high','low','close')) and not r['low']<=min(r['open'],r['close'])<=max(r['open'],r['close'])<=r['high']]
        quality={
            'symbol':symbol,'data_file':filename,'sha256':rec['daily_sha256'],'source_url':url,'retrieved_utc':rec['retrieved_utc'],
            'instrument_metadata':meta,'identity_status':'Candidate until issuer identity and quotation units verified in separate manifest',
            'rows':len(completed),'start_date':completed[0]['date'],'end_date':completed[-1]['date'],'raw_rows':len(rows),
            'completion_rule':'Retain exchange-local dates <=2026-10-20 and earlier than retrieval exchange-local date; exclusive period2 2026-10-21T00:00:00Z.',
            'excluded_rows':[r for r in rows if r not in completed],
            'rows_per_year':dict(Counter(r['date'][:4] for r in completed)),
            'missing_or_nonfinite_by_column':{c:sum(not valid(r,c) for r in completed) for c in ('open','high','low','close','volume','adj_close')},
            'duplicate_dates':len(completed)-len({r['date'] for r in completed}),
            'strictly_increasing_dates':all(completed[i]['date']<completed[i+1]['date'] for i in range(len(completed)-1)),
            'weekend_rows':sum(datetime.fromisoformat(r['date']).weekday()>=5 for r in completed),
            'ohlc_order_violation_count':len(bad),'ohlc_order_violation_dates':bad,
            'nonpositive_price_rows':sum(any(valid(r,c) and r[c]<=0 for c in ('open','high','low','close','adj_close')) for r in completed),
            'zero_volume_rows':sum(r['volume']==0 for r in completed),'negative_volume_rows':sum(valid(r,'volume') and r['volume']<0 for r in completed),
            'zero_high_low_range_rows':sum(valid(r,'high') and valid(r,'low') and r['high']==r['low'] for r in completed),
            'volume_min':min(r['volume'] for r in completed if valid(r,'volume')),'volume_max':max(r['volume'] for r in completed if valid(r,'volume')),
            'corporate_action_event_counts':dict(Counter(x['event_type'] for x in actions)),
            'adjusted_close_differs_from_close_rows':sum(valid(r,'adj_close') and valid(r,'close') and abs(r['adj_close']-r['close'])>1e-5 for r in completed),
            'first_row':completed[0],'last_row':completed[-1],
            'calendar_validation':'No exchange calendar or second market-data vendor used; no claim of missing-session completeness.'
        }
        save(key+'_ETC_quality_report.json',quality)
        rec['quality_summary']={k:v for k,v in quality.items() if k in ('missing_or_nonfinite_by_column','ohlc_order_violation_count','zero_volume_rows','zero_high_low_range_rows','corporate_action_event_counts','adjusted_close_differs_from_close_rows','first_row','last_row')}
    except urllib.error.HTTPError as e:
        rec.update(http_status=e.code,error=str(e))
        with (OUT/(name+'_error.txt')).open('xb') as f: f.write(e.read())
    except Exception as e: rec['error']=type(e).__name__+': '+str(e)
    return rec

with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex: records=list(ex.map(fetch,ITEMS))
save('EUA_ETC_candidates_acquisition_log.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'ticker_limit':['CO2.L','CO2P.L','CARB.L'],'requests':records})
print(json.dumps([{k:v for k,v in r.items() if k not in ('headers','instrument_metadata')} for r in records],ensure_ascii=False,indent=2))
