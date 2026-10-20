from pathlib import Path
import json,csv,hashlib,re,html,datetime
p=Path(__file__).resolve().parent
records=json.loads((p/'ACQUISITION_RECEIPTS.json').read_text())+[json.loads((p/'CALENDAR_ACQUISITION_RECEIPT.json').read_text())]
for r in records:
 if 'local_file' in r:
  b=(p/r['local_file']).read_bytes()
  if len(b)!=r['bytes'] or hashlib.sha256(b).hexdigest()!=r['sha256']: raise ValueError('receipt mismatch')
a=list(csv.DictReader((p/'raw_LOCAL_ONLY/ecb_cny_sample.csv').read_text().splitlines()))
b={r['TIME_PERIOD']:r for r in csv.DictReader((p/'raw_LOCAL_ONLY/ecb_usd_sample.csv').read_text().splitlines())}
rows=[]
for r in a:
 d=r['TIME_PERIOD']
 if not '2018-01-01'<=d<='2021-12-31': raise ValueError('out of scope')
 rows.append({'observation_date':d,'cny_per_usd_cross':float(r['OBS_VALUE'])/float(b[d]['OBS_VALUE']),'available_at_certified':None,'availability_rule':'prior_completed_Europe_calendar_day_only_assumption_not_vintage_certification'})
(p/'FX_SAMPLE_DERIVED_LOCAL_ONLY.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
t=(p/'raw_LOCAL_ONLY/eia_brent_metadata.html').read_text()
chunks=re.findall(r'<tr[^>]*>.*?</tr>',t,re.S|re.I)
chosen=[re.sub(r'\s+',' ',html.unescape(re.sub('<[^>]+>',' ',v))).strip() for v in chunks if re.search(r'2018 Jan-\s*(?:1|8) to',v)][:2]
(p/'BRENT_2018_WEEK_ROWS_LOCAL_ONLY.json').write_text(json.dumps({'rows':chosen,'unit':'USD/barrel','source':'Europe Brent Spot Price FOB; not US WTI','available_at_certified':None,'historical_release_vintage':None},indent=2),encoding='utf-8')
summary={'receipt_count':len(records),'total_received_bytes':sum(r.get('bytes',0) for r in records),'all_byte_hash_checks':'PASS','fx_rows':len(rows),'brent_week_rows':len(chosen),'market_data_read':False,'training':False,'commit_or_push':False,'verified_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(p/'VERIFICATION.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary)); print(chosen)
