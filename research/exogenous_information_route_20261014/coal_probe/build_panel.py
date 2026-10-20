"""Verify frozen receipts then extract only 2018--2021 official coal rows."""
from pathlib import Path
import json,csv,re,html,hashlib,datetime,calendar
ROOT=Path(__file__).resolve().parent
GATE=ROOT.parent/'data_gate'
def clean(s):return re.sub(r'\s+',' ',html.unescape(re.sub(r'<[^>]+>',' ',s))).strip()
def build():
 own=json.loads((ROOT/'ACQUISITION_RECEIPTS.json').read_text(encoding='utf-8'))
 for receipt in own:
  body=(ROOT/receipt['file']).read_bytes()
  if len(body)!=receipt['bytes'] or hashlib.sha256(body).hexdigest()!=receipt['sha256']:raise ValueError('new acquisition receipt mismatch')
 prior=[r for r in json.loads((GATE/'ACQUISITION_RECEIPTS.json').read_text(encoding='utf-8')) if r['id'].startswith('nbs_coal')]
 sources=[(ROOT,r,r['file']) for r in own if r['id']!='archive']+[(GATE,r,r['local_file']) for r in prior]
 rows=[];index=[];manifest=[]
 for base,r,f in sources:
  b=(base/f).read_bytes()
  if len(b)!=r['bytes'] or hashlib.sha256(b).hexdigest()!=r['sha256']:raise ValueError('receipt mismatch')
  if r['status']!=200:raise ValueError('refused input')
  s=b.decode('utf-8');title=clean(re.search(r'<title>(.*?)</title>',s,re.S).group(1))
  pub=re.search(r'(20\d{2})/(\d{2})/(\d{2})\s+(\d{2}:\d{2})',s)
  if not pub:raise ValueError('publication missing')
  date=datetime.date(*map(int,pub.group(1,2,3)));ym=re.search(r'（(20\d{2})年(\d{1,2})月(.*?)）',title)
  if ym:
   year,month=int(ym[1]),int(ym[2]);digits=re.findall(r'\d+',ym[3]);start,end=map(int,digits)
  else:
   ym=re.search(r'(20\d{2})年(\d{1,2})月(上旬|中旬|下旬)',title)
   if not ym:raise ValueError('period missing')
   year,month=int(ym[1]),int(ym[2]);start,end={'上旬':(1,10),'中旬':(11,20),'下旬':(21,calendar.monthrange(year,month)[1])}[ym[3]]
  if not 2018<=year<=2021:raise ValueError('year outside scope')
  event={'source_id':r['id'],'url':r['url'],'title':title,'observation_start':str(datetime.date(year,month,start)),'observation_end':str(datetime.date(year,month,end)),'publication_displayed':date.isoformat()+'T'+pub[4]+':00+08:00','available_date_1day':str(date+datetime.timedelta(days=1)),'available_date_2day':str(date+datetime.timedelta(days=2)),'available_at_certified':None,'source_sha256':r['sha256'],'vintage':'current_snapshot_not_historical_first_version'}
  seen={}
  for tr in re.findall(r'<tr\b[^>]*>(.*?)</tr>',s,re.S|re.I):
   cells=[clean(c) for c in re.findall(r'<t[dh]\b[^>]*>(.*?)</t[dh]>',tr,re.S|re.I)]
   if len(cells)<3 or cells[1]!='吨' or not any(v in cells[0] for v in ['无烟煤','普通混煤','山西大混','山西优混','大同混煤','焦煤','焦炭']):continue
   label=cells[0];price=float(cells[2].replace(',',''))
   if label in seen:
    if seen[label]!=price:raise ValueError('desktop/mobile disagreement')
    continue
   seen[label]=price
   rows.append(dict(event,product_spec_literal=label,price_cny_per_tonne=price,unit_literal='吨',missing=False))
  if len(seen)!=7:raise ValueError('expected seven coal/coke rows')
  index.append(dict(event,coal_coke_rows=len(seen)));manifest.append({'source_id':r['id'],'bytes':len(b),'sha256':r['sha256'],'input_relative_to_probe':str(Path('..')/'data_gate'/f) if base==GATE else f})
 rows.sort(key=lambda r:(r['publication_displayed'],r['product_spec_literal']));index.sort(key=lambda r:r['publication_displayed'])
 def write_csv(name,data):
  with (ROOT/name).open('w',encoding='utf-8-sig',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 write_csv('coal_events_LOCAL_ONLY.csv',rows);write_csv('announcement_index_LOCAL_ONLY.csv',index)
 (ROOT/'INPUT_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
 summary={'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'new_http_requests':len(own),'new_http_200':sum(r.get('status')==200 for r in own),'new_received_bytes':sum(r.get('bytes',0) for r in own),'reused_prior_announcements':len(prior),'announcements':len(index),'coal_coke_rows':len(rows),'observation_years':sorted({int(r['observation_start'][:4]) for r in rows}),'missing_price_rows':sum(r['missing'] for r in rows),'complete_2018_2021_panel':False,'training_performed':False,'carbon_data_read':False,'historical_first_version_certified':False,'byte_and_hash_verification':'PASS','all_7_specs_each_event':'PASS','desktop_mobile_duplicates_collapsed':True,'availability_rule':'displayed_publication_date_plus_1_calendar_day_00:00_Asia_Shanghai; sensitivity_plus_2_days; never_observation_end'}
 (ROOT/'VERIFICATION.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':build()
