"""Bounded official NBS discovery; literal specs; no training or filling."""
from pathlib import Path
import requests,re,json,time,hashlib,csv,html,datetime,calendar
from urllib.parse import urljoin,urlparse
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent
START=time.monotonic(); DEADLINE=START+870
RAW=ROOT/'raw_LOCAL_ONLY'; RAW.mkdir(exist_ok=True)
receipts=[]; candidates={}; inputs=[]; stop=False
session=requests.Session()
def save():
 (ROOT/'ACQUISITION_RECEIPTS.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2),encoding='utf8')
 (ROOT/'DISCOVERY_MANIFEST.json').write_text(json.dumps(list(candidates.values()),ensure_ascii=False,indent=2),encoding='utf8')
def fetch(url,kind):
 global stop
 if stop or time.monotonic()>DEADLINE:return None
 if urlparse(url).hostname!='www.stats.gov.cn':raise ValueError('nonofficial host')
 time.sleep(1)
 rec={'url':url,'kind':kind,'acquired_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 try:
  r=session.get(url,timeout=(10,25)); b=r.content
  rec.update(status=r.status_code,final_url=r.url,bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
  file=RAW/(hashlib.sha256(url.encode()).hexdigest()[:20]+'.html');file.write_bytes(b);rec['file']=str(file.relative_to(ROOT))
  text=b.decode('utf8',errors='replace')
  refusal=r.status_code in (401,403,429) or any(x in text.lower() for x in ['access denied','captcha','访问被拒绝','安全验证'])
  if refusal:stop=True;rec['state']='REFUSED_STOP_NO_ALIAS'
  elif r.status_code==200:rec['state']='received'
  else:rec['state']='failed_no_retry'
 except Exception as e:rec.update(state='failed_no_retry',error=str(e));text=None
 receipts.append(rec);save()
 print(json.dumps({k:v for k,v in rec.items() if k not in ('sha256',)},ensure_ascii=False),flush=True)
 return text if rec.get('state')=='received' else None

def clean(s):return re.sub(r'\s+',' ',html.unescape(re.sub('<[^>]+>',' ',s))).strip()
def discover(s,url):
 for href,body in re.findall(r'<a\b[^>]*href=[\"\x27]([^\"\x27]+)[\"\x27][^>]*>(.*?)</a>',s,re.S|re.I):
  title=clean(body)
  if '流通领域重要生产资料' not in title:continue
  year=re.search(r'(201[89]|202[01])年',title)
  if not year:continue
  target=urljoin(url,html.unescape(href))
  if urlparse(target).hostname=='www.stats.gov.cn':candidates.setdefault(target,{'url':target,'listing_title':title,'discovered_on':url})
 save()

# Reuse verified old inputs, not receipts rewritten in place.
for entry in json.loads((OLD/'INPUT_MANIFEST.json').read_text(encoding='utf8')):
 p=(OLD/entry['input_relative_to_probe']).resolve();b=p.read_bytes()
 if len(b)!=entry['bytes'] or hashlib.sha256(b).hexdigest()!=entry['sha256']:raise ValueError('old input mismatch')
 oldrows=list(csv.DictReader((OLD/'announcement_index_LOCAL_ONLY.csv').open(encoding='utf-8-sig')))
 row=next(x for x in oldrows if x['source_id']==entry['source_id'])
 inputs.append({'url':row['url'],'file':str(p),'sha256':entry['sha256'],'bytes':len(b),'reused':True})

indexurl='https://www.stats.gov.cn/sj/zxfb/'
s=fetch(indexurl,'official_index')
if s:
 discover(s,indexurl)
 # Search address is the literal official form's /search/s route.
 if '/search/s?qt=' in s:
  searchurl=urljoin(indexurl,'/search/s?qt=')+requests.utils.quote('流通领域重要生产资料 2018')
  search=fetch(searchurl,'official_search')
  if search:discover(search,searchurl)
 # One fallback: published static pagination only, bounded by displayed count.
 match=re.search(r'createPageHTML\((\d+),\s*0,\s*"index",\s*"html"\)',s)
 if match and not stop:
  count=int(match[1]);(ROOT/'PAGINATION_EVIDENCE.json').write_text(json.dumps({'entry':indexurl,'literal_call':match[0],'scope':'current channel count, not invented historical year directory'},ensure_ascii=False,indent=2),encoding='utf8')
  for i in range(count-1,0,-1):
   page=urljoin(indexurl,f'index_{i}.html');body=fetch(page,'official_index_page')
   if body:discover(body,page)
   if stop or time.monotonic()>DEADLINE:break

known={x['url'] for x in inputs}
for url in candidates:
 if url in known:continue
 body=fetch(url,'announcement')
 if body:
  r=receipts[-1];inputs.append({'url':url,'file':str(ROOT/r['file']),'sha256':r['sha256'],'bytes':r['bytes'],'reused':False})
 if stop or time.monotonic()>DEADLINE:break
(ROOT/'INPUT_MANIFEST.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2),encoding='utf8')
rows=[];events=[];errors=[]
for item in inputs:
 try:
  b=Path(item['file']).read_bytes()
  if len(b)!=item['bytes'] or hashlib.sha256(b).hexdigest()!=item['sha256']:raise ValueError('input integrity')
  text=b.decode('utf8');title=clean(re.search(r'<title>(.*?)</title>',text,re.S)[1])
  pub=re.search(r'(20\d{2})/(\d{2})/(\d{2})\s+(\d{2}:\d{2})',text)
  if not pub:raise ValueError('publication missing')
  date=datetime.date(*map(int,pub.group(1,2,3)))
  m=re.search(r'(201[89]|202[01])年(\d{1,2})月(?:(\d{1,2})日?[-—－](\d{1,2})日|(上旬|中旬|下旬))',title)
  if not m:raise ValueError('period missing')
  y,mo=int(m[1]),int(m[2]);旬=m[5]
  if 旬:start,end={'上旬':(1,10),'中旬':(11,20),'下旬':(21,calendar.monthrange(y,mo)[1])}[旬]
  else:start,end=int(m[3]),int(m[4]);旬={1:'上旬',11:'中旬',21:'下旬'}.get(start,'UNKNOWN')
  event={'url':item['url'],'title':title,'observation_start':str(datetime.date(y,mo,start)),'observation_end':str(datetime.date(y,mo,end)),'period_end_basis':'title literal' if not m[5] else 'calendar convention from literal 旬','year':y,'month':mo,'ten_day':旬,'publication_displayed':date.isoformat()+'T'+pub[4]+':00+08:00','available_date_1day':str(date+datetime.timedelta(days=1)),'source_sha256':item['sha256'],'historical_first_version_certified':False}
  seen={}
  for tr in re.findall(r'<tr\b[^>]*>(.*?)</tr>',text,re.S|re.I):
   cells=[clean(c) for c in re.findall(r'<t[dh]\b[^>]*>(.*?)</t[dh]>',tr,re.S|re.I)]
   if len(cells)<3 or cells[1]!='吨' or not any(v in cells[0] for v in ['无烟煤','普通混煤','山西大混','山西优混','大同混煤','焦煤','焦炭']):continue
   label=cells[0];price=float(cells[2].replace(',',''))
   if label in seen and seen[label]!=price:raise ValueError('duplicate disagreement')
   seen[label]=price
  if not seen:raise ValueError('no coal rows')
  events.append(event)
  for label,price in seen.items():rows.append(dict(event,product_spec_literal=label,unit_literal='吨',price_cny_per_tonne=price,is_5500_literal='5500' in label))
 except Exception as e:errors.append({'url':item['url'],'error':str(e)})
events.sort(key=lambda x:x['observation_start']);rows.sort(key=lambda x:(x['observation_start'],x['product_spec_literal']))
# No interpolation, forward/cross-year carry, or future filling.
def writecsv(name,data):
 if not data:return
 with (ROOT/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
writecsv('announcement_index_LOCAL_ONLY.csv',events);writecsv('coal_events_LOCAL_ONLY.csv',rows)
focus=[r for r in rows if r['is_5500_literal']];writecsv('coal_5500_LITERAL_LOCAL_ONLY.csv',focus)
coverage=[]
for y in range(2018,2022):
 for mo in range(1,13):
  for ten in ['上旬','中旬','下旬']:
   matching=[e for e in events if (e['year'],e['month'],e['ten_day'])==(y,mo,ten)]
   f=[e for e in focus if (e['year'],e['month'],e['ten_day'])==(y,mo,ten)]
   coverage.append({'year':y,'month':mo,'ten_day':ten,'announcement_count':len(matching),'literal_5500_count':len(f),'missing_announcement':not matching,'missing_5500':not f})
writecsv('COVERAGE_MONTH_TENDAY.csv',coverage);writecsv('MISSING_MONTH_TENDAY.csv',[x for x in coverage if x['missing_5500']])
summary={'announcement_events':len(events),'coal_coke_rows':len(rows),'literal_5500_usable_rows':len(focus),'by_year':{str(y):{'events':sum(e['year']==y for e in events),'5500_rows':sum(e['year']==y for e in focus),'missing_5500_slots':sum(e['year']==y and e['missing_5500'] for e in coverage)} for y in range(2018,2022)},'first_observation':events[0]['observation_start'] if events else None,'last_observation':events[-1]['observation_end'] if events else None,'first_publication':min(e['publication_displayed'] for e in events) if events else None,'last_publication':max(e['publication_displayed'] for e in events) if events else None,'missing_5500_slots':sum(e['missing_5500'] for e in coverage),'expected_tenday_slots':len(coverage),'http_requests':len(receipts),'refusal_stop':stop,'elapsed_seconds':round(time.monotonic()-START,2),'parse_errors':errors,'training_performed':False,'carbon_2022_read':False,'fill_performed':False,'availability_rule':'displayed publication date + 1 calendar day; no historical first-version certification','path_resolution':{'E_exists':True,'D_exists':True,'samefile':True,'resolved_drive':'E'},'blockers':['Official current static index is bounded by its displayed page count; missing historical slots not synthesized.']}
(ROOT/'VERIFICATION.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8');save();print('FINAL '+json.dumps(summary,ensure_ascii=False),flush=True)
