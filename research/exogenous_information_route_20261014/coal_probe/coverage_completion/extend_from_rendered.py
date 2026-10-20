"""Use exact visible official search links; preserve prior batch receipts."""
from pathlib import Path
import json,time,datetime,csv
ROOT=Path(__file__).resolve().parent
source=(ROOT/'acquire_completion.py').read_text(encoding='utf8')
ns={'__file__':str(ROOT/'acquire_completion.py')}
exec(compile(source.split('# Reuse verified old inputs')[0],str(ROOT/'acquire_completion.py'),'exec'),ns)
receipts=json.loads((ROOT/'ACQUISITION_RECEIPTS.json').read_text(encoding='utf8'))
(ROOT/'BATCH_01_RECEIPTS.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2),encoding='utf8')
start=datetime.datetime.fromisoformat(receipts[0]['acquired_utc']); remaining=870-(datetime.datetime.now(datetime.timezone.utc)-start).total_seconds()
ns['START']=time.monotonic()-(datetime.datetime.now(datetime.timezone.utc)-start).total_seconds();ns['DEADLINE']=time.monotonic()+max(0,remaining);ns['receipts']=receipts
inputs=json.loads((ROOT/'INPUT_MANIFEST.json').read_text(encoding='utf8')); ns['inputs']=inputs
seen={(int(r['year']),int(r['month']),r['ten_day']) for r in csv.DictReader((ROOT/'announcement_index_LOCAL_ONLY.csv').open(encoding='utf-8-sig'))}
selected=json.loads((ROOT/'SEARCH_SELECTED_DISCOVERY.json').read_text(encoding='utf8'))
for a in selected:
 m=ns['re'].search(r'(201[89]|202[01])年(\d+)月(?:(\d+)日?[-—－](\d+)日|(上旬|中旬|下旬))',a['title'])
 key=(int(m[1]),int(m[2]),m[5] or {1:'上旬',11:'中旬',21:'下旬'}.get(int(m[3]),'UNKNOWN'))
 if key in seen:continue
 if time.monotonic()>ns['DEADLINE'] or ns['stop']:break
 body=ns['fetch'](a['url'],'announcement_from_rendered_search')
 if body:
  r=ns['receipts'][-1];inputs.append({'url':a['url'],'file':str(ROOT/r['file']),'sha256':r['sha256'],'bytes':r['bytes'],'reused':False});seen.add(key)
 (ROOT/'BATCH_02_INPUT_MANIFEST.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2),encoding='utf8')
(ROOT/'BATCH_02_RECEIPTS.json').write_text(json.dumps(ns['receipts'][75:],ensure_ascii=False,indent=2),encoding='utf8')
exec(compile("(ROOT/'INPUT_MANIFEST.json')"+source.split("(ROOT/'INPUT_MANIFEST.json')",1)[1],str(ROOT/'acquire_completion.py'),'exec'),ns)
