import concurrent.futures, datetime, hashlib, json, time
from pathlib import Path
import requests
import fitz
ROOT=Path(__file__).resolve().parent
SOURCES={
 'segrnn_paper':'https://arxiv.org/pdf/2308.11200',
 'segrnn_model':'https://raw.githubusercontent.com/lss-1138/SegRNN/main/models/SegRNN.py',
 'segrnn_readme':'https://raw.githubusercontent.com/lss-1138/SegRNN/main/README.md',
 'nbeats_paper':'https://arxiv.org/pdf/1905.10437',
 'nbeats_model':'https://raw.githubusercontent.com/ServiceNow/N-BEATS/master/models/nbeats.py',
 'etsformer_paper':'https://arxiv.org/pdf/2202.01381',
 'etsformer_readme':'https://raw.githubusercontent.com/salesforce/ETSformer/main/README.md',
 'etsformer_decoder':'https://raw.githubusercontent.com/salesforce/ETSformer/main/models/etsformer/decoder.py',
 'nhits_paper':'https://arxiv.org/pdf/2201.12886',
 'nhits_model':'https://raw.githubusercontent.com/Nixtla/neuralforecast/main/neuralforecast/models/nhits.py',
 'forecastcf_paper':'https://arxiv.org/pdf/2310.08137',
 'forecastcf_readme':'https://raw.githubusercontent.com/zhendong3wang/counterfactual-explanations-for-forecasting/main/README.md',
 'tangentprop_paper':'https://proceedings.neurips.cc/paper/1991/file/65658fde58ab3c2b6e5132a39fae7cb9-Paper.pdf',
 'selfinterpret_paper':'https://proceedings.mlr.press/v202/yan23d/yan23d.pdf',
 'selfinterpret_readme':'https://raw.githubusercontent.com/Wang-ML-Lab/self-interpretable-time-series/main/README.md',
 'contex_paper':'https://arxiv.org/pdf/2606.18049',
 'star_primary':'https://onlinelibrary.wiley.com/doi/10.1111/agec.12707',
}
def fetch(item):
 name,url=item; start=time.monotonic(); record={'name':name,'url':url,'retrieved_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 try:
  r=None
  attempts=[]
  for attempt in range(3):
   try:
    r=requests.get(url,timeout=35); attempts.append({'attempt':attempt+1,'status':r.status_code}); break
   except requests.RequestException as exc:
    attempts.append({'attempt':attempt+1,'error':str(exc)})
  record['attempts']=attempts
  if r is None: raise RuntimeError('all attempts failed')
  record.update(status=r.status_code,final_url=r.url,bytes=len(r.content),sha256=hashlib.sha256(r.content).hexdigest())
  r.raise_for_status(); suffix='.pdf' if r.content.startswith(b'%PDF') else '.txt'
  path=ROOT/'sources'/(name+suffix); path.parent.mkdir(exist_ok=True); path.write_bytes(r.content); record['file']=str(path.relative_to(ROOT))
  if suffix=='.pdf':
   text='\n'.join(p.get_text() for p in fitz.open(stream=r.content,filetype='pdf')); (ROOT/'sources'/(name+'.txt')).write_text(text,encoding='utf-8'); record['text_chars']=len(text)
 except Exception as exc: record['error']=str(exc)
 record['elapsed_seconds']=time.monotonic()-start
 return record
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool: records=list(pool.map(fetch,SOURCES.items()))
 (ROOT/'primary_receipts.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(records,ensure_ascii=False,indent=2))
