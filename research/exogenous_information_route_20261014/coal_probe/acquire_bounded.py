"""Small official-only batch; never retries a refusal; no model inputs."""
from pathlib import Path
import requests, json, hashlib, datetime, re
ROOT=Path(__file__).resolve().parent
SOURCES={
'archive':'https://www.stats.gov.cn/xxgk/sjfb/zxfb2020/',
'coal_2019_12a':'https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1900559.html',
'coal_2020_12a':'https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1900929.html',
'coal_2021_12b':'https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1901318.html',
}
def main():
 raw=ROOT/'raw_LOCAL_ONLY';raw.mkdir(exist_ok=True)
 receipts=[]
 for key,url in SOURCES.items():
  rec={'id':key,'url':url,'historical_first_version_certified':False,'cap_bytes':1000000}
  try:
   with requests.get(url,timeout=(10,30),stream=True) as r:
    rec.update(status=r.status_code,final_url=r.url,headers={k:v for k,v in r.headers.items() if k.lower() in ['date','etag','last-modified','content-type']})
    b=b''
    for c in r.iter_content(8192):
     b+=c
     if len(b)>1000000:raise ValueError('size cap exceeded')
    f=raw/(key+'.html');f.write_bytes(b)
    rec.update(bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),file='raw_LOCAL_ONLY/'+f.name,state='received' if r.status_code==200 else 'REFUSED_STOP_NO_ALIAS')
  except Exception as e:rec.update(state='failed_no_retry',error=str(e))
  rec['completed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();receipts.append(rec)
  (ROOT/'ACQUISITION_RECEIPTS.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2),encoding='utf-8')
  print(json.dumps(rec,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
