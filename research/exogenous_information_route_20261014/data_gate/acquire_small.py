from pathlib import Path
import requests, hashlib, json, datetime, concurrent.futures
ROOT=Path(__file__).resolve().parent
SOURCES={
 'ecb_cny_sample':'https://data-api.ecb.europa.eu/service/data/EXR/D.CNY.EUR.SP00.A?startPeriod=2018-01-01&endPeriod=2018-01-12&format=csvdata',
 'ecb_usd_sample':'https://data-api.ecb.europa.eu/service/data/EXR/D.USD.EUR.SP00.A?startPeriod=2018-01-01&endPeriod=2018-01-12&format=csvdata',
 'ecb_license':'https://www.ecb.europa.eu/services/using-our-site/disclaimer/html/index.en.html',
 'ecb_release':'https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html',
 'eia_brent_metadata':'https://www.eia.gov/dnav/pet/hist/RBRTED.htm',
 'eia_license':'https://www.eia.gov/about/copyrights_reuse.php',
 'eia_notes':'https://www.eia.gov/dnav/pet/TblDefs/pet_pri_spt_tbldef2.asp',
 'nbs_coal_201812':'https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1900166.html',
 'nbs_coal_201812late':'https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1900192.html',
}
def fetch(pair):
 name,url=pair; record={'id':name,'url':url,'cap_bytes':700000,'historical_first_available_at':None}
 try:
  with requests.get(url,timeout=(15,45),stream=True) as r:
   record.update(status=r.status_code,final_url=r.url,headers={k:v for k,v in r.headers.items() if k.lower() in ['content-type','content-length','last-modified','date','etag']})
   body=b''
   for chunk in r.iter_content(8192):
    body+=chunk
    if len(body)>700000: raise ValueError('bounded size cap exceeded')
   record.update(bytes=len(body),sha256=hashlib.sha256(body).hexdigest())
   filename=name+('.csv' if name.endswith('sample') else '.html')
   (ROOT/'raw_LOCAL_ONLY').mkdir(exist_ok=True)
   (ROOT/'raw_LOCAL_ONLY'/filename).write_bytes(body)
   record['local_file']='raw_LOCAL_ONLY/'+filename
   record['retrieval_state']='received' if r.status_code==200 else 'refused_or_http_error_STOP'
 except Exception as e: record.update(retrieval_state='transport_or_cap_failure',error=str(e))
 record['response_completed_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 return record
if __name__=='__main__':
 records=list(concurrent.futures.ThreadPoolExecutor(max_workers=4).map(fetch,SOURCES.items()))
 (ROOT/'ACQUISITION_RECEIPTS.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(records,ensure_ascii=False,indent=2))
