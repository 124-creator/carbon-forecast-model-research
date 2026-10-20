import requests,json,hashlib,datetime,pathlib,re
root=pathlib.Path('E:/碳价格/独立数据归档/carbon-forecast-model-research/research/broader_direction_scan_20261013')
urls={'mee_notice':'https://www.mee.gov.cn/xxgk2018/xxgk/xxgk06/202110/t20211026_957871.html','copyright_law':'https://www.ncac.gov.cn/xxfb/flfg/flfg_532/202103/t20210309_50530.html'}
for key,url in urls.items():
 try:
  r=requests.get(url,timeout=40);rec={'url':url,'final_url':r.url,'http_status':r.status_code,'bytes':len(r.content),'sha256':hashlib.sha256(r.content).hexdigest(),'returned_at':datetime.datetime.now().astimezone().isoformat(),'type':r.headers.get('Content-Type')}
  if r.status_code==200:
   (root/(key+'_local_original.html')).write_bytes(r.content)
   r.encoding='utf-8';from bs4 import BeautifulSoup
   text=BeautifulSoup(r.text,'html.parser').get_text(' ',strip=True)
   (root/(key+'_local_text.txt')).write_text(text,encoding='utf-8')
   print(key,text[:1200])
 except Exception as e:rec={'url':url,'error':str(e),'returned_at':datetime.datetime.now().astimezone().isoformat()}
 (root/(key+'_receipt.json')).write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(rec,ensure_ascii=False))
