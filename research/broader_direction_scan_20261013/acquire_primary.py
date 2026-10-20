import requests,pathlib,json,datetime,hashlib
root=pathlib.Path('E:/碳价格/独立数据归档/carbon-forecast-model-research/research/broader_direction_scan_20261013')
items={'multi_step_aci':'https://arxiv.org/html/2409.14792v1','transfer_author_pdf':'https://fanchenyou.github.io/docs/icann24.pdf','transfer_journal':'https://sysmath.cjoe.ac.cn/jweb_xtkxysx/EN/10.12341/jssms250520','liquidity_primary':'https://www.mdpi.com/2071-1050/17/12/5407'}
for key,url in items.items():
 rec={'requested_url':url}
 try:
  r=requests.get(url,timeout=40);rec.update(http_status=r.status_code,final_url=r.url,bytes=len(r.content),sha256=hashlib.sha256(r.content).hexdigest(),type=r.headers.get('Content-Type'))
  if r.status_code==200:
   suffix='.pdf' if r.content.startswith(b'%PDF') else '.html'
   dest=root/(key+'_local_original'+suffix);dest.write_bytes(r.content);rec['local_file']=dest.name
 except Exception as e:rec['error']=str(e)
 rec['returned_at']=datetime.datetime.now().astimezone().isoformat();(root/(key+'_receipt.json')).write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(rec,ensure_ascii=False))
