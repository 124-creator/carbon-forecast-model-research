import requests,pathlib,json,datetime,hashlib
root=pathlib.Path('E:/碳价格/独立数据归档/carbon-forecast-model-research/research/broader_direction_scan_20261013')
for key,url in {'grafiti':'https://arxiv.org/abs/2305.12932','event_neighbor':'https://arxiv.org/abs/2607.23426','quebec_metadata':'https://www.environnement.gouv.qc.ca/changements/carbone/documentation-en.htm','joint_observation':'https://arxiv.org/abs/2004.03398'}.items():
 rec={'requested_url':url}
 try:
  r=requests.get(url,timeout=30);rec.update({'http_status':r.status_code,'final_url':r.url,'bytes':len(r.content),'sha256':hashlib.sha256(r.content).hexdigest(),'type':r.headers.get('Content-Type')})
  if r.status_code==200:(root/(key+'_local_original.html')).write_bytes(r.content)
 except Exception as e:rec['error']=str(e)
 rec['returned_at']=datetime.datetime.now().astimezone().isoformat();(root/(key+'_receipt.json')).write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(rec,ensure_ascii=False))
