from pathlib import Path
import urllib.request,json,hashlib,concurrent.futures
p=Path(__file__).resolve().parent
urls=['https://arxiv.org/html/2404.07410v2','https://arxiv.org/abs/2210.08001','https://arxiv.org/abs/2106.14448','https://proceedings.mlr.press/v108/janati20a.html','https://proceedings.mlr.press/v139/zhang21p.html']
def get(u):
 try:
  b=urllib.request.urlopen(u,timeout=30).read(); name=u.split('/')[-1]+'.html'; (p/name).write_bytes(b)
  return dict(url=u,status='fetched',bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),snapshot=name)
 except Exception as e: return dict(url=u,status='failed',error=str(e))
rows=list(concurrent.futures.ThreadPoolExecutor(6).map(get,urls)); (p/'primary_receipts.json').write_text(json.dumps({'queries':['shift consistency regularization prediction invariant','time series forecasting phase consistency optimal transport shift equivariance','optimal transport time series alignment','consistency polyphase'],'sources':rows},indent=2)); print(json.dumps(rows,indent=2))
