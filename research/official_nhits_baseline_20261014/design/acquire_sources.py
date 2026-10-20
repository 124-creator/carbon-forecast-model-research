from pathlib import Path
import requests, hashlib, json, datetime
ROOT=Path(r'E:\碳价格\11\实验十一十一\田\第三次-tian-碳价格实验\official_nhits_baseline_20261014\design')
D=ROOT/'sources'; D.mkdir(parents=True,exist_ok=True)
items=[('nhits_paper.pdf','https://ojs.aaai.org/index.php/AAAI/article/download/25854/25626'),('wpmixer_paper.pdf','https://faculty.eng.usf.edu/yasiny/papers/aaai.pdf'),('spectral_paper.pdf','https://arxiv.org/pdf/1506.03767')]
receipts=[]
for repo,file,name in [('Nixtla/neuralforecast','neuralforecast/models/nhits.py','nhits.py'),('Secure-and-Intelligent-Systems-Lab/WPMixer','README.md','wpmixer_readme.md'),('kwuking/TimeMixer','models/TimeMixer.py','timemixer.py')]:
 try:
  r=requests.get('https://api.github.com/repos/'+repo+'/commits',params={'per_page':1},timeout=40); r.raise_for_status(); commit=r.json()[0]['sha']; (D/(name+'.commit.json')).write_text(r.text,encoding='utf8'); items.append((name,'https://raw.githubusercontent.com/'+repo+'/'+commit+'/'+file))
 except Exception as e: receipts.append({'repo':repo,'error':str(e)})
for name,url in items:
 try:
  r=requests.get(url,timeout=60); r.raise_for_status(); b=r.content; (D/name).write_bytes(b); rec={'name':name,'url':url,'status':r.status_code,'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'retrieved':datetime.datetime.now(datetime.timezone.utc).isoformat()}; receipts.append(rec)
  if name.endswith('.pdf'):
   import fitz
   doc=fitz.open(stream=b,filetype='pdf'); (D/(name+'.txt')).write_text('\n'.join(p.get_text() for p in doc),encoding='utf8')
 except Exception as e: receipts.append({'name':name,'url':url,'error':str(e)})
(ROOT/'source_receipts.json').write_text(json.dumps(receipts,indent=2),encoding='utf8'); print(json.dumps(receipts,indent=2))
