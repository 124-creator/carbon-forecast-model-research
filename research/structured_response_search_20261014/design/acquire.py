from pathlib import Path
import urllib.request,json,hashlib,datetime,concurrent.futures
ROOT=Path(__file__).parent; R=ROOT/'receipts';R.mkdir(exist_ok=True)
def fetch(url):
 req=urllib.request.Request(url,headers={'User-Agent':'structured-response-research'})
 with urllib.request.urlopen(req,timeout=40) as f:return f.read(),f.status,dict(f.headers)
def repo(spec):
 name,path=spec; rec={'repo':name,'path':path,'time':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 try:
  b,status,headers=fetch('https://api.github.com/repos/'+name+'/commits?per_page=1');(R/(name.replace('/','_')+'_commit.json')).write_bytes(b);sha=json.loads(b)[0]['sha']; rec['commit']=sha
  url=f'https://api.github.com/repos/{name}/contents/{path}?ref={sha}'; b,status,headers=fetch(url);o=json.loads(b);import base64; body=base64.b64decode(o['content']); blob=hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest();rec.update(url=url,status=status,blob_expected=o['sha'],blob_actual=blob,verified=blob==o['sha'],bytes=len(body));(R/(name.replace('/','_')+'_'+Path(path).name)).write_bytes(body)
 except Exception as e:rec['error']=repr(e)
 return rec
def paper(spec):
 name,url=spec;rec={'name':name,'url':url}
 try:
  b,status,headers=fetch(url);rec.update(status=status,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),content_type=headers.get('Content-Type'));(R/(name+'.pdf')).write_bytes(b)
 except Exception as e:rec['error']=repr(e)
 return rec
if __name__=='__main__':
 specs=[('ServiceNow/N-BEATS','models/nbeats.py'),('cure-lab/LTSF-Linear','models/DLinear.py'),('thuml/Koopa','models/Koopa.py'),('state-spaces/s4','src/models/sequence/ss/ssm.py')]
 papers=[('koopa','https://papers.neurips.cc/paper_files/paper/2023/file/28b3dc0970fa4624a63278a4268de997-Paper-Conference.pdf'),('nbeats','https://arxiv.org/pdf/1905.10437'),('s4','https://arxiv.org/pdf/2111.00396')]
 with concurrent.futures.ThreadPoolExecutor(max_workers=7) as ex:records=list(ex.map(repo,specs))+list(ex.map(paper,papers))
 (R/'acquisition.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(records,ensure_ascii=False,indent=2))
 for p in R.glob('*.pdf'):
  try:
   import pymupdf
   d=pymupdf.open(p);text='\n'.join(page.get_text() for page in d);p.with_suffix('.txt').write_text(text,encoding='utf-8');print(p.name,'pages',len(d),'textchars',len(text))
  except Exception as e:print('PDF extraction failed',p.name,repr(e))
