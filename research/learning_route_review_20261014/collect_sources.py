from pathlib import Path
import urllib.request, hashlib, json, concurrent.futures
ROOT=Path(__file__).resolve().parent
SOURCES={
 'fraug':'https://arxiv.org/html/2302.09292',
 'cauker':'https://arxiv.org/html/2508.02879',
 'tempopfn':'https://arxiv.org/html/2510.25502',
 'tangent_original':'https://proceedings.neurips.cc/paper/1991/file/65658fde58ab3c2b6e5132a39fae7cb9-Paper.pdf',
 'crfnet':'https://pmc.ncbi.nlm.nih.gov/articles/PMC12019581/',
 'tangentprop':'https://www.deeplearningbook.org/contents/regularization.html'}
def fetch(item):
 name,url=item
 row={'name':name,'url':url}
 try:
  req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
  with urllib.request.urlopen(req,timeout=75) as r: data=r.read();row.update(status=r.status,final_url=r.url,content_type=r.headers.get('Content-Type'))
  ext='.pdf' if data.startswith(b'%PDF') else '.html' if b'<html' in data[:2000].lower() else '.txt'
  p=ROOT/'sources'/(name+ext);p.parent.mkdir(exist_ok=True);p.write_bytes(data)
  row.update(path=str(p),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
  if ext=='.pdf':
   try:
    import fitz
    d=fitz.open(p); out=p.with_suffix('.txt');out.write_text('\n'.join(page.get_text() for page in d),encoding='utf-8');row['text_path']=str(out)
   except Exception as e:row['text_extraction_error']=repr(e)
 except Exception as e:row['error']=repr(e)
 return row
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=7) as ex: rows=list(ex.map(fetch,SOURCES.items()))
 (ROOT/'source_receipts.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(rows,ensure_ascii=False,indent=2))
