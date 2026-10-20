from pathlib import Path
import requests,json,hashlib,base64,pymupdf
D=Path(__file__).parent/'sources'; rec=json.loads((D.parent/'source_receipts.json').read_text())
items=[('pnmax.pdf','https://heungno.net/wp-content/uploads/2012/10/2013-ICCE-Las-vegas.pdf')]
for f in ['models/decomposition.py','models/wavelet_patch_mixer.py']:
 url='https://api.github.com/repos/Secure-and-Intelligent-Systems-Lab/WPMixer/contents/'+f+'?ref=74104c9dddd54d279eb8323f48934b4fd75fcae7'
 try:
  r=requests.get(url,timeout=45);r.raise_for_status(); j=r.json();b=base64.b64decode(j['content']); name='wp_'+Path(f).name;(D/name).write_bytes(b);blob=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest();rec.append({'name':name,'url':url,'sha256':hashlib.sha256(b).hexdigest(),'git_blob':blob,'blob_verified':blob==j['sha']});print(name,b.decode()[:18000])
 except Exception as e:rec.append({'url':url,'error':str(e)})
for name,url in items:
 try:
  r=requests.get(url,timeout=45);r.raise_for_status();b=r.content;(D/name).write_bytes(b);(D/(name+'.txt')).write_text('\n'.join(p.get_text() for p in pymupdf.open(stream=b,filetype='pdf')),encoding='utf8');rec.append({'name':name,'url':url,'sha256':hashlib.sha256(b).hexdigest()})
 except Exception as e:rec.append({'name':name,'error':str(e)})
(D.parent/'source_receipts.json').write_text(json.dumps(rec,indent=2),encoding='utf8')
