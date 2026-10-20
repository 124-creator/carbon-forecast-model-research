exec((__import__('pathlib').Path(__file__).parent/'acquire_sources.py').read_text(encoding='utf8').split('items=')[0])
items=[('nhits_paper_retry.pdf','https://arxiv.org/pdf/2201.12886'),('kernel_pooling.pdf','https://openaccess.thecvf.com/content_cvpr_2017/papers/Cui_Kernel_Pooling_for_CVPR_2017_paper.pdf'),('timemixer_paper.pdf','https://arxiv.org/pdf/2405.14616'),('wpmixer_readme_retry.md','https://raw.githubusercontent.com/Secure-and-Intelligent-Systems-Lab/WPMixer/74104c9dddd54d279eb8323f48934b4fd75fcae7/README.md')]
r=requests.get('https://api.github.com/repos/Secure-and-Intelligent-Systems-Lab/WPMixer/git/trees/74104c9dddd54d279eb8323f48934b4fd75fcae7?recursive=1',timeout=45); (D/'wpmixer_tree.json').write_text(r.text,encoding='utf8'); print([x['path'] for x in r.json().get('tree',[]) if x['path'].endswith('.py')])
receipts=json.loads((ROOT/'source_receipts.json').read_text())
for name,url in items:
 try:
  r=requests.get(url,timeout=60); r.raise_for_status(); (D/name).write_bytes(r.content); receipts.append({'name':name,'url':url,'sha256':hashlib.sha256(r.content).hexdigest(),'bytes':len(r.content),'retrieved':datetime.datetime.now(datetime.timezone.utc).isoformat()})
  if name.endswith('.pdf'):
   import pymupdf
   (D/(name+'.txt')).write_text('\n'.join(p.get_text() for p in pymupdf.open(stream=r.content,filetype='pdf')),encoding='utf8')
 except Exception as e: receipts.append({'name':name,'error':str(e)})
(ROOT/'source_receipts.json').write_text(json.dumps(receipts,indent=2),encoding='utf8');print(json.dumps(receipts[-4:],indent=2))
