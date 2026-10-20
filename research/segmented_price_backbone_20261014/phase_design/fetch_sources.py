from pathlib import Path
import urllib.request, json, concurrent.futures, hashlib
P=Path(__file__).resolve().parent
urls={
 'lps_tree':'https://api.github.com/repos/raymondyeh07/learnable_polyphase_sampling/git/trees/main?recursive=1',
 'phaseformer_tree':'https://api.github.com/repos/neumyor/PhaseFormer_TSL/git/trees/main?recursive=1',
 'patchtst':'https://raw.githubusercontent.com/yuqinie98/PatchTST/main/PatchTST_supervised/layers/PatchTST_backbone.py',
 'segrnn_readme':'https://raw.githubusercontent.com/lss-1138/SegRNN/main/README.md',
 'lps_readme':'https://raw.githubusercontent.com/raymondyeh07/learnable_polyphase_sampling/main/README.md',
 'ismrnn':'https://arxiv.org/html/2407.10768v1',
}
def fetch(k,u):
 try:
  b=urllib.request.urlopen(u,timeout=20).read(); (P/(k+'.txt')).write_bytes(b)
  return {'key':k,'url':u,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'status':'ok'}
 except Exception as e: return {'key':k,'url':u,'status':str(e)}
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
 rows=list(ex.map(lambda kv:fetch(*kv),urls.items()))
(P/'source_fetch.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
for row in rows: print(row)
for k in ['lps_tree','phaseformer_tree']:
 f=P/(k+'.txt')
 if f.exists():
  d=json.loads(f.read_text()); print(k, '\n'.join(x['path'] for x in d.get('tree',[]) if x['path'].endswith('.py') and any(w in x['path'] for w in ['layer','model','poly','logit'])))
