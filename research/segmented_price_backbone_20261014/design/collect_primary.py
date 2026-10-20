"""Primary-source receipts only; never accesses market data."""
from pathlib import Path
import concurrent.futures, datetime, hashlib, json, urllib.request
ROOT = Path(__file__).resolve().parent
SOURCES = {
 'stability_ac_paper': 'https://arxiv.org/pdf/2601.10863v3',
 'stability_ac_author_code': 'https://api.github.com/repos/causify-ai/beyond_accuracy/git/trees/main?recursive=1',
 'forking_current_paper': 'https://arxiv.org/pdf/2510.04487v5',
 'forking_original_paper': 'https://arxiv.org/pdf/1711.11053v2',
 'segrnn_paper': 'https://arxiv.org/pdf/2308.11200',
 'nhits_paper': 'https://ojs.aaai.org/index.php/AAAI/article/download/25854/25626',
 'revision_paper': 'https://public.econ.duke.edu/~ap172/Patton_Timmermann_bounds_JBES_2012.pdf',
 'deepgmm_paper': 'https://arxiv.org/pdf/1905.12495',
 'forking_sequences_paper': 'https://openreview.net/pdf?id=dXdycy7WCX',
 'segrnn_readme': 'https://raw.githubusercontent.com/lss-1138/SegRNN/main/README.md',
 'segrnn_model': 'https://raw.githubusercontent.com/lss-1138/SegRNN/main/models/SegRNN.py',
 'nhits_readme': 'https://raw.githubusercontent.com/cchallu/n-hits/main/README.md',
 'nhits_author_model': 'https://raw.githubusercontent.com/Nixtla/neuralforecast/main/neuralforecast/models/nhits.py',
 'deepgmm_readme': 'https://raw.githubusercontent.com/CausalML/DeepGMM/master/README.md',
 'deepgmm_objective': 'https://api.github.com/repos/CausalML/DeepGMM/contents/game_objectives',
 'segrnn_commit': 'https://api.github.com/repos/lss-1138/SegRNN/commits/main',
 'nhits_commit': 'https://api.github.com/repos/Nixtla/neuralforecast/commits/main',
 'deepgmm_commit': 'https://api.github.com/repos/CausalML/DeepGMM/commits/master',
}
def fetch(item):
 name, url = item
 r = {'name': name, 'url': url, 'retrieved_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
 try:
  request = urllib.request.Request(url, headers={'User-Agent': 'research-source-audit/1.0'})
  with urllib.request.urlopen(request, timeout=35) as response:
   content = response.read(); r.update(status=response.status, final_url=response.url, content_type=response.headers.get('Content-Type'))
  suffix = '.pdf' if content.startswith(b'%PDF') else '.txt'
  path = ROOT / 'sources' / (name + suffix); path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(content)
  r.update(file=str(path.relative_to(ROOT)), bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
  if name.endswith('_commit'): r['commit_sha'] = json.loads(content)['sha']
 except Exception as exc: r['error'] = str(exc)
 return r
if __name__ == '__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor: receipts = list(executor.map(fetch, SOURCES.items()))
 (ROOT / 'source_receipts.json').write_text(json.dumps(receipts, ensure_ascii=False, indent=2), encoding='utf-8')
 print(json.dumps(receipts, ensure_ascii=False, indent=2))
