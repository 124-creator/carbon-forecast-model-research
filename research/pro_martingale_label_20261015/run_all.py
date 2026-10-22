"""One fixed experiment in a NEW output directory, without implicit overwrite."""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
import experiment,baselines,analyze,audit

def main():
 a=argparse.ArgumentParser();a.add_argument('--out',required=True);a.add_argument('--prefix');a.add_argument('--workers',type=int,default=3);o=a.parse_args()
 root=Path(__file__).resolve().parent;out=Path(o.out).resolve()
 prefix=Path(o.prefix).resolve() if o.prefix else root.parents[1]/'data/parsed/figshare_28682348/guangzhou_reported.csv'
 if out.exists():raise FileExistsError('Use a new directory; never overwrite a published run')
 start=time.perf_counter();logs={}
 for flags,key in [([], 'normal'),(['-O'],'optimized')]:
  r=subprocess.run([sys.executable,'-B']+flags+['-m','unittest','discover','-s',str(root),'-p','test_core.py'],capture_output=True,text=True)
  logs[key]=r.stdout+r.stderr
  if r.returncode:raise RuntimeError(logs[key])
 experiment.run(out,prefix,o.workers)
 for key,s in logs.items():(out/f'TESTS_{key}.log').write_text(s)
 baselines.run(out);analyze.run(out);audit.run(out)
 experiment.jwrite(out/'EXECUTION.json',dict(complete=True,total_runner_seconds=time.perf_counter()-start,formal_neural_fits=432,protocol_commit=experiment.core.PROTOCOL_SHA))
 entries=[dict(path=str(f.relative_to(out)),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in sorted(out.rglob('*')) if f.is_file() and f.name!='RESULT_MANIFEST.json']
 experiment.jwrite(out/'RESULT_MANIFEST.json',dict(files=entries,manifest_excluded=True))
if __name__=='__main__':main()
