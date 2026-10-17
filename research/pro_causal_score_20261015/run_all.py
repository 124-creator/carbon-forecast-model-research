"""Execute one frozen experiment into a new directory; no source/holdout changes."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
import mathematical_control,experiment,baselines,analyze,audit

def main():
 root=Path(__file__).resolve().parent;ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--data',default=str(root.parents[1]/'data/parsed/figshare_28682348/guangzhou_reported.csv'));ap.add_argument('--workers',type=int,default=2)
 args=ap.parse_args();out=Path(args.out).resolve()
 if not 1<=args.workers<=3:raise ValueError('workers1..3')
 if out.exists() and any(out.iterdir()):raise FileExistsError('fresh output required')
 tests={}
 for flags,label in [([], 'normal'),(['-O'],'optimized')]:
  proc=subprocess.run([sys.executable,'-B',*flags,str(root/'test_core.py')],capture_output=True,text=True)
  tests[label]={'code':proc.returncode,'stdout':proc.stdout,'stderr':proc.stderr}
  if proc.returncode:raise RuntimeError(proc.stderr)
 mathout=out.parent/(out.name+'_mathematical')
 if mathout.exists() and any(mathout.iterdir()):raise FileExistsError('fresh math output required')
 control=mathematical_control.run(mathout)
 if not control['gate']:raise RuntimeError('STOP_MATHEMATICAL_CONTROL_FAILED')
 hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in root.glob('*.py')}
 start=time.perf_counter();experiment.run(args.data,out,args.workers)
 for f in mathout.iterdir():(out/f.name).write_bytes(f.read_bytes())
 baselines.run(args.data,out);analyze.run(out);audit.run(args.data,out)
 (out/'TESTS.json').write_text(json.dumps(tests,indent=2));(out/'SOURCE_HASHES.json').write_text(json.dumps(hashes,indent=2))
 if hashes!={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in root.glob('*.py')}:raise RuntimeError('execution source changed')
 (out/'EXECUTION.json').write_text(json.dumps({'complete':True,'seconds_after_tests':time.perf_counter()-start,'source_unchanged':True,'protocol_hash':hashlib.sha256((root/'PROTOCOL.md').read_bytes()).hexdigest()},indent=2))
 manifest=[]
 for f in sorted(out.rglob('*')):
  if f.is_file():manifest.append({'path':str(f.relative_to(out)),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
 (out/'MANIFEST.json').write_text(json.dumps({'files':manifest},indent=2))
 print('EXPERIMENT_COMPLETE',out,flush=True)
if __name__=='__main__':main()
