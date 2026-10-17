"""Reproduce a complete bounded study in a NEW output folder, no holdout access."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--data',default=str(R.parents[1]/'data/parsed/figshare_28682348/guangzhou_reported.csv'));p.add_argument('--workers',type=int,default=2);a=p.parse_args()
 out=Path(a.out).resolve()
 if out.exists():raise FileExistsError('Do not overwrite first results; supply a new directory')
 logs=out.parent/(out.name+'_execution');logs.mkdir(exist_ok=False)
 jobs=[('tests',[sys.executable,'-B','test_core.py']),('tests_optimized',[sys.executable,'-B','-O','test_core.py']),
       ('experiment',[sys.executable,'-B','experiment.py','--data',str(Path(a.data).resolve()),'--out',str(out),'--workers',str(a.workers)]),
       ('analysis',[sys.executable,'-B','analyze.py','--out',str(out)]),
       ('audit',[sys.executable,'-B','audit.py','--data',str(Path(a.data).resolve()),'--out',str(out),'--fresh-refits'])]
 receipts=[]
 for name,cmd in jobs:
  st=time.perf_counter()
  with (logs/f'{name}.log').open('w') as f:run=subprocess.run(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT,check=False)
  receipts.append({'stage':name,'returncode':run.returncode,'seconds':time.perf_counter()-st})
  (logs/'EXECUTION.json').write_text(json.dumps(receipts,indent=2)+'\n')
  print(name,run.returncode,round(receipts[-1]['seconds'],3),flush=True)
  if run.returncode:raise RuntimeError(f'{name} failed; keep logs')
 import shutil
 shutil.copytree(logs,out/'execution')
 entries=[]
 for f in sorted(out.rglob('*')):
  if f.is_file():entries.append({'path':str(f.relative_to(out)),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
 (out/'RESULT_MANIFEST.json').write_text(json.dumps({'files':entries,'scope':'all generated results except this manifest'},indent=2)+'\n')
 print('COMPLETE: execution success is not scientific acceptance',flush=True)
if __name__=='__main__':main()
