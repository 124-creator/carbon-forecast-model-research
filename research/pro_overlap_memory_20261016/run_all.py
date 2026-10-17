"""Full frozen experiment in a NEW output directory, never an unseen period."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import argparse,hashlib,json,platform,subprocess,sys,time
import numpy as np,scipy,torch,sklearn,pandas
import core
from experiment import jwrite

ROOT=Path(__file__).resolve().parent

def run(data,out,workers):
    out=Path(out).resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a new output directory; existing first results are not overwritten')
    out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    core.load_prefix(data)
    env=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__,sklearn=sklearn.__version__,pandas=pandas.__version__,dtype='float64',MKLDNN=False,deterministic=True,BLAS_threads=1,workers=workers,torch_build=torch.__config__.show())
    jwrite(out/'ENVIRONMENT.json',env)
    sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')};jwrite(out/'SOURCE_HASHES.json',sources)
    checks=[]
    for optimized in (False,True):
        cmd=[sys.executable]+(['-O'] if optimized else [])+[str(ROOT/'test_core.py')]
        r=subprocess.run(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        (out/('TEST_OPTIMIZED.log' if optimized else 'TEST_NORMAL.log')).write_text(r.stdout)
        checks.append(dict(optimized=optimized,returncode=r.returncode))
        if r.returncode:raise RuntimeError('Implementation tests failed; stop before market fit')
    jwrite(out/'TESTS.json',dict(tests_each_mode=20,runs=checks,scope='implementation/algebra, not market performance'))
    steps=[('training',[str(ROOT/'experiment.py'),'--data',str(Path(data).resolve()),'--out',str(out),'--workers',str(workers)]),
           ('baselines',[str(ROOT/'baselines.py'),'--out',str(out)]),('analysis',[str(ROOT/'analyze.py'),'--out',str(out)]),('audit',[str(ROOT/'audit.py'),'--out',str(out)])]
    receipts=[]
    for name,script in steps:
        s=time.perf_counter();jwrite(out/'PIPELINE_STATUS.json',dict(stage=name,status='running'))
        with (out/(name+'.log')).open('w') as f:r=subprocess.run([sys.executable,'-u',*script],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        receipts.append(dict(stage=name,returncode=r.returncode,seconds=time.perf_counter()-s));jwrite(out/'PIPELINE_RECEIPTS.json',receipts)
        if r.returncode:raise RuntimeError(f'{name} failed; inspect log, do not claim complete execution')
    summary=json.loads((out/'RESULTS.json').read_text());jwrite(out/'PIPELINE_STATUS.json',dict(status='complete',scientific_status=summary['status'],seconds=time.perf_counter()-start,later_history_read=False))
    manifest=[]
    for p in sorted(out.rglob('*')):
        if p.is_file() and p.name!='RESULT_MANIFEST.json':manifest.append(dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    jwrite(out/'RESULT_MANIFEST.json',dict(files=manifest,scope='All output files except this manifest; no scientific-success implication'))
    print(summary['status'],flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);parser.add_argument('--workers',type=int,default=3);parser.add_argument('--data',default=str(ROOT.parents[1]/'data/parsed/figshare_28682348/guangzhou_reported.csv'))
    a=parser.parse_args();run(a.data,a.out,a.workers)
