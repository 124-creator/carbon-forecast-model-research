"""Self-contained replay; writes only a new output directory, never a holdout."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,subprocess,sys,time,json,hashlib
from pathlib import Path
import experiment,baselines,analyze,audit

def main():
    r=Path(__file__).resolve().parent;a=argparse.ArgumentParser()
    a.add_argument('--out',required=True);a.add_argument('--workers',type=int,default=3)
    a.add_argument('--prefix',default=str(r.parents[1]/'data/parsed/figshare_28682348/guangzhou_reported.csv'))
    args=a.parse_args();out=Path(args.out).resolve()
    if out.exists():raise FileExistsError('Use NEW output directory')
    tests={};start=time.perf_counter()
    for mode,flags in [('normal',[]),('optimized',['-O'])]:
        p=subprocess.run([sys.executable,'-B',*flags,'-m','unittest','test_core','-v'],cwd=r,capture_output=True,text=True)
        tests[mode]=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
        if p.returncode:print(p.stderr);raise RuntimeError('Prefit tests failed')
    experiment.run(out,args.prefix,args.workers);experiment.jwrite(out/'TESTS.json',tests)
    baselines.run(out);analyze.run(out);audit.run(out,True)
    experiment.jwrite(out/'EXECUTION.json',dict(complete=True,wall_seconds=time.perf_counter()-start,argv=sys.argv,scope='Fresh execution and representative refits, not unseen confirmation'))
    entries=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='RESULT_MANIFEST.json']
    experiment.jwrite(out/'RESULT_MANIFEST.json',dict(files=entries,manifest_excluded=True))
if __name__=='__main__':main()
