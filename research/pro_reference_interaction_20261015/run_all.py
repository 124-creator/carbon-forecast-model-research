"""Complete reproducibility entry. Writes only a new results directory."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,subprocess,sys,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=3)
    ap.add_argument('--data',type=Path,default=R.parents[1]/'data/parsed/figshare_28682348/guangzhou_reported.csv')
    a=ap.parse_args()
    if a.out.exists():raise FileExistsError('Use a new output directory')
    checks={}
    for label,flags in [('normal',['-B']),('optimized',['-O'])]:
        p=subprocess.run([sys.executable,*flags,str(R/'test_model.py')],cwd=R,capture_output=True,text=True)
        checks[label]=dict(exit_code=p.returncode,log=p.stdout+p.stderr)
        if p.returncode:raise RuntimeError(checks[label]['log'])
    from experiment import run as experiment,dump
    from analyze import run as analyze
    from audit import run as audit
    experiment(a.data,a.out,a.workers)
    dump(a.out/'TESTS.json',checks)
    analyze(a.data,a.out);audit(a.data,a.out)
    manifest=[dict(path=str(p.relative_to(a.out)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
              for p in sorted(a.out.rglob('*')) if p.is_file()]
    dump(a.out/'RESULT_MANIFEST.json',dict(files=manifest,scope='complete generated results; manifest excludes itself'))
    print('EXECUTION_COMPLETE; consult RESULTS.json for scientific status, not CI status')

if __name__=='__main__':main()
