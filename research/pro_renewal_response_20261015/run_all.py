"""Complete replay in a new directory. Existing audited prefix only."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,hashlib,json,sys,time,unittest
from pathlib import Path
import core,experiment,baselines,analyze,audit

def main():
    a=argparse.ArgumentParser();a.add_argument('--out',required=True);a.add_argument('--prefix');a.add_argument('--workers',type=int,default=3);a.add_argument('--no-fresh-audit',action='store_true');o=a.parse_args()
    root=Path(__file__).resolve().parent
    prefix=Path(o.prefix) if o.prefix else root.parents[1]/'data/parsed/figshare_28682348/guangzhou_reported.csv'
    if Path(o.out).exists():raise FileExistsError('Use a new directory, preserving first results')
    if not 1<=o.workers<=3:raise ValueError('workers must be 1..3')
    start=time.perf_counter()
    suite=unittest.defaultTestLoader.discover(str(root),pattern='test_core.py')
    passed=unittest.TextTestRunner(verbosity=2).run(suite)
    if not passed.wasSuccessful():raise RuntimeError('Pre-fit tests failed')
    experiment.run(o.out,prefix,o.workers)
    baselines.run(o.out);analyze.run(o.out);audit.run(o.out,not o.no_fresh_audit)
    dest=Path(o.out)
    experiment.jwrite(dest/'EXECUTION.json',dict(complete=True,protocol_commit=core.PROTOCOL_SHA,base_commit=core.BASE_SHA,tests=passed.testsRun,seconds=time.perf_counter()-start,scope='new full training and arithmetic; not independent holdout'))
    entries=[]
    for p in sorted(dest.rglob('*')):
        if p.is_file() and p.name!='RESULT_MANIFEST.json':entries.append(dict(path=str(p.relative_to(dest)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    experiment.jwrite(dest/'RESULT_MANIFEST.json',dict(files=entries,manifest_excluded=True))

if __name__=='__main__':main()
