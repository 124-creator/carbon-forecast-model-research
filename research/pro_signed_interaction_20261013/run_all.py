"""Replay the frozen Stage F pipeline on existing packaged CSVs only."""
from pathlib import Path
import argparse, json, os, subprocess, sys

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();out=args.out.resolve();base=Path(__file__).resolve().parent
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a new output directory; original results are protected')
    out.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def step(name,cmd):
        (out/'status.json').write_text(json.dumps({'stage':name,'status':'running'}))
        with (out/(name+'.log')).open('w') as log:
            result=subprocess.run(cmd,cwd=base/'src',env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            (out/'status.json').write_text(json.dumps({'stage':name,'status':'execution_failed','returncode':result.returncode}))
            raise RuntimeError(f'{name} failed; inspect its log')
    step('tests_ordinary',[sys.executable,'-B','test_model.py'])
    step('tests_optimized',[sys.executable,'-B','-O','test_model.py'])
    step('synthetic',[sys.executable,'-B','experiment.py','--out',str(out/'synthetic')])
    res=json.loads((out/'synthetic/RESULTS.json').read_text())
    if res['minimum_gate']:
        step('market',[sys.executable,'-B','market.py','--out',str(out/'market'),'--synthetic',str(out/'synthetic/RESULTS.json')])
        res=json.loads((out/'market/RESULTS.json').read_text())
    (out/'status.json').write_text(json.dumps({'stage':'completed','scientific_status':res['scientific_status'],'later_periods_scored':False},indent=2))
    print((out/'status.json').read_text())
if __name__=='__main__':main()
