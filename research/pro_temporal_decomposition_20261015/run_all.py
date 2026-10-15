"""Reproduce in a new output directory. Never opens later periods or new markets."""
from pathlib import Path
import argparse,subprocess,sys,json
from prepare_prefix import prepare
ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--workers',type=int,default=3);ap.add_argument('--source');a=ap.parse_args()
    out=Path(a.out).resolve()
    if out.exists():raise FileExistsError('reproduction requires a new directory')
    out.mkdir(parents=True);data=prepare(a.source);run=out/'run'
    steps=[('tests_normal',[sys.executable,str(ROOT/'test_model.py')]),('tests_optimized',[sys.executable,'-O',str(ROOT/'test_model.py')]),('neural',[sys.executable,str(ROOT/'run_market.py'),'--data',str(data),'--out',str(run),'--workers',str(a.workers)]),('baselines',[sys.executable,str(ROOT/'run_baselines.py'),'--data',str(data),'--out',str(run)]),('analysis',[sys.executable,str(ROOT/'analyze.py'),'--out',str(run)]),('audit',[sys.executable,str(ROOT/'audit_replay.py'),'--data',str(data),'--out',str(run),'--refits'])]
    for name,cmd in steps:
        print('RUN',name,flush=True)
        with (out/f'{name}.log').open('w') as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
        if result.returncode:raise RuntimeError(f'{name} failed: see its log')
    result=json.loads((run/'RESULTS.json').read_text())
    print('Execution complete. Scientific status:',result['status'])
    print('No automatic later-history or additional-market step exists.')
if __name__=='__main__':main()
