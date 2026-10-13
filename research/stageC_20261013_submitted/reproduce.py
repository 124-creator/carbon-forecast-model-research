"""Reproduce in a NEW directory; show status every 10 seconds, never alter originals."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,subprocess,sys,shutil,time,json,os
ROOT=Path(__file__).resolve().parent

def execute(cmd,work:Path,stage:str):
    out=work/'outputs';log=out/(stage+'.log');start=time.monotonic()
    with log.open('w') as stream:
        p=subprocess.Popen([sys.executable,'-u',*cmd],cwd=work/'code',stdout=stream,stderr=subprocess.STDOUT)
        while True:
            rc=p.poll();files=list((out/'predictions').glob('*.npz')) if (out/'predictions').exists() else []
            receipt={'stage':stage,'pid':p.pid,'running':rc is None,'exit_code':rc,'elapsed_seconds':round(time.monotonic()-start,2),'prediction_archives':len(files),'utc':datetime.now(timezone.utc).isoformat(),'log':str(log)}
            (out/'run_status.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
            if rc is not None:
                if rc!=0:raise RuntimeError(f'{stage} failed. See {log}')
                break
            time.sleep(10)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--destination',type=Path,required=True);ap.add_argument('--resume',action='store_true');a=ap.parse_args();work=a.destination.resolve()
    if work==ROOT:raise ValueError('Do not overwrite the delivered experiment')
    if work.exists() and not a.resume:raise FileExistsError('Use a new directory, or --resume for this same reproduction')
    if not work.exists():
        work.mkdir(parents=True)
        for name in ['code','metadata','raw_private']:
            shutil.copytree(ROOT/name,work/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        (work/'outputs').mkdir()
    execute(['download_data.py'],work,'verify_data')
    execute(['-m','unittest','test_models'],work,'tests')
    execute(['run_market.py','--group','all'],work,'market')
    if not (work/'outputs/synthetic_complete.json').exists():execute(['run_synthetic.py'],work,'synthetic')
    execute(['select_and_ablate.py','--ablate'],work,'selection_ablations')
    if not (work/'outputs/evaluation_complete.json').exists():execute(['evaluate.py'],work,'evaluation')
    execute(['quality_audit.py'],work,'quality_audit')
    print('Completed. Results:',work/'outputs')
if __name__=='__main__':main()
