"""Replay one frozen Stage D experiment without fetching market data.

Usage from repository root:
python research/pro_stable_control_20261013/run_all.py --root . --out /tmp/stageD-replay
The output directory must be new. Validation and 2023 are never auto-scored.
"""
from __future__ import annotations
import argparse, hashlib, importlib.metadata, json, os, platform
from pathlib import Path
import subprocess, sys, time

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path('.'))
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    if out.exists():
        raise FileExistsError('Refusing to overwrite a prior experiment: ' + str(out))
    out.mkdir(parents=True)
    here = Path(__file__).resolve().parent
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONHASHSEED='0')
    receipt = {
        'python': sys.version,
        'platform': platform.platform(),
        'versions': {n: importlib.metadata.version(n) for n in ['numpy', 'scipy', 'pandas', 'scikit-learn']},
        'source_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(here.iterdir()) if p.suffix in ['.py', '.json']},
        'steps': [],
        'validation_scored': False,
        'new_history_scored': False,
    }
    def run(name: str, argv: list[str]) -> None:
        receipt['status'] = 'running: ' + name
        (out/'replay_receipt.json').write_text(json.dumps(receipt, indent=2))
        start = time.monotonic()
        print('BEGIN ' + name, flush=True)
        with (out/(name + '.log')).open('w') as fh:
            process = subprocess.run([sys.executable] + argv, cwd=here, env=env, stdout=fh, stderr=subprocess.STDOUT)
        receipt['steps'].append({'name': name, 'returncode': process.returncode, 'seconds': time.monotonic()-start})
        (out/'replay_receipt.json').write_text(json.dumps(receipt, indent=2))
        if process.returncode:
            raise RuntimeError(name + ' failed; see its log in ' + str(out))
        print('END ' + name, flush=True)
    common = ['--out', str(out)]
    run('audit', ['experiment.py', 'audit', '--root', str(root)] + common)
    run('tests', ['-m', 'unittest', '-v', 'test_lowdim'])
    run('tests_optimized', ['-O', '-m', 'unittest', '-v', 'test_lowdim'])
    run('minimal_control', ['lowdim.py'] + common)
    stable = json.loads((out/'control_results.json').read_text())['gate_passed']
    if stable:
        run('candidate_synthetic', ['experiment.py', 'synthetic'] + common)
        candidate_ok = json.loads((out/'candidate_synthetic.json').read_text())['gate_passed']
        if candidate_ok:
            run('market_tuning', ['experiment.py', 'market', '--root', str(root)] + common)
            run('analysis', ['analyze.py', '--root', str(root)] + common)
        else:
            (out/'market_status.json').write_text(json.dumps({'status':'NOT_RUN_SYNTHETIC_GATE_FAILED','validation_scored':False,'new_history_scored':False}, indent=2))
    else:
        (out/'market_status.json').write_text(json.dumps({'status':'NOT_RUN_MINIMAL_CONTROL_FAILED','validation_scored':False,'new_history_scored':False}, indent=2))
    receipt['status'] = 'completed'
    receipt['experiment_status'] = json.loads((out/'market_status.json').read_text())
    (out/'replay_receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt['experiment_status'], indent=2), flush=True)

if __name__ == '__main__':
    main()
