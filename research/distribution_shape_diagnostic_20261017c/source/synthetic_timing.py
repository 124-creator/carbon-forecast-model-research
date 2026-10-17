"""Six fixed end-to-end synthetic fits; no market files are read."""
import argparse
from datetime import date, timedelta
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
from data import Data, H, K, regular_quantiles
from models import FAMILIES
from run import neural_fit, save, utc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    rng = np.random.default_rng(881)
    n = 1300
    p = 1000+np.cumsum(rng.standard_t(5,size=n))
    dates = np.array([(date(2019,1,1)+timedelta(days=i)).isoformat() for i in range(n)])
    data = Data.__new__(Data)
    data.initialize('SYNTHETIC',dates,p,2022,80)
    train = data.batch(data.train[-192:],data.stats_end-1)
    val = data.batch(data.val,data.start-1)
    test = data.batch(data.test[:64],data.end-1)
    base = regular_quantiles(train['y'])
    config = dict(seed=17,learning_rate=.003,weight_decay=.001,batch_size=128,
                  max_epochs=3,patience=3,min_delta=1e-6,gradient_clip=1.)
    t0 = time.monotonic()
    rows = []
    for family in FAMILIES:
        a,prob,av,pv,info = neural_fit(family,train,val,test,base,config,out/family,time.time()+300)
        rows.append(dict(family=family, **info))
        if not np.isfinite(a).all() or np.max(abs(np.sum(a*prob,-1))) > 1e-10:
            raise AssertionError('Invalid synthetic issued measure')
    files = {str(p.relative_to(Path(__file__).resolve().parent.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(Path(__file__).resolve().parent.glob('*.py'))}
    result = dict(created_utc=utc(),status='PASS',synthetic_fits=6,market_fits=0,rows=rows,
                  wall_seconds=time.monotonic()-t0,source_sha256=files,
                  environment=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,
                                   dtype='float64',device='cpu',threads=1))
    save(out/'TIMING.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','source_sha256')},ensure_ascii=False))


if __name__ == '__main__':
    main()
