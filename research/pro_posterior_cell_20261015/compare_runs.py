"""Compare two completed run directories without loading executable model weights."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def compare(left,right):
    left=Path(left);right=Path(right);groups={};count=0
    for pattern in ('fits/*.npz','predictions.npz','reference.npz','evaluation.npz','combination_weights.npz'):
        fs=sorted(left.glob(pattern));arrays=0;different=0;maximum=0.;nan_equal=True
        for f in fs:
            with np.load(f,allow_pickle=False) as a,np.load(right/f.relative_to(left),allow_pickle=False) as b:
                if set(a.files)!=set(b.files):raise ValueError('Array names differ: '+str(f))
                for k in a.files:
                    x,y=a[k],b[k];arrays+=1
                    if x.shape!=y.shape or x.dtype!=y.dtype:raise ValueError('Shape/dtype mismatch')
                    if x.dtype.kind in 'biufc':
                        equal=np.array_equal(x,y,equal_nan=True)
                        nan_equal=nan_equal and np.array_equal(np.isnan(x),np.isnan(y))
                        d=np.logical_xor(x,y) if x.dtype.kind=='b' else np.abs(x-y)
                        maximum=max(maximum,float(np.nanmax(d)) if d.size else 0.)
                    else:equal=np.array_equal(x,y)
                    different+=not equal
        groups[pattern]={'files':len(fs),'arrays':arrays,'nonidentical_arrays':different,'max_absolute_difference':maximum,'nan_masks_equal':bool(nan_equal)}
        count+=arrays
    a=json.loads((left/'ALL_SCORES.json').read_text());b=json.loads((right/'ALL_SCORES.json').read_text())
    if set(a)!=set(b):raise ValueError('Configurations differ')
    diff=[abs(a[k][v]-b[k][v]) for k in a for v in ('score','ensemble')]
    r1=json.loads((left/'FIT_REGISTER.json').read_text());r2=json.loads((right/'FIT_REGISTER.json').read_text())
    identity=lambda r:(r['mode'],r['seed'],r['block'])
    d1={identity(r):r for r in r1};d2={identity(r):r for r in r2}
    if set(d1)!=set(d2):raise ValueError('Fit identities differ')
    changed=[str(k) for k in d1 if d1[k]['selected_epoch']!=d2[k]['selected_epoch']]
    traces=list((left/'logs').glob('*.json'));badtraces=[]
    for f in traces:
        if json.loads(f.read_text())['trace']!=json.loads((right/'logs'/f.name).read_text())['trace']:badtraces.append(f.name)
    manifests={}
    for root in (left,right):
        manifest=json.loads((root/'RESULT_MANIFEST.json').read_text());bad=[]
        for r in manifest['files']:
            p=root/r['path']
            if p.stat().st_size!=r['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:bad.append(r['path'])
        manifests[str(root)]={'entries':len(manifest['files']),'errors':bad}
    return {'comparisons':groups,'arrays_compared':count,'all_arrays_exact':all(g['nonidentical_arrays']==0 for g in groups.values()),'scores_compared':len(diff),'maximum_score_difference':max(diff),'changed_checkpoints':changed,'traces_compared':len(traces),'different_traces':badtraces,'manifests':manifests,'scope':'Numerical artifact comparison, not independent scientific replication or unseen-market confirmation'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--first',required=True);p.add_argument('--second',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    target=Path(a.out)
    if target.exists():raise FileExistsError('Use a new receipt path')
    result=compare(a.first,a.second);target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
