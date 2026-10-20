"""Finite, descriptive diagnosis. Only pre-2021 numeric development values parsed."""
from pathlib import Path
import csv
import hashlib
import json
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent / 'event_feedback_innovation_20261017'


def main():
    scope_path = ROOT / 'provenance/DIAGNOSTIC_SCOPE.json'
    scope = json.loads(scope_path.read_text())
    out = ROOT / 'results/PRE2021_DIAGNOSTIC.json'
    if out.exists():
        raise RuntimeError('Refuse overwrite')
    metadata = json.loads((PARENT / 'provenance/D1_PROTOCOL.json').read_text())
    records = []
    parsed = 0
    for market in metadata['markets']:
        raw = Path(market['path']).read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if sha != market['sha256']:
            raise ValueError('Data identity mismatch')
        rows = [r for r in csv.DictReader(raw.decode('utf-8-sig').splitlines())
                if r['date'][:10] < scope['numeric_diagnostic_end_exclusive']]
        dates = [r['date'][:10] for r in rows]
        p = np.array([float(r['price']) for r in rows])
        if len(p) < 100 or any(a >= b for a, b in zip(dates, dates[1:])):
            raise ValueError('Support error')
        parsed += len(p)
        d = np.diff(p)
        rms = np.sqrt(np.mean(d**2))
        z = d / rms
        nz = np.abs(d) > scope['nonzero_tolerance']
        age = np.zeros(len(p), dtype=int)
        last_sign = np.zeros(len(p), dtype=int)
        for t in range(1, len(p)):
            age[t] = 0 if nz[t-1] else age[t-1] + 1
            last_sign[t] = int(np.sign(d[t-1])) if nz[t-1] else last_sign[t-1]
        bins = []
        for label, lo, hi in [('0',0,0),('1',1,1),('2-4',2,4),('5+',5,10**9)]:
            for sign in [-1,0,1]:
                ids = np.flatnonzero((age[:-1]>=lo)&(age[:-1]<=hi)&(last_sign[:-1]==sign)&(np.arange(len(d))>=60))
                rec = {'age_bin':label, 'last_nonzero_sign':sign, 'n':len(ids)}
                if len(ids) >= scope['minimum_reported_conditional_bin']:
                    moved = nz[ids]
                    rec.update(prob_next_nonzero=float(moved.mean()),
                               mean_next_change_in_pre2021_rms=float(z[ids].mean()),
                               mean_signed_mark_in_pre2021_rms=float(z[ids][moved].mean()) if moved.any() else 0.,
                               sign_positive_given_nonzero=float((z[ids][moved]>0).mean()) if moved.any() else None)
                bins.append(rec)
        origins = np.arange(60,len(p)-1)
        scales = np.array([max(np.sqrt(np.mean(np.diff(p[t-60:t+1])**2)),rms*.01,1e-8) for t in origins])
        localz = d[origins]/scales
        q = np.quantile(scales,[.25,.5,.75])
        weight_groups=[]
        for k in range(4):
            take=np.searchsorted(q,scales,side='right')==k
            weight_groups.append({'quartile':k+1,'n':int(take.sum()),
                'mean_normalized_next_change':float(localz[take].mean()),
                'native_weighted_normalized_next_change':float(np.average(localz[take],weights=scales[take]**2))})
        records.append({'market':market['name'],'source_sha256':sha,'n_pre2021':len(p),
            'first_date':dates[0],'last_date':dates[-1],
            'zero_net_change_fraction':float(1-nz.mean()),'maximum_observed_zero_run':int(age.max()),
            'increment_autocorrelation':{str(k):float(np.corrcoef(z[:-k],z[k:])[0,1]) for k in [1,5,20]},
            'age_sign_bins':bins,'volatility_quartiles':weight_groups})
    receipt={'created_utc':datetime.now(timezone.utc).isoformat(),
        'protocol_sha256':hashlib.sha256(scope_path.read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'records':records,'numeric_prices_parsed':parsed,'numeric_end_exclusive':'2021-01-01',
        'data_byte_hashes_cover_existing_development_files':True,'confirmation_files_opened':0,
        'model_fits':0,'statistical_significance_claim':False}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'output':str(out),'markets':[{'name':r['market'],'n':r['n_pre2021'],
        'zero_fraction':r['zero_net_change_fraction'],'max_zero_run':r['maximum_observed_zero_run'],
        'acf':r['increment_autocorrelation'],'bins':r['age_sign_bins']} for r in records]},ensure_ascii=False))


if __name__=='__main__':
    main()
