"""Simple mature-feedback distribution diagnosis, NOT a new algorithm."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import csv,hashlib,json,time,sys
from pathlib import Path
import numpy as np
from scipy.optimize import nnls
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
H=np.array([1,5,20]); TAU=np.array([.025,.05,.1,.25,.5,.75,.9,.95,.975])
ALPHA=np.array([.5,.2,.1,.05]); LO=np.array([3,2,1,0]); HI=np.array([5,6,7,8])
def dump(n,x): (OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def require(c,s):
    if not c: raise RuntimeError(s)
def read_prefix():
    path=ROOT/'data/parsed/figshare_28682348/guangzhou_reported.csv'; lines=[]
    with path.open('rb',buffering=0) as f:
        for _ in range(951):
            line=bytearray()
            while True:
                b=f.read(1); require(bool(b),'premature EOF'); line+=b
                if b==b'\n': break
            lines.append(bytes(line))
    raw=b''.join(lines); rows=list(csv.DictReader(raw.decode('utf-8-sig').splitlines()))
    d=np.array([r['date'] for r in rows]); p=np.array([float(r['author_reported_price']) for r in rows])
    require(len(p)==950 and d[-1]=='2021-12-31' and np.all(d[1:]>d[:-1]) and np.isfinite(p).all(),'input audit')
    require(hashlib.sha256(raw).hexdigest()=='1366ca4814dcdb8aa43296eb64768d1336d6f0fc9088648d6d7ccdbe43318070','prefix identity')
    return d,p,dict(rows=950,bytes_consumed=len(raw),prefix_sha256=hashlib.sha256(raw).hexdigest(),last_date=d[-1],boundary_record_requested=False,whole_file_read=False,prior_audit_dependency=True)
def configs():
    return [dict(family=f,window=w,**({'decay':l} if f=='EWMA' else {})) for f in ['EMP','EWMA','COND'] for w in [120,240,480] for l in ([.94,.98] if f=='EWMA' else [None])]
def features(p):
    a=np.zeros(len(p))
    for t in range(20,len(p)): a[t]=np.mean(np.abs(np.diff(p[t-20:t+1])))
    return a
def forecast(p,initial,origins,c,override=None):
    # Only init/known prices are accessed; no full-array future target construction.
    a=features(p); out=np.zeros((len(origins),3,9)); logs=[]
    for j,h in enumerate(H):
        init_s=np.arange(20,initial-h); y0=p[init_s+h]-p[init_s]
        floor=max(np.sqrt(np.mean(y0*y0))*.05,1e-6)
        coef=nnls(np.column_stack([np.ones(len(init_s)),a[init_s]]),np.abs(y0))[0] if c['family']=='COND' else None
        def scale(s):
            require(coef is not None,'conditional coefficients unavailable')
            return max(float(coef[0]+coef[1]*a[s]),floor)
        v=max(float(np.mean(y0*y0)),floor*floor); pool=[]; last=19
        for i,t in enumerate(origins):
            for s in range(last+1,int(t-h)+1):
                yy=float(p[s+h]-p[s])
                if override is not None and (s,int(h)) in override: yy=override[(s,int(h))]
                if c['family']=='EWMA':
                    pool.append(yy/max(np.sqrt(v),floor)); v=c['decay']*v+(1-c['decay'])*yy*yy
                elif c['family']=='COND': pool.append(yy/scale(s))
                else: pool.append(yy)
            last=int(t-h)
            z=pool[-c['window']:]; fac=max(np.sqrt(v),floor) if c['family']=='EWMA' else scale(t) if c['family']=='COND' else 1.
            out[i,j]=np.quantile(z,TAU,method='linear')*fac
            logs.append([int(t),int(h),last,last+int(h),len(z),float(fac)])
    return out,logs

def losses(q,y):
    e=y[:,:,None]-q; pin=np.maximum(TAU*e,(TAU-1)*e)
    low=q[:,:,LO]; high=q[:,:,HI]; width=high-low
    inter=width+2/ALPHA*np.maximum(low-y[:,:,None],0)+2/ALPHA*np.maximum(y[:,:,None]-high,0)
    wis=(.5*np.abs(y-q[:,:,4])+np.sum(ALPHA/2*inter,axis=2))/4.5
    return wis,pin,width,(y[:,:,None]>=low)&(y[:,:,None]<=high),y[:,:,None]<low,y[:,:,None]>high

def summarize(q,y):
    w,p,wi,co,lt,ut=losses(q,y)
    return dict(n=len(y),wis_h=w.mean(0).tolist(),pinball_h_tau=p.mean(0).tolist(),width_h_interval=wi.mean(0).tolist(),coverage_h_interval=co.mean(0).tolist(),lower_tail_h_interval=lt.mean(0).tolist(),upper_tail_h_interval=ut.mean(0).tolist())
def bootstrap(delta,years):
    rng=np.random.default_rng(20261013); arr=[]
    for _ in range(1000):
        idx=[]
        for yr in ['2020','2021']:
            pos=np.where(years==yr)[0]; starts=rng.integers(0,len(pos),size=int(np.ceil(len(pos)/20))); idx.extend(pos[((starts[:,None]+np.arange(20))%len(pos)).ravel()[:len(pos)]])
        arr.append(delta[idx].mean(0))
    return np.quantile(arr,[.025,.975],axis=0).tolist()
def verify(d,p,initial,origins,qq,cc,y):
    checks=0
    for k,c in enumerate(cc):
        for t in origins[[0,37,173,300,-1]]:
            t=int(t); full=qq[k,np.where(origins==t)[0][0]]
            cut,_=forecast(p[:t+1],initial,np.array([t]),c)
            changed=p.copy(); changed[t+1:]+=np.arange(len(p)-t-1)*1000+99
            mut,_=forecast(changed,initial,np.array([t]),c)
            # Perturb every horizon-specific label whose target is unmatured.
            masked,_=forecast(p,initial,np.array([t]),c,override={(s,int(h)):1e9 for h in H for s in range(t-int(h)+1,len(p))})
            require(np.array_equal(cut[0],full) and np.array_equal(mut[0],full) and np.array_equal(masked[0],full),'causal replay'); checks+=3
    require(len(origins)==451 and np.all(np.diff(qq,axis=-1)>=-1e-12),'shape/monotonicity')
    # Independent WIS construction from scalar interval formula.
    for q in qq:
        w=losses(q,y)[0]; independent=.5*np.abs(y-q[:,:,4])
        for a,l,u in zip(ALPHA,LO,HI):
            low=q[:,:,l]; high=q[:,:,u]; independent+=a/2*((high-low)+(y<low)*(low-y)*2/a+(y>high)*(y-high)*2/a)
        require(np.allclose(w,independent/4.5,atol=1e-12),'independent WIS')
        pq=q+p[origins,None,None]; py=y+p[origins,None]
        require(np.allclose(losses(pq,py)[0],w,atol=1e-12),'translation invariance')
    return dict(causal_perturbation_and_truncation_checks=checks,all_passed=True,independent_wis=True,price_translation=True,optimized=not __debug__)
def main():
    start=time.monotonic(); d,p,audit=read_prefix(); initial=int(np.sum(d<'2020-01-01'))
    origins=np.array([t for t in range(initial,len(p)-20) if d[t][:4]==d[t+20][:4]])
    y=p[origins[:,None]+H]-p[origins,None]; years=np.array([x[:4] for x in d[origins]]); cc=configs()
    if '--verify' in sys.argv:
        with np.load(OUT/'PREDICTIONS_LOCAL_ONLY.npz',allow_pickle=False) as z:
            require(np.array_equal(z['origins'],origins) and np.array_equal(z['true_changes'],y),'NPZ identity'); qq=z['quantile_changes']
        dump('VERIFICATION_OPTIMIZED.json' if not __debug__ else 'VERIFICATION.json',verify(d,p,initial,origins,qq,cc,y)); print('VERIFIED',not __debug__); return
    dump('CONFIG.json',dict(configurations=cc,horizons=H.tolist(),quantiles=TAU.tolist(),interval_alpha=ALPHA.tolist(),initial_rows=initial,budget_seconds=1200,bootstrap_seed=20261013,bootstrap_repetitions=1000,bootstrap_block_length=20,quantile_method='linear',price_units='source reported price units',initial_scale='2018-19 mature second moment, floor .05 initial RMS',conditional_feature='past20 mean absolute increment',conditional_fit='initial mature nnls abs(Y) on intercept and feature, frozen',center='RW reference zero; empirical quantile median not mean'))
    dump('INPUT_AUDIT.json',audit); qq=[]; all_logs=[]
    for c in cc:
        q,logs=forecast(p,initial,origins,c); qq.append(q); all_logs.append(logs); require(time.monotonic()-start<1200,'budget exceeded')
    qq=np.array(qq); base=next(k for k,c in enumerate(cc) if c==dict(family='EMP',window=240)); bw=losses(qq[base],y)[0]
    feat=features(p); threshold=float(np.median(feat[20:initial])); high=feat[origins]>threshold; results=[]
    for k,c in enumerate(cc):
        sm=summarize(qq[k],y); yr={z:summarize(qq[k,years==z],y[years==z]) for z in ['2020','2021']}
        ratios=np.array([np.array(yr[z]['wis_h'])/losses(qq[base,years==z],y[years==z])[0].mean(0) for z in ['2020','2021']]); ci=bootstrap(losses(qq[k],y)[0]-bw,years)
        coverage=np.array(sm['coverage_h_interval'])[:,2]
        gate=bool(np.all(ratios<=.95) and np.all((coverage>=.85)&(coverage<=.95)) and np.all(np.array(ci)[1]<0))
        groups={z:{g:summarize(qq[k,(years==z)&(high if g=='high' else ~high)],y[(years==z)&(high if g=='high' else ~high)]) for g in ['low','high']} for z in ['2020','2021']}
        gaps=np.array([np.array(groups[z]['high']['coverage_h_interval'])[:,2]-np.array(groups[z]['low']['coverage_h_interval'])[:,2] for z in ['2020','2021']]); stable=bool(np.sum((np.abs(gaps[0])>.1)&(np.abs(gaps[1])>.1)&(gaps[0]*gaps[1]>0))>=2)
        results.append(dict(configuration=c,overall=sm,by_year=yr,volatility_groups=groups,coverage_gap90_high_minus_low=gaps.tolist(),relative_wis_by_year_h=ratios.tolist(),block_bootstrap_delta_wis_ci95=ci,distribution_benefit_gate=gate,stable_mismatch_gate=stable))
    np.savez_compressed(OUT/'PREDICTIONS_LOCAL_ONLY.npz',origins=origins,origin_dates=d[origins],horizons=H,quantiles=TAU,true_changes=y,origin_prices=p[origins],quantile_changes=qq,quantile_prices=qq+p[origins][None,:,None,None],interval_lower_prices=qq[:,:,:,LO]+p[origins][None,:,None,None],interval_upper_prices=qq[:,:,:,HI]+p[origins][None,:,None,None],initial_volatility_threshold=threshold)
    dump('MATURITY_LOGS_LOCAL_ONLY.json',dict(columns=['origin','horizon','max_label_origin','max_target_index','pool_count','current_scale'],by_configuration=all_logs))
    dump('RESULTS.json',results); dump('VERIFICATION.json',verify(d,p,initial,origins,qq,cc,y))
    dump('RUN_RECEIPT.json',dict(seconds=time.monotonic()-start,origins=len(origins),configurations=len(cc),development_only=True,rw_mse_by_year={z:(y[years==z]**2).mean(0).tolist() for z in ['2020','2021']},distribution_gate_passed=[k for k,r in enumerate(results) if r['distribution_benefit_gate']],joint_gate_passed=[k for k,r in enumerate(results) if r['distribution_benefit_gate'] and r['stable_mismatch_gate']],algorithm_proposed=False,reason='Diagnostic only; any future algorithm requires mechanism and novelty review',no_2022_read=True,no_commit_push=True))
    print(json.dumps(json.loads((OUT/'RUN_RECEIPT.json').read_text(encoding='utf-8')),ensure_ascii=False))
if __name__=='__main__': main()
