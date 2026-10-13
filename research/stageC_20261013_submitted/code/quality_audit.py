"""Post-run verification only: no tuning and no changes to predictions."""
from run_market import *
from datetime import datetime,timezone
from sklearn.linear_model import Ridge as SKRidge

def main():
    d=load();checks=[];R0=R
    for c in configs():
        file=P/(c['id']+'.npz');z=np.load(file)
        a=z['delta'];expected=3 if c['family'] in ['free','global','conditional'] else 1
        if a.shape!=(expected,len(d.p),3) or not np.isfinite(a[:,d.first:]).all():raise RuntimeError('Incomplete archive '+c['id'])
        if not (O/('run_'+c['id']+'.json')).exists():raise RuntimeError('Missing receipt '+c['id'])
    checks.append({'check':'all_81_archives_readable_complete','passed':True})
    if pd.DatetimeIndex(d.dates).max().year>=2023:raise RuntimeError('Holdout not truncated')
    checks.append({'check':'2023_not_in_model_data','passed':True})
    for year in [2019,2020,2021,2022]:
        ids=origin_mask(d,year)
        for h in H:
            if not np.all(pd.DatetimeIndex(d.dates[ids+h]).year==year):raise RuntimeError('crossyear target')
    checks.append({'check':'year_and_horizon_common_scoring_support','passed':True})
    # Replay one selected trajectory each for candidate/control plus a full Ridge run.
    sel=json.loads((O/'selected_complete.json').read_text());replays=[]
    for fam in ['conditional','free']:
        c=sel[fam];pp,tt,ll=neural_forecasts(d,fam,**c['params'],seed=11)
        old=np.load(P/(c['id']+'.npz'))['delta'][0]
        diff=float(np.nanmax(np.abs(old-pp)))
        if not np.array_equal(old,pp,equal_nan=True):raise RuntimeError('Replay mismatch '+fam)
        replays.append({'family':fam,'seed':11,'all_forecasts_exact':True,'max_difference':diff})
    c=sel['Ridge_level'];pp=baseline_forecasts(d,'Ridge_level',c['params']);old=np.load(P/(c['id']+'.npz'))['delta'][0]
    if not np.array_equal(old,pp,equal_nan=True):raise RuntimeError('Ridge replay mismatch')
    replays.append({'family':'Ridge_level','all_forecasts_exact':True,'max_difference':0.})
    # Validate original and batching-equivalent HGBT prediction code on a small causal fixture.
    rng=np.random.default_rng(7);price=20+np.cumsum(rng.normal(0,.1,260));small=make_data(price,pd.date_range('2010-01-01',periods=260).values,220);cfg={'leaves':7,'l2':0.,'lr':.05}
    a=baseline_forecasts(small,'HGBT',cfg);b=tree_forecasts_batched(small,cfg)
    if not np.array_equal(a,b,equal_nan=True):raise RuntimeError('Batching changed HGBT forecasts')
    checks.append({'check':'HGBT_batching_equal_to_original_on_fixture','passed':True})
    # Pure algebra: exponential loading equals a rescaled geometric damping sum.
    eq=[];cond=[]
    for tau in [.5,1,5,10,40,400]:
        rho=np.exp(-1/tau);b1=1-np.exp(-H/tau);b2=(1-rho)/rho*np.array([sum(rho**j for j in range(1,int(h)+1)) for h in H]);eq.append(float(np.max(np.abs(b1-b2))))
        J=np.c_[H/20,b1,-H/tau*np.exp(-H/tau)];cond.append({'tau':tau,'condition_number_unit_amplitude':float(np.linalg.cond(J))})
    if max(eq)>1e-12:raise RuntimeError('Damping identity failed')
    checks.append({'check':'exponential_geometric_damping_identity','passed':True,'max_difference':max(eq)})
    # Check exact-source receipt and protocol remain unchanged.
    b=(R/'raw_private/GCP.csv').read_bytes();sha=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
    if sha!='23b1ffc5c68ded3c7b3c03dd46242788f0db52f7':raise RuntimeError('Source differs')
    freeze=json.loads((R/'metadata/protocol_freeze_receipt.json').read_text());now=hashlib.sha256((R/'metadata/protocol_frozen.json').read_bytes()).hexdigest()
    if now!=freeze['sha256']:raise RuntimeError('Frozen protocol changed')
    checks.append({'check':'source_and_frozen_protocol_hashes_unchanged','passed':True})
    result={'utc':datetime.now(timezone.utc).isoformat(),'checks':checks,'representative_replays':replays,'replay_scope':'Two complete neural seed trajectories and one complete linear archive; not full rerun of all 81 configurations','jacobian_algebra_only':cond,'historical_predictions_modified':False,'all_passed':True}
    (O/'quality_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
