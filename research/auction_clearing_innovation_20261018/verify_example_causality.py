"""Bounded preparation audit: metadata joins and artificial future perturbation.

No original market workbook or raw event price column is opened. Real input
arrays are inspected only for dimensions and finite values. Counterfactual
builder runs use newly invented, deterministic records in temporary folders.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

import prepare_examples as production

ROOT = Path(__file__).resolve().parent


def require(test, message):
    if not bool(test):
        raise AssertionError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata_join_review():
    metadata_path=ROOT/'local_only/example_metadata.csv'
    calendar_path=ROOT/'local_only/calendar_vintages_development.csv'
    meta=pd.read_csv(metadata_path,keep_default_na=False)
    columns=['scheduleddate','platform','calendar_year','revision_available_date','valid_until_exclusive','vintage_id']
    cal=pd.read_csv(calendar_path,usecols=columns,keep_default_na=False)
    require((cal.scheduleddate<'2024-01-01').all(),'Calendar contains confirmation event dates')
    require((meta.target_date<'2024-01-01').all(),'Metadata contains confirmation targets')
    years={}
    for (year,vid),group in cal.groupby(['calendar_year','vintage_id'],sort=True):
        dates=set(group.revision_available_date);expiry=set(group.valid_until_exclusive)
        require(len(dates)==1 and len(expiry)==1,'Inconsistent version metadata')
        entry={'id':vid,'available':next(iter(dates)),'expiry':next(iter(expiry)),
               'events':list(zip(group.scheduleddate,group.platform))}
        years.setdefault(int(year),[]).append(entry)
    for versions in years.values():
        versions.sort(key=lambda x:x['available'])
    count=0
    for issue,group in meta.groupby('issue_date',sort=True):
        origins=set(group.origin_date)
        require(len(origins)==1,'Multiple origin dates within one issue date')
        origin=next(iter(origins))
        future=[]
        for year,versions in years.items():
            known=[v for v in versions if v['available']<=issue]
            if not known:
                continue
            last=known[-1]
            require(sum(v['available']==last['available'] for v in known)==1,'Tied known calendar versions')
            if last['expiry'] and issue>=last['expiry']:
                continue
            # Select a whole latest version first, including its deletions.
            future.extend((day,platform,last['id'],last['available'],last['expiry'])
                          for day,platform in last['events'] if day>origin)
        future.sort(key=lambda x:(x[0],x[1]))
        for row in group.itertuples(index=False):
            require(len(future)>=row.horizon,'Issued horizon absent from as-of calendar')
            expected=future[row.horizon-1]
            observed=(row.target_date,row.platform,row.calendar_vintage_id,row.calendar_available_date,row.calendar_valid_until_exclusive)
            require(expected==observed,f'Independent as-of join mismatch: {row.example_id}')
            require(row.history_last_date<row.issue_date<=row.target_date,'History/issue/target chronology violated')
            count+=1
    inputs={}
    with np.load(ROOT/'local_only/examples_development.npz',allow_pickle=False) as arrays:
        for name in ['x','context','base']:
            value=arrays[name]
            require(value.shape[0]==len(meta) and np.isfinite(value).all(),f'Nonfinite or misaligned model input {name}')
            inputs[name]={'shape':list(value.shape),'dtype':str(value.dtype),'finite':True}
    return {'status':'PASS','issued_examples_checked':count,'issue_dates':int(meta.issue_date.nunique()),
            'calendar_versions':sum(len(v) for v in years.values()),'independent_join_algorithm':'Group whole versions by year; select newest available; reject expired latest version; then sort future scheduled date/platform and take horizon rank.',
            'model_input_arrays':inputs,'real_price_or_bid_label_arrays_opened':False,
            'metadata_sha256':digest(metadata_path),'calendar_sha256':digest(calendar_path),
            'examples_sha256':digest(ROOT/'local_only/examples_development.npz')}


def artificial_records():
    dates=pd.date_range('2016-11-01',periods=151,freq='D')
    i=np.arange(len(dates),dtype=float)
    price=100+.02*i+.1*np.sin(i/3)
    ev=pd.DataFrame({'date':dates,'event_id':['artificial_'+str(j) for j in range(len(dates))],
                     'platform':'CAP','event_status':'reported_successful'})
    for j,name in enumerate(production.PRICE_COLS):
        ev[name]=price+.03*j
    for j,name in enumerate(production.VOLUME_COLS):
        ev[name]=(2.5e6 if name=='total_bid_volume_tco2' else 1e6)+100*j
    for j,name in enumerate(production.COUNT_COLS):
        ev[name]=20+j
    cal=pd.DataFrame({'scheduleddate':dates[dates.year==2017],'platform':'CAP','calendar_year':2017,
                      'revision_available_date':pd.Timestamp('2016-12-20'),'valid_until_exclusive':pd.NaT,
                      'vintage_id':'artificial_v1','plannedvolume':1e6,'volume_is_indicative':False})
    return ev,cal


def execute_artificial(root,events,calendar):
    private=root/'local_only';private.mkdir(parents=True)
    events.to_csv(private/'auction_events_development.csv',index=False)
    calendar.to_csv(private/'calendar_vintages_development.csv',index=False)
    saved_argv=sys.argv
    try:
        sys.argv=['prepare_examples.py','--study-dir',str(root),'--horizons','1','5','20']
        with contextlib.redirect_stdout(io.StringIO()):
            production.main()
    finally:
        sys.argv=saved_argv
    meta=pd.read_csv(private/'example_metadata.csv',keep_default_na=False)
    with np.load(private/'examples_development.npz',allow_pickle=False) as source:
        arrays={k:source[k].copy() for k in source.files}
    return meta,arrays


def future_perturbation():
    events,calendar=artificial_records()
    issue='2017-01-16'
    changed=events.copy()
    future=changed.date>=pd.Timestamp(issue)
    for name in production.PRICE_COLS:
        changed.loc[future,name]=10000+np.arange(future.sum())
    for name in production.VOLUME_COLS:
        changed.loc[future,name]=changed.loc[future,name]*11
    changed.loc[future,'event_status']='reported_cancelled_or_failed'
    # Remove the first scheduled outcome altogether: its forecast must remain.
    changed=changed[changed.date!=pd.Timestamp(issue)].copy()
    unpublished=calendar.copy()
    unpublished['vintage_id']='artificial_future_v2'
    unpublished['revision_available_date']=pd.Timestamp('2017-02-01')
    unpublished['plannedvolume']=2e6
    unpublished=unpublished[unpublished.scheduleddate!=pd.Timestamp(issue)]
    future_calendar=pd.concat([calendar,unpublished],ignore_index=True)
    with tempfile.TemporaryDirectory(prefix='eex_causal_synthetic_') as tmp:
        a,xa=execute_artificial(Path(tmp)/'baseline',events,calendar)
        b,xb=execute_artificial(Path(tmp)/'changed_future',changed,future_calendar)
        ia=np.flatnonzero((a.issue_date==issue).to_numpy())
        ib=np.flatnonzero((b.issue_date==issue).to_numpy())
        require(len(ia)==3 and len(ib)==3,'Artificial issue does not retain all three horizons')
        require(a.iloc[ia].example_id.tolist()==b.iloc[ib].example_id.tolist(),'Future labels/unpublished plans changed forecast identities')
        unchanged=[]
        for field in ['x','context','base','anchor','scale','last_bid_median']:
            if field not in xa:
                continue
            require(np.array_equal(xa[field][ia],xb[field][ib],equal_nan=True),'Future information changed '+field)
            unchanged.append(field)
        require(not np.array_equal(xa['price'][ia],xb['price'][ib],equal_nan=True),'Mutation did not change synthetic labels')
        require(not bool(b.iloc[ib[0]].result_found),'Removed synthetic outcome unexpectedly found')
    # A gap after an explicitly expired latest vintage cannot fall back to it.
    gap_cal=future_calendar.copy()
    gap_cal.loc[gap_cal.vintage_id=='artificial_v1','valid_until_exclusive']=pd.Timestamp('2017-01-10')
    result=production.known_plan(gap_cal,pd.Timestamp(issue),pd.Timestamp('2017-01-15'))
    require(result.empty,'Expired whole vintage was reused during a publication gap')
    return {'status':'PASS','artificial_issue':issue,'horizons':[1,5,20],
            'mutations':['All future prices replaced','All future bid/volume values multiplied','Future status text replaced','First target outcome row deleted','Unpublished later calendar added with changed supply and deleted target'],
            'bitwise_unchanged_inputs_and_past_references':unchanged,'target_identities_unchanged':True,
            'labels_changed':True,'missing_outcome_forecast_retained':True,'expired_vintage_no_fallback':True,
            'real_event_price_records_read':0,'market_training_runs':0}


def main():
    source=ROOT/'prepare_examples.py';before=digest(source)
    results={}
    for name,func in [('independent_calendar_join_and_input_finiteness',metadata_join_review),('artificial_future_perturbation',future_perturbation)]:
        try:
            results[name]=func()
        except Exception as exc:
            results[name]={'status':'FAIL','error_type':type(exc).__name__,'error':str(exc)}
    unchanged=digest(source)==before
    passed=all(v['status']=='PASS' for v in results.values()) and unchanged
    report={'overall_status':'PASS' if passed else 'FAIL','prepare_examples_sha256':before,'builder_unchanged_during_review':unchanged,
            'verification_script_sha256':digest(__file__),'checks':results,
            'static_dataflow_review':{'past_rows':'ev.date <= origin; issue = origin + 1 day',
              'historical_anchor':'Last positive observed clearing-price row in past, with its own B/S; after a cancellation this is not necessarily the most recent event.',
              'target_before_outcome_lookup':True,'forward_model_arguments':['x','context','base'],
              'future_price_B_status':'Only target/loss fields and terminal metadata; not x/context/base.',
              'calendar_supply':'Known calendar plannedvolume; final supply mismatch is terminal audit metadata, never an input replacement.',
              'price_only_label_limit':'Price-only history ablation retains common auction calendar context, including ratios to historical B/S.'},
            'limitations':['Annual result fields are assumed known by next-day 00:00 UTC; this check cannot certify historical publication timestamps or revisions.',
                           'Calendar availability dates are accepted from the separately audited metadata; this is a join/dataflow audit, not a new source-publication review.',
                           'No real label/performance arrays or confirmation data were opened; no market model was fitted.']}
    out=ROOT/'results/EXAMPLE_CAUSALITY_CHECK.json'
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'overall_status':report['overall_status'],'prepare_examples_sha256':before,'checks':{k:v['status'] for k,v in results.items()},'failures':{k:v for k,v in results.items() if v['status']!='PASS'}},ensure_ascii=False))
    if not passed:raise SystemExit(2)


if __name__=='__main__':main()
