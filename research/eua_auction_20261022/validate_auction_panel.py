from pathlib import Path
import json,sys,hashlib
import pandas as pd,numpy as np,openpyxl
sys.stdout.reconfigure(encoding='utf-8')
R=Path(__file__).resolve().parent
p=pd.read_csv(R/'daily_auction_panel.csv');e=pd.read_csv(R/'auction_events.csv');u=pd.read_csv(R/'unmatched_auction_dates.csv')
m=json.loads((R/'manifest.json').read_text(encoding='utf-8'))
assert p.date.is_unique and p.date.is_monotonic_increasing
assert p.date.between('2020-07-01','2026-06-08').all()
assert len(e)==sum(x.get('records',0) for x in m['sources'])
within=e[e.date.between('2020-07-01','2026-06-08')]
assert int(p.n_auctions.sum())==len(within)
assert p.has_EUA.sum()==within.loc[within.contract=='T3PA','date'].nunique()
assert p.EUA__n_auctions.sum()==(within.contract=='T3PA').sum()
assert p.EUAA__n_auctions.sum()==(within.contract=='EAA3').sum()
for prefix,contract in [('EUA','T3PA'),('EUAA','EAA3')]:
 for dt,z in within[within.contract==contract].groupby('date'):
  r=p.loc[p.date==dt].iloc[0]
  assert abs(r[prefix+'__auction_volume']-z.auction_volume.sum())<1e-7
  good=z[z.status=='successful']
  if len(good):assert abs(r[prefix+'__cover_ratio_weighted']-np.average(good.cover_ratio,weights=good.auction_volume))<1e-12
  else:assert pd.isna(r[prefix+'__cover_ratio_weighted'])
for v in m['sources']:assert hashlib.sha256((R/v['path']).read_bytes()).hexdigest()==v['sha256']
formula=[]
for v in m['sources']:
 if not v['parsed']:continue
 w=openpyxl.load_workbook(R/v['path'],read_only=True,data_only=False)
 for row in w.active:
  for c in row:
   if c.data_type=='f': formula.append({'source':v['path'],'cell':c.coordinate,'formula':c.value})
 w.close()
res={'checks_passed':True,'source_hashes_unchanged':True,'events':len(e),'panel_dates':len(p),'panel_events':int(p.n_auctions.sum()),'EUA_dates':int(p.has_EUA.sum()),'EUAA_dates':int(p.has_EUAA.sum()),'formula_cells':formula,'event_time_hours_by_contract':{str(k):int(v) for k,v in e.assign(hour=pd.to_datetime(e.time,format='mixed').dt.hour).groupby(['contract','hour']).size().items()},'KRBN_within_span_unmatched':u.loc[(u.asset=='KRBN')&(u.classification=='within_asset_span_not_observed'),'auction_date'].tolist(),'KEUA_extra_unmatched_vs_KCCA':sorted(set(u.loc[(u.asset=='KEUA')&(u.classification=='within_asset_span_not_observed'),'auction_date'])-set(u.loc[(u.asset=='KCCA')&(u.classification=='within_asset_span_not_observed'),'auction_date']))}
(R/'validation.json').write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(res,ensure_ascii=False,indent=2))
