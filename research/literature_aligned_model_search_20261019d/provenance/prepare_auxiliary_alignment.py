#!/usr/bin/env python3
"""Align reported quantity fields to unchanged canonical development dates.

One output row per legacy price row; absent/duplicate/conflicting quotes produce
blank quantities. No price row deletion or filling. Blank means NA, never zero.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from collections import defaultdict, Counter
from pathlib import Path

from audit_public_workbook_candidates import audit_table, digest, number


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--raw-root',type=Path,required=True)
    ap.add_argument('--repo-root',type=Path,required=True)
    ap.add_argument('--out-dir',type=Path,required=True)
    ap.add_argument('--receipt',type=Path,required=True)
    args=ap.parse_args()
    output_paths=[args.out_dir/(name+'_auxiliary.csv') for name in ['GUANGZHOU','HUBEI']]
    if args.receipt.exists() or any(p.exists() for p in output_paths):
        raise FileExistsError('Existing outputs/receipt will not be overwritten')
    result={'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'script_sha256':digest(Path(__file__)),
            'parser_sha256':digest(Path(__file__).with_name('audit_public_workbook_candidates.py')),
            'rule':{'canonical_dates_preserved':True,'cutoff_exclusive':'2024-01-01',
                    'price_match':'exact float equality of reported source and canonical price; no tolerance, rescaling or rounding',
                    'valid_volume':'exactly one author row on date, finite price match and finite volume >= 0',
                    'missing_output':'empty CSV field denotes NA; never filled with zero or a prior value',
                    'duplicate_or_price_conflict':'both quantity fields blank; valid_volume=0',
                    'turnover':'reported finite value when date unique and price matches; not required for valid_volume',
                    'publication':'raw source and prepared row quantities remain outside Git; only this aggregate receipt is public'},
            'fits':0,'sources':{}}
    specs=[('GUANGZHOU','GuangdongCarbonPrice.xls','Date','Closing Price'),
           ('HUBEI','HubeiCarbonPrice.xlsx','Date','Latest')]
    all_outputs=[]
    for market,raw_name,date_name,price_name in specs:
        raw=args.raw_root/'zzy'/raw_name
        summary,records=audit_table(raw,date_name,price_name,'2024-01-01')
        bydate=defaultdict(list)
        for r in records:bydate[r['date']].append(r)
        canonical=args.repo_root/'research/cross_market_innovation_20261017/data'/(market+'_development.csv')
        with canonical.open(encoding='utf-8-sig',newline='') as f:prices=list(csv.DictReader(f))
        assert all(r['date']<'2024-01-01' for r in prices)
        output=[];counts=Counter(); reasons=Counter()
        for row in prices:
            date=row['date'];candidates=bydate.get(date,[])
            match=None;v=None;t=None;valid=False
            if len(candidates)==0:reason='no_author_date'
            elif len(candidates)>1:reason='duplicate_author_date'
            else:
                r=candidates[0]['values'];p=r[price_name];target=number(row['price'])
                if p is None or target is None:reason='price_missing'
                else:
                    match=p==target
                    if not match:reason='price_conflict'
                    else:
                        v=r.get('Volume');t=r.get('Turnover')
                        valid=v is not None and v>=0
                        reason='valid_volume' if valid else 'volume_missing_or_negative'
            if v is not None and v<0:v=None
            counts['price_match_true']+=int(match is True)
            counts['price_match_false']+=int(match is False)
            counts['price_match_unknown']+=int(match is None)
            counts['valid_volume']+=int(valid)
            counts['volume_na']+=int(v is None)
            counts['turnover_na']+=int(t is None)
            counts['reported_volume_zero']+=int(v==0 if v is not None else False)
            reasons[reason]+=1
            output.append({'date':date,'volume':'' if v is None else repr(v),
                           'turnover':'' if t is None else repr(t),
                           'price_match':'' if match is None else int(match),
                           'valid_volume':int(valid)})
        assert [r['date'] for r in output]==[r['date'] for r in prices]
        p=args.out_dir/(market+'_auxiliary.csv')
        all_outputs.append((p,output))
        result['sources'][market]={'author_path':str(raw),'author_sha256':digest(raw),
            'canonical_path':str(canonical),'canonical_sha256':digest(canonical),
            'output_path':str(p),'rows':len(output),'first_date':output[0]['date'],'last_date':output[-1]['date'],
            'counts':dict(counts),'reason_counts':dict(reasons),
            'source_quantity_units':'Header says Volume and Turnover; exact unit, channel and availability not independently authenticated.',
            'source_license':'Zzysty/carbon_dataset contains no LICENSE; public retrieval does not establish redistribution rights.'}
    args.out_dir.mkdir(parents=True,exist_ok=True)
    for p,rows in all_outputs:
        with p.open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['date','volume','turnover','price_match','valid_volume'],lineterminator='\n')
            w.writeheader();w.writerows(rows)
        result['sources'][p.name.split('_auxiliary')[0]]['output_sha256']=digest(p)
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
