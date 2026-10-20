#!/usr/bin/env python3
"""Prepare three licensed author mean-price cohorts without fitting or scoring.

Preserves the old chronological 60/20/20 partition and checks the nine legacy
date/mean_price/source_row exports by hash. New files add the author's reported
volume and turnover. No price repair, calendar filling, or imputation.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path

import xlrd

MARKETS={'北京':'BEIJING','天津':'TIANJIN','重庆':'CHONGQING'}
SOURCE_SHA='5d362db0f8e0a7609d35a04afb25a8bcbbc67167b56c41c3800febf2364cf223'
SCOPE_SHA='59e3e8cdce9e60d744f6b914dcec555c034952db0cf12ecf02aec35d1afae639'
SCHEMA=['date','price','volume_tons','turnover_yuan']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path,data):
    with path.open('x',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False,indent=2);f.write('\n')


def date_value(cell,datemode):
    v=cell.value
    if cell.ctype in (xlrd.XL_CELL_NUMBER,xlrd.XL_CELL_DATE):
        if 19000101<=v<=21001231 and v==int(v):
            return dt.datetime.strptime(str(int(v)),'%Y%m%d').date().isoformat()
        t=xlrd.xldate_as_datetime(v,datemode)
        if t.time()!=dt.time(0):raise ValueError('Non-midnight date')
        return t.date().isoformat()
    return dt.date.fromisoformat(str(v)).isoformat()


def finite(value):
    return float(value) if isinstance(value,(int,float)) and math.isfinite(value) else None


def csv_bytes(header,rows):
    f=io.StringIO(newline='');w=csv.writer(f,lineterminator='\n')
    w.writerow(header);w.writerows(rows)
    return f.getvalue().encode('utf-8')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw-file',type=Path,required=True)
    ap.add_argument('--repo-root',type=Path,required=True)
    ap.add_argument('--out-dir',type=Path,required=True)
    args=ap.parse_args()
    if args.out_dir.exists():raise FileExistsError('Use a new preparation directory')
    prior=args.repo_root/'research/cross_market_innovation_20261017/data_provenance'
    assert sha(args.raw_file)==SOURCE_SHA and args.raw_file.stat().st_size==374272
    assert sha(prior/'INDEPENDENT_SOURCE_SCOPE.json')==SCOPE_SHA
    old_manifest=json.loads((prior/'SCIENCEDB_DATA_MANIFEST.json').read_text())
    old={x['market']:x for x in old_manifest['markets']}
    args.out_dir.mkdir(parents=True,exist_ok=False)
    protocol={
        'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'stage':'source restoration and fixed partition preservation; no model fitting',
        'design_timing':'After v1 development discovery, before this preparation; original 60/20/20 scope was frozen 2026-10-17',
        'source_doi':'10.11922/sciencedb.819','source_file_sha256':SOURCE_SHA,
        'original_scope_sha256':SCOPE_SHA,'original_manifest_sha256':sha(prior/'SCIENCEDB_DATA_MANIFEST.json'),
        'script_sha256':sha(__file__),'markets':MARKETS,
        'schema':SCHEMA,'target':'Author-curated daily transaction mean price in CNY/tonne; copied from the workbook, never relabeled as close or recomputed',
        'quantity_fields':'Author headers explicitly specify volume in tonnes and turnover in CNY',
        'partition':'ascending observed dates; first n*3//5 train, next n//5 validation, all remaining confirmation; preserve legacy boundary dates and hashes',
        'label_support_diagnostic':{'lookback_records':64,'horizons_observation_steps':[1,5,20],
            'rule':'origins t>=64; select origin in each fixed partition only when t+20 is strictly before the partition end; final refit uses all such matured pre-confirmation labels'},
        'source_processing':'Author deleted non-trading dates and reports screening abnormalities relative to monthly weighted means, then replacing verified abnormalities with preceding trading-day data; replacement flags are unavailable.',
        'new_processing':'Date format conversion and ascending sort only; no deletion, calendar filling, smoothing, price repair, or replacement of quantity missingness.',
        'prior_exposure':'Acquired, mechanically parsed, source-compared and partitioned before; no Beijing/Tianjin/Chongqing model-fitting or forecast-scoring record identified in repository audit.',
        'inference_scope':'New-market evaluation of this explicitly defined author-cleaned target; not a claim that source values were never accessed or available as this exact cleaned vintage in real time.',
        'license':'CC BY 4.0, as stated in archived official ScienceDB metadata',
        'market_fits':0,'forecast_scores':0,
    }
    protocol_path=args.out_dir/'PROTOCOL_BEFORE_PARSE.json'
    json_write(protocol_path,protocol)
    book=xlrd.open_workbook(args.raw_file,on_demand=True)
    results=[];files=[]
    for name,code in MARKETS.items():
        sheet=book.sheet_by_name(name)
        headers=sheet.row_values(0)
        assert headers[:4]==['日期','配额日成交量（吨）','配额日成交额（元）','配额日成交均价（元/吨）']
        records=[]
        for i in range(1,sheet.nrows):
            date=date_value(sheet.cell(i,0),book.datemode)
            p=finite(sheet.cell_value(i,3));v=finite(sheet.cell_value(i,1));a=finite(sheet.cell_value(i,2))
            if p is None or p<=0:raise ValueError('Invalid author target; no silent deletion')
            records.append({'date':date,'price':p,'volume_tons':v,'turnover_yuan':a,'source_row':i+1})
        records.sort(key=lambda r:r['date']);n=len(records)
        assert n==old[name]['records'] and len({r['date'] for r in records})==n
        train_n=n*3//5;val_n=n//5;test_start=train_n+val_n
        partitions={'train':(0,train_n),'validation':(train_n,test_start),'confirmation_sealed':(test_start,n)}
        partition_meta={}
        for label,(a,b) in partitions.items():
            rr=records[a:b];identity=old[name]['partitions'][label]
            legacy=csv_bytes(['date','mean_price','source_row'],
                             [[r['date'],repr(r['price']),r['source_row']] for r in rr])
            legacy_sha=hashlib.sha256(legacy).hexdigest()
            assert legacy_sha==identity['sha256'] and len(legacy)==identity['bytes']
            path=args.out_dir/label/(code+'_MEAN.csv');path.parent.mkdir(parents=True,exist_ok=True)
            data=csv_bytes(SCHEMA,[[r['date'],repr(r['price']),
                                   '' if r['volume_tons'] is None else repr(r['volume_tons']),
                                   '' if r['turnover_yuan'] is None else repr(r['turnover_yuan'])] for r in rr])
            with path.open('xb') as f:f.write(data)
            support=[t for t in range(max(64,a),b) if t+20<b]
            partition_meta[label]={'start_row':a,'stop_row_exclusive':b,'rows':b-a,
                'date_min':rr[0]['date'],'date_max':rr[-1]['date'],
                'path':str(path),'sha256':sha(path),'legacy_export_sha256':legacy_sha,
                'legacy_export_exact':True,'mature_h1_h5_h20_origins':len(support),
                'first_origin_row':support[0] if support else None,
                'last_origin_row':support[-1] if support else None}
            files.append({'path':str(path),'sha256':sha(path)})
        path=args.out_dir/(code+'_MEAN.csv')
        data=csv_bytes(SCHEMA,[[r['date'],repr(r['price']),
                               '' if r['volume_tons'] is None else repr(r['volume_tons']),
                               '' if r['turnover_yuan'] is None else repr(r['turnover_yuan'])] for r in records])
        with path.open('xb') as f:f.write(data)
        files.append({'path':str(path),'sha256':sha(path)})
        differences=[abs(r['price']-r['turnover_yuan']/r['volume_tons']) for r in records
                     if r['turnover_yuan'] is not None and r['volume_tons'] is not None and r['volume_tons']>0]
        results.append({'market':name,'code':code,'rows':n,'date_min':records[0]['date'],
            'date_max':records[-1]['date'],'source_headers':headers,'schema':SCHEMA,
            'full_csv':str(path),'sha256':sha(path),'partitions':partition_meta,
            'duplicates':0,'invalid_prices':0,
            'quantity_qc':{'missing_volume':sum(r['volume_tons'] is None for r in records),
                           'negative_volume':sum(r['volume_tons'] is not None and r['volume_tons']<0 for r in records),
                           'zero_volume':sum(r['volume_tons']==0 for r in records),
                           'missing_turnover':sum(r['turnover_yuan'] is None for r in records),
                           'max_abs_mean_price_minus_turnover_per_volume':max(differences) if differences else None},
            'mature_final_refit_origins':max(0,test_start-20-64),
            'first_two_training_rows':[{k:r[k] for k in SCHEMA} for r in records[:2]]})
    book.release_resources()
    manifest={'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'status':'PASS_PINNED_SOURCE_AND_NINE_LEGACY_PARTITIONS','protocol_sha256':sha(protocol_path),
        'source_sha256':SOURCE_SHA,'script_sha256':sha(__file__),
        'old_original_scope_sha256':SCOPE_SHA,'markets':results,'files':files,
        'total_market_rows':sum(r['rows'] for r in results),'market_fits':0,'forecast_scores':0}
    json_write(args.out_dir/'DATA_MANIFEST.json',manifest)
    print(json.dumps({'status':manifest['status'],'protocol_sha256':manifest['protocol_sha256'],
                      'rows':manifest['total_market_rows'],'markets':results},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
