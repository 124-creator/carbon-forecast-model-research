#!/usr/bin/env python3
"""Restore the pinned ScienceDB source and three old-split mean-price cohorts.

Downloads one public 374272-byte file or verifies a supplied local copy; invokes
only the data parser, never a training program. Uses the current Python and
requires xlrd==2.0.1 for XLS parsing. All outputs go to a new directory.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

URL='https://china.scidb.cn/download?fileId=5eeeecf15e972ddb1b4881d7'
SOURCE_SHA='5d362db0f8e0a7609d35a04afb25a8bcbbc67167b56c41c3800febf2364cf223'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(path,obj):
    with path.open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir',type=Path,required=True)
    ap.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[3])
    ap.add_argument('--raw-file',type=Path,help='Use a supplied exact source copy; no network')
    args=ap.parse_args()
    if args.out_dir.exists():raise FileExistsError('Choose a new output directory')
    parser=Path(__file__).with_name('prepare_sciencedb_transfer.py')
    prior=args.repo_root/'research/cross_market_innovation_20261017/data_provenance'
    for p in [parser,prior/'INDEPENDENT_SOURCE_SCOPE.json',prior/'SCIENCEDB_DATA_MANIFEST.json']:
        if not p.is_file():raise FileNotFoundError(p)
    args.out_dir.mkdir(parents=True,exist_ok=False)
    plan={'started_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'source_url':URL,
          'expected_bytes':374272,'expected_sha256':SOURCE_SHA,
          'source_doi':'10.11922/sciencedb.819','license':'CC BY 4.0',
          'network_requested':args.raw_file is None,'script_sha256':sha(__file__),
          'parser_sha256':sha(parser),'original_scope_sha256':sha(prior/'INDEPENDENT_SOURCE_SCOPE.json'),
          'model_fits':0,'forecast_scores':0}
    save(args.out_dir/'SOURCE_RESTORATION_PLAN.json',plan)
    receipt=dict(plan)
    try:
        if args.raw_file:
            body=args.raw_file.read_bytes();receipt['supplied_raw_file']=str(args.raw_file.resolve())
        else:
            request=urllib.request.Request(URL,headers={'User-Agent':'CarbonForecastResearch/1.0'})
            with urllib.request.urlopen(request,timeout=45) as response:
                body=response.read(374273)
                receipt.update(http_status=response.status,final_url=response.url)
        if len(body)!=374272 or hashlib.sha256(body).hexdigest()!=SOURCE_SHA:
            raise ValueError('Source identity mismatch; no parsing or fitting')
        raw=args.out_dir/'sciencedb_819_v6.xls'
        with raw.open('xb') as f:f.write(body)
        receipt.update(source_bytes=len(body),source_sha256=sha(raw),source_md5=hashlib.md5(body).hexdigest(),status='PASS_PINNED_BYTES')
        save(args.out_dir/'ACQUISITION_RECEIPT.json',receipt)
        subprocess.run([sys.executable,str(parser),'--raw-file',str(raw),
                        '--repo-root',str(args.repo_root),'--out-dir',str(args.out_dir/'prepared')],check=True)
        save(args.out_dir/'REPRODUCTION_COMPLETE.json',{
            'status':'PASS','finished_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'source_sha256':SOURCE_SHA,'data_manifest_sha256':sha(args.out_dir/'prepared/DATA_MANIFEST.json'),
            'network_used':args.raw_file is None,'market_fits':0,'forecast_scores':0})
    except Exception as exc:
        save(args.out_dir/'FAILED_STOPPED.json',{'status':'FAILED_STOPPED','error_type':type(exc).__name__,
                                               'error':str(exc),'market_fits':0,'forecast_scores':0})
        raise


if __name__=='__main__':
    main()
