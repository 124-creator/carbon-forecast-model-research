#!/usr/bin/env python3
"""Download six pinned public author files privately; verify bytes and hashes.

Standard library only. No credentials, paid API, model execution, or upload.
Defaults to a new temporary directory. Existing exact files may be reused;
conflicting files and existing receipts are never overwritten.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request

TGS = 'StriveOrange/TGSINFO-VMD-NEQQLattice'
TGS_COMMIT = '3e8a155f034f38cb20430e601c4edb13c22ed7fc'
ZZY = 'Zzysty/carbon_dataset'
ZZY_COMMIT = '8654ea586188634a519310d88a8c582e9685667d'

FILES = [
    dict(repository=TGS,commit=TGS_COMMIT,
         path='TGSINFO-VMD/Data_GuangZhou.xlsx',local='Data_GuangZhou.xlsx',bytes=39135,
         sha256='c5bbe60f6f2c8d3d96d78eb5263f15f6fd842f501acc959d96b4904d2b1a268c',
         git_blob_sha1='0b03be52d87b75fbf5f9d3ca3955dcab89c8c1ba'),
    dict(repository=TGS,commit=TGS_COMMIT,
         path='NEQQLattice/TGSINFO-VMD-GZ.csv',local='TGSINFO-VMD-GZ.csv',bytes=301041,
         sha256='3d2deb6d8a3d9a6407a9f7d1b5b6ed0eceb11ffdf36b9a1dcad68f63a066ba17',
         git_blob_sha1='cf901ea0e656c796375ddc9197933ed5b3db6205'),
    dict(repository=ZZY,commit=ZZY_COMMIT,
         path='GuangdongCarbonPrice.xls',local='zzy/GuangdongCarbonPrice.xls',bytes=513024,
         sha256='1c244f77877cba4fc59deb30ac741e4b0871064cb314152656e27b1d486cb33f',
         git_blob_sha1='de7252c0ae813942682e7afe5ea2855a84bac5e7'),
    dict(repository=ZZY,commit=ZZY_COMMIT,
         path='HubeiCarbonPrice.xlsx',local='zzy/HubeiCarbonPrice.xlsx',bytes=143376,
         sha256='df1de9279d0a34a2860fa749558e67ea61ae17e5b2a14787a06ba7416a58645c',
         git_blob_sha1='294d12e108ee7817cd3da8b3e901f49f37bc8e49'),
    dict(repository=ZZY,commit=ZZY_COMMIT,
         path='ShanghaiCarbonPrice.csv',local='zzy/ShanghaiCarbonPrice.csv',bytes=65503,
         sha256='55196b00e2667d125669b1720cbd23c88aca5aff105563124f6d658d4b3f5992',
         git_blob_sha1='4bfa9b256c7df18d9c6c20db715a294342ad75dd'),
    dict(repository=ZZY,commit=ZZY_COMMIT,
         path='AssociatedExternalFactors.csv',local='zzy/AssociatedExternalFactors.csv',bytes=416592,
         sha256='e2e07cf5c2c18c5580a032d07e40566d8570393fdbecda89b4e642d36c73b65b',
         git_blob_sha1='a5fd1206c4cc2138604ddc76b445e941637a21ec'),
]

NOTEBOOK = dict(repository=TGS,commit=TGS_COMMIT,
    path='NEQQLattice/NEQQLattice.ipynb',local='NEQQLattice.ipynb',bytes=8887431,
    sha256='1e36a5c1f19e271adef449b6232602be8b31b504016776e6436a83ea212feb9e',
    git_blob_sha1='f3398f3134fd2afed2f812e282d6bc20205447aa')


def verified_bytes(data, spec):
    observed={
        'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
        'git_blob_sha1':hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest(),
    }
    for key,value in observed.items():
        if value!=spec[key]:
            raise ValueError(f"Source identity mismatch for {spec['local']}: {key}")
    return observed


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir',type=Path,help='Private local directory; default is a fresh temporary directory')
    ap.add_argument('--receipt',type=Path,help='New receipt path; default is inside out-dir')
    ap.add_argument('--verify-only',action='store_true',help='Check existing bytes without network')
    ap.add_argument('--include-notebook',action='store_true',help='Also retrieve the fixed author notebook for code reading; never executes it')
    args=ap.parse_args()
    if args.verify_only and args.out_dir is None:
        ap.error('--verify-only requires --out-dir')
    out=(args.out_dir or Path(tempfile.mkdtemp(prefix='carbon-author-data-'))).resolve()
    receipt=(args.receipt or out/'AUTHOR_DATA_DOWNLOAD_RECEIPT.json').resolve()
    if receipt.exists():
        raise FileExistsError('Existing receipt is preserved; choose a new --receipt')
    # Keep author originals outside this research checkout. A broader ancestor
    # marker test can reject legitimate /tmp paths in managed workspaces.
    repo=Path(__file__).resolve().parents[3]
    if out.is_relative_to(repo):
        raise ValueError('Choose a private data directory outside this research checkout')
    specs=FILES+([NOTEBOOK] if args.include_notebook else [])
    rows=[]
    for spec in specs:
        dest=out/spec['local']
        url=f"https://raw.githubusercontent.com/{spec['repository']}/{spec['commit']}/{spec['path']}"
        if dest.exists():
            observed=verified_bytes(dest.read_bytes(),spec)
            action='verified_existing'
        elif args.verify_only:
            raise FileNotFoundError(dest)
        else:
            request=urllib.request.Request(url,headers={'User-Agent':'carbon-forecast-source-reproduction/1.0'})
            with urllib.request.urlopen(request,timeout=45) as response:
                data=response.read(spec['bytes']+1)
            observed=verified_bytes(data,spec)
            dest.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
            with dest.open('xb') as f:
                f.write(data)
            dest.chmod(0o600)
            action='downloaded_and_verified'
        rows.append({**spec,**observed,'url':url,'local_path':str(dest),'action':action})
        print(json.dumps({'file':spec['local'],'action':action,'bytes':observed['bytes'],
                          'sha256':observed['sha256']}),flush=True)
    result={'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'status':'PASS_ALL_PINNED_BYTES','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'out_directory':str(out),'network_used':any(r['action']=='downloaded_and_verified' for r in rows),
            'files':rows,'file_count':len(rows),
            'licensing':'Both author repositories had no declared LICENSE at the audited commits. Retrieval is for local source verification, not a grant to redistribute these tables.',
            'scope':'No raw values printed, no source notebook execution, no paid service, no upload, no market fit.',
            'sample_period':'Some original author tables include 2024 records. Downloading exact bytes is not fresh blinded confirmation; downstream preparation applies its declared date cutoff.'}
    receipt.parent.mkdir(parents=True,exist_ok=True)
    with receipt.open('x',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'status':result['status'],'file_count':len(rows),'receipt':str(receipt)}))


if __name__=='__main__':
    main()
