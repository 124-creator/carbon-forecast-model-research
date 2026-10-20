#!/usr/bin/env python3
"""Download two source-identified public CEA/CCER CSVs; do not silently join them.

Source is a third-party curator, not the official trading engine. All rights
remain with the original publisher. No original market rows are committed.
"""
from __future__ import annotations
import argparse,base64,hashlib,io,json,urllib.request
from pathlib import Path
import pandas as pd
SOURCES={
    'cea': {'blob':'287c72ca277233005d54b9e04805d49762a439ad','repo':'YasoneRootflux/carbon-market-blog','path':'data/cea_history.csv','field':'收盘','volume':None,'outfile':'cea_from_public_repository.csv','identity':'reported_close_not_official_average'},
    'ccer':{'blob':'1e6ed6269612b529a833e4ae84af5901e5080d6d','repo':'YasoneRootflux/carbon-market-blog','path':'data/ccer_history.csv','field':'均价','volume':'成交量','outfile':'ccer_from_public_repository.csv','identity':'volume_weighted_average_reported'},
}
def sha(b):return hashlib.sha256(b).hexdigest()
def load(key,out):
    s=SOURCES[key]
    url='https://api.github.com/repos/'+s['repo']+'/git/blobs/'+s['blob']
    req=urllib.request.Request(url,headers={'User-Agent':'academic-carbon-price-research/1.0','Accept':'application/vnd.github+json'})
    with urllib.request.urlopen(req,timeout=45) as h: payload=json.load(h)
    if payload.get('encoding')!='base64':raise ValueError('Unexpected encoding')
    data=base64.b64decode(payload['content'])
    gitsha=hashlib.sha1(('blob '+str(len(data))+'\0').encode()+data).hexdigest()
    if gitsha!=s['blob']:raise ValueError('Source git blob changed; refuse silent replacement')
    f=pd.read_csv(io.BytesIO(data))
    res=pd.DataFrame({'date':f['date'],'price':f[s['field']].astype(float),
                      'volume':f[s['volume']] if s['volume'] is not None else float('nan')})
    dst=out/s['outfile'];res.to_csv(dst,index=False)
    return {'name':key,'identity':s['identity'],'source_url':url,'upstream_blob':s['blob'],'upstream_raw_sha256':sha(data),'canonical_local_sha256':sha(dst.read_bytes()),'rows':len(res),'min_date':res.date.min(),'max_date':res.date.max()}
def main():
    a=argparse.ArgumentParser();a.add_argument('--output',required=True);args=a.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    receipts=[load(k,out) for k in SOURCES]
    (out/'PUBLIC_SOURCE_RECEIPTS.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipts,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
