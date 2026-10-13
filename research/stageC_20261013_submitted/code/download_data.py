"""Download the pinned public author snapshot, then verify its Git blob hash.
This does not authenticate an exchange price definition or grant a data license.
"""
from pathlib import Path
import argparse,hashlib,json,urllib.request
from datetime import datetime,timezone
SOURCE='https://raw.githubusercontent.com/Chen-zhongxiu/Carbon-price-data/6e3ea533832f82fd1c8b4859bb2af8c33bd19a31/GCP.csv'
GIT_SHA='23b1ffc5c68ded3c7b3c03dd46242788f0db52f7'
def verify(body:bytes)->str:
    digest=hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()
    if digest!=GIT_SHA:raise ValueError(f'Source bytes differ from pinned Git blob: {digest}')
    return hashlib.sha256(body).hexdigest()
def main(destination:Path)->None:
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        print('Existing verified source:',verify(destination.read_bytes()));return
    request=urllib.request.Request(SOURCE,headers={'User-Agent':'carbon-method-research/1.0'})
    with urllib.request.urlopen(request,timeout=30) as response:body=response.read()
    sha=verify(body);tmp=destination.with_suffix('.csv.tmp');tmp.write_bytes(body);tmp.replace(destination)
    receipt={'url':SOURCE,'retrieved_utc':datetime.now(timezone.utc).isoformat(),'bytes':len(body),'sha256':sha,'git_blob_sha1':GIT_SHA,'source_grade':'author-published, not exchange-certified'}
    destination.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--destination',type=Path,default=Path(__file__).resolve().parents[1]/'raw_private/GCP.csv');a=p.parse_args();main(a.destination)
