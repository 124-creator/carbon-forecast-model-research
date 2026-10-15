"""Create a working copy of exactly the approved 950 rows, never read row 951."""
from pathlib import Path
import argparse
from model import read_prefix
ROOT=Path(__file__).resolve().parent

def prepare(source=None):
    dest=ROOT/'data/guangzhou_prefix950.csv'
    if dest.exists():read_prefix(dest);return dest
    source=Path(source) if source else ROOT.parents[1]/'data/parsed/figshare_28682348/guangzhou_reported.csv'
    read_prefix(source)
    lines=[]
    with source.open('rb',buffering=0) as f:
        for _ in range(951):
            line=bytearray()
            while True:
                b=f.read(1)
                if not b:raise ValueError('short prefix')
                line+=b
                if b==b'\n':break
            lines.append(bytes(line))
    dest.parent.mkdir(exist_ok=True);dest.write_bytes(b''.join(lines));read_prefix(dest)
    return dest
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source');a=ap.parse_args();print(prepare(a.source))
