"""No whole-file loader. Caller must supply previously audited count."""
import csv,hashlib
from pathlib import Path
def read_prefix(path,records,last_date):
 lines=[]
 with Path(path).open('rb',buffering=0) as f:
  for _ in range(records+1):
   line=bytearray()
   while True:
    b=f.read(1)
    if not b:raise ValueError('short or unterminated audited prefix')
    line+=b
    if b==b'\n':break
   lines.append(bytes(line))
 raw=b''.join(lines);rows=list(csv.DictReader(raw.decode('utf-8-sig').splitlines()))
 if len(rows)!=records or rows[-1]['date']!=last_date:raise ValueError('audit disagreement')
 return rows,dict(records=records,last_date=last_date,boundary_requested=False,prefix_sha256=hashlib.sha256(raw).hexdigest())
