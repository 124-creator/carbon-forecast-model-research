"""Recover only the four whitelisted CSV inputs from the user's archive.

Run outside a public checkout. No network, no extraction of model weights,
no extraction of arbitrary paths, and no modification of an existing source.
"""
import argparse,hashlib,json
from pathlib import Path
from zipfile import ZipFile
EXPECTED={
'EUA_LONG':'142f391305c482eb3bd58dcf8df477db90cf54402495527749a53e126cdf0e7f',
'GUANGZHOU':'8d78a4f13c5a799519546fd3bfa0e0ac5eaa52b0441a983a95ef06f5e5e34051',
'HUBEI':'04655d59c1b585f6c35c02a43469a965fe3589eb613d7d8d8e46996751b7d797',
'SHANGHAI':'425424353bb3ac09e3d96b291680b4179e86f180da626a5d19e5c1e58ed960a1'}
ap=argparse.ArgumentParser();ap.add_argument('--archive',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
out=Path(a.out).resolve()
if out.is_relative_to(Path(__file__).resolve().parents[1]):raise ValueError('Keep market rows outside the public study directory')
out.mkdir(parents=True,exist_ok=False)
receipt={}
with ZipFile(a.archive) as z:
 for name,digest in EXPECTED.items():
  filename=name+'_development.csv';data=z.read('inputs/development/'+filename)
  actual=hashlib.sha256(data).hexdigest()
  if actual!=digest:raise ValueError('Source hash mismatch: '+filename)
  (out/filename).write_bytes(data);receipt[filename]=actual
(out/'RECOVERY.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
