from pathlib import Path
import json,hashlib,urllib.request,zipfile,io,time
R=Path(__file__).resolve().parent
exec(compile((R/'run_synthetic.py').read_text().split('results=[]')[0],str(R/'run_synthetic.py'),'exec'))
meta=json.loads((R/'pypi_1.7.7.json').read_text());wheel=next(x for x in meta['urls'] if x['filename'].endswith('.whl'))
b=urllib.request.urlopen(wheel['url'],timeout=30).read();digest=hashlib.sha256(b).hexdigest()
if digest!=wheel['digests']['sha256']:raise RuntimeError('wheel hash')
(R/wheel['filename']).write_bytes(b)
z=zipfile.ZipFile(io.BytesIO(b));matched=[]
for rel,r in json.loads((R/'source_receipt.json').read_text())['files'].items():
 if rel=='LICENSE':continue
 name='neuralforecast/'+rel
 if hashlib.sha256(z.read(name)).hexdigest()!=r['sha256']:raise RuntimeError('executed wheel source differs')
 matched.append(rel)
(R/'wheel_receipt.json').write_text(json.dumps({'url':wheel['url'],'filename':wheel['filename'],'sha256':digest,'pypi_hash_matches':True,'executed_sources_match_wheel':matched},indent=2))
# Training re-run with changed unseen synthetic suffix, not loaded saved weights.
y=generate('trend');a=y.copy();a[700:]+=10000
states=[];pred=[]
for label,data in [('original',y),('suffix_changed',a)]:
 log=R/f'causality_{label}_loss.jsonl'
 if log.exists():log.unlink()
 nf=NeuralForecast(models=[model(NHITS,17,log)],freq=1);nf.fit(df=frame(data[:700]));states.append({k:v.clone() for k,v in nf.models[0].state_dict().items()});pred.append(nf.predict(df=frame(data[:700]))['NHITS'].to_numpy())
if not all(torch.equal(states[0][k],states[1][k]) for k in states[0]):raise RuntimeError('state suffix sensitivity')
if not np.array_equal(pred[0],pred[1]):raise RuntimeError('prefix forecast suffix sensitivity')
(R/'causality_receipt.json').write_text(json.dumps({'actual_retrains':2,'steps_each':300,'unseen_suffix_changed_from':700,'parameters_equal':True,'prefix_forecast_equal':True,'checkpoint_loaded':False,'scope':'synthetic frozen-prefix pipeline; no forbidden market suffix accessed'},indent=2))
