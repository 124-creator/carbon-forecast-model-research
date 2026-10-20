from pathlib import Path
import json,hashlib,subprocess,sys,re,datetime
ROOT=Path(__file__).parent
SCRIPT=Path('C:/Users/Dreamboat/AppData/Local/hermes/skills/research/grounded-citations/scripts/sources.py')
BASE=[sys.executable,str(SCRIPT),'--ledger',str(ROOT/'ledger.json')]
records=[]
for name in ['acquisition.json','additional_acquisition.json','neighbors_acquisition.json','forecast_observers_acquisition.json']:
 records.extend(json.loads((ROOT/'receipts'/name).read_text(encoding='utf-8')))
files={1:'nbeats.txt',2:'dlinear.txt',3:'koopa.txt',4:'s4.txt',5:'ServiceNow_N-BEATS_nbeats.py',6:'cure-lab_LTSF-Linear_DLinear.py',7:'thuml_Koopa_Koopa.py',8:'state-spaces_s4_ssm.py',9:'deep_impulse.txt',10:'neural_luenberger.txt',11:'ossm.txt',12:'ds3m.txt',13:'FARKANE_APINN-Obs_README.md'}
phrases={1:'basis expansion',2:'moving average',3:'time-variant',4:'state space',5:'class NBeatsBlock',6:'self.Linear_Seasonal = nn.Linear',7:'self.K = torch.linalg.lstsq',8:'next_state = contract',9:'Impulse response estimation',10:'This work proposes',11:'OSSMs explicitly separate',12:'architecture employs discrete',13:'simultaneously estimate system states'}
for i,f in files.items():
 text=(ROOT/'receipts'/f).read_text(encoding='utf-8');m=re.search(re.escape(phrases[i]),text,re.I)
 if m is None: raise RuntimeError(f'Quote phrase absent: {i} {phrases[i]}')
 quote=' '.join(text[m.start():].split()[:14])
 subprocess.run(BASE+['quote',str(i),'--text',quote,'--from',str(ROOT/'receipts'/f)],check=True)
subprocess.run(BASE+['render','--replace-in',str(ROOT/'DESIGN_ZH.md')],check=True)
check=subprocess.run(BASE+['verify',str(ROOT/'DESIGN_ZH.md'),'--strict','--evidence'],capture_output=True,text=True,encoding='utf-8')
queries=[]
for name in ['search_queries.json','search_queries_second.json']:
 queries.extend(json.loads((ROOT/'receipts'/name).read_text(encoding='utf-8')))
summary={'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'query_count':len(queries),'query_failures':sum(not x['response'].get('success') for x in queries),'query_empty':sum(x['response'].get('success',False) and not x['response'].get('data',{}).get('web') for x in queries),'successful_code_blobs':sum(r.get('verified',False) for r in records),'successful_pdfs':sum('sha256' in r for r in records),'acquisition_errors':[r for r in records if 'error' in r],'citation_verification_code':check.returncode,'citation_verification':check.stdout+check.stderr,'learned_model_executed':False,'synthetic_gates_executed':False,'market_data_read':False}
(ROOT/'VERIFICATION.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(summary,ensure_ascii=False,indent=2))
if check.returncode:sys.exit(check.returncode)
