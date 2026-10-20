"""Design qualification only: fixed audited prefixes; no numeric price parsing."""
import csv, hashlib, json
from pathlib import Path
ROOT=Path('E:/碳价格/独立数据归档/carbon-forecast-model-research')
OUT=Path(__file__).resolve().parent
COUNTS={'guangzhou':950,'hubei':936,'eu':999}
AUDIT=ROOT/'research/local_mature_feedback_20261013/INPUT_AUDIT.json'
def bounded(path,n):
    lines=[]
    with path.open('rb',buffering=0) as f:
        for _ in range(n+1):
            line=bytearray()
            while True:
                b=f.read(1)
                if not b: raise ValueError('short prefix')
                line+=b
                if b==b'\n': break
            lines.append(bytes(line))
    raw=b''.join(lines)
    rows=list(csv.DictReader(raw.decode('utf-8-sig').splitlines()))
    if len(rows)!=n: raise ValueError('count disagreement')
    dates=[r['date'] for r in rows]
    if dates[-1]!='2021-12-31' or dates!=sorted(dates) or len(set(dates))!=n or any(d>'2021-12-31' for d in dates): raise ValueError('date disagreement')
    return dates,{'records':n,'bytes_consumed':len(raw),'prefix_sha256':hashlib.sha256(raw).hexdigest(),'boundary_requested':False,'numeric_price_conversion':False}
def main():
    previous=json.loads(AUDIT.read_text(encoding='utf-8'))
    prior={Path(x['file']).name:x['included_rows'] for x in previous}
    all_dates={}; result={'scope':'design qualification, not price training','prior_count_authority':str(AUDIT),'prior_boundary_text_exposure_disclosed':True,'markets':{}}
    for market,n in COUNTS.items():
        path=ROOT/f'data/parsed/figshare_28682348/{market}_reported.csv'
        if prior.get(path.name)!=n: raise ValueError('missing count authority')
        dates,receipt=bounded(path,n); all_dates[market]=dates
        receipt.update(path=str(path),first_date=dates[0],last_date=dates[-1],count_authority='historical audit; independently date-checked bounded prefix')
        result['markets'][market]=receipt
    if result['markets']['guangzhou']['prefix_sha256']!='1366ca4814dcdb8aa43296eb64768d1336d6f0fc9088648d6d7ccdbe43318070': raise ValueError('Guangzhou prefix hash mismatch')
    # Keep existing NHITS rolling initial split; source labels strictly before target training final label date.
    blocks=list(range(459,900,40)); schedules=[]
    for b in blocks:
        cutoff=all_dates['guangzhou'][b-1]
        item={'target_fit_end_exclusive':b,'target_training_last_label_date':cutoff,'source_label_rule':'endpoint_date < target_training_last_label_date','markets':{}}
        for m,dates in all_dates.items():
            n=sum(d<=cutoff if m=='guangzhou' else d<cutoff for d in dates)
            item['markets'][m]={'eligible_quote_records':n,'last_label_date':dates[n-1] if n else None,'mature_pairs_L60':{str(h):max(0,n-60-h) for h in [1,5,20]},'common_H20_origins':max(0,n-60-20)}
        schedules.append(item)
    result['proposed_rolling_schedule']=schedules
    result['window_contract']={'L':60,'H':20,'L_is_differences_requires_61_prices':True,'all_H_labels_mature':True,'target_complete_development_origins_initial459':max(0,950-459-20),'not_new_test':True}
    (OUT/'DATA_QUALIFICATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'markets':result['markets'],'initial_schedule':schedules[0],'window_contract':result['window_contract']},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
