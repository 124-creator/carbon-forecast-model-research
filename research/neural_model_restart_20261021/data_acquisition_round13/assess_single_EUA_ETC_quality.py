from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import csv,json,math,hashlib
OUT=Path(__file__).resolve().parent

def save(name,data):
    with (OUT/name).open('x',encoding='utf-8') as f: json.dump(data,f,indent=2,ensure_ascii=False)

issuer_evidence={
    'source_url':'https://hanetf.com/fund/co2-sparkchange-physical-carbon-eua-etc/',
    'mechanism':'Successful functions.web_fetch HTTP200 text extraction earlier in this continuation. Direct Python GET later returned HTTP403 and is archived separately; no browser verification bypass attempted.',
    'archived_at_utc':datetime.now(timezone.utc).isoformat(),
    'retrieval_time_note':'Web-fetch tool did not expose an exact timestamp; it was read during this 2026-10-21 continuation before the direct requests at 14:38:07 UTC.',
    'verbatim_excerpts':[
        'SparkChange Physical Carbon EUA ETC (CO2) is designed to offer investors a means of accessing physical European Union carbon Allowances (EUAs).',
        'The carbon EUA ETC is 100% physically backed by, and tracks the price of, EUAs (excluding fees).',
        'Inception Date | 18/10/2021',
        'Base Currency | EUR',
        'Replication Method | Physical',
        'ISIN | XS2353177293',
        'Issuer | HANetf ETC Securities Plc',
        'LSE | CO2 LN | CO2.L | BNLYQ70 | € | 2021-11-04',
        'LSE | CO2P LN | CO2P.L | BNLYQC5 | £ | 2021-11-04',
        'As of 20.10.2026',
        'Entitlement Per Security | 0.9566130510'
    ],
    'interpretation':'CO2.L and CO2P.L are two listing/currency lines of the same physically EUA-backed ETC, not two independent instruments. Each listed security is not one EUA over all dates: entitlement changes with accumulated fees.'
}
save('HANetf_CO2_issuer_verified_webfetch_excerpt.json',issuer_evidence)
log=json.loads((OUT/'EUA_ETC_candidates_acquisition_log.json').read_text(encoding='utf-8'))
results=[]
for rec in log['requests']:
    if not rec['name'].startswith('yahoo_') or 'daily_file' not in rec: continue
    symbol=rec['instrument_metadata']['symbol']; key=symbol.replace('.','_')
    with (OUT/rec['daily_file']).open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
    def finite(r):
        return all(r[c] and math.isfinite(float(r[c])) for c in ('open','high','low','close','volume','adj_close'))
    def stats(rs):
        complete=[r for r in rs if finite(r)]
        o=[r for r in complete if not float(r['low'])<=float(r['open'])<=float(r['high'])]
        c=[r for r in complete if not float(r['low'])<=float(r['close'])<=float(r['high'])]
        bad=[r for r in complete if not float(r['low'])<=min(float(r['open']),float(r['close']))<=max(float(r['open']),float(r['close']))<=float(r['high'])]
        positive=[r for r in complete if float(r['volume'])>0]
        return {'rows':len(rs),'complete_rows':len(complete),'missing_rows':len(rs)-len(complete),
            'ohlc_order_violations':len(bad),'open_outside_high_low':len(o),'close_outside_high_low':len(c),
            'high_below_low':sum(float(r['high'])<float(r['low']) for r in complete),
            'material_close_outside_range_gt_1bp_of_close':sum(max(float(r['low'])-float(r['close']),float(r['close'])-float(r['high']),0)>float(r['close'])*1e-4 for r in complete),
            'zero_volume_rows':sum(float(r['volume'])==0 for r in complete),
            'zero_high_low_range_rows':sum(float(r['high'])==float(r['low']) for r in complete),
            'valid_order_positive_volume_positive_range_rows':sum(float(r['low'])<=min(float(r['open']),float(r['close']))<=max(float(r['open']),float(r['close']))<=float(r['high']) and float(r['volume'])>0 and float(r['high'])>float(r['low']) for r in complete),
            'violation_examples':bad[:3]}
    is_physical=symbol in ('CO2.L','CO2P.L')
    listing_date='2021-11-04' if is_physical else '2021-08-27'
    post=[r for r in rows if r['date']>=listing_date]
    pre=[r for r in rows if r['date']<listing_date]
    item={'symbol':symbol,'raw_csv':rec['daily_file'],'sha256':rec['daily_sha256'],
        'issuer_identity_verified':is_physical,
        'identity': 'SparkChange Physical Carbon EUA ETC; ISIN XS2353177293; physically held EUA-backed debt ETC' if is_physical else 'Candidate WisdomTree Carbon; current factsheet search index gives ISIN JE00BP2PWW32, EUA futures-index exposure via swaps. Full authoritative document retrieval blocked; continuity to Yahoo pre-2021 series not verified.',
        'listing_date':listing_date,'listing_date_evidence':'HANetf verified issuer page' if is_physical else 'Official factsheet search index only; candidate status retained',
        'yahoo_instrumentType':rec['instrument_metadata'].get('instrumentType'),
        'instrumentType_note':'Yahoo generic EQUITY classification is not legal product type; issuer identifies ETC.',
        'base_currency':'EUR' if is_physical else 'USD (candidate current product)',
        'yahoo_quote_currency':rec['instrument_metadata'].get('currency'),
        'price_unit':{'CO2.L':'EUR per ETC security','CO2P.L':'GBp (pence sterling) per ETC security; divide OHLC and adj_close by 100 for GBP, leave volume unchanged','CARB.L':'USD per ETC security, per Yahoo; current official factsheet index also lists CARB USD' }[symbol],
        'volume_unit':'ETC securities; no conversion to physical EUA tonnes or EUA futures contracts',
        'same_security_other_currency_line':'CO2P.L' if symbol=='CO2.L' else ('CO2.L' if symbol=='CO2P.L' else None),
        'pre_listing_rows':len(pre),'pre_listing_rows_are_all_zero_volume':all(r['volume'] and float(r['volume'])==0 for r in pre),
        'segments':{'all_rows':stats(rows),'post_listing':stats(post),'since_2024':stats([r for r in post if r['date']>='2024-01-01'])},
        'recommendation':'Do not use as verified OHLC structural modelling input: large range/order inconsistencies persist after listing and in 2024 onward. Preserve raw values; no high/low repair, forward fill, or selective valid-row training dataset produced.',
        'pre_listing_caution':'Yahoo starts on 2021-10-18, issuer inception date, before 2021-11-04 exchange listing. Pre-listing observations must not be treated as traded daily OHLCV.' if is_physical else 'Yahoo history begins before current CARB listing 2021-08-27; possible predecessor/relaunch continuity and current ISIN linkage unresolved. Keep candidate only.'
    }
    results.append(item)
save('EUA_ETC_identity_units_segmented_quality.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'instruments':results,'conclusion':'No candidate passes OHLC-structure usability gate. CO2.L physical EUA ETC identity is verified and EUR units align, but Yahoo quotes do not establish valid candle geometry. CO2P.L same security in GBp, not an independent EUA asset. CARB.L remains candidate due blocked identity-document retrieval and historical linkage uncertainty.','source_scope':'Only three requested tickers and corresponding issuer material; no model results read; no old KRBN data changed.'})
print(json.dumps([{k:r[k] for k in ('symbol','issuer_identity_verified','price_unit','pre_listing_rows','segments')} for r in results],indent=2,ensure_ascii=False))
