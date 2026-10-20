"""Offline final acceptance: verify every receipt, no further network."""
from pathlib import Path
import json,csv,hashlib,datetime,re,html
ROOT=Path(__file__).resolve().parent
receipts=json.loads((ROOT/'ACQUISITION_RECEIPTS.json').read_text(encoding='utf8'))
for r in receipts:
 if 'file' in r:
  b=(ROOT/r['file']).read_bytes()
  if len(b)!=r['bytes'] or hashlib.sha256(b).hexdigest()!=r['sha256']:raise ValueError('receipt bytes mismatch')
manifest=json.loads((ROOT/'INPUT_MANIFEST.json').read_text(encoding='utf8')); unique={}
for x in manifest:
 b=Path(x['file']).read_bytes()
 if len(b)!=x['bytes'] or hashlib.sha256(b).hexdigest()!=x['sha256']:raise ValueError('input bytes mismatch')
 if x['url'] in unique and unique[x['url']]['sha256']!=x['sha256']:raise ValueError('duplicate source conflict')
 unique[x['url']]=x
(ROOT/'FINAL_UNIQUE_INPUT_MANIFEST.json').write_text(json.dumps(list(unique.values()),ensure_ascii=False,indent=2),encoding='utf8')
def readcsv(name):return list(csv.DictReader((ROOT/name).open(encoding='utf-8-sig')))
events=readcsv('announcement_index_LOCAL_ONLY.csv');focus=readcsv('coal_5500_LITERAL_LOCAL_ONLY.csv');allrows=readcsv('coal_events_LOCAL_ONLY.csv');coverage=readcsv('COVERAGE_MONTH_TENDAY.csv');missing=readcsv('MISSING_MONTH_TENDAY.csv')
if len({(r['year'],r['month'],r['ten_day']) for r in events})!=len(events):raise ValueError('duplicate event slots')
if len(focus)!=len(events) or len(allrows)!=7*len(events) or len(coverage)!=144 or len(missing)+len(events)!=144:raise ValueError('count mismatch')
for r in focus:
 if r['product_spec_literal'] not in ('山西优混（ 5500 大卡 ）','山西优混（ 5500 大卡）'):raise ValueError('literal spec changed')
 pub=datetime.datetime.fromisoformat(r['publication_displayed']);avail=datetime.date.fromisoformat(r['available_date_1day'])
 if avail!=pub.date()+datetime.timedelta(days=1):raise ValueError('availability alignment')
parsedurls={e['url'] for e in events};excluded=[]
for url,item in unique.items():
 if url in parsedurls:continue
 s=Path(item['file']).read_text(encoding='utf8');title=html.unescape(re.sub('<[^>]*>','',re.search('<title>(.*?)</title>',s,re.S)[1])).strip()
 written=re.search(r'成文日期</span>([^<]+)',s)
 excluded.append({'url':url,'title':title,'document_written_date_literal':written[1] if written else None,'exclusion':'No literal publication datetime; written date is not automatically publication date','source_sha256':item['sha256']})
(ROOT/'RETRIEVED_UNRESOLVED_PUBLICATION.json').write_text(json.dumps(excluded,ensure_ascii=False,indent=2),encoding='utf8')
summary=json.loads((ROOT/'VERIFICATION.json').read_text(encoding='utf8'))
summary.update(final_unique_source_bodies=len(unique),retrieved_period_events=len(unique),unresolved_publication_events=len(excluded),literal_specs_preserved=sorted({r['product_spec_literal'] for r in focus}),receipt_hash_verification='PASS',output_counts_verification='PASS',network_acquisition_elapsed_seconds=861.63,network_acquisition_closed=True,offline_verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),blockers=['21 of 144 expected period slots remain unavailable for publication-aligned use: 18 not obtained through bounded visible search, 3 official bodies display only written date rather than publication datetime.','Search async renders and similar-result folding limit visible canonical links; incomplete result batches retained as evidence, not completeness claims.','Current-snapshot publication labels do not certify historical first-version bytes.'],parse_errors=excluded)
(ROOT/'FINAL_VERIFICATION.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
lines=['# 煤 panel 有界获取结果','',f'可用发布对齐 events：{len(events)}；字面规格 山西优混（5500大卡） 可用价格：{len(focus)}；煤焦价格行：{len(allrows)}。',f'实际获得唯一公告正文 {len(unique)}；其中 {len(excluded)} 个仅展示成文日期，隔离，不猜发布日期。','', '|年份|可用 events / 36|缺旬|','|---|---:|---:|']
for y,stats in summary['by_year'].items():lines.append(f"|{y}|{stats['events']}|{stats['missing_5500_slots']}|")
lines+=['',f"首末发布：{summary['first_publication']} — {summary['last_publication']}。",'统计期间始于2018-01-01；最后标题为2021年12月下旬，2021-12-31是旬度日历约定而非标题字面末日。数值日标题保持字面末日（例如21—30日不改31日）。','发布日+1自然日；不训练、不跨年carry、不future fill，不读碳2022、不push。2022-01-04是2021年12月下旬煤公告发布日期，不是读取碳2022。','E与D父目录均存在，resolve归于E，samefile=True；所有新产物位于本目录，旧receipts不修改。','新普通HTTP请求192次（浏览器正常页面/依赖请求另计），间隔至少1秒；获取窗口861.63秒，随后只做离线验证；无安全拒绝。','官网搜索可正常渲染；先使用官网可见67页历史分页，随后搜索高级“全部关键词/标题”并按可见分页和缺月定位；未猜公告URL，年度zxfb2020未视作2020历史库。','月份/旬逐项缺口见COVERAGE_MONTH_TENDAY.csv和MISSING_MONTH_TENDAY.csv。','阻塞：18个预期旬未在有界路线中获得正文；另3个正文缺字面发布日期，不能作+1day可用feature。此结果不是完整144旬panel。','浏览器异步分页造成部分空批、社会媒体结果无canonical正文；保留所有批manifest，不把搜索reported total当官方公告数。','离线验证：全部普通HTTP原文长度/哈希、唯一输入、123个唯一期间、861行/每期7规格、123行固定5500字面规格、144旬和21缺旬总计、发布+1day对齐全部通过。']
(ROOT/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
