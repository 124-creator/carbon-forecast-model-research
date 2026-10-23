#!/usr/bin/env python3
"""Fetch a bounded public relay snapshot for research audit (no model fitting).
Raw pages and per-row CSV stay in private local output; redistribution requires review.
"""
import argparse, collections, csv, datetime, hashlib, html.parser, json, pathlib, re, time
import urllib.error, urllib.parse, urllib.request

HEADERS=["交易日期","交易机构","交易品种","开盘价","最高价","最低价","成交均价","收盘价","成交量","成交金额"]
FIELDS=["date","institution","instrument","open_text","high_text","low_text","mean_text","close_text","volume_text","turnover_text"]
BASE="https://ets.sceex.com.cn/internal.htm"

class Tables(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.depth=0; self.tables=[]; self.row=None; self.cell=None
    def handle_starttag(self,tag,attrs):
        if tag=="table":
            self.depth+=1
            if self.depth==1: self.tables.append([])
        if self.depth==1 and tag=="tr": self.row=[]
        if self.depth==1 and tag in ("td","th"): self.cell=[]
    def handle_data(self,data):
        if self.cell is not None: self.cell.append(data)
    def handle_endtag(self,tag):
        if self.depth==1 and tag in ("td","th") and self.cell is not None:
            if self.row is not None: self.row.append(" ".join("".join(self.cell).split()))
            self.cell=None
        if self.depth==1 and tag=="tr" and self.row is not None:
            self.tables[-1].append(self.row); self.row=None
        if tag=="table": self.depth-=1

def valid_date(value):
    try: return datetime.date.fromisoformat(value).isoformat()==value
    except ValueError: return False

def run():
    p=argparse.ArgumentParser()
    p.add_argument("--start",required=True);p.add_argument("--end",required=True)
    p.add_argument("--out",required=True);p.add_argument("--max-pages",type=int,default=40)
    p.add_argument("--page-size",type=int,default=1000);p.add_argument("--delay",type=float,default=0.4)
    a=p.parse_args()
    assert valid_date(a.start) and valid_date(a.end) and a.start<=a.end
    assert 0<a.max_pages<=250 and 0<a.page_size<=1000
    dest=pathlib.Path(a.out).resolve()
    dest.mkdir(parents=True,exist_ok=True)
    raw=dest/"raw";raw.mkdir(exist_ok=True)
    allrows=[];pages=[];errors=[];complete=False;recent=set()
    for i in range(1,a.max_pages+1):
        q={"k":"guo_nei_xing_qing","orderby":"tradeTime desc","url":"mrhq_gn",
           "start_time":a.start,"end_time":a.end,"pageSize":a.page_size,"pageIndex":i}
        url=BASE+"?"+urllib.parse.urlencode(q)
        rec={"page":i,"url":url,"retrieved_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
             "method":"GET","authenticated":False,"requested_range":[a.start,a.end]}
        body=None
        try:
            req=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0 (compatible; CarbonResearchSourceAudit/1.0)"})
            with urllib.request.urlopen(req,timeout=40) as response:
                rec.update(http_status=response.status,final_url=response.geturl())
                body=response.read(8000000)
                if response.read(1): raise ValueError("Response exceeds eight MB")
        except urllib.error.HTTPError as exc:
            rec.update(http_status=exc.code,error=repr(exc));body=exc.read(1000000)
        except Exception as exc:
            rec.update(error=repr(exc))
        if body is not None:
            fname="page_%04d.html"%i
            (raw/fname).write_bytes(body)
            rec.update(raw_file=fname,bytes=len(body),sha256=hashlib.sha256(body).hexdigest())
        if rec.get("http_status")!=200 or body is None:
            errors.append(rec);pages.append(rec);print(json.dumps({"page":i,"failed":rec},ensure_ascii=False),flush=True)
            break
        tables=Tables()
        try:
            tables.feed(body.decode("utf-8"))
            matches=[x for x in tables.tables if x and x[0]==HEADERS]
            if len(matches)!=1: raise ValueError("Ambiguous or missing expected table")
            pr=matches[0][1:]
            if any(len(r)!=len(FIELDS) for r in pr): raise ValueError("Source row width differs from schema")
            if any(not valid_date(r[0]) or not a.start<=r[0]<=a.end for r in pr):
                raise ValueError("Server ignored date filter, or invalid date")
        except Exception as exc:
            rec["parse_error"]=repr(exc);errors.append(rec);pages.append(rec)
            print(json.dumps({"page":i,"parse_error":repr(exc)},ensure_ascii=False),flush=True);break
        rec.update(rows=len(pr),date_min=min((r[0] for r in pr),default=None),
                   date_max=max((r[0] for r in pr),default=None))
        if pr and rec["sha256"] in recent:
            rec["parse_error"]="Duplicate page bytes while paginating"
            pages.append(rec);errors.append(rec);break
        recent.add(rec["sha256"]);pages.append(rec)
        for j,values in enumerate(pr,1):
            one=dict(zip(FIELDS,values));one.update(source_page=i,source_table_row=j,
                                                    source_url=url,source_sha256=rec["sha256"])
            allrows.append(one)
        print(json.dumps({"page":i,"rows":len(pr),"min":rec["date_min"],"max":rec["date_max"]}),flush=True)
        if len(pr)<a.page_size:
            complete=True;break
        if i<a.max_pages:time.sleep(a.delay)
    counter=collections.Counter(x["institution"] for x in allrows)
    product=collections.Counter((x["institution"],x["instrument"]) for x in allrows)
    allkeys=collections.Counter((x["date"],x["institution"],x["instrument"]) for x in allrows)
    identical=collections.Counter(tuple(x[k] for k in FIELDS) for x in allrows)
    missing=collections.Counter()
    for row in allrows:
        for k in FIELDS[3:]:
            if row[k].strip() in ("","-","--","#N/A"):missing[k]+=1
    audit={"retrieved_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "requested_range":[a.start,a.end],"page_size":a.page_size,
           "complete":complete and not errors,"pages":pages,"error_count":len(errors),
           "raw_table_rows":len(allrows),"first_date":min((r["date"] for r in allrows),default=None),
           "last_date":max((r["date"] for r in allrows),default=None),
           "institution_rows":dict(counter),"institution_instrument_rows":
                [{"institution":k[0],"instrument":k[1],"rows":v} for k,v in sorted(product.items())],
           "missing_tokens_by_column":dict(missing),
           "duplicate_market_date_instrument_excess":sum(v-1 for v in allkeys.values()),
           "identical_row_excess":sum(v-1 for v in identical.values()),
           "license":"Unverified. Raw response and all-row data MUST NOT be copied to public GitHub.",
           "observation_status":"Third-party official exchange relay; historical volume/amount units, price and channel identity not certified.",
           "evaluation":{"model_fits":0,"scored_origins":0,"future_targets":0,
                         "deleted_source_rows":0,"imputed_source_values":0,
                         "confirmation_2026":False}}
    if allrows:
        with (dest/"all_rows_local_only.csv").open("w",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
        # A matching SHA-256 does not prove a canonical file contains every
        # page: a previous archive recorded a SHA for a 6,026-row truncation
        # although the 22 source pages contained 21,430 rows. Re-open and
        # verify the complete output before issuing a successful audit.
        with (dest/"all_rows_local_only.csv").open(encoding="utf-8",newline="") as verify:
            reader=csv.DictReader(verify)
            saved=list(reader)
        if len(saved)!=len(allrows) or any(
                any(saved[i][key]!=str(original[key]) for key in allrows[0])
                for i,original in enumerate(allrows)):
            raise RuntimeError("CANONICAL_CSV_INTEGRITY_FAILURE: full-row readback mismatch")
        by_page=collections.Counter(int(record["source_page"]) for record in saved)
        declared=collections.Counter({p["page"]:p["rows"] for p in pages if "rows" in p})
        if by_page!=declared:
            raise RuntimeError("CANONICAL_CSV_INTEGRITY_FAILURE: source-page count mismatch")
        audit["local_csv_readback"]={"verified":True,"row_count":len(saved),
               "sha256":hashlib.sha256((dest/"all_rows_local_only.csv").read_bytes()).hexdigest(),
               "per_page_rows":dict(by_page)}
    (dest/"audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"complete":audit["complete"],"rows":len(allrows),"markets":len(counter),
                      "duplicate_excess":audit["duplicate_market_date_instrument_excess"],
                      "error_count":len(errors)}),flush=True)
    if not audit["complete"]:raise SystemExit(2)

if __name__=="__main__":run()
