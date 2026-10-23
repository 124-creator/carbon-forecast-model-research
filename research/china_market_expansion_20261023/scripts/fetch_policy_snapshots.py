import datetime,hashlib,json,pathlib,urllib.request,urllib.error,time,sys
sources=json.loads(sys.argv[1]);root=pathlib.Path(sys.argv[2]);root.mkdir(parents=True,exist_ok=True)
out=[]
for i,s in enumerate(sources):
 rec=dict(s);rec["retrieved_utc"]=datetime.datetime.now(datetime.timezone.utc).isoformat()
 try:
  req=urllib.request.Request(s["url"],headers={"User-Agent":"Mozilla/5.0 (compatible; AcademicPolicyAudit/1.0)"})
  with urllib.request.urlopen(req,timeout=30) as r:
   rec["http_status"]=r.status;rec["final_url"]=r.geturl();rec["content_type"]=r.headers.get("Content-Type")
   body=r.read(8000000)
   if r.read(1):raise ValueError("Body exceeds 8MB")
   rec["bytes"]=len(body);rec["sha256"]=hashlib.sha256(body).hexdigest()
   suffix=".pdf" if body.startswith(b"%PDF") else ".html"
   target=root/(s["id"]+suffix);target.write_bytes(body)
   rec["local_raw"]=target.name
 except urllib.error.HTTPError as ex:
  rec["http_status"]=ex.code;rec["error"]=str(ex)
  body=ex.read(1000000);rec["error_body_sha256"]=hashlib.sha256(body).hexdigest()
  (root/(s["id"]+".failure.html")).write_bytes(body)
 except Exception as ex:rec["error"]=repr(ex)
 out.append(rec);print(json.dumps({"id":s["id"],"http":rec.get("http_status"),"bytes":rec.get("bytes"),"error":rec.get("error")}),flush=True)
 if i+1<len(sources):time.sleep(0.3)
(root/"receipts.json").write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding="utf-8")
