"""Root-approved three-market source data, description only, no fitting.

All three complete files were explicitly designated development by root.
Numerical summaries are restricted to dates <=2023-12-31 for common coverage.
"""
from pathlib import Path
import csv
import json
import hashlib
import io
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCES = ROOT.parents[1] / "round2_data_sources"


def acf(x, lag):
    if len(x) <= lag + 2 or np.std(x[:-lag]) == 0 or np.std(x[lag:]) == 0:
        return None
    return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])


def run():
    manifest = json.loads((SOURCES / "DATA_PACKET_MANIFEST.json").read_text())
    table, cross_dates, sources = [], {}, []
    for market in ("guangzhou", "hubei", "eu"):
        path = SOURCES / (market + "_reported.csv")
        raw = path.read_bytes()
        expected = next(row for row in manifest["files"]
                        if row["kind"] == "derived_csv" and Path(row["path"]).name == path.name)
        assert len(raw) == expected["bytes"] and hashlib.sha256(raw).hexdigest() == expected["sha256"]
        rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
        assert len(rows) == expected["rows"]
        selected = [row for row in rows if row["date"] <= "2023-12-31"]
        date = np.array([row["date"] for row in selected])
        price = np.array([float(row["author_reported_price"]) for row in selected])
        assert np.all(np.diff(date.astype("datetime64[D]")) > np.timedelta64(0,"D"))
        assert np.isfinite(price).all() and np.all(price > 0)
        sources.append(dict(market=market, total_source_rows=len(rows), described_rows=len(selected),
                            first_date=str(date[0]), last_described_date=str(date[-1]),
                            sha256=expected["sha256"], all_source_explicitly_development=True))
        change = np.diff(price)
        logret = np.diff(np.log(price))
        cross_dates[market] = dict(zip(date[1:], logret))
        for year in map(str, range(2018, 2024)):
            idx = np.flatnonzero(np.char.startswith(date,year))
            pidx = idx[idx > 0]
            delta = price[pidx] - price[pidx-1]
            lr = np.log(price[pidx]/price[pidx-1])
            table.append(dict(market=market, year=year, observations=len(idx), returns=len(pidx),
                mean_price=float(price[idx].mean()), mean_price_change=float(delta.mean()),
                sd_price_change=float(delta.std(ddof=1)), mean_log_return=float(lr.mean()),
                sd_log_return=float(lr.std(ddof=1)), zero_price_change_fraction=float(np.mean(delta==0)),
                zero_price_change_count=int(np.count_nonzero(delta==0)),
                delta_acf1=acf(delta,1), delta_acf5=acf(delta,5),
                logreturn_acf1=acf(lr,1), logreturn_acf5=acf(lr,5)))
    pairs=[]
    for a,b in (("guangzhou","hubei"),("guangzhou","eu"),("hubei","eu")):
        common=sorted(set(cross_dates[a]) & set(cross_dates[b]))
        ar=np.array([cross_dates[a][d] for d in common])
        br=np.array([cross_dates[b][d] for d in common])
        pairs.append(dict(markets=[a,b], same_date_observed_returns=len(common),
                          descriptive_logreturn_correlation=float(np.corrcoef(ar,br)[0,1]),
                          not_a_predictor_feature=True,
                          caveat="Each return spans its own observation gap; contemporaneous date is not synchronized release time"))
    output=dict(scope="explicitly approved expanded development data, description only",
                market_fits=0, numerical_summary_cutoff="2023-12-31", sources=sources,
                market_years=table, same_date_description=pairs,
                rows_described=sum(r["described_rows"] for r in sources),
                no_independent_confirmation_claim=True,
                source_semantics_limitations=manifest["known_unresolved"])
    with (ROOT/"EXPANDED_DEVELOPMENT_DESCRIPTION.json").open("x",encoding="utf-8") as f:
        json.dump(output,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write("\n")
    print(json.dumps(output,ensure_ascii=False))


if __name__ == "__main__":
    run()
