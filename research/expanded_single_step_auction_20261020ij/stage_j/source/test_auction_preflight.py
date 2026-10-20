"""Independently test next-event info boundary; synthetic only and original source SHA."""
from __future__ import annotations
from datetime import datetime,timedelta
from pathlib import Path
import io,zipfile,hashlib
import numpy as np,pandas as pd
from run_auction_mechanism import build_features,open_verified,LOOKBACK,MODELS,SOURCE_SHA,SOURCE_MEMBER,metrics

def main(archive):
    rng=np.random.default_rng(49)
    n=99; dates=pd.date_range('2020-01-02',periods=n,freq='B');price=40+np.cumsum(rng.normal(size=n))
    df=pd.DataFrame({'date':dates,'price':price,'cover_ratio':1.5+np.exp(rng.normal(scale=.1,size=n)),
                     'auction_volume':np.ones(n)*1e6,'total_bid_volume':np.ones(n)*2.7e6,
                     'bidder_count':np.ones(n)*18,'successful_bidder_count':np.ones(n)*12})
    full,origins,anchor,target,dd=build_features(df)
    # Prefix invariance: features for origin 75 cannot depend on future 76..98 rows.
    cut=75; sub=build_features(df.iloc[:cut+2])
    row=np.flatnonzero(origins==cut)[0]
    prefidx=np.flatnonzero(sub[1]==cut)[0]
    gap=max(float(np.max(np.abs(full[k][row]-sub[0][k][prefidx]))) for k in full)
    assert gap<1e-12,gap
    # Explicit future mutation, no observed feature leakage at cut.
    changed=df.copy();changed.loc[cut+1:,'price']*=7
    changed.loc[cut+1:,'cover_ratio']*=15
    feats2,orig2,anc2,tar2,dat2=build_features(changed)
    row2=np.flatnonzero(orig2==cut)[0]
    future_gap=max(float(np.max(np.abs(full[k][row]-feats2[k][row2]))) for k in full)
    assert future_gap<1e-10,future_gap
    assert target[row] != tar2[row2] and anchor[row]==anc2[row2]
    assert set(MODELS)-set(full)=={'rw'}
    data=open_verified(archive)
    assert len(data)==2255 and str(data.date.min().date())=='2012-10-26'
    print({'status':'PASS','synthetic_future_mutation_max_feature_change':future_gap,'prefix_invariance_gap':gap,
           'source_records':len(data),'source_sha256':SOURCE_SHA,'has_2024_values':False})
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--archive',required=True);a=ap.parse_args();main(a.archive)
