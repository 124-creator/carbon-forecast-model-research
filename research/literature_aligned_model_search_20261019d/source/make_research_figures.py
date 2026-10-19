"""Static research figures generated exclusively from the complete result tables."""
from pathlib import Path
import hashlib
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/carbon-model-search-mpl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[1]
DEST = STUDY/"figures"
DEST.mkdir(exist_ok=True)
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "axes.spines.top":False,
                     "axes.spines.right":False, "savefig.dpi":180, "axes.titleweight":"bold"})
NAVY, TEAL, GREY, ORANGE = "#23415a", "#087f8c", "#91a1af", "#c46b34"


def save(fig, name):
    fig.savefig(DEST/f"{name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(DEST/f"{name}.pdf", bbox_inches="tight", facecolor="white",
                metadata={"CreationDate":None,"ModDate":None})
    plt.close(fig)


def table(name):
    return pd.read_csv(STUDY/"results"/name/"CELL_METRICS.csv")


def development():
    v1,v2=table("main_v1"),table("liquidity_v2")
    choices=[
        (v1, ["allocation_full","allocation_readout","allocation_constant","allocation_price","gru","lstm","rw"],
         ["Sequential MLP gate (primary)","Terminal MLP gate","Constant allocation","Price-only allocation","GRU","LSTM","Random walk"],
         "V1 | Four markets, 2022 and 2023"),
        (v2, ["liquidity_full","liquidity_readout","shock_size_only","liquidity_unconstrained","gru","lstm","rw"],
         ["Volume allocation (primary)","Terminal volume gate","Shock-size only","Signed coefficients","GRU with volume","LSTM with volume","Random walk"],
         "V2 | Guangzhou and Hubei, 2022 and 2023"),
    ]
    fig,axes=plt.subplots(1,2,figsize=(13,5.5),layout="constrained")
    for ax,(df,families,labels,title) in zip(axes,choices):
        means=df.groupby("family").mean_seed_mse_over_rw.mean()
        vals=[means[f] for f in families]
        colors=[TEAL if i==0 else NAVY if i==1 else GREY for i in range(len(vals))]
        y=np.arange(len(vals))
        ax.barh(y,vals,color=colors,height=.65)
        for i,v in enumerate(vals): ax.text(1.55,i,f"{v:.4f}",ha="right",va="center",fontsize=9)
        ax.set_yticks(y,labels);ax.invert_yaxis();ax.axvline(1,color=ORANGE,lw=1.3,ls="--")
        ax.set_xlim(0,1.57);ax.set_xlabel("Mean-seed MSE / random-walk MSE (lower is better)")
        ax.set_title(title,fontsize=11,pad=12);ax.grid(axis="x",alpha=.15);ax.set_axisbelow(True)
    fig.suptitle("Development results: different cohorts are shown separately",fontsize=15)
    save(fig,"01_development_results")


def replication():
    df=pd.concat([table("new_market_v3"),table("swewt_gru_supplement")],ignore_index=True)
    families=["liquidity_full","allocation_readout","allocation_constant","shock_size_only","liquidity_readout","swewt_gru_full","swewt_gru_price","gru","lstm","rw"]
    labels=["Volume allocation | primary","Terminal MLP | secondary","Constant allocation","Shock-size only","Terminal volume gate","SWEWT-GRU + inputs*","SWEWT-GRU price*","GRU + inputs","LSTM + inputs","Random walk"]
    fig,axes=plt.subplots(1,4,figsize=(16,6.2),sharey=True,layout="constrained")
    for ax,market,title in zip(axes,["BEIJING_MEAN","TIANJIN_MEAN","MACRO","CHONGQING_MEAN"],
                              ["Beijing | 116 origins","Tianjin | 72 origins","Beijing / Tianjin mean","Chongqing | 28 origins\nSeparate stress test"]):
        subset=df[df.market.isin(["BEIJING_MEAN","TIANJIN_MEAN"])] if market=="MACRO" else df[df.market==market]
        means=subset.groupby("family").mean_seed_mse_over_rw.mean()
        vals=[means[f] for f in families]
        colors=[TEAL if i==0 else NAVY if i==1 else GREY for i in range(len(vals))]
        ax.barh(np.arange(len(vals)),vals,color=colors,height=.66)
        for i,v in enumerate(vals): ax.text(1.24,i,f"{v:.3f}",ha="right",va="center",fontsize=8)
        ax.axvline(1,color=ORANGE,lw=1.1,ls="--");ax.set_xlim(0,1.26)
        ax.set_title(title,fontsize=10);ax.set_xlabel("Mean-seed MSE / RW");ax.grid(axis="x",alpha=.15);ax.set_axisbelow(True)
    axes[0].set_yticks(np.arange(len(families)),labels);axes[0].invert_yaxis()
    fig.suptitle("Additional-market replication on author-cleaned daily mean prices",fontsize=15)
    fig.supxlabel("* SWEWT-GRU is a separately recorded, multi-step controlled adaptation added after partial V3 results.\nPoint estimates shown; dependent-series uncertainty and all other baselines are reported in the audit and full tables.",fontsize=9)
    save(fig,"02_new_market_replication")


def information():
    p=STUDY/"results/information_diagnostic/METRICS.csv"
    df=pd.read_csv(p)
    # Column names are checked against the recorded diagnostic output.
    score="test_mse_over_rw"
    protocol="mode"
    markets=["GUANGZHOU","HUBEI","SHANGHAI_D4","EU_REPORTED"]
    names=["RAW","SLIDING_PAST_ONLY","FULL_SERIES_RETROSPECTIVE"]
    fig,ax=plt.subplots(figsize=(10,5.4),layout="constrained")
    for j,(name,label,color) in enumerate(zip(names,["Raw lags","Historical sliding EWT","Full-series EWT (retrospective)"],[GREY,TEAL,ORANGE])):
        vals=[float(df[(df.market==m)&(df[protocol]==name)][score].iloc[0]) for m in markets]
        x=np.arange(4)+(j-1)*.25
        ax.bar(x,vals,width=.23,label=label,color=color)
        for xx,v in zip(x,vals): ax.text(xx,v+.025,f"{v:.3f}",ha="center",fontsize=8)
    ax.set_xticks(np.arange(4),["Guangzhou","Hubei","Shanghai","EU reported"])
    ax.set_ylim(0,1.23);ax.axhline(1,color=NAVY,lw=1,ls="--")
    ax.set_ylabel("One-step MSE / random-walk MSE");ax.legend(frameon=False,ncol=3,loc="upper center",fontsize=9)
    ax.set_title("Information-scope diagnostic | same Ridge forecast family",fontsize=14,pad=14)
    ax.grid(axis="y",alpha=.15);ax.set_axisbelow(True)
    fig.supxlabel("Separate 70/10/20 experiment; not the annual multi-horizon results. Full-series features use future data.",fontsize=9)
    save(fig,"03_information_scope")


def official_tianjin():
    df=pd.read_csv(STUDY/"results/official_tianjin_v4/PAPER_STYLE_NATIVE_METRICS.csv")
    families=["allocation_readout","allocation_constant","swewt_gru_full","liquidity_full",
              "shock_size_only","liquidity_readout","liquidity_unconstrained","ridge_price",
              "ridge_full","hgb_full","swewt_gru_price","gru","lstm","drift","rw"]
    labels=["Terminal MLP | primary","Constant allocation","SWEWT-GRU + inputs","Volume allocation | secondary",
            "Shock-size only","Terminal volume gate","Signed coefficients","Price Ridge","Full-input Ridge",
            "Gradient boosting","SWEWT-GRU price","GRU + inputs","LSTM + inputs","Drift","Random walk"]
    values=df.pivot(index="family",columns="horizon",values="mean_seed_mse_over_rw").loc[families,[1,5,20]]
    gains=100*(1-values.mean(axis=1).to_numpy())
    fig,axes=plt.subplots(1,2,figsize=(12,8.2),sharey=True,layout="constrained",gridspec_kw={"width_ratios":[1.5,1]})
    y=np.arange(len(families))
    colors=[NAVY if f=="allocation_readout" else TEAL if f=="liquidity_full" else ORANGE if f=="ridge_price" else GREY for f in families]
    axes[0].barh(y,gains,color=colors,height=.65)
    for i,g in enumerate(gains): axes[0].text(12,i,f"{g:+.3f}",ha="right",va="center",fontsize=9)
    axes[0].axvline(0,color=NAVY,lw=1);axes[0].set_xlim(-1.5,12.3)
    axes[0].set_title("Equal-horizon improvement over RW")
    axes[0].set_xlabel("RW percentage points (positive is better)")
    axes[0].set_yticks(y,labels);axes[0].grid(axis="x",alpha=.15);axes[0].set_axisbelow(True)
    from matplotlib.colors import TwoSlopeNorm
    heat=axes[1].imshow(values.to_numpy(),aspect="auto",cmap="RdBu_r",norm=TwoSlopeNorm(vmin=.88,vcenter=1,vmax=1.12))
    for i in y:
        for j in range(3): axes[1].text(j,i,f"{values.iloc[i,j]:.4f}",ha="center",va="center",fontsize=9,
                                      color="white" if abs(values.iloc[i,j]-1)>.075 else "black")
    axes[1].set_xticks([0,1,2],["H1","H5","H20"])
    axes[1].set_title("Mean-seed MSE / RW by horizon")
    axes[1].set_xlabel("Subsequent observed transaction records")
    for spine in axes[1].spines.values():spine.set_visible(False)
    axes[0].set_ylim(len(y)-.5,-.5)
    fig.suptitle("Official Tianjin later-period evaluation | 196 common forecast origins",fontsize=15)
    fig.supxlabel("Test partition: 2023-04-19 to 2026-09-30. All fixed families are shown.\nPrimary terminal MLP and constant control both select epoch 0: their forecasts are exactly equal.\nPoint estimates shown; five prespecified paired uncertainty comparisons are in the independent audit.",fontsize=9)
    save(fig,"04_official_tianjin_replication")


if __name__=="__main__":
    development();replication();information();official_tianjin()
    manifest=[{"path":str(p.relative_to(STUDY)),"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"bytes":p.stat().st_size}
              for p in sorted(DEST.iterdir()) if p.suffix in [".png",".pdf"]]
    (DEST/"MANIFEST.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))
