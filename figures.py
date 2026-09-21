#!/usr/bin/env python3
"""Generate all analysis figures (PNG) into analysis_outputs/figures."""
import json
import os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

BASE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("EV_VR_OUT", BASE / "outputs" / "analysis_outputs"))
FIG = OUT / "figures"; FIG.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({"font.size":10,"axes.grid":True,"grid.alpha":0.25,
                     "axes.spines.top":False,"axes.spines.right":False,"figure.dpi":110})
GC={"G1":"#e4572e","G2":"#f3a712","G3":"#3aa6b9","G4":"#5b5f97"}

S=pd.read_csv(OUT / "station_level_labeled.csv")
P=pd.read_csv(OUT / "participant_summary.csv")
ST=json.load(open(OUT / "stats.json"))

DV=[("dwell_time_ms","Dwell time (ms)"),("fixation_count","Fixation count"),
    ("fixation_duration_ms","Fixation duration (ms)"),("saccade_count","Saccade count"),
    ("gaze_length_deg","Gaze length / scanpath (deg)"),("gaze_dispersion","Gaze dispersion (norm SD)"),
    ("pupil_size_mm","Pupil size (mm)"),("blink_rate_pm","Blink rate (/min)")]

def save(fig,name):
    fig.tight_layout(); fig.savefig(FIG / f"{name}.png",bbox_inches="tight"); plt.close(fig)

# 1. station trend overall (8 panels) w/ regression
fig,axs=plt.subplots(4,2,figsize=(12,14))
for ax,(c,lab) in zip(axs.ravel(),DV):
    m=S.groupby("station")[c].mean(); se=S.groupby("station")[c].sem()
    ax.errorbar(m.index,m.values,yerr=se.values,marker="o",color="#333",capsize=3,lw=1.5)
    td=S[[c,"station"]].dropna()
    sl,ic,r,p,_=stats.linregress(td["station"],td[c])
    xs=np.array([td.station.min(),td.station.max()])
    ax.plot(xs,ic+sl*xs,"--",color="#e4572e",lw=2)
    sig="***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "ns"
    ax.set_title(f"{lab}\nslope={sl:+.3g}/station, r={r:+.2f} {sig}",fontsize=9.5)
    ax.set_xlabel("Station (1–15, corrected)")
fig.suptitle("Dependent variables across stations (cohort mean ± SE) — habituation trend",
             fontsize=13,y=1.005)
save(fig,"01_station_trend_overall")

# 2. trend by group (key DVs)
key=[("dwell_time_ms","Dwell time (ms)"),("fixation_count","Fixation count"),
     ("pupil_size_mm","Pupil size (mm)"),("fixation_duration_ms","Fixation duration (ms)")]
fig,axs=plt.subplots(2,2,figsize=(12,9))
for ax,(c,lab) in zip(axs.ravel(),key):
    for g in ["G1","G2","G3","G4"]:
        d=S[S.group==g].groupby("station")[c].mean()
        ax.plot(d.index,d.values,marker="o",ms=4,color=GC[g],label=g,lw=1.6)
    ax.set_title(lab); ax.set_xlabel("Station (1–15)")
axs[0,0].legend(title="Group",ncol=2,fontsize=8)
fig.suptitle("Per-station trends by experimental group",fontsize=13,y=1.005)
save(fig,"02_station_trend_by_group")

# 3. group bars (8 DVs, mean±SE over participants)
fig,axs=plt.subplots(4,2,figsize=(12,14))
gs=["G1","G2","G3","G4"]
for ax,(c,lab) in zip(axs.ravel(),DV):
    means=[P[P.group==g][c].mean() for g in gs]
    ses=[P[P.group==g][c].sem() for g in gs]
    ax.bar(gs,means,yerr=ses,capsize=4,color=[GC[g] for g in gs])
    ow=ST["dv"][c]["oneway"]
    ax.set_title(f"{lab}\n1-way ANOVA F={ow['F']:.2f}, p={ow['p']:.2f}, η²={ow['eta2']:.2f}",fontsize=9.5)
fig.suptitle("Group means ± SE (participant-level)  [G1 Cyber+AI · G2 Cyber · G3 Anomaly · G4 Anomaly+AI]",
             fontsize=12,y=1.005)
save(fig,"03_group_means")

# 4. 2x2 factorial interaction plots
f2=[("pupil_size_mm","Pupil size (mm)"),("blink_rate_pm","Blink rate (/min)"),
    ("fixation_duration_ms","Fixation duration (ms)"),("dwell_time_ms","Dwell time (ms)")]
fig,axs=plt.subplots(2,2,figsize=(11,9))
for ax,(c,lab) in zip(axs.ravel(),f2):
    for ai,mk,col in [("AI","-o","#5b5f97"),("NoAI","--s","#f3a712")]:
        ys=[P[(P.threat==th)&(P.ai==ai)][c].mean() for th in ["Cyberattack","Anomaly"]]
        ax.plot(["Cyberattack","Anomaly"],ys,mk,color=col,label=ai,lw=2,ms=8)
    tw=ST["dv"][c]["twoway"]
    ax.set_title(f"{lab}\nThreat p={tw['threat']['p']:.2f} · AI p={tw['ai']['p']:.2f} · int p={tw['interaction']['p']:.2f}",fontsize=9)
    ax.set_xlabel("Threat framing")
axs[0,0].legend(title="AI condition")
fig.suptitle("2×2 factorial: Threat framing × AI assistance (participant means)",fontsize=13,y=1.005)
save(fig,"04_factorial_2x2")

# 5. initial vs subsequent (8 DVs)
fig,axs=plt.subplots(4,2,figsize=(12,14))
S["phase"]=np.where(S.station<=5,"Initial\n(1–5)","Subsequent\n(6–15)")
for ax,(c,lab) in zip(axs.ravel(),DV):
    ph=ST["dv"][c].get("phase",{})
    vals=S.groupby("phase")[c].mean(); ses=S.groupby("phase")[c].sem()
    order=["Initial\n(1–5)","Subsequent\n(6–15)"]
    ax.bar(order,[vals[o] for o in order],yerr=[ses[o] for o in order],
           capsize=4,color=["#3aa6b9","#e4572e"])
    d=ph.get("cohens_d",float('nan')); p=ph.get("p",float('nan'))
    sig="***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "ns"
    ax.set_title(f"{lab}\nd={d:+.2f}, p={p:.3f} {sig}",fontsize=9.5)
fig.suptitle("Initial (stations 1–5) vs Subsequent (6–15) — learning/habituation",fontsize=13,y=1.005)
save(fig,"05_initial_vs_subsequent")

# 6. AI-panel attention
aig=pd.read_csv(f"{OUT}/ai_attention_by_group.csv")
aipp=pd.read_csv(f"{OUT}/ai_attention_per_participant.csv")
aipp["ai_cond"]=aipp["group"].map({"G1":"AI","G2":"NoAI","G3":"NoAI","G4":"AI"})
fig,axs=plt.subplots(1,2,figsize=(12,4.8))
axs[0].bar(aig.group,aig.ai_dwell_pct,color=[GC[g] for g in aig.group])
axs[0].set_title("AI-assistant AOI dwell share by group (%)"); axs[0].set_ylabel("% dwell on ai_* AOIs")
ap=ST["ai_panel_attention"]
data=[aipp[aipp.ai_cond=="AI"].ai_dwell_pct,aipp[aipp.ai_cond=="NoAI"].ai_dwell_pct]
bp=axs[1].boxplot(data,labels=[f"AI groups\n(G1,G4)",f"NoAI groups\n(G2,G3)"],patch_artist=True)
for patch,col in zip(bp["boxes"],["#5b5f97","#f3a712"]): patch.set_facecolor(col); patch.set_alpha(.6)
axs[1].set_title(f"AI present vs absent\nt={ap['t']:.2f}, p={ap['p']:.3f} (significant)")
axs[1].set_ylabel("% dwell on ai_* AOIs")
fig.suptitle("Do participants look at the AI assistant more when it is present?  Yes.",fontsize=13,y=1.02)
save(fig,"06_ai_panel_attention")

# 7. AOI dwell-share profile overall + by group heatmap
ov=pd.read_csv(f"{OUT}/aoi_share_overall.csv").head(14)
bg=pd.read_csv(f"{OUT}/aoi_share_by_group.csv")
top=ov.aoi.tolist()
piv=bg[bg.aoi.isin(top)].pivot_table(index="aoi",columns="group",values="share").reindex(top)
fig,axs=plt.subplots(1,2,figsize=(14,7),gridspec_kw={"width_ratios":[1,1.1]})
axs[0].barh(ov.aoi[::-1],ov.share[::-1]*100,color="#3aa6b9")
axs[0].set_title("Top AOIs by dwell share (all participants)"); axs[0].set_xlabel("% of station dwell")
im=axs[1].imshow(piv.values*100,aspect="auto",cmap="inferno")
axs[1].set_xticks(range(4)); axs[1].set_xticklabels(piv.columns)
axs[1].set_yticks(range(len(piv))); axs[1].set_yticklabels(piv.index,fontsize=8)
axs[1].set_title("AOI dwell share (%) by group")
for i in range(len(piv)):
    for j in range(4):
        v=piv.values[i,j]*100
        axs[1].text(j,i,f"{v:.0f}",ha="center",va="center",
                    color="white" if v<piv.values.max()*100*0.6 else "black",fontsize=7)
fig.colorbar(im,ax=axs[1],label="% dwell")
fig.suptitle("Where attention goes: AOI dwell-share profile",fontsize=13,y=1.0)
save(fig,"07_aoi_profile")

# 8. correlation matrix of DVs (station level)
cols=[c for c,_ in DV]
corr=S[cols].corr()
fig,ax=plt.subplots(figsize=(8.5,7))
im=ax.imshow(corr,cmap="RdBu_r",vmin=-1,vmax=1)
ax.set_xticks(range(len(cols))); ax.set_xticklabels([l for _,l in DV],rotation=45,ha="right",fontsize=8)
ax.set_yticks(range(len(cols))); ax.set_yticklabels([l for _,l in DV],fontsize=8)
for i in range(len(cols)):
    for j in range(len(cols)):
        ax.text(j,i,f"{corr.values[i,j]:.2f}",ha="center",va="center",fontsize=7,
                color="white" if abs(corr.values[i,j])>0.6 else "black")
fig.colorbar(im,label="Pearson r")
ax.set_title("Correlation among dependent variables (station level)")
save(fig,"08_dv_correlation")

# 9. per-participant boxplots by group for 4 DVs
fig,axs=plt.subplots(2,2,figsize=(12,9))
for ax,(c,lab) in zip(axs.ravel(),key):
    data=[P[P.group==g][c].dropna() for g in gs]
    bp=ax.boxplot(data,labels=gs,patch_artist=True)
    for patch,g in zip(bp["boxes"],gs): patch.set_facecolor(GC[g]); patch.set_alpha(.6)
    ax.set_title(lab)
fig.suptitle("Participant-level distribution by group",fontsize=13,y=1.005)
save(fig,"09_group_boxplots")

print("figures written:",len(os.listdir(FIG)))
EOF_GUARD = None
