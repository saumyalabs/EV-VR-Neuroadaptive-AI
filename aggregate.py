#!/usr/bin/env python3
"""Enrich extracted metrics with station-remap + group/condition labels,
compute participant/group/factor aggregates, trends, initial-vs-subsequent,
and exploratory stats (one-way ANOVA, 2x2 Threat x AI ANOVA, effect sizes,
station trend regression). Dumps CSVs + stats.json for the report."""
import json, os
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm

BASE = Path(__file__).resolve().parent
SCR = Path(os.environ.get("EV_VR_SCRATCH", BASE / "outputs" / "scratch3"))
OUT = Path(os.environ.get("EV_VR_OUT", BASE / "outputs" / "analysis_outputs"))
OUT.mkdir(parents=True, exist_ok=True)

GROUPS={
 "G1":[1,2,3,4,24,25,26,27,28],
 "G2":[5,6,7,8,9,40],
 "G3":[10,11,12,13,14,38,39],
 "G4":[15,16,17,18,19,20,21,22,23,29,30,31,32,33,34,35,36,37],
}
P2G={p:g for g,ps in GROUPS.items() for p in ps}
THREAT={"G1":"Cyberattack","G2":"Cyberattack","G3":"Anomaly","G4":"Anomaly"}
AI={"G1":"AI","G2":"NoAI","G3":"NoAI","G4":"AI"}

DVS=["gaze_dispersion","fixation_count","fixation_duration_ms","saccade_count",
     "gaze_length_deg","dwell_time_ms","pupil_size_mm","blink_rate_pm"]
RATEDV=["fixation_rate_hz","saccade_rate_hz","scan_rate_degps","onscreen_frac"]

def remap_station(tr):
    return tr if tr<=3 else tr-1            # trial>=5 -> -1 ; trial 4 never present

def label(df):
    df=df.copy()
    df["station"]=df["trial"].apply(remap_station)
    df["group"]=df["participant"].map(P2G)
    df["threat"]=df["group"].map(THREAT)
    df["ai"]=df["group"].map(AI)
    return df

# ---------- load ----------
S=label(pd.read_csv(SCR / "station_level.csv"))
A=label(pd.read_csv(SCR / "aoi_level.csv"))
# rate-normalized DVs (disentangle duration confound)
S["fixation_rate_hz"]=S["fixation_count"]/S["window_s"]
S["saccade_rate_hz"]=S["saccade_count"]/S["window_s"]
S["scan_rate_degps"]=S["gaze_length_deg"]/S["window_s"]
S["onscreen_frac"]=(S["dwell_time_ms"]/1000.0)/S["window_s"]
S.to_csv(OUT / "station_level_labeled.csv",index=False)
A.to_csv(OUT / "aoi_level_labeled.csv",index=False)

ALLDV=DVS+RATEDV
# ---------- participant-level means ----------
P=S.groupby(["participant","group","threat","ai"])[ALLDV].mean().reset_index()
P["n_stations"]=S.groupby("participant").size().values
P.to_csv(OUT / "participant_summary.csv",index=False)

def msd(d,c):
    v=d[c].dropna(); return dict(mean=float(v.mean()),sd=float(v.std()),n=int(v.size))

# ---------- group summary ----------
rows=[]
for g in ["G1","G2","G3","G4"]:
    d=P[P.group==g]
    r={"group":g,"threat":THREAT[g],"ai":AI[g],"n_participants":int(d.participant.nunique())}
    for c in ALLDV:
        s=msd(d,c); r[f"{c}_mean"]=s["mean"]; r[f"{c}_sd"]=s["sd"]
    rows.append(r)
GS=pd.DataFrame(rows); GS.to_csv(OUT / "group_summary.csv",index=False)

# ---------- factor summaries ----------
def factor_summary(col):
    out=[]
    for lev in P[col].unique():
        d=P[P[col]==lev]; r={col:lev,"n":int(d.participant.nunique())}
        for c in ALLDV:
            s=msd(d,c); r[f"{c}_mean"]=s["mean"]; r[f"{c}_sd"]=s["sd"]
        out.append(r)
    return pd.DataFrame(out)
factor_summary("threat").to_csv(OUT / "threat_summary.csv",index=False)
factor_summary("ai").to_csv(OUT / "ai_summary.csv",index=False)

# ---------- 2x2 cell means ----------
cell=P.groupby(["threat","ai"])[ALLDV].agg(["mean","std","count"])
cell.to_csv(OUT / "factorial_2x2_cells.csv")

# ---------- per-station trend (overall + by group) ----------
st_overall=S.groupby("station")[ALLDV].mean().reset_index()
st_overall.to_csv(OUT / "station_trend_overall.csv",index=False)
st_group=S.groupby(["group","station"])[ALLDV].mean().reset_index()
st_group.to_csv(OUT / "station_trend_by_group.csv",index=False)

# ---------- initial vs subsequent ----------
S["phase"]=np.where(S["station"]<=5,"initial(1-5)","subsequent(6-15)")
phase=S.groupby(["group","phase"])[ALLDV].mean().reset_index()
phase.to_csv(OUT / "phase_by_group.csv",index=False)
phase_overall=S.groupby("phase")[ALLDV].mean().reset_index()

# ---------- STATS ----------
def eta_sq_oneway(groups):
    allv=np.concatenate(groups); gm=allv.mean()
    ssb=sum(len(g)*(g.mean()-gm)**2 for g in groups)
    sst=((allv-gm)**2).sum()
    return float(ssb/sst) if sst>0 else np.nan

stats_out={"dv":{}}
for c in ALLDV:
    entry={}
    # one-way ANOVA across 4 groups (participant means)
    gv=[P[P.group==g][c].dropna().values for g in ["G1","G2","G3","G4"]]
    gv=[g for g in gv if len(g)>=2]
    if len(gv)>=2:
        F,p=stats.f_oneway(*gv)
        entry["oneway"]={"F":float(F),"p":float(p),"eta2":eta_sq_oneway(gv)}
    # two-way Threat x AI on participant means
    dd=P[[c,"threat","ai"]].dropna().rename(columns={c:"y"})
    try:
        mdl=smf.ols("y ~ C(threat)*C(ai)",data=dd).fit()
        at=sm.stats.anova_lm(mdl,typ=2)
        ss=at["sum_sq"]; sstot=ss.sum()
        entry["twoway"]={
          "threat":{"F":float(at.loc["C(threat)","F"]),"p":float(at.loc["C(threat)","PR(>F)"]),"eta2":float(ss["C(threat)"]/sstot)},
          "ai":{"F":float(at.loc["C(ai)","F"]),"p":float(at.loc["C(ai)","PR(>F)"]),"eta2":float(ss["C(ai)"]/sstot)},
          "interaction":{"F":float(at.loc["C(threat):C(ai)","F"]),"p":float(at.loc["C(threat):C(ai)","PR(>F)"]),"eta2":float(ss["C(threat):C(ai)"]/sstot)},
        }
    except Exception as e:
        entry["twoway"]={"error":str(e)}
    # station trend (mixed-ish: regress station-level DV on station)
    td=S[[c,"station"]].dropna()
    if len(td)>10:
        sl,ic,r,pp,se=stats.linregress(td["station"],td[c])
        entry["trend"]={"slope":float(sl),"r":float(r),"p":float(pp),
                        "pct_change_1_to_15":float(sl*14/ic*100) if ic else np.nan}
    # initial vs subsequent (independent samples on station rows)
    a1=S[S.phase=="initial(1-5)"][c].dropna(); a2=S[S.phase=="subsequent(6-15)"][c].dropna()
    if len(a1)>2 and len(a2)>2:
        t,p2=stats.ttest_ind(a1,a2,equal_var=False)
        pooled=np.sqrt((a1.var()+a2.var())/2)
        entry["phase"]={"initial_mean":float(a1.mean()),"subsequent_mean":float(a2.mean()),
                        "t":float(t),"p":float(p2),
                        "cohens_d":float((a2.mean()-a1.mean())/pooled) if pooled>0 else np.nan}
    stats_out["dv"][c]=entry

# group sizes w/ data
stats_out["group_sizes"]={g:int(P[P.group==g].participant.nunique()) for g in ["G1","G2","G3","G4"]}
stats_out["n_station_records"]=int(len(S))
stats_out["phase_overall"]=phase_overall.to_dict("records")

# ---------- AOI attention profiles ----------
# dwell share per AOI overall and per group
A["aoi_base"]=A["aoi"].str.replace(r"_(header|coupon|charge_preview|set_amount)$","",regex=True)
def aoi_share(df,by=None):
    if by:
        tot=df.groupby(by)["dwell_time_ms"].transform("sum")
        d=df.assign(share=df["dwell_time_ms"]/tot)
        return d.groupby(by+["aoi"])["share"].sum().reset_index()
    tot=df["dwell_time_ms"].sum()
    return (df.groupby("aoi")["dwell_time_ms"].sum()/tot).reset_index(name="share")
aoi_overall=aoi_share(A).sort_values("share",ascending=False)
aoi_overall.to_csv(OUT / "aoi_share_overall.csv",index=False)
aoi_by_group=aoi_share(A,["group"])
aoi_by_group.to_csv(OUT / "aoi_share_by_group.csv",index=False)

# AI-panel attention by group (key hypothesis: AI groups attend ai_* more)
A["is_ai"]=A["aoi"].str.startswith("ai_")
ai_attn=A.groupby(["group"]).apply(
    lambda d:(d.loc[d.is_ai,"dwell_time_ms"].sum()/d["dwell_time_ms"].sum()*100),
    include_groups=False).reset_index(name="ai_dwell_pct")
# per participant ai attention for stats
A["is_ai"]=A["aoi"].str.startswith("ai_")
ai_pp=A.groupby(["participant","group"]).apply(
    lambda d:(d.loc[d.is_ai,"dwell_time_ms"].sum()/d["dwell_time_ms"].sum()*100),
    include_groups=False).reset_index(name="ai_dwell_pct")
ai_pp.to_csv(OUT / "ai_attention_per_participant.csv",index=False)
ai_attn.to_csv(OUT / "ai_attention_by_group.csv",index=False)
# AI attention: AI vs NoAI groups
ai_pp["ai_cond"]=ai_pp["group"].map(AI)
aiY=ai_pp[ai_pp.ai_cond=="AI"]["ai_dwell_pct"]; aiN=ai_pp[ai_pp.ai_cond=="NoAI"]["ai_dwell_pct"]
t,p=stats.ttest_ind(aiY,aiN,equal_var=False)
stats_out["ai_panel_attention"]={"AI_groups_mean_pct":float(aiY.mean()),"NoAI_groups_mean_pct":float(aiN.mean()),
                                  "t":float(t),"p":float(p)}

# coupon attention overall
A["is_coupon"]=A["aoi"].str.contains("coupon")
coupon_pct=A.loc[A.is_coupon,"dwell_time_ms"].sum()/A["dwell_time_ms"].sum()*100
stats_out["coupon_dwell_pct_overall"]=float(coupon_pct)

json.dump(stats_out,open(OUT / "stats.json","w"),indent=1)
print("Aggregation complete. Files in analysis_outputs/")
print("Groups w/ data:",stats_out["group_sizes"])
print("\nGroup means (selected DVs):")
print(GS[["group","threat","ai","fixation_duration_ms_mean","pupil_size_mm_mean","blink_rate_pm_mean","gaze_dispersion_mean","fixation_rate_hz_mean"]].round(3).to_string(index=False))
print("\nAI-panel attention by group (%):")
print(ai_attn.round(2).to_string(index=False))
print("\n2x2 interaction p-values (exploratory):")
for c in DVS:
    tw=stats_out["dv"][c].get("twoway",{})
    if "threat" in tw:
        print(f"  {c:22s} threat p={tw['threat']['p']:.3f}  ai p={tw['ai']['p']:.3f}  interax p={tw['interaction']['p']:.3f}")
