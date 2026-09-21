#!/usr/bin/env python3
"""
Per-participant extraction of 8 eye-tracking dependent variables, AOI-linked,
per station, by joining the AOI mapper outputs (NNN_gaze_aoi.csv) with the
raw Varjo gaze exports in the Participants folders.

DVs:
  gaze_dispersion, fixation_count, fixation_duration, saccade_count,
  gaze_length(scanpath), dwell_time, pupil_size, blink_rate

Run per batch:  python extract.py <batch_key>
Writes (append) to SCRATCH/aoi_level.csv and SCRATCH/station_level.csv
"""
import sys, os, glob, json, csv, re
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(os.environ.get("EV_VR_BASE", Path(__file__).resolve().parent))
SCRATCH = Path(os.environ.get("EV_VR_SCRATCH", BASE / "outputs" / "scratch3"))
SCRATCH.mkdir(parents=True, exist_ok=True)

OUT_FOLDERS = {
 "1_10":  ("aoi_output_1_to_10",  "dataset (1 - 20)/Participants 1 -10"),
 "11_20": ("aoi_output_11_to_20", "dataset (1 - 20)/Participants 11 - 20"),
 "21_30": ("aoi_output_21_to_30", "dataset (1 - 20)/Participants 21 - 30"),
 "31_40": ("aoi_output_31_to_40", "dataset (1 - 20)/Participants 31 - 40"),
}

VARJO_NA = ["-nan(ind)","nan(ind)","-nan","-1.#IND","1.#IND","-1.#INF","1.#INF"]
GAZE_TS = "relative_to_unix_epoch_timestamp"

# --- method parameters (documented in report) ---
IVT_VEL_THRESH = 30.0     # deg/s  (Salvucci & Goldberg I-VT standard)
MIN_FIX_MS     = 60.0     # min fixation duration
MAX_STEP_MS    = 25.0     # break continuity if gap larger (5 samples @200Hz)
MIN_SACC_MS    = 8.0      # min saccade run
MAX_SACC_MS    = 120.0    # max plausible saccade run
DWELL_CAP_MS   = 50.0     # cap inter-sample dt for dwell (gap guard, matches mapper)
BLINK_OPEN_THR = 0.30     # mean eye-openness below = eye closed
BLINK_MIN_MS   = 40.0     # min closed run to count as a blink
BLINK_MAX_MS   = 600.0    # longer closed run = not a normal blink
PUPIL_MIN, PUPIL_MAX = 1.5, 8.5   # plausible pupil diameter mm

RAW_COLS = [GAZE_TS,"status","gaze_forward_x","gaze_forward_y","gaze_forward_z",
            "left_pupil_diameter_in_mm","right_pupil_diameter_in_mm",
            "left_eye_openness","right_eye_openness"]


def parse_intervals(inter_path):
    rows=[]
    with open(inter_path, newline="", encoding="utf-8", errors="replace") as f:
        for rec in csv.DictReader(f):
            if rec.get("channel")!="parsed_json" or rec.get("route")!="UI": continue
            page=(rec.get("page") or "").strip()
            if not page: continue
            try: t=int(rec["timestamp_unix_ms"])
            except (KeyError,TypeError,ValueError): continue
            tr=(rec.get("trial") or "").strip()
            trial=int(float(tr)) if tr else -1
            rows.append((t,page,trial))
    df=pd.DataFrame(rows,columns=["t_ms","page","trial"]).sort_values("t_ms").reset_index(drop=True)
    if df.empty: return df
    keep=(df["page"].ne(df["page"].shift()))|(df["trial"].ne(df["trial"].shift()))
    df=df[keep].reset_index(drop=True)
    df["t0"]=df["t_ms"].astype(float)
    df["t1"]=df["t0"].shift(-1)
    df.loc[df.index[-1],"t1"]=df["t0"].iloc[-1]+30000
    return df[df["page"]=="station"][["trial","t0","t1"]].reset_index(drop=True)


def load_raw(raw_path):
    hdr=pd.read_csv(raw_path,nrows=0).columns
    keep=[c for c in RAW_COLS if c in hdr]
    df=pd.read_csv(raw_path, usecols=keep, na_values=VARJO_NA,
                   engine="c", low_memory=False)
    for c in keep:
        df[c]=pd.to_numeric(df[c],errors="coerce")
    df["t_ms"]=df[GAZE_TS]/1e6
    df["openness"]=df[["left_eye_openness","right_eye_openness"]].mean(axis=1)
    pl=df["left_pupil_diameter_in_mm"]; pr=df["right_pupil_diameter_in_mm"]
    pl=pl.where((pl>=PUPIL_MIN)&(pl<=PUPIL_MAX)); pr=pr.where((pr>=PUPIL_MIN)&(pr<=PUPIL_MAX))
    df["pupil"]=pd.concat([pl,pr],axis=1).mean(axis=1)
    df=df[df["t_ms"].notna()].sort_values("t_ms").reset_index(drop=True)
    return df


def ivt_fixations(sub):
    """sub: per-station merged frame sorted by t_ms with cols t_ms,u,v,aoi,fx,fy,fz,valid.
       Returns (fixations list of dict, n_saccades, scan_deg, scan_norm, disp)."""
    t=sub["t_ms"].to_numpy(float)
    fx=sub["fx"].to_numpy(float); fy=sub["fy"].to_numpy(float); fz=sub["fz"].to_numpy(float)
    u=sub["u"].to_numpy(float); v=sub["v"].to_numpy(float)
    aoi=sub["aoi"].to_numpy(object)
    n=len(t)
    if n<3:
        return [],0,0.0,0.0,np.nan
    dt=np.diff(t)
    dot=np.clip(fx[1:]*fx[:-1]+fy[1:]*fy[:-1]+fz[1:]*fz[:-1],-1,1)
    ang=np.degrees(np.arccos(dot))                # deg between consecutive gaze vectors
    with np.errstate(divide="ignore",invalid="ignore"):
        vel=np.where(dt>0, ang/(dt/1000.0), np.inf)
    gap=(dt>MAX_STEP_MS)|(dt<=0)
    # per-sample classification: sample i fixation if step i-1->i slow (and no gap)
    is_fix_step = (vel<IVT_VEL_THRESH)&(~gap)
    # scanpath over continuous steps
    cont=~gap
    scan_deg=float(np.nansum(ang[cont]))
    du=np.diff(u); dv=np.diff(v)
    scan_norm=float(np.nansum(np.hypot(du,dv)[cont]))
    disp=float(np.sqrt(np.nanvar(u)+np.nanvar(v)))
    # build sample-level state then RLE with gap breaks
    state=np.empty(n,dtype=int); state[0]=0
    state[1:]=np.where(is_fix_step,1,2)        # 1=fix,2=sacc
    brk=np.zeros(n,dtype=bool); brk[1:]=gap
    fixations=[]
    i=1
    while i<n:
        j=i
        while j+1<n and state[j+1]==state[i] and not brk[j+1]:
            j+=1
        run_t0=t[i-1] if state[i]==1 else t[i]
        run_t1=t[j]
        dur=run_t1-run_t0
        if state[i]==1 and dur>=MIN_FIX_MS:
            idx=slice(i-1,j+1)
            labs=aoi[idx]
            lab=pd.Series(labs).mode()
            lab=lab.iloc[0] if len(lab) else "none"
            fixations.append(dict(t0=run_t0,t1=run_t1,dur=dur,
                                  u=float(np.nanmean(u[idx])),v=float(np.nanmean(v[idx])),
                                  aoi=lab,nsamp=j-i+2))
        i=j+1
    n_sacc=sum(1 for k in range(1,len(fixations))
               if fixations[k]["t0"]-fixations[k-1]["t1"] < 400.0)
    return fixations,n_sacc,scan_deg,scan_norm,disp


def blink_pupil_window(raw, t0, t1):
    w=raw[(raw["t_ms"]>=t0)&(raw["t_ms"]<t1)]
    dur_ms=t1-t0
    if len(w)<5 or dur_ms<=0:
        return np.nan, np.nan, dur_ms
    op=w["openness"].to_numpy(float); tt=w["t_ms"].to_numpy(float)
    closed=op<BLINK_OPEN_THR
    nb=0; i=0; m=len(closed)
    while i<m:
        if closed[i]:
            j=i
            while j+1<m and closed[j+1]: j+=1
            run=tt[j]-tt[i]
            if BLINK_MIN_MS<=run<=BLINK_MAX_MS: nb+=1
            i=j+1
        else: i+=1
    blink_rate=nb/(dur_ms/60000.0)            # blinks per minute
    pupil=float(np.nanmean(w["pupil"]))
    return blink_rate, pupil, dur_ms


def process(pid, out_csv, raw_path, inter_path):
    g=pd.read_csv(out_csv)                      # participant, t_ms, trial, page, u, v, aoi
    g=g[g["page"]=="station"].copy()
    if g.empty: return [],[]
    raw=load_raw(raw_path)
    iv=parse_intervals(inter_path)
    has_fwd=all(c in raw.columns for c in ("gaze_forward_x","gaze_forward_y","gaze_forward_z"))
    cols=["t_ms","pupil"]+(["gaze_forward_x","gaze_forward_y","gaze_forward_z"] if has_fwd else [])
    rsel=raw[cols].dropna(subset=["t_ms"]).sort_values("t_ms")
    g=g.sort_values("t_ms")
    m=pd.merge_asof(g, rsel, on="t_ms", direction="nearest", tolerance=6.0)
    if has_fwd:
        m=m.rename(columns={"gaze_forward_x":"fx","gaze_forward_y":"fy","gaze_forward_z":"fz"})
        m=m.dropna(subset=["fx","fy","fz"])
    else:
        m["fx"]=np.nan; m["fy"]=np.nan; m["fz"]=np.nan
    aoi_rows=[]; st_rows=[]
    # blink/pupil per station from full raw window
    bp={int(r.trial):blink_pupil_window(raw,r.t0,r.t1) for r in iv.itertuples()}
    for trial,sub in m.groupby("trial"):
        sub=sub.sort_values("t_ms")
        fixs,n_sacc,scan_deg,scan_norm,disp=ivt_fixations(sub)
        if not has_fwd:
            n_sacc=np.nan; scan_deg=np.nan
        # dwell per sample (capped dt)
        dts=sub["t_ms"].diff().clip(lower=0,upper=DWELL_CAP_MS).fillna(0)
        sub=sub.assign(dt=dts.to_numpy())
        fixdf=pd.DataFrame(fixs)
        br,pu,wdur=bp.get(int(trial),(np.nan,np.nan,np.nan))
        # ---- station level ----
        n_fix=(len(fixdf) if has_fwd else np.nan)
        mean_fix=float(fixdf["dur"].mean()) if len(fixdf) else np.nan
        total_dwell=float(sub["dt"].sum())
        pupil_samp=float(np.nanmean(sub["pupil"])) if sub["pupil"].notna().any() else np.nan
        st_rows.append(dict(participant=pid,trial=int(trial),
            gaze_dispersion=disp, fixation_count=n_fix, fixation_duration_ms=mean_fix,
            saccade_count=n_sacc, gaze_length_deg=scan_deg, gaze_length_norm=scan_norm,
            dwell_time_ms=total_dwell, pupil_size_mm=(pu if not np.isnan(pu) else pupil_samp),
            blink_rate_pm=br, window_s=(wdur/1000.0 if not np.isnan(wdur) else np.nan),
            n_samples=len(sub), n_fix_samples=int(fixdf["nsamp"].sum()) if len(fixdf) else 0))
        # ---- AOI level ----
        fix_by_aoi=(fixdf.groupby("aoi")["dur"].agg(["count","mean"]) if len(fixdf) else
                    pd.DataFrame(columns=["count","mean"]))
        for aoi,s2 in sub.groupby("aoi"):
            disp_a=float(np.sqrt(np.nanvar(s2["u"])+np.nanvar(s2["v"])))
            fc=(int(fix_by_aoi["count"].get(aoi,0)) if len(fix_by_aoi) else 0) if has_fwd else np.nan
            fd=float(fix_by_aoi["mean"].get(aoi,np.nan)) if len(fix_by_aoi) else np.nan
            aoi_rows.append(dict(participant=pid,trial=int(trial),aoi=aoi,
                n_samples=len(s2), dwell_time_ms=float(s2["dt"].sum()),
                fixation_count=fc, fixation_duration_ms=fd,
                gaze_dispersion=disp_a, pupil_size_mm=float(np.nanmean(s2["pupil"]))))
    return aoi_rows, st_rows


def find_raw_inter(pdir, pid):
    sess=sorted(glob.glob(os.path.join(pdir,f"session_*_{pid}")))
    if not sess: return None,None
    d=sess[-1]
    gz=sorted(glob.glob(os.path.join(d,"varjo_gaze_output_*.csv"))) or sorted(glob.glob(os.path.join(d,f"{pid}.csv")))
    it=sorted(glob.glob(os.path.join(d,"interactions_*.csv")))
    return (gz[-1] if gz else None),(it[-1] if it else None)


def main():
    batch=sys.argv[1]
    out_folder,part_folder=OUT_FOLDERS[batch]
    outdir=os.path.join(BASE,out_folder); pdir=os.path.join(BASE,part_folder)
    pids=sorted({re.match(r"(\d{3})_gaze_aoi\.csv",os.path.basename(p)).group(1)
                 for p in glob.glob(os.path.join(outdir,"*_gaze_aoi.csv"))})
    if len(sys.argv)>2:                     # optional explicit pid subset
        want=set(sys.argv[2].split(","))
        pids=[p for p in pids if p in want]
    A=[]; S=[]
    for pid in pids:
        out_csv=os.path.join(outdir,f"{pid}_gaze_aoi.csv")
        raw_path,inter_path=find_raw_inter(pdir,pid)
        if not raw_path or not inter_path:
            print(f"  P{pid}: missing raw/inter -> skip"); continue
        try:
            a,s=process(pid,out_csv,raw_path,inter_path)
            A+=a; S+=s
            print(f"  P{pid}: {len(s)} stations, {len(a)} aoi-rows")
        except Exception as e:
            print(f"  P{pid}: ERROR {e}")
    mode="a" if os.path.exists(os.path.join(SCRATCH,"aoi_level.csv")) else "w"
    hdr = mode=="w"
    pd.DataFrame(A).to_csv(os.path.join(SCRATCH,"aoi_level.csv"),mode=mode,header=hdr,index=False)
    pd.DataFrame(S).to_csv(os.path.join(SCRATCH,"station_level.csv"),mode=mode,header=hdr,index=False)
    print(f"batch {batch} done: +{len(S)} station rows +{len(A)} aoi rows")

if __name__=="__main__":
    main()
