#!/usr/bin/env python3
"""Assemble a self-contained HTML analysis report with embedded figures."""
import base64
import json
import os
from pathlib import Path
import pandas as pd, numpy as np

BASE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("EV_VR_OUT", BASE / "outputs" / "analysis_outputs"))
FIG = OUT / "figures"
ST=json.load(open(OUT / "stats.json"))

def img(name):
    b=base64.b64encode(open(FIG / f"{name}.png","rb").read()).decode()
    return f'<img src="data:image/png;base64,{b}" style="width:100%;max-width:1000px;border:1px solid #e2e2e2;border-radius:8px;margin:8px 0"/>'

def table(df,fmt=None,maxrows=None):
    d=df.copy()
    if maxrows: d=d.head(maxrows)
    if fmt:
        for c in d.columns:
            if d[c].dtype.kind in "fc":
                d[c]=d[c].map(lambda x:fmt.format(x) if pd.notna(x) else "—")
    return d.to_html(index=False,border=0,classes="tbl",escape=False)

gs=ST["group_sizes"]
ap=ST["ai_panel_attention"]
coup=ST["coupon_dwell_pct_overall"]

def trend_row(c):
    e=ST["dv"][c]; tr=e.get("trend",{}); ph=e.get("phase",{})
    sig=lambda p:"***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "ns"
    return (tr.get("slope",float('nan')),tr.get("r",float('nan')),tr.get("p",float('nan')),
            ph.get("initial_mean",float('nan')),ph.get("subsequent_mean",float('nan')),
            ph.get("cohens_d",float('nan')),ph.get("p",float('nan')),sig(ph.get("p",1)))

DVMETA=[("dwell_time_ms","Dwell time","ms","Total time gaze was on the station screen during a visit (Σ capped inter-sample Δt)."),
 ("fixation_count","Fixation count","n","Number of fixations (I-VT, &lt;30°/s, ≥60 ms) in the visit."),
 ("fixation_duration_ms","Fixation duration","ms","Mean duration of those fixations."),
 ("saccade_count","Saccade count","n","Transitions between consecutive fixations (&lt;400 ms gap)."),
 ("gaze_length_deg","Gaze length / scanpath","deg","Σ angular displacement between consecutive valid gaze vectors."),
 ("gaze_dispersion","Gaze dispersion","norm SD","√(var(u)+var(v)) of mapped gaze — spatial spread of attention."),
 ("pupil_size_mm","Pupil size","mm","Mean of left/right pupil diameter over valid samples."),
 ("blink_rate_pm","Blink rate","/min","Blinks (mean eye-openness &lt;0.30 for 40–600 ms) per minute over the visit window.")]

# build DV definition table rows
dv_defs="".join(
 f"<tr><td><b>{n}</b></td><td>{u}</td><td>{d}</td></tr>" for _,n,u,d in DVMETA)

# trend summary table
tr_rows=""
for c,n,u,_ in DVMETA:
    sl,r,p,im,sm,d,pp,sg=trend_row(c)
    arrow="▼" if sl<0 else "▲"
    tr_rows+=(f"<tr><td><b>{n}</b> <span class='muted'>({u})</span></td>"
              f"<td>{sl:+.3g}</td><td>{r:+.2f}</td><td>{p:.3f}</td>"
              f"<td>{im:.1f}</td><td>{sm:.1f}</td><td>{d:+.2f}</td>"
              f"<td class='{ 'sig' if pp<.05 else '' }'>{pp:.3f} {sg}</td></tr>")

# group summary table (selected)
GSdf=pd.read_csv(f"{OUT}/group_summary.csv")
sel=["group","threat","ai","n_participants",
     "dwell_time_ms_mean","fixation_count_mean","fixation_duration_ms_mean",
     "saccade_count_mean","gaze_length_deg_mean","gaze_dispersion_mean",
     "pupil_size_mm_mean","blink_rate_pm_mean"]
GSshow=GSdf[sel].rename(columns=lambda c:c.replace("_mean","").replace("_"," "))
Pdf=pd.read_csv(f"{OUT}/participant_summary.csv")
aoi_ov=pd.read_csv(f"{OUT}/aoi_share_overall.csv").head(12)
aoi_ov["share"]=(aoi_ov["share"]*100).round(1)

twoway_rows=""
for c,n,u,_ in DVMETA:
    tw=ST["dv"][c].get("twoway",{})
    if "threat" in tw:
        f=lambda k:f"{tw[k]['p']:.3f}"
        cls=lambda k:"sig" if tw[k]['p']<.05 else ""
        twoway_rows+=(f"<tr><td><b>{n}</b></td>"
            f"<td class='{cls('threat')}'>{f('threat')}</td>"
            f"<td class='{cls('ai')}'>{f('ai')}</td>"
            f"<td class='{cls('interaction')}'>{f('interaction')}</td></tr>")

ow_rows=""
for c,n,u,_ in DVMETA:
    ow=ST["dv"][c]["oneway"]
    ow_rows+=f"<tr><td><b>{n}</b></td><td>{ow['F']:.2f}</td><td>{ow['p']:.3f}</td><td>{ow['eta2']:.3f}</td></tr>"

HTML=f"""<!doctype html><html><head><meta charset="utf-8">
<title>EV Charging VR — Eye-Tracking AOI Analysis</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:1060px;
margin:0 auto;padding:32px 26px;color:#1c1c20;line-height:1.6;background:#fff}}
h1{{font-size:30px;margin:0 0 4px;letter-spacing:-.3px}}
h2{{font-size:23px;margin:42px 0 8px;padding-top:14px;border-top:2px solid #ececf0}}
h3{{font-size:18px;margin:26px 0 6px;color:#33333a}}
.sub{{color:#6a6a73;font-size:15px;margin-bottom:6px}}
.muted{{color:#8a8a93;font-weight:400}}
.tbl{{border-collapse:collapse;width:100%;font-size:13.5px;margin:10px 0}}
.tbl th{{background:#f4f4f7;text-align:left;padding:7px 9px;border-bottom:2px solid #e0e0e6}}
.tbl td{{padding:6px 9px;border-bottom:1px solid #eee}}
.tbl tr:hover td{{background:#fafafc}}
.sig{{color:#0a7d32;font-weight:600}}
.card{{background:#f7f9fb;border:1px solid #e6ebef;border-radius:10px;padding:16px 20px;margin:16px 0}}
.key{{background:#eef7f0;border-left:4px solid #1c8a43}}
.warn{{background:#fdf6ec;border-left:4px solid #d9920a}}
.flag{{background:#fcefef;border-left:4px solid #c0392b}}
code{{background:#f0f0f4;padding:1px 5px;border-radius:4px;font-size:13px}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:8px 22px}}
figcaption{{font-size:13px;color:#6a6a73;margin:2px 0 18px}}
ul{{margin:8px 0}} li{{margin:4px 0}}
</style></head><body>

<h1>Eye-Tracking AOI Analysis — EV Charging Station VR Study</h1>
<div class="sub">39 participants · 302 station visits · 8 dependent variables × 4 independent variables ·
AOI-linked oculometrics from Varjo gaze joined to the AOI-mapper outputs</div>

<div class="card key">
<b>Executive summary.</b> Two effects stand out from the data.
<b>(1) The AI assistant works as a manipulation:</b> participants in the AI-present groups (G1, G4)
spent <b>7.5%</b> of their station dwell on the AI-assistant AOIs vs <b>4.8%</b> in the AI-absent groups
(G2, G3); t={ap['t']:.2f}, <b>p={ap['p']:.3f}</b>. The AOIs confirm people actually look at the feature when it exists.
<b>(2) Strong habituation across stations:</b> nearly every effort metric falls sharply from the first five
stations to the later ones — dwell time {trend_row('dwell_time_ms')[3]:.0f}→{trend_row('dwell_time_ms')[4]:.0f} ms
(d={trend_row('dwell_time_ms')[5]:+.2f}, p&lt;.001), fixation count
{trend_row('fixation_count')[3]:.0f}→{trend_row('fixation_count')[4]:.0f} (d={trend_row('fixation_count')[5]:+.2f}, p&lt;.001) —
while mean fixation <i>duration</i> stays flat. Participants make <i>fewer</i> but equally-long fixations:
classic growth of interface efficiency/automaticity.
By contrast, the between-group manipulations (threat framing, AI presence) produced
<b>no significant differences in the global eye-movement DVs</b> (all one-way ANOVA p&gt;0.35), an expected
result given the small, heavily imbalanced groups. Coupons dominate attention overall ({coup:.0f}% of dwell).
</div>

<h2>1 · Data, cohort and design</h2>
<p>Each of 39 participants (001–039; P040 produced no station data — its interaction log contains zero
<code>station</code> events) drove a VR EV-charging security simulation while a Varjo headset recorded gaze
at ~200 Hz. Analysis is restricted to the <b>station-selection screen</b>, the only screen with the AOI layout
of interest. Metrics were computed by joining the AOI-mapper outputs (<code>NNN_gaze_aoi.csv</code>: per-sample
<code>u,v,aoi</code>) with the raw Varjo exports in the <i>Participants</i> folders (3D gaze vectors, pupil
diameter, eye-openness) — the timestamp join matched 100% within 3 ms.</p>

<h3>Station-numbering correction</h3>
<p>The logged station/trial labels skip 4: they run <code>1,2,3,5,6,…,16</code>. As instructed, every label
≥5 was shifted down by one (<code>5→4 … 16→15</code>), giving a contiguous <b>stations 1–15</b>. Station 4 never
appears in any participant's log, confirming the off-by-one was a labelling artefact, not missing data.</p>

<h3>Groups and the 2×2 factor structure</h3>
<p>The four groups map cleanly onto a <b>Threat framing × AI assistance</b> factorial:</p>
<table class="tbl">
<tr><th>Group</th><th>Threat framing</th><th>AI assistance</th><th>n (with data)</th></tr>
<tr><td><b>G1</b></td><td>Cyberattack</td><td>AI</td><td>{gs['G1']}</td></tr>
<tr><td><b>G2</b></td><td>Cyberattack</td><td>No AI</td><td>{gs['G2']} <span class="muted">(P040 lost)</span></td></tr>
<tr><td><b>G3</b></td><td>Anomaly</td><td>No AI</td><td>{gs['G3']}</td></tr>
<tr><td><b>G4</b></td><td>Anomaly</td><td>AI</td><td>{gs['G4']}</td></tr>
</table>
<div class="card warn"><b>Design caveat (read first).</b> The groups are between-subjects and badly imbalanced
(G2=5 vs G4=18). This gives low power for group contrasts and means the absence of group effects below should
be read as <i>"not detectable in this sample,"</i> not <i>"no effect exists."</i> All inferential p-values are
reported as <b>exploratory</b>, accompanied by effect sizes (η², Cohen's d), per the agreed approach.</div>

<h2>2 · The eight dependent variables — definitions &amp; method</h2>
<p>Fixations and saccades use the velocity-threshold algorithm <b>I-VT</b> (Salvucci &amp; Goldberg, 2000) on the
3D gaze-direction vectors: angular velocity below <b>30°/s</b> marks fixation samples, runs ≥<b>60 ms</b> become
fixations, and saccades are the transitions between successive fixations. Blinks are detected from mean
eye-openness dropping below 0.30 for 40–600 ms. Pupil size is the mean of both eyes' diameter (mm) over valid
samples. Each variable is computed <b>per participant × per station</b>, and the AOI-decomposable ones
(dwell, fixation count/duration, dispersion, pupil) also <b>per AOI</b>.</p>
<table class="tbl"><tr><th>Variable</th><th>Unit</th><th>Definition / computation</th></tr>{dv_defs}</table>
<p class="muted">Because fixation/saccade/scanpath counts scale with how long a participant lingered, the
pipeline also stores per-second <b>rate</b> versions (<code>fixation_rate_hz</code>, <code>saccade_rate_hz</code>,
<code>scan_rate_degps</code>, <code>onscreen_frac</code>) so attentional intensity can be separated from sheer
dwell duration. P032 lacks the 3D gaze vector (reduced 34-column export), so its fixation/saccade DVs are
recorded as missing while its dwell, dispersion, pupil and blink values are retained.</p>

<h2>3 · Where attention goes — the AOIs verify the behaviour</h2>
<figure>{img('07_aoi_profile')}<figcaption>Left: top AOIs by dwell share across all participants.
Right: the same AOIs broken out by group.</figcaption></figure>
<p>The AOI profile is dominated by the <b>kiosk coupon/promo panels</b> — collectively <b>{coup:.0f}%</b> of all
station dwell — with a strong rightmost-kiosk gradient (kiosk 3 ≫ kiosk 2 ≫ kiosk 1). This is exactly the
signature of a <b>deal-comparison task</b>: participants spend their time reading offers, not headers or stats.
The pattern is stable across all four groups, which is itself a useful validity check — the mapping and AOI
definitions reproduce the same behavioural structure independently in each group.</p>

<h3>The AI-assistant AOI — the clearest condition effect</h3>
<figure>{img('06_ai_panel_attention')}<figcaption>AI-assistant AOI dwell share by group, and pooled by AI
presence. AI-present groups look at the assistant significantly more.</figcaption></figure>
<p>Pooling to the factor level, the AI-present groups devote <b>{ap['AI_groups_mean_pct']:.1f}%</b> of dwell to
the <code>ai_*</code> AOIs versus <b>{ap['NoAI_groups_mean_pct']:.1f}%</b> for AI-absent groups
(t={ap['t']:.2f}, <b>p={ap['p']:.3f}</b>). This is the analysis's strongest condition effect and a clean
demonstration that AOI-level measures detect the manipulation even where global eye-movement metrics do not —
the difference is <i>spatial</i> (where they look), not in <i>how</i> their eyes move.</p>

<h2>4 · Group &amp; factorial comparisons (global DVs)</h2>
<figure>{img('03_group_means')}<figcaption>Participant-level group means ± SE for all eight DVs, with one-way
ANOVA across G1–G4.</figcaption></figure>
<table class="tbl"><tr><th>DV</th><th>F</th><th>p</th><th>η²</th></tr>{ow_rows}</table>
<p>No DV separates the four groups (all p&gt;0.35; η²≤0.09). The largest numeric group differences are in
<b>blink rate</b> (G2 elevated) and <b>pupil size</b> (G4 largest), but neither survives as significant.</p>

<figure>{img('04_factorial_2x2')}<figcaption>Threat framing × AI assistance interaction plots
(non-parallel lines would indicate interaction).</figcaption></figure>
<table class="tbl"><tr><th>DV</th><th>Threat p</th><th>AI p</th><th>Threat×AI p</th></tr>{twoway_rows}</table>
<p>The 2×2 ANOVA on participant means yields no significant main effects or interactions for any global DV.
Combined with the AOI result above, the picture is consistent: <b>the manipulations changed <i>what</i> people
looked at (AI AOI), not the gross statistics of their eye movements.</b></p>
<figure>{img('09_group_boxplots')}<figcaption>Participant-level distributions by group for four key DVs —
note the wide within-group spread that swamps between-group differences at these sample sizes.</figcaption></figure>

<h2>5 · The dominant effect — habituation across stations</h2>
<figure>{img('01_station_trend_overall')}<figcaption>Cohort mean ± SE of each DV across stations 1–15 with
OLS trend line. *** p&lt;.001, ** p&lt;.01, * p&lt;.05.</figcaption></figure>
<table class="tbl">
<tr><th>DV</th><th>slope/station</th><th>r</th><th>p(trend)</th><th>Init (1–5)</th><th>Subseq (6–15)</th><th>Cohen's d</th><th>p(init vs subseq)</th></tr>
{tr_rows}</table>
<p>Effort and exploration fall steadily as participants progress:
<b>dwell time</b>, <b>fixation count</b>, <b>saccade count</b>, <b>scanpath length</b> and the rate-normalised
measures all decline significantly, while <b>fixation duration is flat</b> and <b>pupil size edges upward</b>.
The combination — fewer fixations and saccades, shorter scanpaths, similar fixation length — is the textbook
signature of <b>skill acquisition / habituation</b>: once the layout is learned, participants locate the
relevant coupon and commit faster, scanning the screen far less. The slight pupil increase runs opposite to a
pure "less cognitive load" story and more likely reflects task fatigue or display adaptation; it is a small
effect (d={trend_row('pupil_size_mm')[5]:+.2f}) and should be treated cautiously.</p>

<figure>{img('05_initial_vs_subsequent')}<figcaption>Initial (stations 1–5) vs subsequent (6–15) for every DV,
with paired effect size.</figcaption></figure>
<figure>{img('02_station_trend_by_group')}<figcaption>The habituation trend holds within every group — the
decline is a general learning effect, not driven by one condition.</figcaption></figure>
<p>Crucially the downward trend appears in <b>all four groups</b>, confirming it is a task-general learning
effect rather than an artefact of any condition. This is the single most reliable finding in the dataset and
the one with the largest effect sizes.</p>

<h2>6 · Relationships among the DVs</h2>
<figure>{img('08_dv_correlation')}<figcaption>Pearson correlations among station-level DVs.</figcaption></figure>
<p>The "effort" cluster — dwell time, fixation count, saccade count, scanpath length — is strongly
inter-correlated (they rise and fall together within a visit), which is why they move in lockstep across stations.
<b>Pupil size, fixation duration and blink rate are largely independent</b> of that cluster, so they carry
separate information rather than restating dwell. This justifies reporting them as distinct DVs and warns against
treating, say, fixation count as independent evidence from dwell time.</p>

<h2>7 · Group means (all DVs)</h2>
{table(GSshow, fmt="{:.2f}")}

<h3>Top AOIs overall</h3>
{table(aoi_ov.rename(columns={'share':'dwell share %'}), fmt="{:.1f}")}

<h2>8 · Conclusions</h2>
<div class="card key"><ul>
<li><b>The AOIs validate the intended behaviour.</b> Attention concentrates on coupon panels ({coup:.0f}%) in a
clear deal-comparison pattern, and the AI-assistant AOI draws significantly more dwell when the assistant is
present (p={ap['p']:.3f}) — the eye data confirm participants engaged the feature as designed.</li>
<li><b>Learning dominates the experiment.</b> The largest, most reliable effects are temporal: dwell, fixations,
saccades and scanning drop ~30–40% from early to late stations (d up to {abs(trend_row('dwell_time_ms')[5]):.2f},
p&lt;.001) while fixation duration holds — participants become rapidly more efficient at the interface.</li>
<li><b>Threat framing and AI presence did not move the global oculomotor DVs</b> in this sample. The effect of the
manipulations is expressed spatially (which AOI), not in gross eye-movement statistics — and group contrasts are
underpowered, so this is a "not detected here," not a null proof.</li>
<li><b>Methodologically</b>, the per-second rate metrics and the DV correlation structure show the eight variables
are not all independent: four form an effort cluster; pupil, fixation duration and blink rate stand apart.</li>
</ul></div>

<h2>9 · Limitations</h2>
<div class="card flag"><ul>
<li><b>Severe group imbalance</b> (G2=5, G4=18) — group/factorial tests are low-powered; treat all p-values as exploratory.</li>
<li><b>AOI mapping is the per-participant <code>auto</code> approximation</b> (no HMD pose); ~17% of dwell is unmapped.
Fine-grained within-card AOIs (set-amount vs preview) are the least reliable layer — re-run high-unmapped
participants (P004, P014, P021, P028, P029) with homography before trusting that granularity.</li>
<li><b>Fixation detection runs on eye-in-head 3D vectors</b>, not gaze-on-stimulus angle; thresholds (30°/s, 60 ms)
are standard defaults and adjustable — absolute fixation counts shift with them, though relative comparisons are robust.</li>
<li><b>Uneven station coverage</b>: participants contribute different station counts (mean ≈ 7.7), and P021 (1 visit) /
P028 (2 visits) are thin. Station-level trend uses all available visits but later stations have fewer participants.</li>
<li><b>Pupillometry is uncalibrated</b> (raw mm, bright VR display → small pupils ~2.5 mm); usable for relative
contrasts only, not absolute cognitive-load claims.</li>
</ul></div>

<h2>10 · File manifest</h2>
<p>All computed data and reproducible code are in <code>Claude Output/analysis_outputs/</code>:</p>
<ul>
<li><code>station_level_labeled.csv</code> — every participant×station: all 8 DVs + rate metrics + group/threat/AI/station.</li>
<li><code>aoi_level_labeled.csv</code> — every participant×station×AOI: dwell, fixation count/duration, dispersion, pupil.</li>
<li><code>participant_summary.csv</code> — per-participant DV means.</li>
<li><code>group_summary.csv</code>, <code>threat_summary.csv</code>, <code>ai_summary.csv</code>, <code>factorial_2x2_cells.csv</code> — condition aggregates.</li>
<li><code>station_trend_overall.csv</code>, <code>station_trend_by_group.csv</code>, <code>phase_by_group.csv</code> — trend tables.</li>
<li><code>aoi_share_overall.csv</code>, <code>aoi_share_by_group.csv</code>, <code>ai_attention_*.csv</code> — AOI attention.</li>
<li><code>stats.json</code> — all test statistics. <code>figures/</code> — the 9 figures. Scripts: <code>extract.py</code>, <code>aggregate.py</code>, <code>figures.py</code>, <code>build_report.py</code>.</li>
</ul>
<p class="muted">Generated for Naman. Inferential statistics are exploratory given the imbalanced between-subjects design; effect sizes accompany every test.</p>
</body></html>"""

open(f"{OUT}/ANALYSIS_REPORT.html","w").write(HTML)
print("report written:",len(HTML),"bytes")
