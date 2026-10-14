"""
Clinical wellness scale vs electrophysiology
============================================
Reads Wellness-scale.xlsx and joins it to the session features, to see whether the
LFP/CSD features follow how SICK each mouse is, and whether that helps to tell
NPC from EFV.

The wellness scale: 10 criteria, +1 each if present -> 0 (healthy) ... 10 (worst).
Scored at weeks 7, 10, 12 and 14. Recordings happened between week 10 (after
surgery) and week 14, so by default each mouse gets the MEAN of weeks 10, 12 and 14
("score during the recordings"). Change WELLNESS_VAR to use another one.

Keep this script in the same folder as the PCA settings script you want to use
(CONFIG below): pca_analysis_new.py (selected features, all sessions) or
pca_analysis_first_sessions.py (selected features, first sessions).

Figures (RESULTSNatalia/wellness<tag>/):
  01_wellness_over_time.png     score of every mouse at weeks 7-14, by group
  02_symptoms_by_group.png      % of mice with each criterion, per group and week
  03_maps_colored_by_wellness.png  t-SNE and PCA, color = wellness score
  03b_maps_group_and_intensity.png  t-SNE and PCA with group COLOR (WT blue, NPC red,
                                EFV green) and INTENSITY = wellness (top row) or
                                position in the mouse's recordings (bottom row)
  04_feature_vs_wellness.png    which features change with the score (correlation)
  05_top_features_scatter.png   per mouse: feature vs score, NPC and EFV lines
Excel: wellness_results.xlsx (tidy scores, per-mouse values, correlations)

!! Only ~16 mice: every correlation here has few points. Read them as hints.
!! Do NOT add the wellness score as a FEATURE to separate NPC from EFV: it is a
   clinical measure that already differs between them (the treatment improves it),
   so it would separate them trivially without saying anything about the LFP/CSD.
"""

import importlib
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.cm import ScalarMappable

# =============================================================================
# SETTINGS
# =============================================================================
CONFIG = "pca_analysis_new"   # or "pca_analysis_first_sessions" (must be in this folder)
cfg = importlib.import_module(CONFIG)

WELLNESS_FILE = cfg.baseDir / "Wellness-scale.xlsx"   # put the file in ANALISIS, or change this
SUMMARY_SHEET = "Seguimiento global"                  # sheet with the total score per week
WEEKS = [7, 10, 12, 14]
RECORDING_WEEKS = [10, 12, 14]
WELLNESS_VAR = "score_during_recordings"
#   "score_during_recordings" = mean of weeks 10, 12, 14 (default)
#   "score_w7", "score_w10", "score_w12", "score_w14" = one week
#   "progression_rate" = increase of the score per week (slope weeks 7 -> 14)
PERPLEXITY = 15
RANDOM_STATE = 42
N_TOP = 6                       # features shown in figure 05
WELLNESS_COLOR_RANGE = None     # figure 03b: None = range of the NPC + EFV mice (WT, ~0, shown
                                # lightest); or e.g. (0, 8) for the full scale
OUTPUT_DIR = cfg.RESULTS_DIR / f"wellness{getattr(cfg, 'RUN_TAG', '')}"
GROUPS = ["WT", "NPC", "EFV"]
MARKERS = {"WT": "o", "NPC": "s", "EFV": "^"}
SAME_CRITERION = {"Clumsiness": "Ataxia", "Apathy": "Lethargy"}   # names used in one block
# =============================================================================

POWER_RE = re.compile(r"^(Delta|Theta|Alpha|SlowGamma|MidGamma|FastGamma|Ripple)_(PYR|SLM)_(Theta|LIA)_(mean|max)$",
                      re.IGNORECASE)


def family(name):
    if POWER_RE.match(name):
        return "power"
    if name.upper().startswith("CSD_"):
        return "CSD"
    if name.lower().startswith("ripple_"):
        return "ripple"
    return "other"


def mouse_number(name):
    """'WT 23', 'CTR-23', 'Efv 263', 218, 'NPC_249' -> 23, 23, 263, 218, 249."""
    m = re.findall(r"\d+", str(name))
    return int(m[-1]) if m else None


def group_from_name(name):
    s = str(name).upper()
    if s.startswith(("WT", "CTR")):
        return "WT"
    if s.startswith("NPC"):
        return "NPC"
    if s.startswith("EFV"):
        return "EFV"
    return None


# -----------------------------------------------------------------------------
# Reading the Excel
# -----------------------------------------------------------------------------
def read_total_scores(path):
    """Long table: animal number, group, week, total score (from the summary sheet)."""
    raw = pd.read_excel(path, sheet_name=SUMMARY_SHEET, header=None)
    hdr = raw.index[raw.iloc[:, 0].astype(str).str.strip().str.lower() == "week"][0]
    names = raw.iloc[hdr, 1:].dropna()
    rows = []
    for r in range(hdr + 1, len(raw)):
        week = pd.to_numeric(raw.iloc[r, 0], errors="coerce")
        if pd.isna(week):
            continue
        for c, name in names.items():
            v = pd.to_numeric(raw.iloc[r, c], errors="coerce")
            if not pd.isna(v):
                rows.append({"mouse": mouse_number(name), "group": group_from_name(name),
                             "name_in_wellness": str(name), "week": int(week), "score": float(v)})
    return pd.DataFrame(rows)


def read_criteria(path, known_mice):
    """Long table of the 10 criteria (from the 'Cohorte' sheets). Mice not in the study are skipped."""
    rows = []
    xl = pd.ExcelFile(path)
    for sheet in xl.sheet_names:
        if sheet == SUMMARY_SHEET:
            continue
        raw = pd.read_excel(path, sheet_name=sheet, header=None)
        week, header = None, None
        for _, row in raw.iterrows():
            first, second = row.iloc[0], row.iloc[1] if len(row) > 1 else None
            if isinstance(first, str) and "week" in first.lower():          # 'Week 10: ...'
                week, header = int(re.findall(r"\d+", first)[0]), None
                continue
            if isinstance(second, str) and "week" in second.lower():        # date | '9 weeks'
                week, header = int(re.findall(r"\d+", second)[0]), None
                continue
            if isinstance(first, str) and first.strip().lower() == "animal":
                header = [SAME_CRITERION.get(str(h).strip(), str(h).strip()) for h in row.iloc[1:12]]
                continue
            if header is None or week is None or pd.isna(first):
                continue
            num = mouse_number(first)
            if num not in known_mice:
                continue
            vals = pd.to_numeric(row.iloc[1:12], errors="coerce").to_numpy()
            for h, v in zip(header, vals):
                if h.lower() != "total score":
                    rows.append({"mouse": num, "week_scored": week, "criterion": h, "value": v})
    crit = pd.DataFrame(rows)
    if len(crit):   # weeks 9 and 13 (EFV218/220 block) -> 10 and 14, as in the summary sheet
        crit["week"] = crit["week_scored"].map(lambda w: min(WEEKS, key=lambda x: (abs(x - w), -x)))
    return crit


def per_mouse_table(scores):
    wide = scores.pivot_table(index=["mouse", "group"], columns="week", values="score").reset_index()
    out = wide[["mouse", "group"]].copy()
    for w in WEEKS:
        out[f"score_w{w}"] = wide.get(w)
    out["score_during_recordings"] = wide[[w for w in RECORDING_WEEKS if w in wide]].mean(axis=1)
    weeks = np.array([w for w in WEEKS if w in wide])
    out["progression_rate"] = [np.polyfit(weeks, r, 1)[0] if np.isfinite(r).sum() >= 2 else np.nan
                               for r in wide[list(weeks)].to_numpy(float)]
    return out


def relative_time(ids):
    """0 = first session of the mouse, 1 = last (from the date in the session name)."""
    def dt(name):
        m = re.search(r"(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})", str(name))
        if m:
            return pd.Timestamp(*(int(x) for x in m.groups()))
        m = re.search(r"(\d{4})[-_.](\d{1,2})[-_.](\d{1,2})", str(name))
        return pd.Timestamp(*(int(x) for x in m.groups())) if m else pd.NaT
    d = ids["session"].map(dt)
    order = d.groupby(ids["animal"]).rank(method="first")
    n = order.groupby(ids["animal"]).transform("max")
    return ((order - 1) / (n - 1)).where(n > 1).to_numpy(float)


def group_cmap(color):
    """From a very light version of the group color (low) to a dark version (high)."""
    c = np.array(to_rgb(color))
    light = 0.82 * np.ones(3) + 0.18 * c
    dark = 0.45 * c
    return LinearSegmentedColormap.from_list("g", [light, c, dark])


def plot_group_intensity(maps, groups, values, label, axes_row, color, vmin, vmax):
    """Same color family per group, darker = higher value. Grey = no value."""
    for ax, (Z, title) in zip(axes_row, maps):
        nan = np.isnan(values)
        ax.scatter(Z[nan, 0], Z[nan, 1], s=34, color="#d9d9d9", edgecolor="k", linewidth=0.3)
        for g in GROUPS:
            m = (groups == g) & ~nan
            ax.scatter(Z[m, 0], Z[m, 1], s=46, c=values[m], cmap=group_cmap(color[g]),
                       vmin=vmin, vmax=vmax, edgecolor="k", linewidth=0.4)
        ax.set_title(f"{title}\nintensity = {label}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])


# -----------------------------------------------------------------------------
def ancova_group_p(y, w, is_efv):
    """Mouse level: y ~ a + b*wellness + c*EFV. p-values of b (wellness) and c (group)."""
    ok = np.isfinite(y) & np.isfinite(w)
    y, w, g = y[ok], w[ok], is_efv[ok].astype(float)
    if len(y) < 5 or g.min() == g.max():
        return np.nan, np.nan
    Xd = np.column_stack([np.ones(len(y)), w, g])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta
    dof = len(y) - Xd.shape[1]
    s2 = resid @ resid / dof
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(Xd.T @ Xd)))
    p = 2 * stats.t.sf(np.abs(beta / se), dof)
    return p[1], p[2]


def bh_fdr(p):
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.sum():
        pv = p[ok]; n = len(pv); o = np.argsort(pv)
        r = np.minimum.accumulate((pv[o] * n / np.arange(1, n + 1))[::-1])[::-1]
        tmp = np.empty(n); tmp[o] = np.minimum(r, 1); q[ok] = tmp
    return q


# -----------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    color = {g: cfg.color_for(g, i) for i, g in enumerate(GROUPS)}

    # ---- wellness ----
    if not WELLNESS_FILE.exists():
        raise FileNotFoundError(f"{WELLNESS_FILE} not found: put Wellness-scale.xlsx there or change WELLNESS_FILE")
    scores = read_total_scores(WELLNESS_FILE)
    mice = per_mouse_table(scores)
    try:
        crit = read_criteria(WELLNESS_FILE, set(mice["mouse"]))
    except Exception as e:
        print(f"[!] Could not read the individual criteria ({e}); figure 02 skipped")
        crit = pd.DataFrame()
    print(f"Wellness: {len(mice)} mice, weeks {sorted(int(w) for w in scores['week'].unique())}")
    print(mice.groupby("group")[["score_w7", "score_w10", "score_w12", "score_w14"]].mean().round(1)
          .reindex(GROUPS).to_string())

    # ---- features + join by mouse number ----
    ids, feats = cfg.load_data(cfg.EXCEL_PATH, cfg.SHEET, cfg.ID_COLS)
    groups = ids[cfg.GROUP_COL].astype(str).to_numpy()
    num = ids["animal"].map(mouse_number).to_numpy()
    w_by_mouse = mice.set_index("mouse")[WELLNESS_VAR]
    w_sess = pd.Series(num).map(w_by_mouse).to_numpy(float)
    missing = sorted(set(ids.loc[np.isnan(w_sess), "animal"]))
    if missing:
        print(f"[!] No wellness score for: {missing}")
    bad = mice.set_index("mouse")["group"]
    clash = [a for a, n_, g in zip(ids["animal"], num, groups) if n_ in bad.index and bad[n_] != g]
    if clash:
        print(f"[!] Group in the wellness file differs for: {sorted(set(clash))}")
    X = StandardScaler().fit_transform(feats.to_numpy())
    print(f"\n{len(ids)} sessions x {X.shape[1]} features; wellness variable = '{WELLNESS_VAR}'")

    # ================= FIGURE 1: score over time =================
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axvspan(min(RECORDING_WEEKS), max(RECORDING_WEEKS), color="#f2f2f2", zorder=0,
               label="recording period")
    rng = np.random.default_rng(0)
    for _, r in mice.iterrows():
        y = [r[f"score_w{w}"] for w in WEEKS]
        jit = rng.uniform(-0.12, 0.12)
        ax.plot(np.array(WEEKS) + jit, np.array(y) + rng.uniform(-0.08, 0.08), "-",
                marker=MARKERS[r["group"]], color=color[r["group"]], alpha=0.6, lw=1, ms=5)
    for g in GROUPS:
        m = mice[mice["group"] == g]
        ax.plot(WEEKS, [m[f"score_w{w}"].mean() for w in WEEKS], "-", color=color[g], lw=3,
                marker=MARKERS[g], ms=9, markeredgecolor="k", label=f"{g} mean (n={len(m)})")
    ax.set_xticks(WEEKS); ax.set_xlabel("week"); ax.set_ylabel("wellness score (0 = healthy)")
    ax.set_title("Wellness score of every mouse (thin) and group mean (thick)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "01_wellness_over_time.png", dpi=200)

    # ================= FIGURE 2: symptoms =================
    if len(crit):
        crit["group"] = crit["mouse"].map(mice.set_index("mouse")["group"])
        crit_names = list(dict.fromkeys(crit["criterion"]))
        weeks_c = sorted(crit["week"].unique())
        fig2, axes2 = plt.subplots(1, len(weeks_c), figsize=(3.3 * len(weeks_c) + 1.5, 4.6),
                                   sharey=True, squeeze=False)
        for ax, w in zip(axes2[0], weeks_c):
            tab = (crit[crit["week"] == w].groupby(["criterion", "group"])["value"].mean()
                   .unstack().reindex(index=crit_names, columns=[g for g in GROUPS]))
            im = ax.imshow(tab.to_numpy(float) * 100, cmap="Reds", vmin=0, vmax=100, aspect="auto")
            ax.set_xticks(range(tab.shape[1]), tab.columns)
            ax.set_yticks(range(len(crit_names)), crit_names, fontsize=8)
            for i in range(tab.shape[0]):
                for j in range(tab.shape[1]):
                    v = tab.iloc[i, j]
                    if pd.notna(v):
                        ax.text(j, i, f"{v * 100:.0f}", ha="center", va="center", fontsize=7)
            ax.set_title(f"week {w}")
        fig2.colorbar(im, ax=axes2, shrink=0.8, label="% of mice with the symptom")
        fig2.suptitle("Which symptoms does each group have? (WT 90 has no detailed criteria)")
        fig2.savefig(OUTPUT_DIR / "02_symptoms_by_group.png", dpi=200, bbox_inches="tight")

    # ================= FIGURE 3: maps colored by wellness =================
    emb_file = cfg.RESULTS_DIR / f"embedding_results{getattr(cfg, 'RUN_TAG', '')}" / "embeddings.xlsx"
    cols = [f"tSNE_p{PERPLEXITY}_1", f"tSNE_p{PERPLEXITY}_2"]
    Y = None
    if emb_file.exists():
        emb = pd.read_excel(emb_file)
        if all(c in emb for c in cols):
            key = ids["animal"].astype(str) + "|" + ids["session"].astype(str)
            ekey = emb["animal"].astype(str) + "|" + emb["session"].astype(str)
            Y = emb.set_index(ekey)[cols].reindex(key).to_numpy()
            if np.isnan(Y).any():
                Y = None
    if Y is None:
        Y = TSNE(n_components=2, perplexity=min(PERPLEXITY, len(X) - 1), init="pca",
                 learning_rate="auto", random_state=RANDOM_STATE).fit_transform(X)
    pca = PCA(n_components=2).fit(X)
    P = pca.transform(X)
    vr = pca.explained_variance_ratio_ * 100
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5.2), layout="constrained")
    vmin, vmax = np.nanmin(mice[WELLNESS_VAR]), np.nanmax(mice[WELLNESS_VAR])
    for ax, Z, title in [(axes3[0], Y, f"t-SNE (perplexity {PERPLEXITY})"),
                         (axes3[1], P, f"PCA: PC1 ({vr[0]:.0f}%) vs PC2 ({vr[1]:.0f}%)")]:
        ax.scatter(Z[np.isnan(w_sess), 0], Z[np.isnan(w_sess), 1], s=30, color="lightgray",
                   edgecolor="k", linewidth=0.3)
        for g in GROUPS:
            m = (groups == g) & ~np.isnan(w_sess)
            sc = ax.scatter(Z[m, 0], Z[m, 1], c=w_sess[m], cmap="magma_r", vmin=vmin, vmax=vmax,
                            s=48, marker=MARKERS[g], edgecolor="k", linewidth=0.5, label=g)
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=8, title="marker = group")
    fig3.colorbar(sc, ax=axes3, shrink=0.8, label=f"{WELLNESS_VAR}  (light = healthy, dark = sick)")
    fig3.suptitle("Sessions colored by the wellness score of their mouse")
    fig3.savefig(OUTPUT_DIR / "03_maps_colored_by_wellness.png", dpi=200)

    # ================= FIGURE 3b: group color + intensity =================
    rel_t = relative_time(ids)
    fig6, axes6 = plt.subplots(2, 2, figsize=(13, 11), layout="constrained")
    maps = [(Y, f"t-SNE (perplexity {PERPLEXITY})"), (P, f"PCA: PC1 ({vr[0]:.0f}%) vs PC2 ({vr[1]:.0f}%)")]
    if WELLNESS_COLOR_RANGE is None:          # spread the colors over the sick mice
        sick_w = w_sess[np.isin(groups, ["NPC", "EFV"])]
        wmin, wmax = np.floor(np.nanmin(sick_w)), np.ceil(np.nanmax(sick_w))
    else:
        wmin, wmax = WELLNESS_COLOR_RANGE
    print(f"Figure 03b: wellness intensity from {wmin:g} (light) to {wmax:g} (dark); lower values = lightest")
    plot_group_intensity(maps, groups, w_sess, f"wellness ({WELLNESS_VAR}): light = healthy, dark = sick",
                         axes6[0], color, wmin, wmax)
    plot_group_intensity(maps, groups, rel_t, "recording time: light = first session, dark = last",
                         axes6[1], color, 0, 1)
    for row, (vmin_, vmax_, lab) in enumerate([(wmin, wmax, "wellness score"),
                                                 (0, 1, "relative time (0 = first, 1 = last)")]):
        for g in GROUPS:
            sm = ScalarMappable(norm=Normalize(vmin_, vmax_), cmap=group_cmap(color[g]))
            cb = fig6.colorbar(sm, ax=axes6[row], shrink=0.6, aspect=25, pad=0.01)
            cb.set_label(f"{g}: {lab}", fontsize=8)
            cb.ax.tick_params(labelsize=7)
    fig6.suptitle("Color = group (WT blue, NPC red, EFV green); intensity = wellness (top) or time (bottom).\n"
                  "Grey = no value (no score, or no date in the session name)")
    fig6.savefig(OUTPUT_DIR / "03b_maps_group_and_intensity.png", dpi=200)

    # ================= correlations (MOUSE level) =================
    fm = feats.copy()
    fm["mouse"] = num
    fm["group"] = groups
    per_mouse = fm.groupby(["mouse", "group"]).median().reset_index()
    per_mouse["wellness"] = per_mouse["mouse"].map(w_by_mouse)
    sick = per_mouse["group"].isin(["NPC", "EFV"]).to_numpy()
    is_efv = (per_mouse["group"] == "EFV").to_numpy()
    rows = []
    for f in feats.columns:
        y = per_mouse[f].to_numpy(float)
        w = per_mouse["wellness"].to_numpy(float)
        ok = np.isfinite(y) & np.isfinite(w)
        r_all = stats.spearmanr(w[ok], y[ok]) if ok.sum() > 3 else (np.nan, np.nan)
        oks = ok & sick
        r_sick = stats.spearmanr(w[oks], y[oks]) if oks.sum() > 3 else (np.nan, np.nan)
        p_w, p_g = ancova_group_p(y[sick], w[sick], is_efv[sick])
        rows.append({"feature": f, "family": family(f),
                     "rho_all_mice": r_all[0], "p_all_mice": r_all[1],
                     "rho_NPC_EFV": r_sick[0], "p_NPC_EFV": r_sick[1],
                     "p_wellness_given_group": p_w, "p_group_given_wellness": p_g,
                     "n_mice": int(ok.sum()), "n_NPC_EFV_mice": int(oks.sum())})
    corr = pd.DataFrame(rows)
    corr["q_NPC_EFV"] = bh_fdr(corr["p_NPC_EFV"])
    corr = corr.reindex(corr["rho_NPC_EFV"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    print("\nFEATURES vs WELLNESS (one value per mouse = median of its sessions)")
    print("  rho_all_mice : includes WT (mostly WT vs sick)")
    print("  rho_NPC_EFV  : only sick mice -> does the feature follow HOW sick a mouse is?")
    print("  p_group_given_wellness: NPC vs EFV difference at the SAME wellness score")
    print(corr.head(15).to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    # ================= FIGURE 4: correlation bars =================
    fam_c = {"ripple": "#7b3294", "CSD": "#e66101", "power": "#018571", "other": "gray"}
    c2 = corr.sort_values("rho_NPC_EFV")
    fig4, axes4 = plt.subplots(1, 2, figsize=(12, max(4, 0.28 * len(c2) + 1.5)), sharey=True)
    for ax, col, title in [(axes4[0], "rho_all_mice", "All mice (WT + NPC + EFV)"),
                           (axes4[1], "rho_NPC_EFV", "Only NPC + EFV mice")]:
        pcol = "p_all_mice" if col == "rho_all_mice" else "p_NPC_EFV"
        ax.barh(range(len(c2)), c2[col], color=[fam_c[f] for f in c2["family"]],
                edgecolor=["k" if p < 0.05 else "none" for p in c2[pcol]])
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlim(-1, 1); ax.set_xlabel("Spearman rho with wellness score")
        ax.set_title(title)
    axes4[0].set_yticks(range(len(c2)), c2["feature"], fontsize=7)
    present = [f for f in fam_c if f in set(c2["family"])]
    axes4[1].legend([plt.Rectangle((0, 0), 1, 1, color=fam_c[f]) for f in present], present,
                    fontsize=8, loc="lower right")
    fig4.suptitle("Does each feature change with the wellness score?  (black edge = p < 0.05; "
                  "+ = higher when sicker)")
    fig4.tight_layout(); fig4.savefig(OUTPUT_DIR / "04_feature_vs_wellness.png", dpi=200)

    # ================= FIGURE 5: top features, per mouse =================
    top = corr["feature"].head(N_TOP).tolist()
    ncol = min(3, len(top))
    nrow = int(np.ceil(len(top) / ncol))
    fig5, axes5 = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.7 * nrow), squeeze=False)
    for ax, f in zip(axes5.flat, top):
        for g in GROUPS:
            ms = groups == g                         # sessions (faint)
            ax.scatter(w_sess[ms] + rng.uniform(-0.08, 0.08, ms.sum()), feats[f].to_numpy()[ms],
                       s=8, color=color[g], alpha=0.25, edgecolor="none")
            mm = per_mouse["group"] == g             # mice (big)
            ax.scatter(per_mouse.loc[mm, "wellness"], per_mouse.loc[mm, f], s=60, marker=MARKERS[g],
                       color=color[g], edgecolor="k", label=g, zorder=3)
            if g in ("NPC", "EFV") and mm.sum() >= 3:
                xw, yf = per_mouse.loc[mm, "wellness"].to_numpy(float), per_mouse.loc[mm, f].to_numpy(float)
                ok = np.isfinite(xw) & np.isfinite(yf)
                if ok.sum() >= 3 and np.ptp(xw[ok]) > 0:
                    b = np.polyfit(xw[ok], yf[ok], 1)
                    xx = np.linspace(xw[ok].min(), xw[ok].max(), 10)
                    ax.plot(xx, np.polyval(b, xx), "-", color=color[g], lw=2)
        r = corr.set_index("feature").loc[f]
        ax.set_title(f"{f}\nrho (NPC+EFV) = {r['rho_NPC_EFV']:.2f}, p = {r['p_NPC_EFV']:.2g}", fontsize=8)
        ax.set_xlabel(WELLNESS_VAR, fontsize=8)
        if POWER_RE.match(f):
            ax.set_yscale("log")
    for ax in axes5.flat[len(top):]:
        ax.axis("off")
    axes5.flat[0].legend(fontsize=7)
    fig5.suptitle("Big markers = one mouse (median of its sessions); faint dots = sessions.\n"
                  "If NPC and EFV mice fall on ONE line, EFV looks like 'milder NPC'; "
                  "two separate lines = treatment changes the feature beyond severity.", fontsize=9)
    fig5.tight_layout(); fig5.savefig(OUTPUT_DIR / "05_top_features_scatter.png", dpi=200)

    # ---- save ----
    out = ids.copy()
    out["wellness"] = w_sess
    with pd.ExcelWriter(OUTPUT_DIR / "wellness_results.xlsx") as w:
        scores.to_excel(w, sheet_name="scores_long", index=False)
        mice.to_excel(w, sheet_name="per_mouse_scores", index=False)
        if len(crit):
            crit.to_excel(w, sheet_name="criteria_long", index=False)
        per_mouse.to_excel(w, sheet_name="per_mouse_features", index=False)
        corr.to_excel(w, sheet_name="feature_vs_wellness", index=False)
        out.to_excel(w, sheet_name="sessions", index=False)
    print(f"\nSaved in {OUTPUT_DIR}")
    plt.show()


if __name__ == "__main__":
    main()