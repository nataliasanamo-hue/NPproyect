"""
Clinical wellness scale vs electrophysiology  (features selected per comparison)
================================================================================
New version of wellness_analysis.py that:
  1. uses the features chosen by select_features.py for EACH comparison
     (settings and data come from pca_analysis_new_selection.py, same folder), and
  2. FIXES the mismatch between the "wellness over time" plot and the maps.

Why the old maps (03b) did not match the "wellness over time" plot
------------------------------------------------------------------
* In the old script every session of a mouse got ONE value: the mean score of
  weeks 10, 12 and 14. So the FIRST session of a sick NPC mouse was painted just
  as dark as its LAST one. The top row of 03b was a per-MOUSE colour, never a
  per-SESSION one, which is why "early" sessions could look very sick.
* The bottom row ("recording time") was the RANK of the session inside its own
  mouse (0 = first, 1 = last), not the age of the mouse. "Light" only meant
  "first session of that mouse", not "young / early disease".
* The colour range was stretched over the sick mice only (3 -> 7), so small
  differences looked big and WT was clipped to the lightest colour.
* And the important biological point: the part of the curve where NPC and EFV
  are close to WT is WEEK 7, which is BEFORE the surgery. There are no
  recordings there. By week 10 (first recordings) NPC already scores ~5.

What this version does instead
------------------------------
Each session gets an ESTIMATED WEEK OF AGE from the date in its name:
    week = RECORDING_START_WEEK + (days since that mouse's first session) / 7
(or, for mice in KNOWN_REFERENCE_DATES, from a date whose week is known).
The wellness score of THAT session is the mouse's score interpolated at that
week (between the scorings of weeks 10, 12, 14). Set SESSION_SCORE_MODE =
"per_mouse" to get the old behaviour back and compare.

Figures (RESULTSNatalia/wellness_new_selection/<comparison>/)
  01_recordings_on_wellness_curve.png  each mouse's score curve with its recording
                                       sessions drawn ON the curve -> which part of
                                       the disease do the recordings really sample?
  02_maps_group_and_intensity.png      t-SNE and PCA, colour = group, intensity =
                                       session score (top) / estimated week (bottom),
                                       same fixed scales for every group
  03_old_vs_new_score.png              per-mouse constant (old) vs per-session (new)
  04_feature_vs_wellness.png           correlation BETWEEN mice and WITHIN mice
  05_top_features_within_mouse.png     the features that change most as the SAME
                                       mouse gets sicker
Excel: wellness_results.xlsx

!! Few mice: read every correlation as a hint, not a result.
!! Do NOT add the wellness score as a FEATURE for the classifiers: it is a clinical
   measure that already differs between groups.
"""

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import pca_analysis_new_selection as cfg   # settings + load_data (same folder)

# =============================================================================
# SETTINGS
# =============================================================================
COMPARISONS_TO_RUN = cfg.COMPARISONS_TO_RUN   # or e.g. ["WT_vs_NPC"]

WELLNESS_FILE = cfg.baseDir / "Wellness-scale.xlsx"   # put the file in ANALISIS, or change this
SUMMARY_SHEET = "Seguimiento global"                  # sheet with the total score per week
WEEKS = [7, 10, 12, 14]                               # weeks scored in the summary sheet

SESSION_SCORE_MODE = "per_session"
#   "per_session" = score of the mouse at the week of THAT session (interpolated)  <- new
#   "per_mouse"   = mean of weeks 10, 12, 14 for every session (old behaviour)

RECORDING_START_WEEK = 10   # age (weeks) of each mouse's FIRST session. The sheet says
                            # week 10 = "post surgery, pre training" and week 12 = "after
                            # some recordings", so the true value may be ~10.5-11: adjust it
RECORDING_END_WEEK = 14     # last scoring ("post recordings, pre sacrifice")

# Mice for which you know the date of one scored week. Their session weeks are computed
# from that date instead of from RECORDING_START_WEEK. Add more if you have the dates.
#   mouse number: ("YYYY-MM-DD", week at that date)
KNOWN_REFERENCE_DATES = {
    218: ("2025-12-05", 7),   # Cohorte 2 sheet: 2025-12-05 = 7 weeks
    220: ("2025-12-05", 7),
}

SCORE_COLOR_RANGE = (0, 8)  # fixed colour scale for the score (same for every group)
PERPLEXITY = 15             # reuse this t-SNE from nonlinear_embedding_new_selection.py
RANDOM_STATE = 42
N_TOP = 6                   # features shown in figure 05
MIN_SESSIONS_WITHIN = 3     # a mouse needs >= this many sessions for the within-mouse test

OUTPUT_ROOT = cfg.RESULTS_DIR / "wellness_new_selection"
GROUPS = ["WT", "NPC", "EFV"]
MARKERS = {"WT": "o", "NPC": "s", "EFV": "^"}
# =============================================================================


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
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


def session_date(name):
    """Date in the session folder name, e.g. '..._2026-01-14_12-30-05' or '2026_01_14'."""
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})", str(name))
    if m:
        return pd.Timestamp(*(int(x) for x in m.groups()))
    m = re.search(r"(\d{4})[-_.](\d{1,2})[-_.](\d{1,2})", str(name))
    return pd.Timestamp(*(int(x) for x in m.groups())) if m else pd.NaT


def group_cmap(color):
    """Very light version of the group colour (low) -> dark version (high)."""
    c = np.array(to_rgb(color))
    return LinearSegmentedColormap.from_list("g", [0.85 + 0.15 * c, c, 0.45 * c])


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
# Wellness file
# -----------------------------------------------------------------------------
def read_total_scores(path):
    """Long table: mouse number, group, week, total score (summary sheet)."""
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


def per_mouse_scores(scores):
    wide = scores.pivot_table(index=["mouse", "group"], columns="week", values="score").reset_index()
    out = wide[["mouse", "group"]].copy()
    for w in WEEKS:
        out[f"score_w{w}"] = wide.get(w)
    rec = [w for w in WEEKS if RECORDING_START_WEEK <= w <= RECORDING_END_WEEK and w in wide]
    if not rec:
        rec = [w for w in WEEKS if w >= 10 and w in wide]
    out["score_mean_recordings"] = wide[rec].mean(axis=1)
    return out


def score_at(mouse_row, week):
    """Score of one mouse at any week, by linear interpolation between its scorings."""
    xs = np.array([w for w in WEEKS if pd.notna(mouse_row.get(f"score_w{w}"))], float)
    ys = np.array([mouse_row[f"score_w{int(w)}"] for w in xs], float)
    if len(xs) == 0 or not np.isfinite(week):
        return np.nan
    return float(np.interp(week, xs, ys))   # outside the range -> nearest scoring


# -----------------------------------------------------------------------------
# One wellness value PER SESSION
# -----------------------------------------------------------------------------
def session_table(ids, mice):
    """ids (animal, session, group) -> + mouse, date, est_week, score_session, score_old."""
    t = ids.copy()
    t["mouse"] = t["animal"].map(mouse_number)
    t["date"] = t["session"].map(session_date)
    first = t.groupby("animal")["date"].transform("min")
    t["est_week"] = RECORDING_START_WEEK + (t["date"] - first).dt.days / 7
    for num, (d, w) in KNOWN_REFERENCE_DATES.items():
        m = t["mouse"] == num
        if m.any():
            t.loc[m, "est_week"] = w + (t.loc[m, "date"] - pd.Timestamp(d)).dt.days / 7
    t["week_source"] = np.where(t["mouse"].isin(KNOWN_REFERENCE_DATES), "known date",
                                f"first session = week {RECORDING_START_WEEK}")
    t.loc[t["date"].isna(), "week_source"] = "no date in session name"

    by_mouse = mice.set_index("mouse")
    t["score_old"] = t["mouse"].map(by_mouse["score_mean_recordings"])
    t["score_session"] = [score_at(by_mouse.loc[m], w) if m in by_mouse.index else np.nan
                          for m, w in zip(t["mouse"], t["est_week"])]
    # sessions without a date: fall back to the mouse mean so they are not lost
    t["score_session"] = t["score_session"].fillna(t["score_old"])
    t["score"] = t["score_session"] if SESSION_SCORE_MODE == "per_session" else t["score_old"]

    late = t["est_week"] > RECORDING_END_WEEK + 0.5
    if late.any():
        print(f"[!] {late.sum()} session(s) estimated after week {RECORDING_END_WEEK} "
              f"(score clamped to the last scoring): {sorted(set(t.loc[late, 'animal']))}")
    nod = t["date"].isna()
    if nod.any():
        print(f"[!] {nod.sum()} session(s) without a date in the name -> mouse mean used")
    missing = sorted(set(t.loc[t["score_old"].isna(), "animal"]))
    if missing:
        print(f"[!] No wellness score for: {missing}")
    return t


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
def fig_curve(mice, st, color, groups_here, out):
    """Each mouse's wellness curve + its recording sessions placed ON the curve."""
    fig, axes = plt.subplots(1, len(groups_here), figsize=(5 * len(groups_here), 4.3),
                             sharey=True, squeeze=False)
    for ax, g in zip(axes[0], groups_here):
        ax.axvspan(RECORDING_START_WEEK, RECORDING_END_WEEK, color="#f2f2f2", zorder=0)
        mm = mice[mice["group"] == g]
        for _, r in mm.iterrows():
            ax.plot(WEEKS, [r[f"score_w{w}"] for w in WEEKS], "-", color=color[g], alpha=0.35, lw=1)
        ax.plot(WEEKS, [mm[f"score_w{w}"].mean() for w in WEEKS], "-", color=color[g], lw=3,
                label=f"{g} mean score")
        s = st[(st["group"] == g) & st["est_week"].notna()]
        ax.scatter(s["est_week"], s["score_session"], s=22, color=color[g], edgecolor="k",
                   lw=0.4, zorder=3, label=f"recording sessions (n={len(s)})")
        ax.set_title(g); ax.set_xticks(WEEKS); ax.set_xlabel("week of age")
        ax.legend(fontsize=7, loc="upper left")
    axes[0, 0].set_ylabel("wellness score (0 = healthy)")
    fig.suptitle("Where do the recordings fall on the disease curve?  (grey = recording period; "
                 "the 'close to WT' week 7 has NO recordings)", fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=200)


def fig_maps(Y, P, vr, st, color, groups_here, out, comparison):
    maps = [(Y, f"t-SNE (perplexity {PERPLEXITY})"), (P, f"PCA: PC1 ({vr[0]:.0f}%) vs PC2 ({vr[1]:.0f}%)")]
    wk = st["est_week"].to_numpy(float)
    wmin, wmax = np.nanmin(wk) if np.isfinite(wk).any() else RECORDING_START_WEEK, \
        np.nanmax(wk) if np.isfinite(wk).any() else RECORDING_END_WEEK
    rows = [(st["score"].to_numpy(float), *SCORE_COLOR_RANGE,
             f"wellness score of the session ({SESSION_SCORE_MODE})", "score"),
            (wk, np.floor(wmin), np.ceil(wmax), "estimated week of age (same scale for all mice)", "week")]
    g_arr = st["group"].astype(str).to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10.5), layout="constrained")
    for r, (vals, vmin, vmax, label, short) in enumerate(rows):
        for ax, (Z, title) in zip(axes[r], maps):
            nan = np.isnan(vals)
            ax.scatter(Z[nan, 0], Z[nan, 1], s=34, color="#d9d9d9", edgecolor="k", linewidth=0.3)
            for g in groups_here:
                m = (g_arr == g) & ~nan
                ax.scatter(Z[m, 0], Z[m, 1], s=50, c=vals[m], cmap=group_cmap(color[g]),
                           vmin=vmin, vmax=vmax, marker=MARKERS[g], edgecolor="k", linewidth=0.4)
            ax.set_title(f"{title}\nintensity = {label}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        for g in groups_here:
            cb = fig.colorbar(ScalarMappable(Normalize(vmin, vmax), group_cmap(color[g])),
                              ax=axes[r], shrink=0.6, aspect=25, pad=0.01)
            cb.set_label(f"{g}: {short}", fontsize=8); cb.ax.tick_params(labelsize=7)
    fig.suptitle(f"{comparison} - colour = group, darker = sicker (top) / older (bottom). "
                 "Grey = no value", fontsize=10)
    fig.savefig(out, dpi=200)


def fig_old_vs_new(st, color, groups_here, out):
    fig, ax = plt.subplots(figsize=(5, 4.6))
    rng = np.random.default_rng(0)
    for g in groups_here:
        s = st[st["group"] == g]
        ax.scatter(s["score_old"] + rng.uniform(-0.1, 0.1, len(s)), s["score_session"],
                   s=20, color=color[g], edgecolor="k", lw=0.3, alpha=0.8, label=g)
    lim = [SCORE_COLOR_RANGE[0] - 0.5, SCORE_COLOR_RANGE[1] + 0.5]
    ax.plot(lim, lim, "k--", lw=0.8); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("old: one value per mouse (mean weeks 10-14)")
    ax.set_ylabel("new: score at the week of the session")
    ax.set_title("Each dot = a session. Vertical spread =\nwhat the old score was hiding", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=200)


# -----------------------------------------------------------------------------
# Correlations: BETWEEN mice and WITHIN mice
# -----------------------------------------------------------------------------
def correlations(feats, st):
    """
    between: one point per mouse (median of its sessions) vs its mean score.
             -> do sicker MICE have different features?  (mixes group + severity)
    within : each session minus its mouse's mean, for feature AND score, pooled.
             -> when the SAME mouse gets sicker, does the feature change?
             (removes everything constant per mouse: group, electrode position...)
    """
    df = feats.copy()
    df["mouse"], df["group"], df["score"] = st["mouse"].values, st["group"].values, st["score"].values
    sick = df["group"].isin(["NPC", "EFV"])
    per_mouse = df.groupby(["mouse", "group"]).median(numeric_only=True).reset_index()
    pm_sick = per_mouse["group"].isin(["NPC", "EFV"])

    n_sess = df.groupby("mouse")["score"].transform("size")
    ok_w = sick & (n_sess >= MIN_SESSIONS_WITHIN)
    dev = df[ok_w].copy()
    cols = list(feats.columns) + ["score"]
    dev[cols] = dev[cols] - dev.groupby("mouse")[cols].transform("mean")

    rows = []
    for f in feats.columns:
        r = {"feature": f}
        for lab, sub in [("all_mice", per_mouse), ("NPC_EFV_mice", per_mouse[pm_sick])]:
            ok = sub[[f, "score"]].dropna()
            rr = stats.spearmanr(ok["score"], ok[f]) if len(ok) > 3 and ok["score"].nunique() > 1 else (np.nan, np.nan)
            r[f"rho_between_{lab}"], r[f"p_between_{lab}"], r[f"n_{lab}"] = rr[0], rr[1], len(ok)
        ok = dev[[f, "score"]].dropna()
        ok = ok[ok["score"].abs() > 1e-9] if (ok["score"].abs() > 1e-9).sum() > 3 else ok
        rr = stats.spearmanr(ok["score"], ok[f]) if len(ok) > 3 and ok["score"].nunique() > 1 else (np.nan, np.nan)
        r["rho_within"], r["p_within"], r["n_sessions_within"] = rr[0], rr[1], len(ok)
        rows.append(r)
    corr = pd.DataFrame(rows)
    corr["q_within"] = bh_fdr(corr["p_within"])
    corr["q_between_NPC_EFV_mice"] = bh_fdr(corr["p_between_NPC_EFV_mice"])
    corr = corr.reindex(corr["rho_within"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    return corr, per_mouse, dev


def fig_corr(corr, out, comparison):
    c2 = corr.sort_values("rho_within")
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, 0.28 * len(c2) + 1.5)), sharey=True)
    for ax, col, pcol, title in [
            (axes[0], "rho_between_NPC_EFV_mice", "p_between_NPC_EFV_mice",
             "BETWEEN mice (NPC + EFV, one point per mouse)"),
            (axes[1], "rho_within", "p_within",
             "WITHIN mice (same mouse, sessions as it gets sicker)")]:
        ax.barh(range(len(c2)), c2[col], color=np.where(c2[col] > 0, "#c0504d", "#4f81bd"),
                edgecolor=["k" if p < 0.05 else "none" for p in c2[pcol]])
        ax.axvline(0, color="k", lw=0.8); ax.set_xlim(-1, 1)
        ax.set_xlabel("Spearman rho with wellness score"); ax.set_title(title, fontsize=9)
    axes[0].set_yticks(range(len(c2)), c2["feature"], fontsize=7)
    fig.suptitle(f"{comparison}: does each feature follow the wellness score?  "
                 "(black edge = p < 0.05; red = higher when sicker)", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=200)


def fig_top_within(corr, dev, st, color, out):
    top = corr["feature"].head(N_TOP).tolist()
    if not top or dev.empty:
        return
    ncol = min(3, len(top)); nrow = int(np.ceil(len(top) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.6 * nrow), squeeze=False)
    g = st.loc[dev.index, "group"].astype(str).to_numpy()
    for ax, f in zip(axes.flat, top):
        for gg in ["NPC", "EFV"]:
            m = g == gg
            ax.scatter(dev.loc[m, "score"], dev.loc[m, f], s=16, color=color[gg],
                       edgecolor="k", lw=0.3, alpha=0.8, label=gg)
        r = corr.set_index("feature").loc[f]
        ax.axhline(0, color="gray", lw=0.6); ax.axvline(0, color="gray", lw=0.6)
        ax.set_title(f"{f}\nwithin rho = {r['rho_within']:.2f}, p = {r['p_within']:.2g}", fontsize=8)
        ax.set_xlabel("score - mouse mean", fontsize=8); ax.set_ylabel("feature - mouse mean", fontsize=8)
    for ax in axes.flat[len(top):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Within each mouse: right = sicker than its own average", fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=200)


# -----------------------------------------------------------------------------
def load_tsne(ids, comparison):
    f = cfg.output_dir("tsne", comparison) / "tsne_embeddings.xlsx"
    cols = [f"tSNE_p{PERPLEXITY}_1", f"tSNE_p{PERPLEXITY}_2"]
    if f.exists():
        emb = pd.read_excel(f)
        if all(c in emb for c in cols):
            key = ids["animal"].astype(str) + "|" + ids["session"].astype(str)
            ekey = emb["animal"].astype(str) + "|" + emb["session"].astype(str)
            Y = emb.set_index(ekey)[cols].reindex(key).to_numpy()
            if not np.isnan(Y).any():
                print(f"[i] t-SNE taken from {f}")
                return Y
    return None


def run_comparison(comparison, scores, mice):
    out_dir = OUTPUT_ROOT / comparison
    out_dir.mkdir(parents=True, exist_ok=True)
    ids, feats = cfg.load_data(cfg.excel_for(comparison), cfg.SHEET, cfg.ID_COLS)
    ids = ids.rename(columns={cfg.GROUP_COL: "group"})
    groups_here = [g for g in GROUPS if g in set(ids["group"].astype(str))]
    color = {g: cfg.color_for(g, i) for i, g in enumerate(GROUPS)}
    print(f"{len(ids)} sessions x {feats.shape[1]} features; groups {groups_here}")

    st = session_table(ids, mice)
    clash = st[st["mouse"].map(mice.set_index("mouse")["group"]).fillna(st["group"]) != st["group"]]
    if len(clash):
        print(f"[!] Group differs from the wellness file for: {sorted(set(clash['animal']))}")

    # quick check: does the session score rise with the session's week in sick mice?
    s = st[st["group"].isin(["NPC", "EFV"])].dropna(subset=["est_week", "score_session"])
    if len(s) > 3:
        rho, p = stats.spearmanr(s["est_week"], s["score_session"])
        print(f"Check: session week vs session score (NPC+EFV): rho = {rho:.2f} (p = {p:.2g})")
        print(st.groupby("group")[["est_week", "score_old", "score_session"]].agg(["min", "max"])
              .round(1).to_string())

    fig_curve(mice[mice["group"].isin(groups_here)], st, color, groups_here,
              out_dir / "01_recordings_on_wellness_curve.png")

    X = StandardScaler().fit_transform(feats.to_numpy())
    Y = load_tsne(ids, comparison)
    if Y is None:
        Y = TSNE(n_components=2, perplexity=min(PERPLEXITY, len(X) - 1), init="pca",
                 learning_rate="auto", random_state=RANDOM_STATE).fit_transform(X)
    pca = PCA(n_components=2).fit(X)
    fig_maps(Y, pca.transform(X), pca.explained_variance_ratio_ * 100, st, color, groups_here,
             out_dir / "02_maps_group_and_intensity.png", comparison)
    fig_old_vs_new(st, color, groups_here, out_dir / "03_old_vs_new_score.png")

    corr, per_mouse, dev = correlations(feats, st)
    print("\nFEATURES vs WELLNESS (sorted by |rho_within|)")
    print(corr.head(12)[["feature", "rho_within", "p_within", "q_within",
                         "rho_between_NPC_EFV_mice", "p_between_NPC_EFV_mice"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    if corr["rho_within"].notna().any():
        fig_corr(corr, out_dir / "04_feature_vs_wellness.png", comparison)
        fig_top_within(corr, dev, st, color, out_dir / "05_top_features_within_mouse.png")
    else:
        print("[i] No NPC/EFV mice with enough sessions: figures 04-05 skipped")

    with pd.ExcelWriter(out_dir / "wellness_results.xlsx") as w:
        st.drop(columns=["score"]).assign(score_used=st["score"]).to_excel(w, sheet_name="sessions", index=False)
        mice.to_excel(w, sheet_name="per_mouse_scores", index=False)
        per_mouse.to_excel(w, sheet_name="per_mouse_features", index=False)
        corr.to_excel(w, sheet_name="feature_vs_wellness", index=False)
    if not cfg.SHOW_FIGURES:
        plt.close("all")
    print(f"Saved in {out_dir}")


def main():
    if not WELLNESS_FILE.exists():
        raise FileNotFoundError(f"{WELLNESS_FILE} not found: put Wellness-scale.xlsx there or change WELLNESS_FILE")
    scores = read_total_scores(WELLNESS_FILE)
    mice = per_mouse_scores(scores)
    print(f"Wellness: {len(mice)} mice; mean score per group and week:")
    print(mice.groupby("group")[[f"score_w{w}" for w in WEEKS]].mean().round(1).reindex(GROUPS).to_string())
    print(f"Session score mode: {SESSION_SCORE_MODE}")
    for c in COMPARISONS_TO_RUN:
        print("\n" + "=" * 70 + f"\n{c}\n" + "=" * 70)
        try:
            run_comparison(c, scores, mice)
        except FileNotFoundError as e:
            print(f"[!] {e}")
    if cfg.SHOW_FIGURES:
        plt.show()


if __name__ == "__main__":
    main()