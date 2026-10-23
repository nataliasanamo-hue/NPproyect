"""
Select the features that differ between study groups (WT / NPC / EFV)
=====================================================================
Independent script: reads RESULTSNatalia/features_final.xlsx, tests every feature,
and writes a new Excel with ONLY the selected features, in the same format, so the
PCA / t-SNE / UMAP / k-means scripts can use it just by changing their EXCEL_PATH.

HOW EACH FEATURE IS TESTED (and why)
------------------------------------
1. Sessions of the same mouse are NOT independent (same electrode, same brain).
   Treating 89 sessions as 89 independent mice makes p-values far too small
   ("pseudoreplication"). So the main test is a LINEAR MIXED MODEL:
        feature ~ group + (random effect of animal)
   = "is there a group difference, taking into account that sessions of the
   same mouse are similar?". p-value from a likelihood-ratio test with a
   small-sample correction (F distribution with df = n_animals - n_groups),
   because there are only ~16 mice.
   Two other tests are also reported, for comparison:
     - "animal":  Kruskal-Wallis on the MEDIAN of each mouse (16 values; safe but weak)
     - "session": Kruskal-Wallis on all sessions (what the old MATLAB did; too optimistic)
2. ~70 features are tested at once: by pure chance ~3-4 of them would give
   p < 0.05 even with NO real difference. So p-values are corrected with the
   False Discovery Rate (Benjamini-Hochberg) -> "q-value". By default a feature is
   selected if q < ALPHA (SELECT_BY = "fdr"); "p" and "top" are looser options
   for exploring.
3. Some features are the same measurement summarized twice (mean and max of the
   same band/layer/state; mean and median of the same CSD layer). With
   REDUNDANCY = "twins" only the more significant of each pair is kept, while
   every frequency band, layer and brain state stays separate.
   Every feature also gets an "evidence" label: strong (q < 0.05),
   moderate (q < 0.10), weak (only p < 0.05, not after correction).
4. For each selected feature, pairwise tests (WT-NPC, WT-EFV, NPC-EFV) tell you
   WHICH groups differ.

!! IMPORTANT: selecting features with the group labels and then looking at PCA /
clusters is partly CIRCULAR: the groups will separate better *because* we chose
features that separate them. Fine for exploring and describing, but for the final
machine-learning model the selection must be repeated INSIDE the cross-validation.

OUTPUT (RESULTSNatalia/feature_selection/)
  feature_tests.xlsx        every feature: group medians, p-values, q-values, selected?
  features_selected.xlsx    animal, session, group + selected features (sheet "features")
  01_significance.png       -log10(q) of every feature, by family
  02_selected_features.png  boxplots of the selected features by group
"""

import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2, kruskal, spearmanr
from scipy.stats import f as f_dist

# =============================================================================
# SETTINGS
# =============================================================================
baseDir = Path(r"D:\Violeta\ANALISIS")
RESULTS_DIR = baseDir / "RESULTSNatalia"
EXCEL_PATH = RESULTS_DIR / "features_final.xlsx"
SHEET = "features"
OUTPUT_DIR = RESULTS_DIR / "feature_selection"

ID_COLS = ["animal", "session", "group"]
GROUPS = ["WT", "NPC", "EFV"]
# not biological features (same rule as PCA_analysis.py)
EXCLUDE_PREFIXES = ["ch_", "config_", "power_pyr_channel", "power_slm_channel",
                    "power_channels_match", "n_"]
EXCLUDE_SUFFIXES = ["_n"]

TEST = "mixed"            # test used to SELECT: "mixed" (recommended), "animal" or "session"
SELECT_BY = "p"           # "p":   p-value < ALPHA  (screening: keeps every feature with a hint
                          #        of a group difference; some may be false positives)
                          # "fdr": q-value < ALPHA  (strict: only differences you can claim)
                          # "top": the TOP_N features with the smallest p
ALPHA = 0.05
TOP_N = 10
LOG_POWER = True          # test log(power): power values are very skewed
REDUNDANCY = "twins"      # "twins": only merge the SAME measurement summarized twice
                          #   (mean/max of one band+layer+state; mean/median of one CSD layer).
                          #   Every frequency band, layer and state is KEPT separately.
                          # "correlation": also merge any features with |r| > CORR_THRESHOLD
                          #   (can remove different bands/layers that move together!)
                          # "none": keep everything that passes the test
CORR_THRESHOLD = 0.90
EXCLUDE_ODD_SESSIONS = False   # True = leave out the odd sessions listed below
# The 11 odd sessions (flattened CSD + weak SLM power; see compare_odd_sessions.py).
# Edit this list if you decide to add or remove sessions.
ODD_SESSIONS = {
    "EFV_218": ["2026-01-16_17-50-35"],
    "EFV_220": ["2026-01-13_14-12-19", "2026-01-13_15-18-23", "2026-01-15_18-37-42"],
    "EFV_277": ["2026-02-20_12-02-47"],
    "NPC_249": ["2026-02-04_16-01-16", "2026-02-05_16-14-02", "2026-02-05_16-40-43"],
    "NPC_29":  ["P-7"],
    "NPC_30":  ["P4-HI"],
    "NPC_47":  ["P3-HD"],
}
N_PLOT = 24               # selected features shown as boxplots
GROUP_COLORS = {"WT": "#1f77b4", "NPC": "#d62728", "EFV": "#2ca02c"}
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


# -----------------------------------------------------------------------------
# Linear mixed model with a random intercept per animal (fitted by maximum
# likelihood). Written here so no extra package (statsmodels) is needed.
# -----------------------------------------------------------------------------
def _mixed_loglik(y, X, animal_idx):
    """Max log-likelihood of y = X b + u_animal + e, u ~ N(0, t2), e ~ N(0, s2)."""
    blocks = [np.where(animal_idx == a)[0] for a in np.unique(animal_idx)]

    def negll(theta):
        s2, t2 = np.exp(theta)
        XtVX = np.zeros((X.shape[1], X.shape[1]))
        XtVy = np.zeros(X.shape[1])
        logdet = 0.0
        parts = []
        for b in blocks:
            n = len(b)
            c = t2 / (s2 + n * t2)          # V^-1 = (I - c 11') / s2
            Xb, yb = X[b], y[b]
            sx, sy = Xb.sum(0), yb.sum()
            XtVX += (Xb.T @ Xb - c * np.outer(sx, sx)) / s2
            XtVy += (Xb.T @ yb - c * sx * sy) / s2
            logdet += (n - 1) * np.log(s2) + np.log(s2 + n * t2)
            parts.append((Xb, yb, c))
        beta = np.linalg.solve(XtVX, XtVy)
        quad = 0.0
        for Xb, yb, c in parts:
            r = yb - Xb @ beta
            quad += (r @ r - c * r.sum() ** 2) / s2
        return 0.5 * (logdet + quad + len(y) * np.log(2 * np.pi))

    best = None
    for start in ([0.0, -1.0], [-1.0, 0.0], [0.0, -6.0]):
        res = minimize(negll, start, method="L-BFGS-B", bounds=[(-12, 6), (-18, 6)])
        if best is None or res.fun < best.fun:
            best = res
    return -best.fun


def mixed_lrt(y, groups, animals):
    """p-value of 'group' in y ~ group + (1|animal), likelihood-ratio test."""
    y = np.asarray(y, float)
    ok = ~np.isnan(y)
    y, groups, animals = y[ok], np.asarray(groups)[ok], np.asarray(animals)[ok]
    levels = [g for g in pd.unique(groups)]
    if len(levels) < 2 or np.nanstd(y) == 0:
        return np.nan
    y = (y - y.mean()) / y.std()
    a_idx = pd.factorize(animals)[0]
    X0 = np.ones((len(y), 1))
    X1 = np.column_stack([X0] + [(groups == g).astype(float) for g in levels[1:]])
    try:
        lr = 2 * (_mixed_loglik(y, X1, a_idx) - _mixed_loglik(y, X0, a_idx))
    except np.linalg.LinAlgError:
        return np.nan
    # Small-sample correction: with only ~16 mice the plain chi-square test is too
    # optimistic. Compare LR/df1 with an F distribution whose 2nd df = number of
    # animals - number of groups (the real amount of independent information).
    df1 = len(levels) - 1
    df2 = len(np.unique(a_idx)) - len(levels)
    if df2 < 1:
        return float(chi2.sf(max(lr, 0.0), df=df1))
    return float(f_dist.sf(max(lr, 0.0) / df1, df1, df2))


def kw(values, groups):
    samples = [values[groups == g] for g in pd.unique(groups)]
    samples = [s[~np.isnan(s)] for s in samples]
    samples = [s for s in samples if len(s) > 0]
    if len(samples) < 2 or np.nanstd(np.concatenate(samples)) == 0:
        return np.nan
    return float(kruskal(*samples).pvalue)


def bh_fdr(p):
    """Benjamini-Hochberg q-values (NaN stays NaN)."""
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    n = len(pv)
    if n == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1)
    q[ok] = out
    return q


def twin_key(name):
    """Features with the same key are the same measurement summarized differently."""
    m = POWER_RE.match(name)
    if m:                                   # band, layer, state  (mean / max are twins)
        return ("power", m.group(1).lower(), m.group(2).upper(), m.group(3).lower())
    m = re.match(r"^CSD_([A-Za-z]+)_(mean|median|std)$", name, re.IGNORECASE)
    if m:                                   # mean / median are twins, std is different
        kind = "spread" if m.group(2).lower() == "std" else "centre"
        return ("CSD", m.group(1).upper(), kind)
    return (name,)


def cohen_d(a, b):
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan


# -----------------------------------------------------------------------------
def load():
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET)
    df = df[df["group"].astype(str).isin(GROUPS)].reset_index(drop=True)

    if EXCLUDE_ODD_SESSIONS:
        wanted = {f"{a.strip()}|{s.strip()}" for a, ss in ODD_SESSIONS.items() for s in ss}
        key = (df["animal"].astype(str).str.strip() + "|" + df["session"].astype(str).str.strip())
        found = set(key[key.isin(wanted)])
        print(f"[i] EXCLUDE_ODD_SESSIONS = True: leaving out {len(found)} of the {len(wanted)} listed odd sessions")
        for k in sorted(wanted - found):
            print(f"    [!] not found in the Excel (check the name): {k.replace('|', '  ')}")
        df = df[~key.isin(wanted)].reset_index(drop=True)
    else:
        print("[i] EXCLUDE_ODD_SESSIONS = False: using ALL sessions")

    cand = [c for c in df.columns if c not in ID_COLS
            and not c.startswith(tuple(EXCLUDE_PREFIXES)) and not c.endswith(tuple(EXCLUDE_SUFFIXES))]
    feats = df[cand].apply(pd.to_numeric, errors="coerce")
    feats = feats.loc[:, feats.notna().sum() >= 10]                 # enough data
    feats = feats.loc[:, feats.std(skipna=True) > 0]                # not constant
    return df, feats


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df, feats = load()
    groups = df["group"].astype(str).to_numpy()
    animals = df["animal"].astype(str).to_numpy()
    n_anim = df.groupby("group")["animal"].nunique().reindex(GROUPS)
    print(f"{len(df)} sessions, {feats.shape[1]} features | animals per group: {n_anim.to_dict()}")
    rule = {"fdr": f"q < {ALPHA}", "p": f"p < {ALPHA} (no correction)", "top": f"top {TOP_N} by p"}[SELECT_BY]
    print(f"Selecting with test = '{TEST}', rule: {rule}\n")

    rows = []
    for i, f in enumerate(feats.columns, 1):
        x = feats[f].to_numpy(float)
        if LOG_POWER and family(f) == "power" and np.nanmin(x) > 0:
            x_test, transf = np.log10(x), "log10"
        else:
            x_test, transf = x, ""

        # per-animal medians for the "animal" test
        per_animal = (pd.DataFrame({"x": x_test, "g": groups, "a": animals})
                      .groupby(["a", "g"])["x"].median().reset_index())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = {
                "feature": f, "family": family(f), "tested_as": transf,
                **{f"median_{g}": np.nanmedian(x[groups == g]) for g in GROUPS},
                "p_mixed": mixed_lrt(x_test, groups, animals),
                "p_animal": kw(per_animal["x"].to_numpy(), per_animal["g"].to_numpy()),
                "p_session": kw(x_test, groups),
            }
        rows.append(r)
        print(f"\r  testing {i}/{feats.shape[1]}", end="")
    print()

    res = pd.DataFrame(rows)
    for t in ("mixed", "animal", "session"):
        res[f"q_{t}"] = bh_fdr(res[f"p_{t}"])
    res = res.sort_values(f"p_{TEST}").reset_index(drop=True)
    if SELECT_BY == "fdr":
        res["significant"] = res[f"q_{TEST}"] < ALPHA
    elif SELECT_BY == "p":
        res["significant"] = res[f"p_{TEST}"] < ALPHA
    else:
        res["significant"] = res.index < TOP_N

    # ---- pairwise comparisons (only for significant features) ----
    pairs = [("WT", "NPC"), ("WT", "EFV"), ("NPC", "EFV")]
    for a, b in pairs:
        res[f"p_{a}_vs_{b}"] = np.nan
        res[f"d_{a}_vs_{b}"] = np.nan
    for idx, r in res[res["significant"]].iterrows():
        x = feats[r["feature"]].to_numpy(float)
        if r["tested_as"] == "log10":
            x = np.log10(x)
        for a, b in pairs:
            m = np.isin(groups, [a, b])
            if TEST == "mixed":
                p = mixed_lrt(x[m], groups[m], animals[m])
            elif TEST == "animal":
                pa = pd.DataFrame({"x": x[m], "g": groups[m], "a": animals[m]}).groupby(["a", "g"])["x"].median().reset_index()
                p = kw(pa["x"].to_numpy(), pa["g"].to_numpy())
            else:
                p = kw(x[m], groups[m])
            res.loc[idx, f"p_{a}_vs_{b}"] = p
            res.loc[idx, f"d_{a}_vs_{b}"] = cohen_d(x[groups == a], x[groups == b])

    # ---- redundancy among significant features ----
    res["redundant_with"] = ""
    kept = []
    for idx, r in res[res["significant"]].iterrows():      # most significant first
        f = r["feature"]
        twin = None
        if REDUNDANCY in ("twins", "correlation"):
            twin = next((k for k in kept if twin_key(k) == twin_key(f)), None)
        if twin is None and REDUNDANCY == "correlation":
            for k in kept:
                rho = spearmanr(feats[f], feats[k], nan_policy="omit").correlation
                if abs(rho) > CORR_THRESHOLD:
                    twin = k
                    break
        if twin is None:
            kept.append(f)
        else:
            res.loc[idx, "redundant_with"] = twin
    res["selected"] = res["feature"].isin(kept)
    q = res[f"q_{TEST}"]
    res["evidence"] = np.select([q < 0.05, q < 0.10, res[f"p_{TEST}"] < 0.05],
                                ["strong", "moderate", "weak"], default="")

    # ---- report ----
    n_sig = int(res["significant"].sum())
    print("\nTOP 10 features by p-value (whether selected or not):")
    print(res[["feature", "family", f"p_{TEST}", f"q_{TEST}", "significant"]].head(10)
          .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"\nPassing the rule: {n_sig} of {len(res)}")
    print(f"Selected after merging twins (REDUNDANCY = '{REDUNDANCY}'): {len(kept)}")
    print("Evidence of the selected ones: " +
          ", ".join(f"{k} {v}" for k, v in res.loc[res["selected"], "evidence"].value_counts().items()))
    print("By family (significant / tested):")
    print(res.groupby("family")["significant"].agg(["sum", "count"]).to_string())
    print(f"\nNumber of features with q < {ALPHA} / p < {ALPHA} for each test (to compare):")
    for t in ("mixed", "animal", "session"):
        print(f"  {t:8s}: {int((res[f'q_{t}'] < ALPHA).sum()):3d} / {int((res[f'p_{t}'] < ALPHA).sum()):3d}"
              + ("   <- too optimistic: treats sessions of a mouse as independent" if t == "session" else ""))
    if kept:
        cols = ["feature", f"p_{TEST}", f"q_{TEST}", "evidence"] + [f"median_{g}" for g in GROUPS] + \
               [f"p_{a}_vs_{b}" for a, b in pairs]
        for fam in ("ripple", "CSD", "power", "other"):
            sel = res[res["selected"] & (res["family"] == fam)]
            if len(sel):
                print(f"\nSELECTED {fam.upper()} FEATURES ({len(sel)}):")
                print(sel[cols].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
        print_power_map(res)

    with pd.ExcelWriter(OUTPUT_DIR / "feature_tests.xlsx") as w:
        res.to_excel(w, sheet_name="all_features", index=False)
        pd.DataFrame({"setting": ["TEST", "SELECT_BY", "ALPHA", "TOP_N", "LOG_POWER", "REDUNDANCY",
                                  "CORR_THRESHOLD", "EXCLUDE_ODD_SESSIONS", "n_sessions"],
                      "value": [TEST, SELECT_BY, ALPHA, TOP_N, LOG_POWER, REDUNDANCY,
                                CORR_THRESHOLD, EXCLUDE_ODD_SESSIONS, len(df)]}
                     ).to_excel(w, sheet_name="settings", index=False)

    if len(kept) >= 2:
        out = pd.concat([df[ID_COLS], feats[kept]], axis=1)
        out.to_excel(OUTPUT_DIR / "features_selected.xlsx", sheet_name="features", index=False)
        print(f"\nfeatures_selected.xlsx written with {len(kept)} features")
    else:
        print(f"\n[!] Only {len(kept)} feature(s) selected: features_selected.xlsx NOT written "
              "(PCA / clustering need at least 2). To explore anyway, try SELECT_BY = 'p' or 'top', "
              "or ALPHA = 0.10, knowing they are less strict.")

    plot_significance(res)
    if kept:
        plot_selected(res, feats, groups, animals, kept[:N_PLOT])
    print(f"Results in {OUTPUT_DIR}")
    plt.show()


# -----------------------------------------------------------------------------
def print_power_map(res):
    """Grid band x (layer, state): which power features differ (any summary)."""
    rows = []
    for _, r in res.iterrows():
        m = POWER_RE.match(r["feature"])
        if m:
            rows.append({"band": m.group(1), "where": f"{m.group(2).upper()}-{m.group(3)}",
                         "sig": r["significant"], "ev": r["evidence"]})
    if not rows:
        return
    d = pd.DataFrame(rows)
    order = ["Delta", "Theta", "Alpha", "SlowGamma", "MidGamma", "FastGamma", "Ripple"]
    mark = {"strong": "***", "moderate": "**", "weak": "*", "": "."}
    rank = {"strong": 3, "moderate": 2, "weak": 1, "": 0}
    best = (d.assign(r=d["ev"].map(rank)).groupby(["band", "where"])["r"].max()
            .unstack().reindex([b for b in order if b in set(d["band"])]))
    inv = {v: k for k, v in rank.items()}
    print("\nPOWER differences by band and layer-state  (*** strong, ** moderate, * weak, . none):")
    print(best.apply(lambda col: col.map(lambda v: mark[inv[int(v)]] if pd.notna(v) else " ")).to_string())


def plot_significance(res):
    col = f"{'q' if SELECT_BY == 'fdr' else 'p'}_{TEST}"
    r = res.sort_values(["family", "feature"]).reset_index(drop=True)
    fam_colors = {"ripple": "#7b3294", "CSD": "#e66101", "power": "#018571", "other": "gray"}
    fig, ax = plt.subplots(figsize=(max(8, 0.16 * len(r)), 4.5))
    ax.bar(range(len(r)), -np.log10(r[col].clip(lower=1e-12)),
           color=[fam_colors[f] for f in r["family"]],
           edgecolor=["k" if s else "none" for s in r["selected"]])
    ax.axhline(-np.log10(ALPHA), color="k", ls="--", lw=1, label=f"{col.split('_')[0]} = {ALPHA}")
    ax.set_xticks(range(len(r)), r["feature"], rotation=90, fontsize=5)
    ax.set_ylabel(f"-log10({col.split('_')[0]})   (higher = more different)")
    ax.set_title(f"Group differences per feature (test: {TEST}); black edge = selected")
    present = [f for f in fam_colors if f in set(r["family"])]
    handles = [plt.Rectangle((0, 0), 1, 1, color=fam_colors[f]) for f in present]
    ax.legend(handles + [ax.get_lines()[0]], present + [f"threshold {ALPHA}"], fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_significance.png", dpi=200)


def plot_selected(res, feats, groups, animals, show):
    ncol = min(4, len(show))
    nrow = int(np.ceil(len(show) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.3 * nrow), squeeze=False)
    rng = np.random.default_rng(0)
    for n, f in enumerate(show):
        ax = axes.flat[n]
        x = feats[f].to_numpy(float)
        for gi, g in enumerate(GROUPS):
            m = groups == g
            v = x[m]
            ax.boxplot(v[~np.isnan(v)], positions=[gi], widths=0.55, showfliers=False)
            ax.scatter(gi + rng.uniform(-0.18, 0.18, m.sum()), v, s=8, alpha=0.45,
                       color=GROUP_COLORS[g], edgecolor="none")
            am = pd.Series(v).groupby(animals[m]).median()      # one diamond per mouse
            ax.scatter(np.full(len(am), gi + 0.32), am, marker="D", s=22,
                       color=GROUP_COLORS[g], edgecolor="k", linewidth=0.5, zorder=3)
        r = res.set_index("feature").loc[f]
        ax.set_xticks(range(len(GROUPS)), GROUPS)
        if r["tested_as"] == "log10":
            ax.set_yscale("log")
        ax.set_title(f"{f}\np = {r[f'p_{TEST}']:.2g}, q = {r[f'q_{TEST}']:.2g}", fontsize=8)
    for ax in axes.flat[len(show):]:
        ax.axis("off")
    fig.suptitle("Selected features (dots = sessions, diamonds = median of each mouse)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_selected_features.png", dpi=200)


if __name__ == "__main__":
    main()