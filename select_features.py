"""
Select the features that differ between study groups, ONE EXCEL PER COMPARISON
==============================================================================
The selection uses the SAME statistics as the group_comparison figures:
it reads RESULTSNatalia/group_comparison/group_comparison_pvalues.csv (made by
plot_group_comparison.py: one-way ANOVA of the 3 groups on sessions + Tukey-Kramer)
and keeps, for each comparison, every feature that has a star in the figures:

    WT_vs_NPC         -> star in the WT-NPC bracket   (p_tukey_WT_vs_NPC  < ALPHA)
    NPC_vs_EFV        -> star in the NPC-EFV bracket  (p_tukey_NPC_vs_EFV < ALPHA)
    WT_vs_EFV         -> star in the WT-EFV bracket   (p_tukey_WT_vs_EFV  < ALPHA)
    WT_vs_NPC_vs_EFV  -> a star in ANY of the three brackets

Every frequency band, layer and brain state is kept separately. Two settings decide
what happens with mean and max of the same band/layer/state (and mean and median of
the same CSD layer), which are the same measurement summarized twice:
    KEEP_BOTH = True  -> every feature with a star is kept, mean AND max (default)
    KEEP_BOTH = False -> only one of them: the mean when both have a star, the max
                         (or the CSD median) when only that one has a star
    USE_MAX = False   -> the max of power is never selected (only the mean)
The CSD std is a different measurement (ripple-to-ripple variability), so it is
always treated on its own.

ORDER OF THE SCRIPTS (every time features_final changes, e.g. new mice):
    1. build_feature_table.py     -> features_final.xlsx / .csv
    2. plot_group_comparison.py   -> figures + group_comparison_pvalues.csv
    3. select_features.py         -> this script
If group_comparison_pvalues.csv is OLDER than features_final, this script stops and
tells you to run plot_group_comparison.py first (CHECK_DATES = True).

!! IMPORTANT: the figures' test treats sessions as independent, so it finds more
differences than a test per mouse would (some will be false positives). That is fine
for an exploratory selection; for the honest accuracy of the models, the selection
is repeated INSIDE the cross-validation.

THE SAME RULE INSIDE THE MODELS (no leakage)
select_in_fold() applies exactly this rule, but recomputing the ANOVA + Tukey only on the
sessions it is given (e.g. all the mice except the one hidden in a fold).
feature_stability.py and the classifiers import it from here, so this file must stay in
the same folder as them. Running this script is not affected by it.

INPUT : RESULTSNatalia/features_final.xlsx (sheet "features")
        RESULTSNatalia/group_comparison/group_comparison_pvalues.csv
OUTPUT: RESULTSNatalia/feature_selection/, one set per comparison <C>:
        features_selected_<C>.xlsx   animal, session, group + selected features
                                     (sheet "features"; only the sessions of the
                                     groups of that comparison)
        feature_tests_<C>.xlsx       every feature: p-values, star, selected or why not,
                                     + sheet "settings" with the settings of this run
        The OLD features_selected.xlsx (used by PCA / t-SNE / k-means) is never touched.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# SETTINGS
# =============================================================================
baseDir = Path(r"D:\Violeta\ANALISIS")
RESULTS_DIR = baseDir / "RESULTSNatalia"
FEATURES_EXCEL = RESULTS_DIR / "features_final.xlsx"
FEATURES_CSV = RESULTS_DIR / "features_final.csv"       # only used for the date check
SHEET = "features"
PVALUES_CSV = RESULTS_DIR / "group_comparison" / "group_comparison_pvalues.csv"
OUTPUT_DIR = RESULTS_DIR / "feature_selection"

# Comparisons to run, one after the other (each one writes its own files).
COMPARISONS_TO_RUN = ["WT_vs_NPC", "NPC_vs_EFV", "WT_vs_EFV", "WT_vs_NPC_vs_EFV"]

KEEP_BOTH = True     # True  = keep EVERY feature with a star: mean AND max (and CSD mean AND
                     #         median) are both kept when both have a star
                     # False = only one per measurement (the mean when both have a star)
USE_MAX = True       # True  = the max of power can be selected
                     # False = the max of power is never selected (only the mean)
                     # (feature_stability.py and the classifiers use these same settings)
ALPHA = 0.05         # a "star" = Tukey p < ALPHA (the same as in the figures)
CHECK_DATES = True   # stop if group_comparison_pvalues.csv is older than features_final
# =============================================================================

# groups of each comparison + the Tukey columns of group_comparison_pvalues.csv it uses
COMPARISONS = {
    "WT_vs_NPC": (["WT", "NPC"], ["p_tukey_WT_vs_NPC"]),
    "NPC_vs_EFV": (["NPC", "EFV"], ["p_tukey_NPC_vs_EFV"]),
    "WT_vs_EFV": (["WT", "EFV"], ["p_tukey_WT_vs_EFV"]),
    "WT_vs_NPC_vs_EFV": (["WT", "NPC", "EFV"],
                         ["p_tukey_WT_vs_NPC", "p_tukey_WT_vs_EFV", "p_tukey_NPC_vs_EFV"]),
}
ID_COLS = ["animal", "session", "group"]
# columns of features_final that are not biological features (same rule as the models)
EXCLUDE_PREFIXES = ["ch_", "config_", "power_pyr_channel", "power_slm_channel",
                    "power_channels_match", "n_"]
EXCLUDE_SUFFIXES = ["_n"]
# when two summaries of the same measurement both have a star, the first one in this list wins
PREFERENCE = ["mean", "median", "max"]


def twin_key(feature):
    """Features with the same key are the same measurement summarized differently
    (power: mean / max of one band+layer+state; CSD: mean / median of one layer).
    CSD std and the ripple features have their own key."""
    m = re.match(r"^(.+)_(mean|max)$", feature)
    if m and not feature.startswith(("CSD_", "ripple_")):
        return m.group(1)
    m = re.match(r"^(CSD_[A-Za-z]+)_(mean|median)$", feature)
    if m:
        return m.group(1)
    return feature


def summary_of(feature):
    m = re.search(r"_(mean|median|max)$", feature)
    return m.group(1) if m else ""


def star_text(p):
    if pd.isna(p) or p >= ALPHA:
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*"


# -----------------------------------------------------------------------------
# THE SELECTION RULE (used here AND inside every fold of feature_stability.py and
# the classifiers, so all of them choose features in exactly the same way)
# -----------------------------------------------------------------------------
def decide(res, use_max=None, keep_both=None):
    """res: table with columns feature, summary, p_used, star (one row per feature).
    Adds 'selected' and 'reason' following the rule: a star -> candidate; with keep_both
    every candidate is kept; otherwise mean/max (and mean/median for CSD) of the same
    measurement -> only one is kept (PREFERENCE order). With use_max False the power max
    is never selected."""
    use_max = USE_MAX if use_max is None else use_max
    keep_both = KEEP_BOTH if keep_both is None else keep_both
    res = res.copy()
    res["selected"] = False
    res["reason"] = ""
    res.loc[res["star"] == "", "reason"] = "no star"
    if not use_max:
        is_max = (res["summary"] == "max") & ~res["feature"].str.startswith(("CSD_", "ripple_"))
        res.loc[is_max & (res["star"] != ""), "reason"] = "max not used (USE_MAX = False)"
        candidates = res[(res["star"] != "") & ~is_max]
    else:
        candidates = res[res["star"] != ""]

    if keep_both:                     # every feature with a star is kept
        res.loc[candidates.index, "selected"] = True
        res.loc[candidates.index, "reason"] = "selected"
        return res

    for key, grp in candidates.groupby(candidates["feature"].map(twin_key), sort=False):
        order = sorted(grp.index, key=lambda i: PREFERENCE.index(res.at[i, "summary"])
                       if res.at[i, "summary"] in PREFERENCE else -1)
        winner = order[0]
        res.at[winner, "selected"] = True
        res.at[winner, "reason"] = "selected"
        for i in order[1:]:
            res.at[i, "reason"] = f"same measurement as {res.at[winner, 'feature']} (kept that one)"
    return res


def plotted_features(columns):
    """The features that have a figure in plot_group_comparison.py (same list, same order):
    power {Band}_{PYR|SLM}_{Theta|LIA}_{mean,max}, CSD_{layer}_{mean,median,std},
    ripple frequency / amplitude / power."""
    cols = list(columns)
    out = []
    for c in cols:
        m = re.match(r"^([A-Za-z]+)_(PYR|SLM)_(Theta|LIA)_mean$", c)
        if m:
            out += [f for f in (c, c[:-len("mean")] + "max") if f in cols]
    for layer in ("SO", "PYR", "RAD", "SLM"):
        out += [f for f in (f"CSD_{layer}_mean", f"CSD_{layer}_median", f"CSD_{layer}_std") if f in cols]
    out += [f for f in ("ripple_frequency_mean", "ripple_amplitude_mean", "ripple_power_mean") if f in cols]
    return out


def tukey_table(df):
    """The same statistics as plot_group_comparison.py (one-way ANOVA of WT, NPC, EFV on
    sessions + Tukey-Kramer, raw values, NaN ignored), computed on the sessions in df.
    Returns one row per feature with p_tukey_WT_vs_NPC, p_tukey_WT_vs_EFV, p_tukey_NPC_vs_EFV."""
    from itertools import combinations
    from scipy.stats import tukey_hsd
    rows = []
    groups = df["group"].to_numpy()
    for f in plotted_features(df.columns):
        x = pd.to_numeric(df[f], errors="coerce").to_numpy(float)
        names = [g for g in ("WT", "NPC", "EFV") if np.any((groups == g) & ~np.isnan(x))]
        arrays = [x[(groups == g) & ~np.isnan(x)] for g in names]
        row = {"feature": f}
        if len(names) >= 2 and np.ptp(np.concatenate(arrays)) > 0:
            try:
                res = tukey_hsd(*arrays)
                for i, j in combinations(range(len(names)), 2):
                    row[f"p_tukey_{names[i]}_vs_{names[j]}"] = float(res.pvalue[i, j])
            except ValueError:
                pass
        rows.append(row)
    out = pd.DataFrame(rows)
    for c in ("p_tukey_WT_vs_NPC", "p_tukey_WT_vs_EFV", "p_tukey_NPC_vs_EFV"):
        if c not in out:
            out[c] = np.nan
    return out


_Q_CRIT = {}


def _q_crit(k, dof):
    """Critical value of the studentized range for p = ALPHA (computed once per k, dof)."""
    from scipy.stats import studentized_range
    key = (k, dof, ALPHA)
    if key not in _Q_CRIT:
        _Q_CRIT[key] = float(studentized_range.ppf(1 - ALPHA, k, dof))
    return _Q_CRIT[key]


def star_table(df):
    """FAST version of tukey_table for the selection inside the folds: same ANOVA + Tukey-Kramer,
    but instead of the (slow) exact p-value it compares each Tukey statistic with the critical
    value for p = ALPHA. The decision "p < ALPHA" is exactly the same.
    Returns one row per feature with True/False columns star_WT_vs_NPC, star_WT_vs_EFV,
    star_NPC_vs_EFV and q_<pair> (the Tukey statistic; bigger = more different)."""
    from itertools import combinations
    groups = df["group"].to_numpy()
    rows = []
    for f in plotted_features(df.columns):
        x = pd.to_numeric(df[f], errors="coerce").to_numpy(float)
        names = [g for g in ("WT", "NPC", "EFV") if np.any((groups == g) & ~np.isnan(x))]
        arrays = [x[(groups == g) & ~np.isnan(x)] for g in names]
        row = {"feature": f}
        n_tot, k = sum(len(a) for a in arrays), len(names)
        if k >= 2 and n_tot > k and np.ptp(np.concatenate(arrays)) > 0:
            dof = n_tot - k
            mse = sum(((a - a.mean()) ** 2).sum() for a in arrays) / dof
            qc = _q_crit(k, dof)
            for i, j in combinations(range(k), 2):
                a, b = arrays[i], arrays[j]
                q = abs(a.mean() - b.mean()) / np.sqrt(mse / 2 * (1 / len(a) + 1 / len(b)))
                row[f"q_{names[i]}_vs_{names[j]}"] = q
                row[f"star_{names[i]}_vs_{names[j]}"] = q > qc
        rows.append(row)
    out = pd.DataFrame(rows)
    for pair in ("WT_vs_NPC", "WT_vs_EFV", "NPC_vs_EFV"):
        if f"q_{pair}" not in out:
            out[f"q_{pair}"] = np.nan
        if f"star_{pair}" not in out:
            out[f"star_{pair}"] = False
        out[f"star_{pair}"] = out[f"star_{pair}"].fillna(False).astype(bool)
    return out


def select_in_fold(df_train, comparison, use_max=None, keep_both=None):
    """The selection of select_features.py, but computed ONLY on the sessions of df_train
    (e.g. every mouse except the hidden one). Option A: the Tukey test uses the three groups
    that are in df_train, and the star of the comparison is taken from it.
    Returns the list of selected features (never empty: if no feature has a star, the one
    with the largest Tukey statistic is returned, so a model can still be trained)."""
    pairs = [c.replace("p_tukey_", "") for c in COMPARISONS[comparison][1]]
    st = star_table(df_train)
    res = st[["feature"]].copy()
    res["summary"] = res["feature"].map(summary_of)
    res["star"] = np.where(st[[f"star_{p}" for p in pairs]].any(axis=1), "*", "")
    res = decide(res, use_max, keep_both)
    kept = list(res.loc[res["selected"], "feature"])
    if not kept:
        q = st[[f"q_{p}" for p in pairs]].max(axis=1)
        if q.notna().any():
            kept = [st.loc[q.idxmax(), "feature"]]
    return kept


# -----------------------------------------------------------------------------
def check_inputs():
    for f in (FEATURES_EXCEL, PVALUES_CSV):
        if not f.exists():
            raise SystemExit(f"[!] File not found: {f}")
    if CHECK_DATES:
        features_time = max(f.stat().st_mtime for f in (FEATURES_EXCEL, FEATURES_CSV) if f.exists())
        if PVALUES_CSV.stat().st_mtime < features_time:
            raise SystemExit(
                f"[!] {PVALUES_CSV.name} is OLDER than features_final: the p-values may not match\n"
                f"    the current data. Run plot_group_comparison.py first, then this script again.\n"
                f"    (To skip this check, set CHECK_DATES = False.)")


def load():
    df = pd.read_excel(FEATURES_EXCEL, sheet_name=SHEET)
    df["group"] = df["group"].astype(str).str.strip()
    df["animal"] = df["animal"].astype(str).str.strip()
    pv = pd.read_csv(PVALUES_CSV)

    # features of the table that have no figure (they can never be selected)
    feats = [c for c in df.columns if c not in ID_COLS
             and not c.startswith(tuple(EXCLUDE_PREFIXES)) and not c.endswith(tuple(EXCLUDE_SUFFIXES))]
    no_figure = [c for c in feats if c not in set(pv["feature"])]
    missing = [c for c in pv["feature"] if c not in df.columns]
    if no_figure:
        print(f"[i] {len(no_figure)} feature(s) of features_final have no figure in group_comparison, "
              f"so they cannot be selected: {', '.join(no_figure)}")
    if missing:
        print(f"[!] {len(missing)} feature(s) of the csv are not in features_final (ignored): {', '.join(missing)}")
        pv = pv[~pv["feature"].isin(missing)]
    return df, pv.reset_index(drop=True)


def run_comparison(name, df, pv):
    groups, pcols = COMPARISONS[name]
    res = pv[["figure", "feature"]].copy()
    res["summary"] = res["feature"].map(summary_of)
    res["p_used"] = pv[pcols].min(axis=1)                    # for 3 groups: the best bracket
    res["star"] = res["p_used"].map(star_text)
    for c in ["p_anova", "kw_p", "p_tukey_WT_vs_NPC", "p_tukey_WT_vs_EFV", "p_tukey_NPC_vs_EFV"]:
        res[c] = pv[c]
    for g in groups:
        res[f"n_{g}"] = pv[f"n_{g}"]
        res[f"mean_{g}"] = pv[f"mean_{g}"]

    # ---- decide feature by feature (the rule lives in decide()) ----
    res = decide(res)
    kept = list(res.loc[res["selected"], "feature"])

    # ---- report ----
    print(f"Features with a star: {int((res['star'] != '').sum())} of {len(res)} | selected: {len(kept)}")
    for fam, prefix in (("ripple", "ripple_"), ("CSD", "CSD_")):
        sel = [f for f in kept if f.startswith(prefix)]
        if sel:
            print(f"  {fam:7s}: {', '.join(sel)}")
    power = [f for f in kept if not f.startswith(("ripple_", "CSD_"))]
    if power:
        print(f"  power  : {', '.join(power)}")

    # ---- save ----
    sessions = df[df["group"].isin(groups)].reset_index(drop=True)
    n_anim = sessions.groupby("group")["animal"].nunique().reindex(groups)
    sel_file = OUTPUT_DIR / f"features_selected_{name}.xlsx"
    tests_file = OUTPUT_DIR / f"feature_tests_{name}.xlsx"
    if kept:
        sessions[ID_COLS + kept].to_excel(sel_file, sheet_name="features", index=False)
        print(f"  -> {sel_file.name}: {len(sessions)} sessions, {len(kept)} features")
    else:
        if sel_file.exists():
            sel_file.unlink()   # don't leave an old list that no longer matches
        print(f"  [!] no feature with a star: {sel_file.name} NOT written")

    res = res.sort_values(["selected", "p_used"], ascending=[False, True])
    with pd.ExcelWriter(tests_file) as w:
        res.to_excel(w, sheet_name="all_features", index=False)
        pd.DataFrame({
            "setting": ["COMPARISON", "groups", "Tukey column(s) used", "KEEP_BOTH", "USE_MAX", "ALPHA",
                        "p-values file", "p-values file date", "sessions", "animals per group",
                        "features selected"],
            "value": [name, ", ".join(groups), ", ".join(pcols), KEEP_BOTH, USE_MAX, ALPHA,
                      str(PVALUES_CSV),
                      pd.Timestamp(PVALUES_CSV.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M"),
                      len(sessions), str(n_anim.to_dict()), len(kept)],
        }).to_excel(w, sheet_name="settings", index=False)

    return {"comparison": name, "sessions": len(sessions),
            "animals": ", ".join(f"{g} {n}" for g, n in n_anim.items()),
            "with_star": int((res["star"] != "").sum()), "selected": len(kept)}


def main():
    check_inputs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df, pv = load()
    print(f"KEEP_BOTH = {KEEP_BOTH} | USE_MAX = {USE_MAX} | star = Tukey p < {ALPHA} | p-values from {PVALUES_CSV.name}")
    summary = []
    for name in COMPARISONS_TO_RUN:
        print("\n" + "=" * 70 + f"\n{name}\n" + "=" * 70)
        summary.append(run_comparison(name, df, pv))
    print("\n" + "=" * 70 + "\nSUMMARY\n" + "=" * 70)
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"\nResults in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()