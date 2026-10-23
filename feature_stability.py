"""
Stability of the feature selection (leave-one-mouse-out)
========================================================
How robust is the selection of select_features.py? For each comparison, the selection
is repeated once per mouse, each time LEAVING OUT all the sessions of that mouse, with
exactly the same rule (select_in_fold of select_features.py: ANOVA + Tukey of the three
groups on sessions, star for the pair, mean/max rule with KEEP_BOTH and USE_MAX).

    WT_vs_NPC, NPC_vs_EFV, WT_vs_EFV : one repetition per mouse of the two groups
    WT_vs_NPC_vs_EFV                 : one repetition per mouse of the three groups

For every feature we count in how many repetitions it was selected:
    robust   : selected in >= 90 % of the repetitions (does not depend on any mouse)
    moderate : 50 - 90 %
    fragile  : < 50 %   (a few mice decide whether it appears)
and which mice make it disappear when they are left out.

These are the same selections the classifiers make in their folds (one per hidden
mouse), so this shows in advance which features the models will rely on.
No model is trained here.

INPUT : RESULTSNatalia/features_final.xlsx  (all features; the p-values are recomputed
        in every repetition, they are NOT taken from group_comparison_pvalues.csv)
OUTPUT: RESULTSNatalia/feature_stability/
        feature_stability.xlsx     sheet "summary" (feature x comparison, % of repetitions)
                                   + one sheet per comparison (details per feature)
                                   + sheet "selections" (features chosen in each repetition)
        01_stability_heatmap.png   features x comparisons, % of repetitions
        02_stability_<C>.png       one bar chart per comparison

Keep this file in the same folder as select_features.py (it imports the rule from there).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import select_features as sf   # the selection rule + paths + USE_MAX (same folder)

# =============================================================================
# SETTINGS
# =============================================================================
COMPARISONS_TO_RUN = ["WT_vs_NPC", "NPC_vs_EFV", "WT_vs_EFV", "WT_vs_NPC_vs_EFV"]
ROBUST, MODERATE = 0.90, 0.50      # thresholds of the labels (fraction of repetitions)
OUTPUT_DIR = sf.RESULTS_DIR / "feature_stability"
SHOW_FIGURES = False               # True = open the figures at the end (always saved)
# KEEP_BOTH, USE_MAX and ALPHA are taken from select_features.py, so both scripts always agree.
# =============================================================================


def label(frac):
    if frac >= ROBUST:
        return "robust"
    return "moderate" if frac >= MODERATE else "fragile"


def load():
    df = pd.read_excel(sf.FEATURES_EXCEL, sheet_name=sf.SHEET)
    df["group"] = df["group"].astype(str).str.strip()
    df["animal"] = df["animal"].astype(str).str.strip()
    return df[df["group"].isin(["WT", "NPC", "EFV"])].reset_index(drop=True)


def run_comparison(name, df):
    groups = sf.COMPARISONS[name][0]
    mice = sorted(df.loc[df["group"].isin(groups), "animal"].unique())

    reference = sf.select_in_fold(df, name)          # all mice = what select_features.py selects
    selections = []
    print("  ", end="")
    for i, mouse in enumerate(mice, 1):
        train = df[df["animal"] != mouse]            # option A: the other two/three groups stay
        kept = sf.select_in_fold(train, name)
        selections.append({"comparison": name, "left_out_mouse": mouse,
                           "group_of_mouse": df.loc[df["animal"] == mouse, "group"].iloc[0],
                           "n_features": len(kept), "features": ", ".join(kept), "_set": set(kept)})
        print("." if i < len(mice) else f". {len(mice)} repetitions done\n", end="", flush=True)

    n = len(mice)
    all_feats = list(dict.fromkeys(reference + [f for s in selections for f in sorted(s["_set"])]))
    rows = []
    for f in all_feats:
        k = sum(f in s["_set"] for s in selections)
        lost = [s["left_out_mouse"] for s in selections if f not in s["_set"]]
        rows.append({"feature": f, "in_selection_all_mice": f in reference,
                     "repetitions_selected": k, "repetitions_total": n,
                     "fraction": k / n, "stability": label(k / n),
                     "missing_when_leaving_out": ", ".join(lost)})
    tab = (pd.DataFrame(rows)
           .sort_values(["fraction", "in_selection_all_mice"], ascending=[False, False])
           .reset_index(drop=True))

    counts = tab.loc[tab["in_selection_all_mice"], "stability"].value_counts()
    print(f"  {len(mice)} mice | {len(reference)} features selected with all the mice: "
          + ", ".join(f"{counts.get(s, 0)} {s}" for s in ("robust", "moderate", "fragile")))
    extra = tab[~tab["in_selection_all_mice"]]
    if len(extra):
        print(f"  {len(extra)} other feature(s) appear only in some repetitions: "
              + ", ".join(f"{r.feature} ({r.repetitions_selected}/{n})" for r in extra.itertuples()))
    weak = tab[tab["in_selection_all_mice"] & (tab["stability"] != "robust")]
    for r in weak.itertuples():
        print(f"    {r.stability:8s} {r.feature:28s} {r.repetitions_selected}/{n}  "
              f"(missing when leaving out: {r.missing_when_leaving_out})")
    return tab, selections


def plot_bars(name, tab):
    t = tab.iloc[::-1]
    colors = t["stability"].map({"robust": "#1b7837", "moderate": "#e08214", "fragile": "#b2182b"})
    fig, ax = plt.subplots(figsize=(7.5, 0.28 * len(t) + 1.6))
    ax.barh(range(len(t)), 100 * t["fraction"], color=colors,
            edgecolor=["k" if s else "none" for s in t["in_selection_all_mice"]], linewidth=0.8)
    ax.set_yticks(range(len(t)), t["feature"], fontsize=7)
    for thr in (ROBUST, MODERATE):
        ax.axvline(100 * thr, color="gray", ls="--", lw=0.8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of repetitions (one mouse left out each time) in which it was selected")
    ax.set_title(f"{name}: stability of the selection\n"
                 f"green = robust, orange = moderate, red = fragile; black edge = selected with all mice",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"02_stability_{name}.png", dpi=200)
    if not SHOW_FIGURES:
        plt.close(fig)


def plot_heatmap(summary):
    fig, ax = plt.subplots(figsize=(1.3 * summary.shape[1] + 4, 0.26 * len(summary) + 1.8))
    im = ax.imshow(100 * summary.to_numpy(float), cmap="Greens", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(summary.shape[1]), summary.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(summary)), summary.index, fontsize=7)
    for r in range(summary.shape[0]):
        for c in range(summary.shape[1]):
            v = summary.iat[r, c]
            if pd.notna(v) and v > 0:
                ax.text(c, r, f"{100 * v:.0f}", ha="center", va="center", fontsize=6,
                        color="w" if v > 0.6 else "k")
    fig.colorbar(im, ax=ax, label="% of repetitions selected", shrink=0.6)
    ax.set_title("Stability of the selection (leave one mouse out)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_stability_heatmap.png", dpi=200)
    if not SHOW_FIGURES:
        plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    print(f"{len(df)} sessions, mice per group: {df.groupby('group')['animal'].nunique().to_dict()}")
    print(f"Rule of select_features.py: star = Tukey p < {sf.ALPHA}, "
          f"KEEP_BOTH = {sf.KEEP_BOTH}, USE_MAX = {sf.USE_MAX}")

    tables, all_sel = {}, []
    for name in COMPARISONS_TO_RUN:
        print("\n" + "=" * 70 + f"\n{name}\n" + "=" * 70)
        tab, sel = run_comparison(name, df)
        tables[name] = tab
        all_sel += sel
        plot_bars(name, tab)

    summary = pd.concat({n: t.set_index("feature")["fraction"] for n, t in tables.items()}, axis=1)
    order = summary.fillna(0).mean(axis=1).sort_values(ascending=False).index
    summary = summary.loc[order]
    plot_heatmap(summary)

    with pd.ExcelWriter(OUTPUT_DIR / "feature_stability.xlsx") as w:
        (summary * 100).round(0).reset_index().rename(columns={"index": "feature"}) \
            .to_excel(w, sheet_name="summary", index=False)
        for name, tab in tables.items():
            tab.to_excel(w, sheet_name=name[:31], index=False)
        pd.DataFrame(all_sel).drop(columns="_set").to_excel(w, sheet_name="selections", index=False)
        pd.DataFrame({"setting": ["KEEP_BOTH", "USE_MAX", "ALPHA", "ROBUST", "MODERATE", "input", "sessions"],
                      "value": [sf.KEEP_BOTH, sf.USE_MAX, sf.ALPHA, ROBUST, MODERATE,
                                str(sf.FEATURES_EXCEL), len(df)]}
                     ).to_excel(w, sheet_name="settings", index=False)
    print(f"\nResults in {OUTPUT_DIR}")
    if SHOW_FIGURES:
        plt.show()


if __name__ == "__main__":
    main()