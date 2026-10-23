"""
Decision tree with leave-one-MOUSE-out, feature selection INSIDE every fold (no leakage)
and its own PERMUTATION TEST
=======================================================================================
A separate script for the decision tree, done in exactly the same way as
classifiers_new_selection.py (LDA / SVM / LogReg), so the results can be compared, but
with its own files and folder so nothing gets mixed up.

WHAT A DECISION TREE IS
It asks yes/no questions about one feature at a time ("is CSD_SLM_mean <= -0.3?") and
follows the branches down to a leaf, which gives the group. It is NOT linear like the
other three models: it can catch thresholds and combinations they cannot, but with few
mice it easily learns noise, so its size is limited (TREE_MAX_DEPTH, TREE_MIN_SAMPLES_LEAF).
It needs no z-scores (each question uses one feature alone), so it works with the real
values (log10 for power), which makes its questions readable.
Probability of each group = its (balanced) proportion in the leaf where the session lands.

PART 1 - CLASSIFICATION, for each comparison (WT_vs_NPC, NPC_vs_EFV, WT_vs_EFV,
WT_vs_NPC_vs_EFV), for every mouse of the groups of the comparison:
    1. HIDE that mouse (all its sessions).
    2. SELECT the features with the rule of select_features.py (select_in_fold), using ONLY
       the other mice (option A: the ANOVA + Tukey uses the three groups, the star of the
       pair decides).
    3. TRAIN the tree with those features, with the training mice of the groups of the
       comparison only (the third group never enters the model). Missing values are filled
       with the median of the training mice.
    4. PREDICT the hidden mouse.
Results per SESSION and per MOUSE (mouse = average probability of its sessions; the mouse
is the honest unit). Balanced accuracy = mean of the % correct of each class
(chance = 1 / number of classes).

PART 2 - PERMUTATION TEST (if RUN_PERMUTATION): the real chance level. The labels are
shuffled PER MOUSE (all the sessions of a mouse keep the same fake label; each group keeps
its number of mice; in pairs the third group keeps its label) and the WHOLE of part 1 is
repeated with the fake labels, many times. If a pair has at most N_PERMUTATIONS possible
shufflings, all of them are run (exact test). The shufflings are made exactly like in
permutation_test_new_selection.py (same seed, same rule), so the tree is tested against
the same fake datasets as LDA / SVM / LogReg.
    chance_mean : mean balanced accuracy of the shuffled runs (usually BELOW the theoretical
                  value with leave-one-mouse-out and few mice)
    p_value     : fraction of shuffled runs as good as or better than the real one
                  (small p = the real accuracy is unlikely to be luck; smallest p ~ 1/N)

TREE IMPORTANCE: how much each feature "cleans" the groups in the questions where the tree
uses it (Gini importance), averaged over the folds (0 where it was not selected or not used).

THE TREE PICTURE (figure 07): ONE tree trained on ALL the mice of the comparison, with the
features chosen in at least half of the folds, so you can read its questions. It is only a
picture (like the UMAP): the accuracy always comes from the hidden mice, never from it.

INPUT : RESULTSNatalia/features_final.xlsx
OUTPUT: everything inside RESULTSNatalia/tree_new_selection/
          <comparison>/predictions.xlsx   summary, confusion per session and per mouse, per
                                          mouse, per session, features + tree size of every
                                          fold, how often each feature was chosen, tree
                                          importance, settings
          <comparison>/permutation.xlsx   summary (real, chance level, p) + every shuffled run
          <comparison>/01_umap_real_vs_predicted.png (or 01_tsne_...)   map, picture only
          <comparison>/02_mouse_predictions.png      (2-class comparisons)
          <comparison>/04_confusion_matrix.png       heatmap per session
          <comparison>/05_permutation_test.png       shuffled runs vs the real accuracy
          <comparison>/06_tree_importance.png
          <comparison>/07_tree_example.png           picture only
          tree_summary.xlsx    one row per comparison: accuracy, chance level and p
                               (sessions and mice)
        (Figure numbers are the same as in the other classifiers; 03 does not exist here.)

Keep this file in the same folder as select_features.py (it imports the rule from there).
It does NOT use classifiers_new_selection.py: changing one never changes the other.
Requirements: pip install numpy pandas matplotlib scikit-learn openpyxl joblib
"""

import re
import time
from itertools import combinations
from math import factorial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

import select_features as sf   # the selection rule + paths + KEEP_BOTH / USE_MAX (same folder)

# =============================================================================
# SETTINGS
# =============================================================================
COMPARISONS_TO_RUN = ["WT_vs_NPC", "NPC_vs_EFV", "WT_vs_EFV", "WT_vs_NPC_vs_EFV"]

# --- the tree ---
TREE_MAX_DEPTH = None         # max number of questions from the top to a leaf. Small = simple
                           # tree (1 = a single question). With so few mice keep it small:
                           # a deep tree memorises the training mice.
TREE_MIN_SAMPLES_LEAF = 1  # min number of SESSIONS in each leaf (bigger = fewer, safer
                           # questions; stops leaves made of one session)
LOG_POWER = True           # the tree uses log10 of power; the SELECTION always uses the raw
                           # values, exactly like the figures (same as the other classifiers)

# --- permutation test ---
RUN_PERMUTATION = True     # False = only part 1 (quick, e.g. to look at the figures first)
N_PERMUTATIONS = 100       # shuffled runs per comparison (fewer possible -> all of them, exact)
PERMUTATION_SEED = 0       # same seed as permutation_test_new_selection.py -> same shufflings
N_JOBS = -2                # processor cores used at the same time:
                           #   -2 = all except one (default: the laptop stays usable)
                           #   -1 = all of them;  1 = only one (slowest); 2, 3... = that number

# --- data (keep them the same as in classifiers_new_selection.py) ---
EXCLUDE_ODD_SESSIONS = False     # True = leave out the odd sessions
ODD_SESSIONS = {
    "EFV_218": ["2026-01-16_17-50-35"],
    "EFV_220": ["2026-01-13_14-12-19", "2026-01-13_15-18-23", "2026-01-15_18-37-42"],
    "EFV_277": ["2026-02-20_12-02-47"],
    "NPC_249": ["2026-02-04_16-01-16", "2026-02-05_16-14-02", "2026-02-05_16-40-43"],
    "NPC_29":  ["P-7"],
    "NPC_30":  ["P4-HI"],
    "NPC_47":  ["P3-HD"],
}

# --- figures ---
EMBEDDING = "umap"   # map of figure 01 (picture only): "umap", "tsne" or "both"
UMAP_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
TSNE_PERPLEXITY = 15
RANDOM_STATE = 42
PLOT_MOUSE_SUMMARY = True
SHOW_FIGURES = False             # True = open the figures at the end (they are always saved)
GROUP_COLORS = {"WT": "#1f77b4", "NPC": "#d62728", "EFV": "#2ca02c"}
# KEEP_BOTH, USE_MAX and ALPHA (the selection rule) are taken from select_features.py.
# =============================================================================

OUT_ROOT = sf.RESULTS_DIR / "tree_new_selection"
MODEL = "Tree"


def is_power(col):
    return (col.split("_")[0] in ("Delta", "Theta", "Alpha", "SlowGamma", "MidGamma", "FastGamma", "Ripple")
            and col.endswith(("_mean", "_max")) and ("_PYR_" in col or "_SLM_" in col))


def load():
    df = pd.read_excel(sf.FEATURES_EXCEL, sheet_name=sf.SHEET)
    df["group"] = df["group"].astype(str).str.strip()
    df["animal"] = df["animal"].astype(str).str.strip()
    df = df[df["group"].isin(["WT", "NPC", "EFV"])].reset_index(drop=True)
    if EXCLUDE_ODD_SESSIONS:
        wanted = {f"{a}|{s}" for a, ss in ODD_SESSIONS.items() for s in ss}
        key = df["animal"] + "|" + df["session"].astype(str).str.strip()
        print(f"[i] Leaving out {int(key.isin(wanted).sum())} odd sessions")
        df = df[~key.isin(wanted)].reset_index(drop=True)

    feats = sf.plotted_features(df.columns)          # the only features that can be selected
    raw = df[feats].apply(pd.to_numeric, errors="coerce")
    model_values = raw.copy()
    if LOG_POWER:                                    # value by value: no information leaks
        for c in feats:
            if is_power(c) and model_values[c].min(skipna=True) > 0:
                model_values[c] = np.log10(model_values[c])
    sel_df = pd.concat([df[sf.ID_COLS], raw], axis=1)  # raw values, for the selection
    return df[sf.ID_COLS].copy(), sel_df, model_values


def comparison_data(comparison, ids, values):
    """Sessions, model values and class numbers of the groups of one comparison."""
    names = list(sf.COMPARISONS[comparison][0])
    in_comp = ids["group"].isin(names).to_numpy()
    cids = ids[in_comp].reset_index(drop=True)
    cvals = values[in_comp].reset_index(drop=True)
    y = cids["group"].map({g: i for i, g in enumerate(names)}).to_numpy()
    return names, cids, cvals, y


def make_tree():
    clf = DecisionTreeClassifier(max_depth=TREE_MAX_DEPTH, min_samples_leaf=TREE_MIN_SAMPLES_LEAF,
                                 class_weight="balanced", random_state=RANDOM_STATE)
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("clf", clf)])


# -----------------------------------------------------------------------------
# PART 1: leave-one-mouse-out
# -----------------------------------------------------------------------------
def leave_one_mouse_out(comparison, cids, cvals, sel_df, y, n_classes, progress=False):
    """The 4 steps of every fold. sel_df carries the labels used for the SELECTION (real or
    shuffled); y the labels used to TRAIN. Returns the probabilities (each session predicted
    in the fold where its own mouse was hidden) and the details of every fold."""
    animals = cids["animal"].to_numpy()
    prob = np.full((len(cids), n_classes), np.nan)
    folds = []
    for mouse in sorted(np.unique(animals)):
        feats = sf.select_in_fold(sel_df[sel_df["animal"] != mouse], comparison)
        train, test = animals != mouse, animals == mouse
        model = make_tree().fit(cvals.loc[train, feats], y[train])
        prob[test] = model.predict_proba(cvals.loc[test, feats])
        tree = model.named_steps["clf"]
        folds.append({"left_out_mouse": mouse, "group": cids.loc[test, "group"].iloc[0],
                      "n_features": len(feats), "features": ", ".join(feats),
                      "tree_depth": tree.get_depth(), "tree_leaves": tree.get_n_leaves(),
                      "_feats": feats, "_imp": dict(zip(feats, tree.feature_importances_))})
        if progress:
            print(".", end="", flush=True)
    return prob, folds


def balanced_acc(cids, y, prob):
    """Balanced accuracy per session and per mouse (mouse = mean probability of its sessions)."""
    ba_s = balanced_accuracy_score(y, prob.argmax(axis=1))
    t = pd.DataFrame(prob).groupby(cids["animal"].to_numpy()).mean()
    ym = pd.Series(y).groupby(cids["animal"].to_numpy()).first().loc[t.index]
    ba_m = balanced_accuracy_score(ym, t.to_numpy().argmax(axis=1))
    return ba_s, ba_m


def summarize(cids, y, prob, names):
    pcols = [f"P_{n}" for n in names]
    ses = cids.copy()
    ses["true_class"] = [names[v] for v in y]
    ses[pcols] = prob
    ses["predicted"] = [names[i] for i in prob.argmax(axis=1)]
    ses["correct"] = ses["true_class"] == ses["predicted"]

    mouse = (ses.groupby(["animal", "group", "true_class"], as_index=False)
             .agg(n_sessions=("session", "size"), **{c: (c, "mean") for c in pcols},
                  sessions_correct=("correct", "mean")))
    mouse["predicted"] = [names[i] for i in mouse[pcols].to_numpy().argmax(axis=1)]
    mouse["correct"] = mouse["true_class"] == mouse["predicted"]

    rows = []
    for level, t in (("session", ses), ("mouse", mouse)):
        c = confusion_matrix(t["true_class"], t["predicted"], labels=names)
        row = {"level": level, "n": len(t), "accuracy": (t["true_class"] == t["predicted"]).mean(),
               "balanced_accuracy": balanced_accuracy_score(t["true_class"], t["predicted"])}
        for i, n in enumerate(names):
            row[f"{n}_correct"] = f"{c[i, i]}/{c[i].sum()}"
        rows.append(row)
    # confusion matrices: rows = real group, columns = what the tree said
    cms = {level: pd.DataFrame(confusion_matrix(t["true_class"], t["predicted"], labels=names),
                               index=[f"real {n}" for n in names], columns=[f"said {n}" for n in names])
           for level, t in (("session", ses), ("mouse", mouse))}
    return ses, mouse, pd.DataFrame(rows), cms


def tree_importance(folds):
    """Gini importance of the tree of every fold, averaged over ALL the folds
    (0 where the feature was not selected or the tree did not ask about it)."""
    n = len(folds)
    rows = []
    for f in sorted({f for fo in folds for f in fo["_feats"]}):
        vals = np.array([fo["_imp"].get(f, 0.0) for fo in folds])
        rows.append({"feature": f, "folds_selected": sum(f in fo["_feats"] for fo in folds),
                     "folds_used_by_tree": int((vals > 0).sum()),
                     "mean_importance": vals.sum() / n, "sd_importance": vals.std()})
    return pd.DataFrame(rows).sort_values("mean_importance", ascending=False).reset_index(drop=True)


# -----------------------------------------------------------------------------
# PART 2: permutation test
# -----------------------------------------------------------------------------
def n_cores_used():
    n = cpu_count()
    return max(1, n + 1 + N_JOBS) if N_JOBS < 0 else max(1, min(N_JOBS, n))


def shufflings(mouse_groups, names, rng):
    """Fake assignments mouse -> group (each group keeps its number of mice), made exactly
    like in permutation_test_new_selection.py. exact = True when ALL the possible ones are
    listed (the real one is then one of them)."""
    mice = list(mouse_groups.index)
    sizes = [int((mouse_groups == g).sum()) for g in names]
    n_total = factorial(len(mice))
    for s in sizes:
        n_total //= factorial(s)
    if len(names) == 2 and n_total <= N_PERMUTATIONS:
        out = [pd.Series({m: (names[0] if m in first else names[1]) for m in mice})
               for first in combinations(mice, sizes[0])]
        return out, True
    labels = mouse_groups.to_numpy()
    return [pd.Series(rng.permutation(labels), index=mice) for _ in range(N_PERMUTATIONS)], False


def one_run(comparison, fake, cids, cvals, sel_df, names):
    """The whole of part 1 with fake labels -> (balanced accuracy sessions, mice)."""
    sel_fake = sel_df.copy()
    in_comp = sel_fake["animal"].isin(fake.index)
    sel_fake.loc[in_comp, "group"] = sel_fake.loc[in_comp, "animal"].map(fake)
    cids_fake = cids.copy()
    cids_fake["group"] = cids_fake["animal"].map(fake)
    y_fake = cids_fake["group"].map({g: k for k, g in enumerate(names)}).to_numpy()
    prob, _ = leave_one_mouse_out(comparison, cids_fake, cvals, sel_fake, y_fake, len(names))
    return balanced_acc(cids_fake, y_fake, prob)


def p_value(real, null, exact):
    """Exact test: the real labelling is one of the runs. Random runs: (k + 1) / (N + 1)."""
    k = int(np.sum(null >= real - 1e-12))
    return k / len(null) if exact else (k + 1) / (len(null) + 1)


def permutation_test(comparison, cids, cvals, sel_df, names, real):
    mouse_groups = cids.groupby("animal")["group"].first()
    fakes, exact = shufflings(mouse_groups, names, np.random.default_rng(PERMUTATION_SEED))
    print(f"  permutation test: {len(fakes)} " +
          ("shufflings = ALL the possible ones (exact test)" if exact else "random shufflings"),
          end=" ", flush=True)
    t0 = time.time()
    n_workers = n_cores_used()
    chunk = max(len(fakes) // 20, 4 * n_workers)
    results = []
    with Parallel(n_jobs=n_workers) as parallel:
        for start in range(0, len(fakes), chunk):
            results += parallel(delayed(one_run)(comparison, f, cids, cvals, sel_df, names)
                                for f in fakes[start:start + chunk])
            print(".", end="", flush=True)
    print(f" done in {time.time() - t0:.0f} s")
    runs = pd.DataFrame({"run": range(1, len(fakes) + 1),
                         "BA_sessions": [r[0] for r in results],
                         "BA_mice": [r[1] for r in results],
                         "is_real_labels": [bool((f == mouse_groups.loc[f.index]).all()) for f in fakes]})
    rows = []
    for level, col, k in (("session", "BA_sessions", 0), ("mouse", "BA_mice", 1)):
        null = runs[col].to_numpy()
        rows.append({"level": level, "real": real[k], "chance_mean": null.mean(),
                     "chance_sd": null.std(), "p_value": p_value(real[k], null, exact),
                     "theoretical_chance": 1 / len(names), "n_runs": len(null),
                     "type": "exact (all shufflings)" if exact else f"random (seed {PERMUTATION_SEED})"})
    return pd.DataFrame(rows), runs, exact


# -----------------------------------------------------------------------------
def run_comparison(comparison, ids, sel_df, values):
    names, cids, cvals, y = comparison_data(comparison, ids, values)
    n_mice = cids.groupby("group")["animal"].nunique().to_dict()
    print(f"{len(cids)} sessions, mice per group: {n_mice}")
    print("  folds: ", end="", flush=True)
    prob, folds = leave_one_mouse_out(comparison, cids, cvals, sel_df, y, len(names), progress=True)
    print(f" {len(folds)} done")

    counts = pd.Series([f for fo in folds for f in fo["_feats"]]).value_counts()
    chosen = (counts / len(folds)).rename("fraction_of_folds").rename_axis("feature").reset_index()
    print(f"  features per fold: {min(fo['n_features'] for fo in folds)}-"
          f"{max(fo['n_features'] for fo in folds)}; tree depth per fold: "
          f"{min(fo['tree_depth'] for fo in folds)}-{max(fo['tree_depth'] for fo in folds)}")

    ses, mouse, summary, cms = summarize(cids, y, prob, names)
    imp = tree_importance(folds)
    s = summary.set_index("level")
    real = (s.loc["session", "balanced_accuracy"], s.loc["mouse", "balanced_accuracy"])
    print(f"  Tree balanced accuracy: sessions {real[0]:.2f}, mice {real[1]:.2f}   "
          f"(theoretical chance {1 / len(names):.2f})")

    out_dir = OUT_ROOT / comparison
    out_dir.mkdir(parents=True, exist_ok=True)
    save_predictions(comparison, cids, folds, chosen, ses, mouse, summary, cms, imp, out_dir)
    plot_confusion(comparison, cms, names, out_dir)
    for method in (["umap", "tsne"] if EMBEDDING == "both" else [EMBEDDING]):
        plot_map(method, comparison, cvals, chosen, ses, summary, names, out_dir)
    if PLOT_MOUSE_SUMMARY and len(names) == 2:
        plot_mice(comparison, ses, mouse, names, out_dir)
    plot_tree_importance(comparison, imp, len(folds), out_dir)
    plot_tree_example(comparison, cvals, y, chosen, names, out_dir)

    row = {"comparison": comparison, "BA_sessions": real[0], "BA_mice": real[1],
           "theoretical_chance": 1 / len(names), "mice": len(folds), "sessions": len(cids)}
    if RUN_PERMUTATION:
        perm, runs, exact = permutation_test(comparison, cids, cvals, sel_df, names, real)
        with pd.ExcelWriter(out_dir / "permutation.xlsx") as w:
            perm.to_excel(w, sheet_name="summary", index=False)
            runs.to_excel(w, sheet_name="runs", index=False)
        plot_permutation(comparison, perm, runs, exact, out_dir)
        p = perm.set_index("level")
        print(f"  chance level: sessions {p.loc['session', 'chance_mean']:.2f} "
              f"(p = {p.loc['session', 'p_value']:.3f}) | mice {p.loc['mouse', 'chance_mean']:.2f} "
              f"(p = {p.loc['mouse', 'p_value']:.3f})")
        row.update({"chance_sessions": p.loc["session", "chance_mean"],
                    "p_sessions": p.loc["session", "p_value"],
                    "chance_mice": p.loc["mouse", "chance_mean"],
                    "p_mice": p.loc["mouse", "p_value"],
                    "n_runs": len(runs), "exact": exact})
    if not SHOW_FIGURES:
        plt.close("all")
    return row


def save_predictions(comparison, cids, folds, chosen, ses, mouse, summary, cms, imp, out_dir):
    with pd.ExcelWriter(out_dir / "predictions.xlsx") as w:
        summary.to_excel(w, sheet_name="summary", index=False)
        cms["session"].to_excel(w, sheet_name="confusion_per_session")
        cms["mouse"].to_excel(w, sheet_name="confusion_per_mouse")
        mouse.to_excel(w, sheet_name="per_mouse", index=False)
        ses.to_excel(w, sheet_name="per_session", index=False)
        pd.DataFrame(folds).drop(columns=["_feats", "_imp"]).to_excel(
            w, sheet_name="features_per_fold", index=False)
        chosen.to_excel(w, sheet_name="features_chosen", index=False)
        imp.to_excel(w, sheet_name="tree_importance", index=False)
        pd.DataFrame({"setting": ["MODEL", "COMPARISON", "selection", "KEEP_BOTH", "USE_MAX", "ALPHA",
                                  "TREE_MAX_DEPTH", "TREE_MIN_SAMPLES_LEAF", "LOG_POWER",
                                  "EXCLUDE_ODD_SESSIONS", "n_sessions"],
                      "value": [MODEL, comparison, "inside every fold (select_in_fold, option A)",
                                sf.KEEP_BOTH, sf.USE_MAX, sf.ALPHA, TREE_MAX_DEPTH,
                                TREE_MIN_SAMPLES_LEAF, LOG_POWER, EXCLUDE_ODD_SESSIONS, len(cids)]}
                     ).to_excel(w, sheet_name="settings", index=False)


# -----------------------------------------------------------------------------
# FIGURES
# -----------------------------------------------------------------------------
_UMAP_WARNED = False


def plot_map(method, comparison, cvals, chosen, ses, summary, names, out_dir):
    """Picture only: UMAP or t-SNE of the features chosen in at least half of the folds."""
    global _UMAP_WARNED
    feats = list(chosen.loc[chosen["fraction_of_folds"] >= 0.5, "feature"]) or list(chosen["feature"])
    X = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(cvals[feats]))
    if X.shape[1] < 2:
        return
    if method == "umap":
        try:
            import umap
        except ImportError:
            if not _UMAP_WARNED:
                print("[!] UMAP not installed (pip install umap-learn): using t-SNE for figure 01")
                _UMAP_WARNED = True
            method = "tsne"
    if method == "umap":
        k = min(UMAP_NEIGHBORS, len(X) - 1)
        Y = umap.UMAP(n_components=2, n_neighbors=k, min_dist=UMAP_MIN_DIST,
                      random_state=RANDOM_STATE).fit_transform(X)
        label, param = "UMAP", f"n_neighbors {k}"
    else:
        perp = min(TSNE_PERPLEXITY, len(X) - 1)
        Y = TSNE(n_components=2, perplexity=perp, init="pca", learning_rate="auto",
                 random_state=RANDOM_STATE).fit_transform(X)
        label, param = "t-SNE", f"perplexity {perp}"
    wrong = ~ses["correct"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    for g in names:
        m = (ses["group"] == g).to_numpy()
        axes[0].scatter(Y[m, 0], Y[m, 1], s=45, alpha=0.85, color=GROUP_COLORS[g],
                        edgecolor="k", linewidth=0.4, label=g)
    axes[0].legend(fontsize=9)
    axes[0].set_title("REAL labels")
    pred = ses["predicted"].to_numpy()
    for g in names:
        m = pred == g
        axes[1].scatter(Y[m, 0], Y[m, 1], s=45, alpha=0.85, color=GROUP_COLORS[g],
                        edgecolor="k", linewidth=0.4, label=f"said {g}")
    n, n_wrong = len(wrong), int(wrong.sum())
    axes[1].scatter(Y[wrong, 0], Y[wrong, 1], s=170, facecolor="none", edgecolor="k", linewidth=1.3,
                    label=f"wrong {n_wrong}/{n} ({100 * n_wrong / n:.0f}%)")
    handles, labels = axes[1].get_legend_handles_labels()
    handles.insert(len(names), plt.Line2D([], [], ls=""))
    labels.insert(len(names), f"correct {n - n_wrong}/{n} ({100 * (n - n_wrong) / n:.0f}%)")
    axes[1].legend(handles, labels, fontsize=9)
    axes[1].set_title(f"Labels ASSIGNED by the {MODEL} (mouse hidden)")
    for ax in axes:
        ax.set_xlabel(f"{label} 1")
    axes[0].set_ylabel(f"{label} 2")
    s = summary.set_index("level")
    fig.suptitle(f"{MODEL} - {comparison} (leave-one-mouse-out, selection inside each fold) | "
                 f"balanced accuracy: sessions {s.loc['session', 'balanced_accuracy']:.2f}, "
                 f"mice {s.loc['mouse', 'balanced_accuracy']:.2f} (chance {1 / len(names):.2f})\n"
                 f"{label} of the {len(feats)} features chosen in >= half of the folds, {param}")
    fig.tight_layout()
    fig.savefig(out_dir / f"01_{label.replace('-', '').lower()}_real_vs_predicted.png", dpi=200)


def plot_mice(comparison, ses, mouse, names, out_dir):
    pcol = f"P_{names[1]}"
    mouse = mouse.sort_values(["true_class", pcol]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(mouse)), 4.5))
    for i, r in mouse.iterrows():
        c = GROUP_COLORS.get(r["group"], "gray")
        s = ses.loc[ses["animal"] == r["animal"], pcol]
        ax.scatter(np.full(len(s), i), s, s=18, color=c, alpha=0.45, edgecolor="none")
        ax.scatter(i, r[pcol], marker="D", s=70, color=c, edgecolor="k" if r["correct"] else "red",
                   linewidth=1.2 if r["correct"] else 2.5, zorder=3)
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set_xticks(range(len(mouse)), mouse["animal"], rotation=60, fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel(f"P({names[1]}) predicted while the mouse was hidden")
    ax.set_title(f"{MODEL} - {comparison}: dots = sessions, diamond = mouse mean (red edge = wrong)")
    fig.tight_layout()
    fig.savefig(out_dir / "02_mouse_predictions.png", dpi=200)


def plot_confusion(comparison, cms, names, out_dir):
    """Heatmap of the confusion matrix PER SESSION: rows = real group, columns = what the
    tree said. Colour and % = share of that REAL group (each row adds up to 100 %)."""
    cm = cms["session"].to_numpy()
    pct = 100 * cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.6 + 1.6, 3.0 + 0.6 * len(names)))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n({pct[i, j]:.0f}%)", ha="center", va="center",
                    fontsize=11, color="w" if pct[i, j] > 60 else "k")
    ax.set_xticks(range(len(names)), [f"said {n}" for n in names])
    ax.set_yticks(range(len(names)), [f"real {n}" for n in names])
    ax.set_title(f"{MODEL} - {comparison}: confusion matrix per session\n"
                 f"({cm.sum()} sessions, {np.trace(cm)} correct; leave-one-mouse-out)", fontsize=10)
    fig.colorbar(im, ax=ax, label="% of the real group", shrink=0.8)
    fig.savefig(out_dir / "04_confusion_matrix.png", dpi=200, bbox_inches="tight")


def plot_permutation(comparison, perm, runs, exact, out_dir):
    s = perm.set_index("level")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, level, col in zip(axes, ("session", "mouse"), ("BA_sessions", "BA_mice")):
        null = runs[col].to_numpy()
        real = s.loc[level, "real"]
        ax.hist(null, bins=np.linspace(0, 1, 26), color="#bdbdbd", edgecolor="k", linewidth=0.5,
                label=f"shuffled labels ({len(null)} runs)")
        ax.axvline(null.mean(), color="#4d4d4d", ls="--", lw=1.5, label=f"chance level = {null.mean():.2f}")
        ax.axvline(real, color="#b2182b", lw=2.5, label=f"real = {real:.2f}")
        ax.set_xlim(0, 1)
        ax.set_xlabel("balanced accuracy")
        ax.set_ylabel("number of shuffled runs")
        ax.set_title(f"per {level.upper()}: p = {s.loc[level, 'p_value']:.3f}", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
    kind = "all possible shufflings, exact test" if exact else "random shufflings"
    fig.suptitle(f"{MODEL} - {comparison}: permutation test, labels shuffled per mouse ({kind})\n"
                 f"p = fraction of shuffled runs as good as or better than the real one", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "05_permutation_test.png", dpi=200)


def plot_tree_importance(comparison, imp, n_folds, out_dir):
    """Mean Gini importance over the folds; only features the tree used at least once."""
    t = imp[imp["folds_used_by_tree"] > 0].iloc[::-1]
    if t.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 0.3 * len(t) + 1.8))
    ax.barh(range(len(t)), t["mean_importance"], xerr=t["sd_importance"], color="#8c6d31",
            ecolor="#4d4d4d", capsize=2)
    ax.set_yticks(range(len(t)), [f"{f} (used {u}/{n_folds})" for f, u in
                                  zip(t["feature"], t["folds_used_by_tree"])], fontsize=7)
    ax.set_xlabel("tree importance (Gini), mean over all the folds (bar = SD)")
    ax.set_xlim(0, 1)
    ax.set_title(f"{MODEL} - {comparison}: which features the tree asks about\n"
                 f"(in brackets: folds whose tree used the feature)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "06_tree_importance.png", dpi=200)


def plot_tree_example(comparison, cvals, y, chosen, names, out_dir):
    """PICTURE ONLY: one tree trained on ALL the mice of the comparison, with the features
    chosen in at least half of the folds. Not used for any accuracy."""
    feats = list(chosen.loc[chosen["fraction_of_folds"] >= 0.5, "feature"]) or list(chosen["feature"])
    tree = make_tree().fit(cvals[feats], y).named_steps["clf"]
    fig, ax = plt.subplots(figsize=(max(9, 3.2 * tree.get_n_leaves()), 2.6 * (tree.get_depth() + 1) + 1))
    art = plot_tree(tree, feature_names=feats, class_names=names, filled=False, rounded=True,
                    impurity=False, proportion=False, precision=2, fontsize=8, ax=ax)
    # colour each box with the colour of the group it predicts (darker = purer box)
    for a in art:
        box = a.get_bbox_patch()
        txt = a.get_text()
        mc = re.search(r"class = (\w+)", txt)
        mv = re.search(r"value = \[([^\]]+)\]", txt)
        if box is None or not mc or not mv or mc.group(1) not in GROUP_COLORS:
            continue
        v = np.array([float(x) for x in mv.group(1).split(",")])
        purity = v.max() / v.sum() if v.sum() > 0 else 0
        rgb = np.array(plt.matplotlib.colors.to_rgb(GROUP_COLORS[mc.group(1)]))
        w = 0.15 + 0.65 * max(0.0, (purity - 1 / len(names)) / (1 - 1 / len(names)))
        box.set_facecolor(tuple(w * rgb + (1 - w)))   # mixed with white, text stays readable
    log_note = " (power features in log10)" if LOG_POWER else ""
    ax.set_title(f"{MODEL} - {comparison}: EXAMPLE tree trained on ALL the mice (picture only, the "
                 f"accuracy comes from the hidden mice)\n{len(feats)} features chosen in >= half of "
                 f"the folds{log_note}; 'value' = balanced weight of each group in the box", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "07_tree_example.png", dpi=200)


# -----------------------------------------------------------------------------
def main():
    ids, sel_df, values = load()
    print(f"{len(ids)} sessions | decision tree: max depth {TREE_MAX_DEPTH}, min {TREE_MIN_SAMPLES_LEAF} "
          f"sessions per leaf | selection: star = Tukey p < {sf.ALPHA}, KEEP_BOTH = {sf.KEEP_BOTH}, "
          f"USE_MAX = {sf.USE_MAX}" +
          (f" | permutation test: {N_PERMUTATIONS} runs, {n_cores_used()} of {cpu_count()} cores"
           if RUN_PERMUTATION else " | permutation test OFF"))
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for comparison in COMPARISONS_TO_RUN:
        print("\n" + "=" * 70 + f"\n{comparison}\n" + "=" * 70)
        rows.append(run_comparison(comparison, ids, sel_df, values))
    table = pd.DataFrame(rows)
    table.to_excel(OUT_ROOT / "tree_summary.xlsx", index=False)
    print("\n" + "=" * 70 + "\nSUMMARY - decision tree (balanced accuracy, mouse hidden in every fold)\n"
          + "=" * 70)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nResults in {OUT_ROOT}")
    if SHOW_FIGURES:
        plt.show()


if __name__ == "__main__":
    main()