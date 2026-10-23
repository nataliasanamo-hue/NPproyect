"""
Classifiers (LDA, SVM, logistic regression) with leave-one-MOUSE-out and the feature
selection done INSIDE every fold (no leakage)
=====================================================================================
For each comparison (WT_vs_NPC, NPC_vs_EFV, WT_vs_EFV, WT_vs_NPC_vs_EFV):

  for every mouse of the groups of the comparison:
    1. HIDE that mouse (all its sessions).
    2. SELECT the features with the rule of select_features.py (select_in_fold), using
       ONLY the other mice (option A: the ANOVA + Tukey uses the three groups, the star
       of the pair decides). The hidden mouse never takes part in choosing features.
    3. TRAIN each model with those features, using only the training mice of the groups
       of the comparison (the third group never enters the model). Missing values and
       z-scores are also computed from the training mice only.
    4. PREDICT the hidden mouse using only those features.

The selection of each fold is computed once and shared by the three models, so they
always use exactly the same features and can be compared fairly.

Results are given per SESSION and per MOUSE (mouse = average probability of its
sessions). The mouse-level result is the honest one: the mouse is the independent unit.
Balanced accuracy = mean of the % correct of each class (chance = 1 / number of classes),
so a bigger group is not favoured.

FEATURE IMPORTANCE (models in IMPORTANCE_MODELS, by default the SVM)
"Take one feature out and see how much the labels change" (drop-column importance):
in every fold the model is trained again WITHOUT each selected feature, one at a time,
and the hidden mouse is predicted again. For each feature:
    BA_drop_session / BA_drop_mouse : balanced accuracy with all features minus without it
                                      (positive = the model needs it; ~0 = the other
                                      features already carry its information;
                                      negative = it was adding noise)
    labels_changed                  : % of sessions whose predicted label changes without it

CHANCE LEVEL: the permutation test is in its own script, permutation_test_new_selection.py
(it uses the folds and models of this file, so both always agree).

INPUT : RESULTSNatalia/features_final.xlsx (all features; the selection is redone in every fold)
OUTPUT: RESULTSNatalia/<model>_new_selection/<comparison>/   (lda / svm / logreg)
          predictions.xlsx   summary, confusion per session and per mouse, per mouse,
                             per session, features of every fold, how often each feature
                             was chosen, importance (SVM), settings
          01_umap_real_vs_predicted.png and/or 01_tsne_real_vs_predicted.png  (EMBEDDING)
          02_mouse_predictions.png      (2-class comparisons, if PLOT_MOUSE_SUMMARY)
          03_feature_importance.png     (SVM)
          04_confusion_matrix.png       heatmap per session (count + % of the real group)
        RESULTSNatalia/classifiers_new_selection_summary.xlsx   all models x comparisons

Keep this file in the same folder as select_features.py (it imports the rule from there).
Requirements: pip install numpy pandas matplotlib scikit-learn openpyxl
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import select_features as sf   # the selection rule + paths + KEEP_BOTH / USE_MAX (same folder)

# =============================================================================
# SETTINGS
# =============================================================================
COMPARISONS_TO_RUN = ["WT_vs_NPC", "NPC_vs_EFV", "WT_vs_EFV", "WT_vs_NPC_vs_EFV"]
MODELS_TO_RUN = ["LDA", "SVM", "LogReg"]
IMPORTANCE_MODELS = ["SVM"]      # models for which the drop-column importance is computed

SVM_C = 1.0          # small (0.1) = smoother, more tolerant boundary; big (10) = stricter
LOGREG_C = 1.0       # small (0.1) = smaller weights, smoother; big (10) = stricter
LOG_POWER = True     # the MODELS use log10 of power (very skewed); the SELECTION always uses
                     # the raw values, exactly like the figures

EXCLUDE_ODD_SESSIONS = False     # True = leave out the odd sessions of select_features.py
ODD_SESSIONS = {
    "EFV_218": ["2026-01-16_17-50-35"],
    "EFV_220": ["2026-01-13_14-12-19", "2026-01-13_15-18-23", "2026-01-15_18-37-42"],
    "EFV_277": ["2026-02-20_12-02-47"],
    "NPC_249": ["2026-02-04_16-01-16", "2026-02-05_16-14-02", "2026-02-05_16-40-43"],
    "NPC_29":  ["P-7"],
    "NPC_30":  ["P4-HI"],
    "NPC_47":  ["P3-HD"],
}

EMBEDDING = "umap"   # map of figure 01 (only a picture, the models never use it):
                     # "umap", "tsne" or "both" (one figure each)
UMAP_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
TSNE_PERPLEXITY = 15
RANDOM_STATE = 42
PLOT_MOUSE_SUMMARY = True
SHOW_FIGURES = False             # True = open the figures at the end (they are always saved)
GROUP_COLORS = {"WT": "#1f77b4", "NPC": "#d62728", "EFV": "#2ca02c"}
# KEEP_BOTH, USE_MAX and ALPHA (the selection rule) are taken from select_features.py.
# =============================================================================

FOLDER = {"LDA": "lda", "SVM": "svm", "LogReg": "logreg"}


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


def make_model(name, n_classes):
    if name == "LDA":
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto",
                                         priors=np.full(n_classes, 1 / n_classes))
    elif name == "SVM":
        clf = SVC(kernel="linear", C=SVM_C, class_weight="balanced", decision_function_shape="ovr")
    else:
        clf = LogisticRegression(C=LOGREG_C, class_weight="balanced", max_iter=5000)
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("clf", clf)])


def class_probabilities(model, X):
    """One column per class, rows sum to 1, highest = predicted class.
    The SVM gives distances to the boundary, turned into probability-like values (softmax)."""
    if hasattr(model.named_steps["clf"], "predict_proba"):
        return model.predict_proba(X)
    d = model.decision_function(X)
    if d.ndim == 1:
        d = np.column_stack([np.zeros_like(d), d])
    e = np.exp(d - d.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def fit_predict(name, n_classes, Xtr, ytr, Xte):
    model = make_model(name, n_classes)
    model.fit(Xtr, ytr)
    return class_probabilities(model, Xte)


# -----------------------------------------------------------------------------
def run_comparison(comparison, ids, sel_df, values):
    groups = sf.COMPARISONS[comparison][0]
    names = list(groups)
    in_comp = ids["group"].isin(groups).to_numpy()
    cids = ids[in_comp].reset_index(drop=True)
    cvals = values[in_comp].reset_index(drop=True)
    y = cids["group"].map({g: i for i, g in enumerate(names)}).to_numpy()
    animals = cids["animal"].to_numpy()
    mice = sorted(np.unique(animals))
    n_classes = len(names)
    print(f"{len(cids)} sessions, mice per group: {cids.groupby('group')['animal'].nunique().to_dict()}")

    print("  folds: ", end="", flush=True)
    prob, prob_without, folds = leave_one_mouse_out(comparison, cids, cvals, sel_df, y, n_classes,
                                                    with_importance=True, progress=True)
    print(f" {len(mice)} done")

    # features: how often each one was chosen
    counts = pd.Series([f for fo in folds for f in fo["_feats"]]).value_counts()
    chosen = (counts / len(folds)).rename("fraction_of_folds").rename_axis("feature").reset_index()
    print(f"  features per fold: {min(fo['n_features'] for fo in folds)}-"
          f"{max(fo['n_features'] for fo in folds)}; chosen in every fold: "
          f"{int((chosen['fraction_of_folds'] == 1).sum())}")

    results = {}
    for m in MODELS_TO_RUN:
        ses, mouse_t, summary, cms = summarize(cids, y, prob[m], names)
        imp = None
        if m in prob_without:
            imp = importance(cids, y, prob[m], prob_without[m], folds, names)
        results[m] = {"ses": ses, "mouse": mouse_t, "summary": summary, "cms": cms, "imp": imp}
        s = summary.set_index("level")
        print(f"  {m:6s} balanced accuracy: sessions {s.loc['session', 'balanced_accuracy']:.2f}, "
              f"mice {s.loc['mouse', 'balanced_accuracy']:.2f}   (chance {1 / n_classes:.2f})")
        save(m, comparison, cids, cvals, folds, chosen, ses, mouse_t, summary, cms, imp, names)
    return results


def comparison_data(comparison, ids, values):
    """Sessions, model values and class numbers of the groups of one comparison."""
    groups = sf.COMPARISONS[comparison][0]
    names = list(groups)
    in_comp = ids["group"].isin(groups).to_numpy()
    cids = ids[in_comp].reset_index(drop=True)
    cvals = values[in_comp].reset_index(drop=True)
    y = cids["group"].map({g: i for i, g in enumerate(names)}).to_numpy()
    return names, cids, cvals, y


def leave_one_mouse_out(comparison, cids, cvals, sel_df, y, n_classes, with_importance=False,
                        progress=False, models=None):
    """The 4 steps of every fold (hide a mouse, select, train, predict the hidden mouse).
    sel_df carries the labels used for the SELECTION (real or shuffled); y the labels used
    to TRAIN. Returns the predicted probabilities of every model (each session predicted in
    the fold where its own mouse was hidden), the importance runs and the fold details."""
    models = MODELS_TO_RUN if models is None else models
    animals = cids["animal"].to_numpy()
    prob = {m: np.full((len(cids), n_classes), np.nan) for m in models}
    prob_without = ({m: {} for m in models if m in IMPORTANCE_MODELS}
                    if with_importance else {})
    folds = []
    for mouse in sorted(np.unique(animals)):
        # 1-2) select with every mouse except the hidden one (all three groups: option A)
        feats = sf.select_in_fold(sel_df[sel_df["animal"] != mouse], comparison)
        folds.append({"left_out_mouse": mouse, "group": cids.loc[animals == mouse, "group"].iloc[0],
                      "n_features": len(feats), "features": ", ".join(feats), "_feats": feats})
        train, test = animals != mouse, animals == mouse
        Xtr, Xte = cvals.loc[train, feats], cvals.loc[test, feats]
        # 3-4) train only with the groups of the comparison, predict the hidden mouse
        for m in models:
            prob[m][test] = fit_predict(m, n_classes, Xtr, y[train], Xte)
            if m in prob_without and len(feats) > 1:
                for f in feats:
                    keep = [c for c in feats if c != f]
                    arr = prob_without[m].setdefault(f, prob[m].copy() * np.nan)
                    arr[test] = fit_predict(m, n_classes, Xtr[keep], y[train], Xte[keep])
        if progress:
            print(".", end="", flush=True)
    return prob, prob_without, folds


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
    # confusion matrices: rows = real group, columns = what the model said
    cms = {level: pd.DataFrame(confusion_matrix(t["true_class"], t["predicted"], labels=names),
                               index=[f"real {n}" for n in names], columns=[f"said {n}" for n in names])
           for level, t in (("session", ses), ("mouse", mouse))}
    return ses, mouse, pd.DataFrame(rows), cms


def balanced_acc(cids, y, prob, names):
    pred = prob.argmax(axis=1)
    ba_s = balanced_accuracy_score(y, pred)
    t = pd.DataFrame(prob).groupby(cids["animal"].to_numpy()).mean()
    ym = pd.Series(y).groupby(cids["animal"].to_numpy()).first().loc[t.index]
    ba_m = balanced_accuracy_score(ym, t.to_numpy().argmax(axis=1))
    return ba_s, ba_m, pred


def importance(cids, y, prob_full, prob_without, folds, names):
    ba_s, ba_m, pred_full = balanced_acc(cids, y, prob_full, names)
    rows = []
    for f, arr in prob_without.items():
        p = np.where(np.isnan(arr), prob_full, arr)     # folds where f was not selected: unchanged
        s, mm, pred = balanced_acc(cids, y, p, names)
        rows.append({"feature": f,
                     "folds_selected": sum(f in fo["_feats"] for fo in folds),
                     "BA_drop_session": ba_s - s, "BA_drop_mouse": ba_m - mm,
                     "labels_changed_%": 100 * np.mean(pred != pred_full)})
    if not rows:
        return None
    return (pd.DataFrame(rows)
            .sort_values(["BA_drop_session", "labels_changed_%"], ascending=False)
            .reset_index(drop=True))


# -----------------------------------------------------------------------------
def save(model, comparison, cids, cvals, folds, chosen, ses, mouse, summary, cms, imp, names):
    out_dir = sf.RESULTS_DIR / f"{FOLDER[model]}_new_selection" / comparison
    out_dir.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_dir / "predictions.xlsx") as w:
        summary.to_excel(w, sheet_name="summary", index=False)
        cms["session"].to_excel(w, sheet_name="confusion_per_session")
        cms["mouse"].to_excel(w, sheet_name="confusion_per_mouse")
        mouse.to_excel(w, sheet_name="per_mouse", index=False)
        ses.to_excel(w, sheet_name="per_session", index=False)
        pd.DataFrame(folds).drop(columns="_feats").to_excel(w, sheet_name="features_per_fold", index=False)
        chosen.to_excel(w, sheet_name="features_chosen", index=False)
        if imp is not None:
            imp.to_excel(w, sheet_name="importance", index=False)
        pd.DataFrame({"setting": ["MODEL", "COMPARISON", "selection", "KEEP_BOTH", "USE_MAX", "ALPHA",
                                  "SVM_C", "LOGREG_C", "LOG_POWER", "EXCLUDE_ODD_SESSIONS", "n_sessions"],
                      "value": [model, comparison, "inside every fold (select_in_fold, option A)",
                                sf.KEEP_BOTH, sf.USE_MAX, sf.ALPHA, SVM_C, LOGREG_C, LOG_POWER,
                                EXCLUDE_ODD_SESSIONS, len(cids)]}
                     ).to_excel(w, sheet_name="settings", index=False)

    plot_confusion(model, comparison, cms, names, out_dir)

    methods = ["umap", "tsne"] if EMBEDDING == "both" else [EMBEDDING]
    for method in methods:
        plot_map(method, model, comparison, cvals, chosen, ses, summary, names, out_dir)
    if PLOT_MOUSE_SUMMARY and len(names) == 2:
        plot_mice(model, comparison, ses, mouse, names, out_dir)
    if imp is not None:
        plot_importance(model, comparison, imp, len(folds), out_dir)
    if not SHOW_FIGURES:
        plt.close("all")


_UMAP_WARNED = False


def plot_map(method, model, comparison, cvals, chosen, ses, summary, names, out_dir):
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
    axes[1].set_title(f"Labels ASSIGNED by the {model} (mouse hidden)")
    for ax in axes:
        ax.set_xlabel(f"{label} 1")
    axes[0].set_ylabel(f"{label} 2")
    s = summary.set_index("level")
    fig.suptitle(f"{model} - {comparison} (leave-one-mouse-out, selection inside each fold) | "
                 f"balanced accuracy: sessions {s.loc['session', 'balanced_accuracy']:.2f}, "
                 f"mice {s.loc['mouse', 'balanced_accuracy']:.2f} (chance {1 / len(names):.2f})\n"
                 f"{label} of the {len(feats)} features chosen in >= half of the folds, {param}")
    fig.tight_layout()
    fig.savefig(out_dir / f"01_{label.replace('-', '').lower()}_real_vs_predicted.png", dpi=200)


def plot_mice(model, comparison, ses, mouse, names, out_dir):
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
    ax.set_title(f"{model} - {comparison}: dots = sessions, diamond = mouse mean (red edge = wrong)")
    fig.tight_layout()
    fig.savefig(out_dir / "02_mouse_predictions.png", dpi=200)


def plot_confusion(model, comparison, cms, names, out_dir):
    """Heatmap of the confusion matrix PER SESSION: rows = real group, columns = what the
    model said. Colour and % = share of that REAL group (each row adds up to 100 %)."""
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
    ax.set_title(f"{model} - {comparison}: confusion matrix per session\n"
                 f"({cm.sum()} sessions, {np.trace(cm)} correct; leave-one-mouse-out)", fontsize=10)
    fig.colorbar(im, ax=ax, label="% of the real group", shrink=0.8)
    fig.savefig(out_dir / "04_confusion_matrix.png", dpi=200, bbox_inches="tight")


def plot_importance(model, comparison, imp, n_folds, out_dir):
    t = imp.iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.27 * len(t) + 1.8), sharey=True)
    axes[0].barh(range(len(t)), 100 * t["BA_drop_session"],
                 color=np.where(t["BA_drop_session"] > 0, "#1b7837", "#b2182b"))
    axes[0].axvline(0, color="k", lw=0.8)
    axes[0].set_xlabel("drop in balanced accuracy (sessions, % points) when the feature is removed")
    axes[1].barh(range(len(t)), t["labels_changed_%"], color="#4d4d4d")
    axes[1].set_xlabel("% of sessions whose predicted label changes")
    axes[0].set_yticks(range(len(t)), [f"{f} ({k}/{n_folds})" for f, k in
                                       zip(t["feature"], t["folds_selected"])], fontsize=7)
    fig.suptitle(f"{model} - {comparison}: drop-column importance "
                 f"(in brackets: folds in which the feature was selected)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "03_feature_importance.png", dpi=200)


# -----------------------------------------------------------------------------
def main():
    ids, sel_df, values = load()
    print(f"{len(ids)} sessions | selection rule of select_features.py: star = Tukey p < {sf.ALPHA}, "
          f"KEEP_BOTH = {sf.KEEP_BOTH}, USE_MAX = {sf.USE_MAX}")
    rows = []
    for comparison in COMPARISONS_TO_RUN:
        print("\n" + "=" * 70 + f"\n{comparison}\n" + "=" * 70)
        res = run_comparison(comparison, ids, sel_df, values)
        for m, r in res.items():
            s = r["summary"].set_index("level")
            row = {"comparison": comparison, "model": m,
                   "BA_sessions": s.loc["session", "balanced_accuracy"],
                   "BA_mice": s.loc["mouse", "balanced_accuracy"],
                   "theoretical_chance": 1 / len(sf.COMPARISONS[comparison][0])}
            row.update({"mice": s.loc["mouse", "n"], "sessions": s.loc["session", "n"]})
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_excel(sf.RESULTS_DIR / "classifiers_new_selection_summary.xlsx", index=False)
    print("\n" + "=" * 70 + "\nSUMMARY (balanced accuracy, mouse hidden in every fold)\n" + "=" * 70)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if SHOW_FIGURES:
        plt.show()


if __name__ == "__main__":
    main()