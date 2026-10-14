"""
LDA classifier with leave-one-MOUSE-out cross-validation
=========================================================
Linear Discriminant Analysis (LDA): draws a straight boundary between the groups
using the mean and spread of each group.
(svm_classifier.py does the same with a linear SVM, to compare.)

Three possible questions (list the ones you want in QUESTIONS_TO_RUN below):
    "WT_vs_NPC_vs_EFV" : the model answers WT, NPC or EFV   (chance = 33 %)
    "WT_vs_sick"       : WT  vs  sick (NPC + EFV together)  (chance = 50 %)
    "NPC_vs_EFV"       : NPC vs  EFV  (does the treatment change the profile?)

HOW IT WORKS
------------
Leave-one-mouse-out: with N mice we train N models. Each time, ALL sessions
of one mouse are hidden, the model learns from the other mice, and then it
predicts the hidden mouse. So the model is always tested on a mouse it has
never seen (sessions of the same mouse are too similar to be split between
training and testing).

Everything that "learns" from the data is done INSIDE each fold, only with the
training mice, so the hidden mouse never leaks into the model:
    1. fill missing values (median of the training mice)
    2. standardize (z-score)
    3. (optional, K_FEATURES) keep the K features that best separate the classes
    4. LDA with shrinkage (stable with few mice and many features)

Results are given per SESSION and per MOUSE (mouse = average probability
of its sessions). The mouse-level result is the honest one: the mouse is the
real independent unit.

Accuracy can be misleading when one class is bigger (e.g. 5 WT vs 11 sick:
always saying "sick" gives 69 %). That is why we report BALANCED accuracy
(mean of the % correct in each class): chance = 1 / number of classes.

INPUT : USE_SELECTED = True  -> feature_selection/features_selected.xlsx
        (chosen with ALL mice, so accuracy is somewhat optimistic)
        USE_SELECTED = False -> features_final.xlsx (use K_FEATURES = 10 so the
        selection happens inside each fold: the honest number)
OUTPUT: RESULTSNatalia/lda_results_selected/<question>/   (one folder per question)
        predictions.xlsx   per session, per mouse, summary, confusion, settings
        01_tsne_real_vs_predicted.png   (same t-SNE map: real labels vs model labels)
        02_mouse_predictions.png        (only if PLOT_MOUSE_SUMMARY = True)

Requirements: pip install numpy pandas matplotlib scikit-learn openpyxl
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# =============================================================================
# SETTINGS
# =============================================================================
baseDir = Path(r"D:\Violeta\ANALISIS")
RESULTS_DIR = baseDir / "RESULTSNatalia"
SHEET = "features"

# Questions to run, one after the other (each one gets its own results + figure).
#   "WT_vs_NPC_vs_EFV" : the model answers WT, NPC or EFV
#   "WT_vs_sick"       : the model answers WT or sick
#   "NPC_vs_EFV"       : the model answers NPC or EFV
QUESTIONS_TO_RUN = ["WT_vs_NPC_vs_EFV", "WT_vs_sick", "NPC_vs_EFV"]

MODEL = "LDA"     # (used in titles and folder names; don't edit)
QUESTION = QUESTIONS_TO_RUN[0]   # (changed automatically while running; don't edit)

USE_SELECTED = True          # True  = features chosen by select_features.py
                             #         (run select_features.py FIRST with the same
                             #          EXCLUDE_ODD_SESSIONS value as here)
                             # False = all the features of features_final.xlsx
if USE_SELECTED:
    EXCEL_PATH = RESULTS_DIR / "feature_selection" / "features_selected.xlsx"
    RUN_TAG = "_selected"
else:
    EXCEL_PATH = RESULTS_DIR / "features_final.xlsx"
    RUN_TAG = ""

K_FEATURES = None            # None = use ALL the features in the Excel (recommended
                             #        with USE_SELECTED = True: they are already chosen)
                             # a number (5, 10, 20) = keep only the K best inside each fold
LOG_POWER = True             # log10 of power features (very skewed values)
MAX_MISSING_COL = 0.20       # drop a feature if > 20 % of sessions have no value

# same non-biological columns as in pca_analysis_new.py / select_features.py
ID_COLS = ["animal", "session", "group"]
EXCLUDE_PREFIXES = ["ch_", "config_", "power_pyr_channel", "power_slm_channel",
                    "power_channels_match", "n_"]
EXCLUDE_SUFFIXES = ["_n"]

EXCLUDE_ODD_SESSIONS = False   # False = keep ALL sessions (odd ones included)
                               # True  = leave out the odd sessions listed below
ODD_SESSIONS = {
    "EFV_218": ["2026-01-16_17-50-35"],
    "EFV_220": ["2026-01-13_14-12-19", "2026-01-13_15-18-23", "2026-01-15_18-37-42"],
    "EFV_277": ["2026-02-20_12-02-47"],
    "NPC_249": ["2026-02-04_16-01-16", "2026-02-05_16-14-02", "2026-02-05_16-40-43"],
    "NPC_29":  ["P-7"],
    "NPC_30":  ["P4-HI"],
    "NPC_47":  ["P3-HD"],
}

GROUP_COLORS = {"WT": "#1f77b4", "NPC": "#d62728", "EFV": "#2ca02c"}

# t-SNE figure (same settings as nonlinear_embedding_new.py)
TSNE_PERPLEXITY = 15
RANDOM_STATE = 42
PLOT_MOUSE_SUMMARY = False     # True = also save the old per-mouse probability plot
# =============================================================================

# For each question: which groups take part, and which class each one becomes.
QUESTIONS = {
    "WT_vs_NPC_vs_EFV": {"WT": 0, "NPC": 1, "EFV": 2},   # 3 answers
    "WT_vs_sick": {"WT": 0, "NPC": 1, "EFV": 1},         # 2 answers (NPC+EFV merged)
    "NPC_vs_EFV": {"NPC": 0, "EFV": 1},                  # 2 answers
}
CLASS_NAMES = {
    "WT_vs_NPC_vs_EFV": ["WT", "NPC", "EFV"],
    "WT_vs_sick": ["WT", "sick"],
    "NPC_vs_EFV": ["NPC", "EFV"],
}

POWER_BANDS = ("Delta_", "Theta_", "Alpha_", "SlowGamma_", "MidGamma_", "FastGamma_", "Ripple_")


def is_power(col):
    return col.startswith(POWER_BANDS) and col.endswith(("_mean", "_max")) \
        and ("_PYR_" in col or "_SLM_" in col)


# -----------------------------------------------------------------------------
def load_data():
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET)
    df["group"] = df["group"].astype(str).str.strip()
    df["animal"] = df["animal"].astype(str).str.strip()

    if EXCLUDE_ODD_SESSIONS:
        wanted = {f"{a}|{s}" for a, ss in ODD_SESSIONS.items() for s in ss}
        key = df["animal"] + "|" + df["session"].astype(str).str.strip()
        print(f"[i] Leaving out {key.isin(wanted).sum()} odd sessions")
        df = df[~key.isin(wanted)]

    # keep only the groups of this question
    mapping = QUESTIONS[QUESTION]
    df = df[df["group"].isin(mapping)].reset_index(drop=True)
    y = df["group"].map(mapping).to_numpy()

    # feature columns
    cand = [c for c in df.columns if c not in ID_COLS
            and not c.startswith(tuple(EXCLUDE_PREFIXES))
            and not c.endswith(tuple(EXCLUDE_SUFFIXES))]
    X = df[cand].apply(pd.to_numeric, errors="coerce")
    X = X.loc[:, X.isna().mean() <= MAX_MISSING_COL]
    X = X.loc[:, X.std(skipna=True) > 0]

    if LOG_POWER:   # a transformation value by value: no information leaks
        for c in X.columns:
            if is_power(c) and X[c].min(skipna=True) > 0:
                X[c] = np.log10(X[c])

    return df[ID_COLS].copy(), X, y


def n_used(n_features):
    return n_features if K_FEATURES is None else min(K_FEATURES, n_features)


def make_model(n_features):
    n_classes = len(CLASS_NAMES[QUESTION])
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        # with k = all features this step keeps everything (so the code stays the same)
        ("select", SelectKBest(f_classif, k=n_used(n_features))),
        # shrinkage="auto" (Ledoit-Wolf) keeps LDA stable with few samples;
        # equal priors so the bigger class is not favoured
        ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto",
                                           priors=np.full(n_classes, 1 / n_classes))),
    ])


def class_probabilities(model, X):
    """Probability of each class (one column per class)."""
    return model.predict_proba(X)


def leave_one_mouse_out(ids, X, y):
    """Return, for every session, the probability of EACH class predicted by a
    model that never saw that mouse, plus which features each fold chose."""
    names = CLASS_NAMES[QUESTION]
    animals = ids["animal"].to_numpy()
    prob = np.full((len(y), len(names)), np.nan)    # one column per class
    chosen = []

    for train, test in LeaveOneGroupOut().split(X, y, groups=animals):
        mouse = animals[test][0]
        model = make_model(X.shape[1])
        model.fit(X.iloc[train], y[train])
        prob[test] = class_probabilities(model, X.iloc[test])
        mask = model.named_steps["select"].get_support()
        chosen.append({"left_out_mouse": mouse, "features": list(X.columns[mask])})
        said = names[int(prob[test].mean(axis=0).argmax())]
        print(f"  fold {len(chosen):2d}: left out {mouse:10s} -> model says {said}")
    return prob, chosen


def summarize(ids, y, prob):
    """The predicted class is the one with the highest probability.
    For a mouse: average the probabilities of its sessions, then take the highest."""
    names = CLASS_NAMES[QUESTION]
    pcols = [f"P_{n}" for n in names]
    ses = ids.copy()
    ses["true_class"] = [names[v] for v in y]
    ses[pcols] = prob
    ses["predicted"] = [names[i] for i in prob.argmax(axis=1)]
    ses["correct"] = ses["true_class"] == ses["predicted"]

    mouse = (ses.groupby(["animal", "group", "true_class"], as_index=False)
             .agg(n_sessions=("session", "size"),
                  **{c: (c, "mean") for c in pcols},
                  sessions_correct=("correct", "mean")))
    mouse["predicted"] = [names[i] for i in mouse[pcols].to_numpy().argmax(axis=1)]
    mouse["correct"] = mouse["true_class"] == mouse["predicted"]

    rows = []
    for level, t in (("session", ses), ("mouse", mouse)):
        cm = confusion_matrix(t["true_class"], t["predicted"], labels=names)
        row = {"level": level, "n": len(t),
               "accuracy": (t["true_class"] == t["predicted"]).mean(),
               "balanced_accuracy": balanced_accuracy_score(t["true_class"], t["predicted"])}
        for i, n in enumerate(names):
            row[f"{n}_correct"] = f"{cm[i, i]}/{cm[i].sum()}"
        rows.append(row)

    # confusion matrix per mouse: rows = real group, columns = what the model said
    cm_mouse = pd.DataFrame(confusion_matrix(mouse["true_class"], mouse["predicted"], labels=names),
                            index=[f"real {n}" for n in names],
                            columns=[f"said {n}" for n in names])
    return ses, mouse, pd.DataFrame(rows), cm_mouse


def plot_tsne(X, ses, summary, out_dir):
    """Same t-SNE map twice: colored by the REAL class (left) and by the class
    the model PREDICTED while that mouse was hidden (right)."""
    names = CLASS_NAMES[QUESTION]
    # t-SNE is only a picture: it uses all sessions together and does not
    # influence the classifier at all
    Xs = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X))
    perp = min(TSNE_PERPLEXITY, len(Xs) - 1)
    Y = TSNE(n_components=2, perplexity=perp, init="pca", learning_rate="auto",
             random_state=RANDOM_STATE).fit_transform(Xs)

    wrong = ~ses["correct"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)

    def dot(color, label, **kw):
        return plt.Line2D([], [], marker="o", ls="", color=color, markeredgecolor="k",
                          markersize=8, label=label, **kw)

    # LEFT: real group (WT blue, NPC red, EFV green)
    groups = list(QUESTIONS[QUESTION])
    for g in groups:
        m = (ses["group"] == g).to_numpy()
        axes[0].scatter(Y[m, 0], Y[m, 1], s=45, alpha=0.85, color=GROUP_COLORS[g],
                        edgecolor="k", linewidth=0.4)
    axes[0].legend(handles=[dot(GROUP_COLORS[g], g) for g in groups], fontsize=9)
    axes[0].set_title("REAL labels")

    # RIGHT: class assigned by the LDA, with the same 3 colors
    pred = ses["predicted"].to_numpy()
    grp = ses["group"].to_numpy()
    handles = []
    # one color = the label the model says. In WT_vs_sick the model only says
    # "WT" or "sick" (never NPC/EFV), so "sick" is drawn in red.
    pred_colors = {"WT": GROUP_COLORS["WT"], "sick": GROUP_COLORS["NPC"],
                   "NPC": GROUP_COLORS["NPC"], "EFV": GROUP_COLORS["EFV"]}
    for cls in names:
        m = pred == cls
        axes[1].scatter(Y[m, 0], Y[m, 1], s=45, alpha=0.85, color=pred_colors[cls],
                        edgecolor="k", linewidth=0.4)
        handles.append(dot(pred_colors[cls], f"said {cls}"))
    axes[1].scatter(Y[wrong, 0], Y[wrong, 1], s=170, facecolor="none",
                    edgecolor="k", linewidth=1.3)   # ring = model got it wrong
    n, n_wrong = len(wrong), int(wrong.sum())
    handles.append(plt.Line2D([], [], ls="", label=f"correct {n - n_wrong}/{n} ({100 * (n - n_wrong) / n:.0f}%)"))
    handles.append(plt.Line2D([], [], marker="o", ls="", markersize=12, markerfacecolor="none",
                              color="k", label=f"wrong {n_wrong}/{n} ({100 * n_wrong / n:.0f}%)"))
    axes[1].legend(handles=handles, fontsize=9)
    axes[1].set_title(f"Labels ASSIGNED by the {MODEL}")

    for ax in axes:
        ax.set_xlabel("t-SNE 1")
    axes[0].set_ylabel("t-SNE 2")

    s = summary.set_index("level")
    fig.suptitle(f"{MODEL} - {QUESTION} (leave-one-mouse-out) | balanced accuracy: "
                 f"sessions {s.loc['session', 'balanced_accuracy']:.2f}, "
                 f"mice {s.loc['mouse', 'balanced_accuracy']:.2f}   "
                 f"(chance = {1 / len(names):.2f})\n"
                 f"t-SNE perplexity {perp}")
    fig.tight_layout()
    fig.savefig(out_dir / "01_tsne_real_vs_predicted.png", dpi=200)


def plot_mice(ses, mouse, out_dir):
    names = CLASS_NAMES[QUESTION]
    if len(names) != 2:
        print("[i] The per-mouse plot is only made for the 2-class questions")
        return
    pcol = f"P_{names[1]}"
    mouse = mouse.sort_values(["true_class", pcol]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(mouse)), 4.5))
    for i, r in mouse.iterrows():
        c = GROUP_COLORS.get(r["group"], "gray")
        s = ses.loc[ses["animal"] == r["animal"], pcol]
        ax.scatter(np.full(len(s), i), s, s=18, color=c, alpha=0.45, edgecolor="none")
        ax.scatter(i, r[pcol], marker="D", s=70, color=c,
                   edgecolor="k" if r["correct"] else "red", linewidth=1.2 if r["correct"] else 2.5,
                   zorder=3)
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set_xticks(range(len(mouse)), mouse["animal"], rotation=60, fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel(f"P({names[1]})  predicted while the mouse was hidden")
    ax.set_title(f"{QUESTION}: each mouse predicted by a model that never saw it\n"
                 f"dots = sessions, diamond = mouse mean (red edge = wrong)")
    handles = [plt.Line2D([], [], marker="D", ls="", color=GROUP_COLORS[g], label=g)
               for g in QUESTIONS[QUESTION]]
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "02_mouse_predictions.png", dpi=200)


def main():
    global QUESTION
    for q in QUESTIONS_TO_RUN:
        QUESTION = q
        print("\n" + "=" * 70 + f"\n{q}\n" + "=" * 70)
        run_question()
    plt.show()   # opens all the figures at the end


def run_question():
    tag = RUN_TAG + ("_no_odd" if EXCLUDE_ODD_SESSIONS else "")
    out_dir = RESULTS_DIR / f"{MODEL.lower()}_results{tag}" / QUESTION
    out_dir.mkdir(parents=True, exist_ok=True)

    ids, X, y = load_data()
    names = CLASS_NAMES[QUESTION]
    n_mice = ids.groupby("group")["animal"].nunique().to_dict()
    print(f"\nQuestion: {QUESTION}  (possible answers: {', '.join(names)})")
    print(f"Excel: {EXCEL_PATH.name}  |  odd sessions excluded: {EXCLUDE_ODD_SESSIONS}")
    print(f"{len(y)} sessions, {X.shape[1]} candidate features, "
          f"{n_used(X.shape[1])} used in each fold")
    print(f"Mice per group: {n_mice}\n")

    prob, chosen = leave_one_mouse_out(ids, X, y)
    ses, mouse, summary, cm_mouse = summarize(ids, y, prob)

    print(f"\nRESULTS (chance level for balanced accuracy = {1 / len(names):.2f}):")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\nConfusion matrix per MOUSE (rows = real, columns = what the model said):")
    print(cm_mouse.to_string())
    print("\nPer mouse:")
    print(mouse.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    # how often each feature was chosen across the folds (stability)
    counts = pd.Series([f for c in chosen for f in c["features"]]).value_counts()
    freq = (counts / len(chosen)).rename("fraction_of_folds").reset_index()
    freq.columns = ["feature", "fraction_of_folds"]
    print("\nFeatures chosen most often across folds:")
    print(freq.head(15).to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    with pd.ExcelWriter(out_dir / "predictions.xlsx") as w:
        summary.to_excel(w, sheet_name="summary", index=False)
        cm_mouse.to_excel(w, sheet_name="confusion_per_mouse")
        mouse.to_excel(w, sheet_name="per_mouse", index=False)
        ses.to_excel(w, sheet_name="per_session", index=False)
        freq.to_excel(w, sheet_name="features_chosen", index=False)
        pd.DataFrame({"setting": ["MODEL", "QUESTION", "EXCEL", "K_FEATURES", "LOG_POWER",
                                  "MAX_MISSING_COL", "EXCLUDE_ODD_SESSIONS", "n_sessions", "n_features"],
                      "value": [MODEL, QUESTION, EXCEL_PATH.name,
                                str(K_FEATURES), LOG_POWER, MAX_MISSING_COL,
                                EXCLUDE_ODD_SESSIONS, len(y), X.shape[1]]}
                     ).to_excel(w, sheet_name="settings", index=False)

    plot_tsne(X, ses, summary, out_dir)
    if PLOT_MOUSE_SUMMARY:
        plot_mice(ses, mouse, out_dir)
    print(f"\nResults saved in {out_dir}")


if __name__ == "__main__":
    main()