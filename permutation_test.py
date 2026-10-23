"""
Permutation test: the real CHANCE LEVEL of each classifier and comparison
=========================================================================
What accuracy would the classifiers get by pure chance, with exactly our data and our
procedure? To find out, the group labels are shuffled and the WHOLE analysis of
classifiers_new_selection.py is repeated with the fake labels (leave-one-mouse-out,
feature selection inside every fold, LDA / SVM / LogReg), many times.

HOW THE LABELS ARE SHUFFLED
  - PER MOUSE: all the sessions of a mouse keep the same fake label (shuffling sessions
    one by one would give a mouse sessions with different labels, and the model could
    "cheat" by recognising the mouse).
  - Each group keeps its number of mice (e.g. 3 WT and 8 NPC).
  - For pairs, only the mice of the two groups are shuffled; the third group keeps its
    label (it only takes part in the Tukey test of option A, never in the model).
  - If a comparison has at most N_PERMUTATIONS possible shufflings (WT_vs_EFV has 84;
    WT_vs_NPC has 165), ALL of them are run once (exact test). Otherwise N_PERMUTATIONS
    random ones.

RESULTS (per model and comparison, per session and per mouse)
    real        : balanced accuracy with the real labels (the same as classifiers_new_selection.py)
    chance_mean : mean balanced accuracy of the shuffled runs = the real chance level.
                  With leave-one-mouse-out and few mice it is usually BELOW the theoretical
                  0.50 (each hidden mouse leaves its own group smaller in training).
    p_value     : fraction of shuffled runs that did as well as or better than the real one.
                  Small p (e.g. < 0.05) = the real accuracy is unlikely to be luck.
                  With N runs the smallest possible p is about 1/N.

It runs several shuffles at the same time on the cores of the processor (N_JOBS; by
default all except one, so the computer stays usable). This is safe: the computer may get
warm and the fan loud, and other programs may feel slower meanwhile. It takes a while:
run it once, when the classifier results are final (with the charger plugged in).

INPUT : RESULTSNatalia/features_final.xlsx (via classifiers_new_selection.py)
OUTPUT: RESULTSNatalia/<model>_new_selection/<comparison>/   (next to the classifier results)
          permutation.xlsx           sheet "summary" (real, chance level, p) + "runs" (every shuffle)
          05_permutation_test.png    histogram of the shuffled runs vs the real accuracy
        RESULTSNatalia/permutation_test_new_selection_summary.xlsx   all models x comparisons

Keep this file in the same folder as classifiers_new_selection.py and select_features.py.
All the settings of the models (SVM_C, LOG_POWER, EXCLUDE_ODD_SESSIONS...) and of the
selection (KEEP_BOTH, USE_MAX...) are taken from those two files, so the shuffled runs are
always done exactly like the real one.
"""

import time
from itertools import combinations
from math import factorial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed

import classifiers_new_selection as cl   # folds, models, data loading (same folder)
import select_features as sf             # paths + selection rule (same folder)

# =============================================================================
# SETTINGS
# =============================================================================
COMPARISONS_TO_RUN = ["WT_vs_NPC", "NPC_vs_EFV", "WT_vs_EFV", "WT_vs_NPC_vs_EFV"]
MODELS_TO_RUN = ["LDA", "SVM", "LogReg"]
N_PERMUTATIONS = 100     # shuffled runs per comparison (fewer possible -> all of them, exact)
PERMUTATION_SEED = 0     # fixed seed -> the same shufflings every time
N_JOBS = -2              # processor cores used at the same time:
                         #   -2 = all except one (default: the laptop stays usable)
                         #   -1 = all of them (fastest, the laptop may feel slow meanwhile)
                         #    1 = only one (slowest, like a normal script); 2, 3... = that number
SHOW_FIGURES = False     # True = open the figures at the end (they are always saved)
# =============================================================================


def n_cores_used():
    """How many cores N_JOBS means on this computer (-1 = all, -2 = all except one...)."""
    n = cpu_count()
    return max(1, n + 1 + N_JOBS) if N_JOBS < 0 else max(1, min(N_JOBS, n))


def shufflings(mouse_groups, names, rng):
    """Fake assignments mouse -> group (each group keeps its number of mice).
    Returns (list of assignments, exact). exact = True when ALL the possible ones are listed
    (the real one is then one of them)."""
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


def one_run(comparison, fake, cids, cvals, sel_df, names, models):
    """The whole classifier analysis with fake labels -> balanced accuracy of every model."""
    sel_fake = sel_df.copy()
    in_comp = sel_fake["animal"].isin(fake.index)
    sel_fake.loc[in_comp, "group"] = sel_fake.loc[in_comp, "animal"].map(fake)
    cids_fake = cids.copy()
    cids_fake["group"] = cids_fake["animal"].map(fake)
    y_fake = cids_fake["group"].map({g: k for k, g in enumerate(names)}).to_numpy()
    prob, _, _ = cl.leave_one_mouse_out(comparison, cids_fake, cvals, sel_fake, y_fake, len(names),
                                        models=models)
    out = {}
    for m in models:
        ba_s, ba_m, _ = cl.balanced_acc(cids_fake, y_fake, prob[m], names)
        out[m] = (ba_s, ba_m)
    return out


def p_value(real, null, exact):
    """Fraction of shuffled runs with a balanced accuracy >= the real one.
    Exact test: the real labelling is one of the runs. Random runs: (k + 1) / (N + 1)."""
    k = int(np.sum(null >= real - 1e-12))
    return k / len(null) if exact else (k + 1) / (len(null) + 1)


def run_comparison(comparison, ids, sel_df, values):
    names, cids, cvals, y = cl.comparison_data(comparison, ids, values)
    mouse_groups = cids.groupby("animal")["group"].first()
    print(f"{len(cids)} sessions, mice per group: {mouse_groups.value_counts().to_dict()}")

    # real labels (the same result as classifiers_new_selection.py)
    prob, _, _ = cl.leave_one_mouse_out(comparison, cids, cvals, sel_df, y, len(names),
                                        models=MODELS_TO_RUN)
    real = {m: cl.balanced_acc(cids, y, prob[m], names)[:2] for m in MODELS_TO_RUN}

    rng = np.random.default_rng(PERMUTATION_SEED)
    fakes, exact = shufflings(mouse_groups, names, rng)
    print(f"  {len(fakes)} " + ("shufflings = ALL the possible ones (exact test)" if exact
                                else "random shufflings") + " of the labels per mouse")
    t0 = time.time()
    results = []
    n_workers = n_cores_used()
    chunk = max(len(fakes) // 20, 4 * n_workers)   # batches big enough to keep every core busy
    print("  ", end="", flush=True)
    with Parallel(n_jobs=n_workers) as parallel:
        for start in range(0, len(fakes), chunk):
            part = fakes[start:start + chunk]
            results += parallel(delayed(one_run)(comparison, f, cids, cvals, sel_df, names, MODELS_TO_RUN)
                                for f in part)
            print(".", end="", flush=True)
    print(f" done in {time.time() - t0:.0f} s")

    summary_rows = []
    for m in MODELS_TO_RUN:
        runs = pd.DataFrame({"run": range(1, len(fakes) + 1),
                             "BA_sessions": [r[m][0] for r in results],
                             "BA_mice": [r[m][1] for r in results],
                             "is_real_labels": [bool((f == mouse_groups.loc[f.index]).all()) for f in fakes]})
        rows = []
        for level, col, k in (("session", "BA_sessions", 0), ("mouse", "BA_mice", 1)):
            null = runs[col].to_numpy()
            rows.append({"level": level, "real": real[m][k], "chance_mean": null.mean(),
                         "chance_sd": null.std(), "p_value": p_value(real[m][k], null, exact),
                         "theoretical_chance": 1 / len(names), "n_runs": len(null),
                         "type": "exact (all shufflings)" if exact else f"random (seed {PERMUTATION_SEED})"})
        summary = pd.DataFrame(rows)
        s = summary.set_index("level")
        print(f"  {m:6s} sessions: real {s.loc['session', 'real']:.2f} vs chance "
              f"{s.loc['session', 'chance_mean']:.2f} (p = {s.loc['session', 'p_value']:.3f}) | "
              f"mice: real {s.loc['mouse', 'real']:.2f} vs chance {s.loc['mouse', 'chance_mean']:.2f} "
              f"(p = {s.loc['mouse', 'p_value']:.3f})")

        out_dir = sf.RESULTS_DIR / f"{cl.FOLDER[m]}_new_selection" / comparison
        out_dir.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(out_dir / "permutation.xlsx") as w:
            summary.to_excel(w, sheet_name="summary", index=False)
            runs.to_excel(w, sheet_name="runs", index=False)
        plot(m, comparison, summary, runs, exact, out_dir)
        summary_rows.append({"comparison": comparison, "model": m,
                             "real_sessions": s.loc["session", "real"],
                             "chance_sessions": s.loc["session", "chance_mean"],
                             "p_sessions": s.loc["session", "p_value"],
                             "real_mice": s.loc["mouse", "real"],
                             "chance_mice": s.loc["mouse", "chance_mean"],
                             "p_mice": s.loc["mouse", "p_value"],
                             "n_runs": len(fakes), "exact": exact})
    return summary_rows


def plot(model, comparison, summary, runs, exact, out_dir):
    s = summary.set_index("level")
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
    fig.suptitle(f"{model} - {comparison}: permutation test, labels shuffled per mouse ({kind})\n"
                 f"p = fraction of shuffled runs as good as or better than the real one", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "05_permutation_test.png", dpi=200)
    if not SHOW_FIGURES:
        plt.close(fig)


def main():
    ids, sel_df, values = cl.load()
    print(f"{len(ids)} sessions | models: {', '.join(MODELS_TO_RUN)} | "
          f"selection rule: KEEP_BOTH = {sf.KEEP_BOTH}, USE_MAX = {sf.USE_MAX} | "
          f"cores used: {n_cores_used()} of {cpu_count()}")
    rows = []
    for comparison in COMPARISONS_TO_RUN:
        print("\n" + "=" * 70 + f"\n{comparison}\n" + "=" * 70)
        rows += run_comparison(comparison, ids, sel_df, values)
    table = pd.DataFrame(rows)
    table.to_excel(sf.RESULTS_DIR / "classifiers_permutation_test_new_selection.xlsx", index=False)
    print("\n" + "=" * 70 + "\nSUMMARY (balanced accuracy: real vs chance level)\n" + "=" * 70)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if SHOW_FIGURES:
        plt.show()


if __name__ == "__main__":
    main()