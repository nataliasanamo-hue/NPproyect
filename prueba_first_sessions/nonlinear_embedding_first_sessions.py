"""
t-SNE and UMAP of the session features (nonlinear dimensionality reduction)
===========================================================================
Uses EXACTLY the same data and feature selection as pca_analysis_first_sessions.py
(it imports its settings and load_data function), so both analyses are comparable.
It reads the same Excel as pca_analysis_first_sessions.py (features_final.xlsx, or
features_selected.xlsx if USE_SELECTED = True there).
Keep this file in the same folder as pca_analysis_first_sessions.py.

Idea
----
PCA keeps straight-line directions of maximum variance (global structure).
t-SNE and UMAP instead try to keep NEIGHBOURS together: sessions that are similar
in the 19-D feature space end up close in the 2-D map. They can reveal groups that
PCA misses, but:
  - distances BETWEEN far-apart clusters and cluster SIZES are not meaningful,
  - the axes have no units or meaning (unlike PC1, PC2),
  - the result depends on a parameter (perplexity / n_neighbors), so we try several.

Each figure has two rows:
  row 1: colored by GROUP (WT / NPC / EFV)  -> do the groups separate?
  row 2: colored by ANIMAL (marker = group) -> are the "clusters" just animals?

Silhouette score (shown in titles): how well the GROUPS form separate clusters.
  ~1 = perfectly separated, ~0 = overlapping, < 0 = mixed / wrong side.

Requirements:  pip install umap-learn   (plus the ones of pca_analysis.py)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Settings (Excel path, columns, excluded features) come from the PCA script, so both
# analyses use exactly the same data. It must be in the same folder as this file.
# Python needs the exact file name, capital letters included.
import pca_analysis_first_sessions as cfg   # settings come from pca_analysis_first_sessions.py (same folder)

# =============================================================================
# SETTINGS
# =============================================================================
PERPLEXITIES = [5, 15, 30]   # t-SNE: roughly "how many neighbours each point looks at"
N_NEIGHBORS = [5, 15, 30]    # UMAP: same idea (small = local detail, large = global shape)
MIN_DIST = 0.1               # UMAP: how tightly points can pack together
RANDOM_STATE = 42            # fixed seed -> same picture every time you run it

ANIMAL_COL = "animal"
OUTPUT_DIR = cfg.RESULTS_DIR / f"embedding_results{getattr(cfg, 'RUN_TAG', '')}"  # *_selected if USE_SELECTED
GROUP_MARKERS = {"WT": "o", "NPC": "s", "EFV": "^"}
# =============================================================================


def silhouette(Y, labels):
    return silhouette_score(Y, labels) if len(np.unique(labels)) > 1 else np.nan


def run_tsne(X, perplexity):
    return TSNE(n_components=2, perplexity=perplexity, init="pca",
                learning_rate="auto", random_state=RANDOM_STATE).fit_transform(X)


def run_umap(X, n_neighbors):
    import umap
    return umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=MIN_DIST,
                     random_state=RANDOM_STATE).fit_transform(X)


def plot_grid(results, labels, animals, method, fname):
    """results = list of (parameter text, 2-D embedding)."""
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(4.8 * n, 9.5), squeeze=False)
    groups = pd.unique(labels)
    animal_list = sorted(pd.unique(animals))
    cmap = plt.get_cmap("tab20", max(len(animal_list), 1))
    animal_color = {a: cmap(i) for i, a in enumerate(animal_list)}

    for col, (param, Y) in enumerate(results):
        # Row 1: groups
        ax = axes[0, col]
        for gi, g in enumerate(groups):
            m = labels == g
            ax.scatter(Y[m, 0], Y[m, 1], s=40, color=cfg.color_for(g, gi),
                       edgecolor="k", alpha=0.85, label=g)
        ax.set_title(f"{method} – {param}\nsilhouette (groups) = {silhouette(Y, labels):.2f}")
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("Colored by GROUP")
            ax.legend()

        # Row 2: animals (marker = group)
        ax = axes[1, col]
        for a in animal_list:
            for g in groups:
                m = (animals == a) & (labels == g)
                if m.any():
                    ax.scatter(Y[m, 0], Y[m, 1], s=40, color=animal_color[a],
                               marker=GROUP_MARKERS.get(g, "o"), edgecolor="k",
                               alpha=0.9, label=f"{a}" if col == n - 1 else None)
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("Colored by ANIMAL (marker = group)")

    axes[1, -1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=6,
                       title="animal", title_fontsize=7)
    fig.suptitle(f"{method}: try several parameters – trust only patterns that appear in all")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / fname, dpi=200, bbox_inches="tight")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ids, feats = cfg.load_data(cfg.EXCEL_PATH, cfg.SHEET, cfg.ID_COLS)
    labels = ids[cfg.GROUP_COL].astype(str).to_numpy()
    animals = (ids[ANIMAL_COL].astype(str).to_numpy() if ANIMAL_COL in ids
               else np.array(["?"] * len(ids)))
    print(f"\nData: {feats.shape[0]} sessions x {feats.shape[1]} features, "
          f"{len(np.unique(animals))} animals")

    X = StandardScaler().fit_transform(feats.to_numpy())
    print(f"Silhouette of the groups in the ORIGINAL feature space: {silhouette(X, labels):.2f}")

    out = ids.copy()

    # ---- t-SNE ----
    tsne_res = []
    for p in PERPLEXITIES:
        if p >= len(X):
            print(f"[!] perplexity {p} skipped (must be < number of sessions)")
            continue
        Y = run_tsne(X, p)
        tsne_res.append((f"perplexity {p}", Y))
        out[f"tSNE_p{p}_1"], out[f"tSNE_p{p}_2"] = Y[:, 0], Y[:, 1]
        print(f"t-SNE perplexity {p:>2}: silhouette = {silhouette(Y, labels):.2f}")
    plot_grid(tsne_res, labels, animals, "t-SNE", "01_tsne.png")

    # ---- UMAP ----
    try:
        import umap  # noqa: F401
    except ImportError:
        print("\n[!] UMAP not installed. Run:  pip install umap-learn   and run again.")
    else:
        umap_res = []
        for k in N_NEIGHBORS:
            if k >= len(X):
                print(f"[!] n_neighbors {k} skipped (must be < number of sessions)")
                continue
            Y = run_umap(X, k)
            umap_res.append((f"n_neighbors {k}", Y))
            out[f"UMAP_k{k}_1"], out[f"UMAP_k{k}_2"] = Y[:, 0], Y[:, 1]
            print(f"UMAP n_neighbors {k:>2}: silhouette = {silhouette(Y, labels):.2f}")
        plot_grid(umap_res, labels, animals, "UMAP", "02_umap.png")

    out.to_excel(OUTPUT_DIR / "embeddings.xlsx", index=False)
    print(f"\nFigures and embeddings.xlsx saved in '{OUTPUT_DIR}'")
    plt.show()


if __name__ == "__main__":
    main()