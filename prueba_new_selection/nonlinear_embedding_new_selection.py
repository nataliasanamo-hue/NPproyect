"""
t-SNE and UMAP with the features selected FOR EACH COMPARISON
=============================================================
Same analysis as nonlinear_embedding_new.py, but for every comparison in
COMPARISONS_TO_RUN (set in pca_analysis_new_selection.py) it reads
    RESULTSNatalia/feature_selection/features_selected_<comparison>.xlsx
and saves each method in its own folder, so nothing older is overwritten:
    RESULTSNatalia/tsne_new_selection/<comparison>/   01_tsne.png, tsne_embeddings.xlsx
    RESULTSNatalia/umap_new_selection/<comparison>/   02_umap.png, umap_embeddings.xlsx

It imports the settings and load_data of pca_analysis_new_selection.py, so both use
exactly the same data. Keep this file in the same folder as that one.

Idea
----
PCA keeps straight-line directions of maximum variance (global structure).
t-SNE and UMAP instead try to keep NEIGHBOURS together: sessions that are similar
in the feature space end up close in the 2-D map. They can reveal groups that
PCA misses, but:
  - distances BETWEEN far-apart clusters and cluster SIZES are not meaningful,
  - the axes have no units or meaning (unlike PC1, PC2),
  - the result depends on a parameter (perplexity / n_neighbors), so we try several.

Each figure has two rows:
  row 1: colored by GROUP                   -> do the groups separate?
  row 2: colored by ANIMAL (marker = group) -> are the "clusters" just animals?

Silhouette score (shown in titles): how well the GROUPS form separate clusters.
  ~1 = perfectly separated, ~0 = overlapping, < 0 = mixed / wrong side.
The printed silhouette in the ORIGINAL feature space is the honest one
(the 2-D maps can exaggerate separation).

Requirements:  pip install umap-learn   (plus the ones of the PCA script)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import pca_analysis_new_selection as cfg   # settings + data (same folder)

# =============================================================================
# SETTINGS
# =============================================================================
PERPLEXITIES = [5, 15, 30]   # t-SNE: roughly "how many neighbours each point looks at"
N_NEIGHBORS = [5, 15, 30]    # UMAP: same idea (small = local detail, large = global shape)
MIN_DIST = 0.1               # UMAP: how tightly points can pack together
RANDOM_STATE = 42            # fixed seed -> same picture every time you run it

ANIMAL_COL = "animal"
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


def plot_grid(results, labels, animals, method, out_file, comparison):
    """results = list of (parameter text, 2-D embedding)."""
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(4.8 * n, 9.5), squeeze=False)
    groups = pd.unique(labels)
    animal_list = sorted(pd.unique(animals))
    cmap = plt.get_cmap("tab20", max(len(animal_list), 1))
    animal_color = {a: cmap(i) for i, a in enumerate(animal_list)}

    for col, (param, Y) in enumerate(results):
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
    fig.suptitle(f"{comparison} – {method}: trust only patterns that appear with every parameter")
    fig.tight_layout()
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    if not cfg.SHOW_FIGURES:
        plt.close(fig)


def run_comparison(comparison, has_umap):
    ids, feats = cfg.load_data(cfg.excel_for(comparison), cfg.SHEET, cfg.ID_COLS)
    labels = ids[cfg.GROUP_COL].astype(str).to_numpy()
    animals = ids[ANIMAL_COL].astype(str).to_numpy()
    print(f"Data: {feats.shape[0]} sessions x {feats.shape[1]} features, "
          f"{len(np.unique(animals))} animals")

    X = StandardScaler().fit_transform(feats.to_numpy())
    print(f"Silhouette of the groups in the ORIGINAL feature space: {silhouette(X, labels):.2f}")

    # ---- t-SNE ----
    out_dir = cfg.output_dir("tsne", comparison)
    out_dir.mkdir(parents=True, exist_ok=True)
    out, res = ids.copy(), []
    for p in PERPLEXITIES:
        if p >= len(X):
            print(f"[!] perplexity {p} skipped (must be < number of sessions)")
            continue
        Y = run_tsne(X, p)
        res.append((f"perplexity {p}", Y))
        out[f"tSNE_p{p}_1"], out[f"tSNE_p{p}_2"] = Y[:, 0], Y[:, 1]
        print(f"t-SNE perplexity {p:>2}: silhouette = {silhouette(Y, labels):.2f}")
    if res:
        plot_grid(res, labels, animals, "t-SNE", out_dir / "01_tsne.png", comparison)
        out.to_excel(out_dir / "tsne_embeddings.xlsx", index=False)
        print(f"Saved in {out_dir}")

    # ---- UMAP ----
    if not has_umap:
        return
    out_dir = cfg.output_dir("umap", comparison)
    out_dir.mkdir(parents=True, exist_ok=True)
    out, res = ids.copy(), []
    for k in N_NEIGHBORS:
        if k >= len(X):
            print(f"[!] n_neighbors {k} skipped (must be < number of sessions)")
            continue
        Y = run_umap(X, k)
        res.append((f"n_neighbors {k}", Y))
        out[f"UMAP_k{k}_1"], out[f"UMAP_k{k}_2"] = Y[:, 0], Y[:, 1]
        print(f"UMAP n_neighbors {k:>2}: silhouette = {silhouette(Y, labels):.2f}")
    if res:
        plot_grid(res, labels, animals, "UMAP", out_dir / "02_umap.png", comparison)
        out.to_excel(out_dir / "umap_embeddings.xlsx", index=False)
        print(f"Saved in {out_dir}")


def main():
    try:
        import umap  # noqa: F401
        has_umap = True
    except ImportError:
        has_umap = False
        print("[!] UMAP not installed (pip install umap-learn): only t-SNE will be made")

    for c in cfg.COMPARISONS_TO_RUN:
        print("\n" + "=" * 70 + f"\n{c}\n" + "=" * 70)
        try:
            run_comparison(c, has_umap)
        except FileNotFoundError as e:
            print(f"[!] {e}")
    if cfg.SHOW_FIGURES:
        plt.show()


if __name__ == "__main__":
    main()