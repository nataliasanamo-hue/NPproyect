"""
PCA (Principal Component Analysis) of LFP / CSD ripple features
================================================================
Groups: WT, NPC, EFV

What this script does
---------------------
1. Loads RESULTSNatalia/features_final.xlsx (one row = one session), made by
   build_feature_table.py. Results go to RESULTSNatalia/pca_results/.
2. Standardizes every feature (mean 0, std 1) so features measured in
   different units (µV, power, CSD...) count equally.
3. Runs a full PCA and draws:
     - the scree plot: eigenvalue (lambda) of each component
     - the explained-variance plot: % of variance per component + cumulative %
4. Reports the cumulative % of variance explained for N = 3 ... 6 and
   picks the smallest N that reaches VARIANCE_TARGET.
5. Visualizes the data in the new space:
     - 3D scatter of PC1, PC2, PC3 colored by group
     - "pair plot" of all N components (every PC against every other PC)
     - loadings heatmap: how much each original feature contributes to each PC
6. Saves everything (figures + an Excel with scores, loadings and variance).

Requirements:  pip install numpy pandas matplotlib scikit-learn openpyxl
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# =============================================================================
# SETTINGS - change these to match your Excel
# =============================================================================
baseDir = Path(r"D:\Violeta\ANALISIS")
RESULTS_DIR = baseDir / "RESULTSNatalia"
EXCEL_PATH = RESULTS_DIR / "features_final.xlsx"   # made by build_feature_table.py
SHEET = "features"
GROUP_COL = "group"                       # column with WT / NPC / EFV
ID_COLS = ["animal", "session", "group"]  # identify each row, not used in PCA

# Columns that are NOT biological features (electrode channels, checks, counts).
# Any column whose name starts with one of these prefixes is left out of the PCA.
EXCLUDE_PREFIXES = ["ch_", "config_", "power_pyr_channel", "power_slm_channel",
                    "power_channels_match", "n_"]
EXCLUDE_SUFFIXES = ["_n"]                 # e.g. CSD_PYR_n (number of ripples)

MAX_MISSING_COL = 0.20   # drop a feature if > 20 % of sessions have no value

N_MIN, N_MAX = 3, 6             # range of dimensions to try
VARIANCE_TARGET = 0.80          # 0.80 = we want >= 80 % explained variance
N_FORCE = None                  # set e.g. N_FORCE = 4 to choose N yourself

OUTPUT_DIR = RESULTS_DIR / "pca_results"
GROUP_COLORS = {"WT": "#1f77b4", "NPC": "#d62728", "EFV": "#2ca02c"}
# =============================================================================


def color_for(group, i):
    fallback = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return GROUP_COLORS.get(group, fallback[i % len(fallback)])


def load_data(path, sheet, id_cols):
    """Return (id table, numeric feature table) with incomplete rows removed."""
    df = pd.read_excel(path, sheet_name=sheet)
    ids_present = [c for c in id_cols if c in df.columns]

    feats = df.drop(columns=ids_present)
    excluded = [c for c in feats.columns
                if c.startswith(tuple(EXCLUDE_PREFIXES)) or c.endswith(tuple(EXCLUDE_SUFFIXES))]
    feats = feats.drop(columns=excluded)
    print(f"[i] Left out (channels / checks / counts): {excluded}")

    numeric = feats.select_dtypes(include=np.number)
    ignored = [c for c in feats.columns if c not in numeric.columns]
    if ignored:
        print(f"[!] Non-numeric columns ignored: {ignored}")

    # Features missing in many sessions would make us lose those sessions
    too_missing = numeric.columns[numeric.isna().mean() > MAX_MISSING_COL].tolist()
    numeric = numeric.drop(columns=too_missing)
    if too_missing:
        print(f"[!] Removed features missing in > {MAX_MISSING_COL:.0%} of sessions: {too_missing}")

    # Columns with no variation carry no information for PCA
    constant = numeric.columns[numeric.std(ddof=0) == 0].tolist()
    numeric = numeric.drop(columns=constant)
    if constant:
        print(f"[!] Removed constant columns: {constant}")

    # PCA cannot handle missing values: drop rows with any NaN
    complete = numeric.notna().all(axis=1)
    if (~complete).any():
        print(f"[!] Removed {(~complete).sum()} row(s) with missing values")

    ids = df.loc[complete, ids_present].reset_index(drop=True)
    numeric = numeric.loc[complete].reset_index(drop=True)
    return ids, numeric


def plot_scree(pca_full, n_min, n_max, out_dir):
    eig = pca_full.explained_variance_                 # lambda of each PC
    ratio = pca_full.explained_variance_ratio_ * 100   # lambda / sum(lambda)
    cum = np.cumsum(ratio)
    k = np.arange(1, len(eig) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(k, eig, "o-", color="black")
    ax1.axhline(1, ls="--", color="gray", label="λ = 1 (Kaiser criterion)")
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Eigenvalue (λ)")
    ax1.set_title("Scree plot")
    ax1.set_xticks(k)
    ax1.legend()

    ax2.bar(k, ratio, alpha=0.6, label="Variance of each PC")
    ax2.plot(k, cum, "o-", color="red", label="Cumulative variance")
    ax2.axvspan(n_min - 0.5, n_max + 0.5, color="orange", alpha=0.15,
                label=f"N = {n_min}–{n_max}")
    for n in range(n_min, min(n_max, len(k)) + 1):
        ax2.annotate(f"{cum[n - 1]:.0f}%", (n, cum[n - 1]),
                     textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=8)
    ax2.set_xlabel("Principal component")
    ax2.set_ylabel("Explained variance (%)")
    ax2.set_title("Explained variance")
    ax2.set_xticks(k)
    ax2.set_ylim(0, 105)
    ax2.legend(loc="center right")

    fig.tight_layout()
    fig.savefig(out_dir / "01_scree_explained_variance.png", dpi=200)


def plot_3d(scores, labels, ratio, out_dir):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(projection="3d")
    for i, g in enumerate(pd.unique(labels)):
        m = labels == g
        ax.scatter(scores[m, 0], scores[m, 1], scores[m, 2], s=45,
                   color=color_for(g, i), edgecolor="k", alpha=0.85, label=g)
    ax.set_xlabel(f"PC1 ({ratio[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({ratio[1]:.1f}%)")
    ax.set_zlabel(f"PC3 ({ratio[2]:.1f}%)")
    ax.set_title(f"PCA space – PC1-3 explain {ratio[:3].sum():.1f}% of variance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "02_pca_3d.png", dpi=200)


def plot_pairs(scores, labels, ratio, n, out_dir):
    groups = pd.unique(labels)
    fig, axes = plt.subplots(n, n, figsize=(2.6 * n, 2.6 * n))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            for gi, g in enumerate(groups):
                m = labels == g
                c = color_for(g, gi)
                if i == j:   # diagonal: distribution of each PC per group
                    ax.hist(scores[m, i], bins=15, alpha=0.5, color=c)
                else:
                    ax.scatter(scores[m, j], scores[m, i], s=12, alpha=0.8,
                               color=c, label=g if (i, j) == (0, 1) else None)
            if i == n - 1:
                ax.set_xlabel(f"PC{j + 1} ({ratio[j]:.0f}%)")
            if j == 0:
                ax.set_ylabel(f"PC{i + 1} ({ratio[i]:.0f}%)")
            ax.tick_params(labelsize=7)
    fig.legend(loc="upper right")
    fig.suptitle(f"All pairs of the first {n} principal components")
    fig.tight_layout()
    fig.savefig(out_dir / "03_pca_pairs.png", dpi=200)


def plot_loadings(loadings, out_dir):
    n_feat, n_pc = loadings.shape
    fig, ax = plt.subplots(figsize=(1.1 * n_pc + 4, 0.35 * n_feat + 2))
    im = ax.imshow(loadings.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n_pc), loadings.columns)
    ax.set_yticks(range(n_feat), loadings.index, fontsize=8)
    for r in range(n_feat):
        for c in range(n_pc):
            ax.text(c, r, f"{loadings.values[r, c]:.2f}", ha="center",
                    va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Loading (weight of feature in PC)")
    ax.set_title("Loadings: which features build each PC")
    fig.tight_layout()
    fig.savefig(out_dir / "04_loadings.png", dpi=200)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load
    ids, feats = load_data(EXCEL_PATH, SHEET, ID_COLS)
    labels = (ids[GROUP_COL].astype(str).to_numpy() if GROUP_COL in ids
              else np.array(["all"] * len(feats)))
    print(f"\nData: {feats.shape[0]} sessions x {feats.shape[1]} features")
    print("Features used:", list(feats.columns))
    print("Observations per group:",
          pd.Series(labels).value_counts().to_dict())

    # 2) Standardize (z-score each feature)
    X = StandardScaler().fit_transform(feats.to_numpy())

    # 3) Full PCA to see ALL eigenvalues
    pca_full = PCA().fit(X)
    cum = np.cumsum(pca_full.explained_variance_ratio_) * 100
    n_avail = pca_full.n_components_
    plot_scree(pca_full, N_MIN, N_MAX, OUTPUT_DIR)

    # 4) Variance for N = N_MIN ... N_MAX and choice of N
    n_range = range(N_MIN, min(N_MAX, n_avail) + 1)
    print("\nCumulative explained variance:")
    for n in n_range:
        print(f"  N = {n}: {cum[n - 1]:5.1f} %")
    if N_FORCE:
        n_final = N_FORCE
    else:
        n_final = next((n for n in n_range if cum[n - 1] >= VARIANCE_TARGET * 100),
                       min(N_MAX, n_avail))
    print(f"\n-> Using N = {n_final} ({cum[n_final - 1]:.1f} % of variance)")
    if cum[n_final - 1] < VARIANCE_TARGET * 100:
        print(f"   (target of {VARIANCE_TARGET:.0%} not reached within N <= {N_MAX})")

    # 5) PCA with the chosen N and visualizations
    pca = PCA(n_components=n_final).fit(X)
    scores = pca.transform(X)
    ratio = pca.explained_variance_ratio_ * 100
    pc_names = [f"PC{i + 1}" for i in range(n_final)]

    plot_3d(scores, labels, ratio, OUTPUT_DIR)
    plot_pairs(scores, labels, ratio, n_final, OUTPUT_DIR)
    loadings = pd.DataFrame(pca.components_.T, index=feats.columns, columns=pc_names)
    plot_loadings(loadings, OUTPUT_DIR)

    # 6) Save numbers
    variance_table = pd.DataFrame({
        "PC": [f"PC{i + 1}" for i in range(n_avail)],
        "eigenvalue_lambda": pca_full.explained_variance_,
        "explained_variance_%": pca_full.explained_variance_ratio_ * 100,
        "cumulative_%": cum,
    })
    scores_df = pd.concat([ids, pd.DataFrame(scores, columns=pc_names)], axis=1)
    with pd.ExcelWriter(OUTPUT_DIR / "pca_results.xlsx") as writer:
        scores_df.to_excel(writer, sheet_name="scores", index=False)
        loadings.to_excel(writer, sheet_name="loadings")
        variance_table.to_excel(writer, sheet_name="variance", index=False)

    print(f"\nFigures and pca_results.xlsx saved in '{OUTPUT_DIR}/'")
    plt.show()   # opens the windows; the 3D one can be rotated with the mouse


if __name__ == "__main__":
    main()