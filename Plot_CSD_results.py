# Plot_CSD_results.py
# Translation of Plot_CSD_results.m
#
# For every session it takes the per-ripple CSD values of each layer (SO, PYR, RAD, SLM),
# averages them over a central window, and compares the three groups (WT, NPC, EFV)
# with a boxplot + all individual ripples, Kruskal-Wallis and pairwise comparisons.
#
# FOLDERS (nothing from the MATLAB pipeline is read or overwritten):
#   reads : <session>/analyset_python/rippleCSDs.mat      (made by processRippleSession.py)
#   writes: <session>/ProcesamientoNatalia/CSD_central_means.csv   (one table per session)
#           <baseDir>/ProcesamientoNatalia/{soCSD,pyrCSD,RadCSD,slmCSD}.png   (group figures)
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import kruskal, rankdata, studentized_range

# ================== SETTINGS ==================
baseDir = Path(r"D:\Violeta\ANALISIS")  # folder with the animals (MATLAB used E:\...)
INPUT_FOLDER = "analyset_python"  # where processRippleSession.py saved rippleCSDs.mat
SESSION_OUT_FOLDER = "ProcesamientoNatalia"  # inside every session folder
SAVE_SESSION_TABLE = True  # per-session CSV with the central means (extra, not in MATLAB)

# Colours of each group (RGB/256, as in MATLAB)
GROUP_NAMES = ["WT", "NPC", "EFV"]
GROUP_COLORS = np.array([[76, 153, 102], [70, 120, 180], [150, 110, 170]]) / 256

HALF_WINDOW = 25  # MATLAB: halfWindow = 25  (-> 50 samples)

# (field in rippleCSDs, console name, y label, title, file name)
LAYERS = [
    ("so", "SO", "CSD So", "CSD So by Group", "soCSD.png"),
    ("pyr", "PYR", "CSD Pyr", "CSD Pyr by Group", "pyrCSD.png"),
    ("rad", "Radial", "CSD Radial", "CSD Radial by Group", "RadCSD.png"),
    ("slm", "SLM", "CSD SLM", "CSD SLM by Group", "slmCSD.png"),
]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def detect_group(animal_name):
    """MATLAB: contains(name,'WT') / 'NPC' / 'EFV' (case-sensitive, first match wins)."""
    for g in GROUP_NAMES:
        if g in animal_name:
            return g
    return None


def central_window_idx(n_time, half=HALF_WINDOW):
    """
    MATLAB (1-based): centerIdx = ceil(nTime/2); idx = centerIdx-half+1 : centerIdx+half
    Same samples in 0-based Python: centerIdx-half ... centerIdx+half-1.
    For nTime = 251 -> 0-based 101..150 (50 samples; the t=0 sample is index 125).
    """
    center = int(np.ceil(n_time / 2.0))
    return np.arange(center - half, center + half)


def layer_values(rippleCSDs, layer):
    """[time x ripples] matrix. A session with a single ripple comes squeezed to 1-D."""
    v = np.asarray(rippleCSDs[layer]["values"], dtype=float)
    if v.ndim == 1:
        v = v.reshape(-1, 1)
    return v


def matlab_prctile(x_sorted, p):
    """MATLAB prctile (what boxplot uses): piecewise linear through the points (i-0.5)/n."""
    n = len(x_sorted)
    pos = np.asarray(p) / 100.0 * n + 0.5
    return np.interp(pos, np.arange(1, n + 1), x_sorted)


def box_stats(x):
    """Box statistics computed like MATLAB boxplot (quartiles by prctile, whiskers at 1.5 IQR)."""
    xs = np.sort(x)
    q1, med, q3 = matlab_prctile(xs, [25, 50, 75])
    iqr = q3 - q1
    return {
        "med": med,
        "q1": q1,
        "q3": q3,
        "whislo": xs[xs >= q1 - 1.5 * iqr].min(),
        "whishi": xs[xs <= q3 + 1.5 * iqr].max(),
        "fliers": [],
    }


def kruskalwallis_multcompare(groups):
    """
    MATLAB: [p,tbl,stats] = kruskalwallis(data,g,'off'); c = multcompare(stats,'Display','off')
    groups: list of 1-D arrays without NaN (groups with no data already removed).
    Returns chi2, df, p and a list of (i, j, p_value) with i, j indexes into `groups`.
    multcompare after Kruskal-Wallis = Tukey-Kramer on the mean ranks (infinite df).
    """
    sizes = np.array([len(g) for g in groups])
    k = len(groups)
    allx = np.concatenate(groups)
    N = len(allx)

    chi2, p = kruskal(*groups)  # includes the tie correction, like MATLAB
    df = k - 1

    ranks = rankdata(allx)  # average rank for ties
    meanranks = np.array([r.mean() for r in np.split(ranks, np.cumsum(sizes)[:-1])])

    _, t = np.unique(allx, return_counts=True)
    sumt = np.sum(t.astype(float) ** 3 - t)
    s2 = N * (N + 1) / 12.0 - sumt / (12.0 * (N - 1))

    pairs = []
    for i, j in combinations(range(k), 2):  # (1,2),(1,3),(2,3): same order as multcompare
        se = np.sqrt(s2 / 2.0 * (1.0 / sizes[i] + 1.0 / sizes[j]))
        q = abs(meanranks[i] - meanranks[j]) / se
        pairs.append((i, j, float(studentized_range.sf(q, k, np.inf))))
    return chi2, df, p, pairs


def star_text(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    return "*"


# ----------------------------------------------------------------------------
# One figure (SO / PYR / RAD / SLM)
# ----------------------------------------------------------------------------
def plot_layer(values, labels, console_name, ylabel, title, out_file, rng):
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)

    if values.size == 0 or np.all(np.isnan(values)):
        print(f"\n[{console_name}] no data, figure not made.")
        return

    fig, ax = plt.subplots(figsize=(5.6, 4.2))  # MATLAB default figure: 560x420 px

    groups = [values[labels == g] for g in GROUP_NAMES]
    groups_clean = [g[~np.isnan(g)] for g in groups]

    # ---- Boxplot with fixed order (black, no outlier symbols) ----
    positions, bstats = [], []
    for i, g in enumerate(groups_clean, start=1):
        if len(g):
            positions.append(i)
            bstats.append(box_stats(g))
    ax.bxp(
        bstats,
        positions=positions,
        widths=0.5,
        showfliers=False,
        boxprops=dict(color="k", linewidth=1.5),  # LineWidth 1.5 on the boxes
        medianprops=dict(color="k", linewidth=0.75),
        whiskerprops=dict(color="k", linewidth=0.75, linestyle="--"),
        capprops=dict(color="k", linewidth=0.75),
    )

    # ---- Scatter of all ripples (jittered) ----
    for iG, g in enumerate(groups, start=1):
        if len(g) == 0:
            continue
        ax.scatter(
            iG + 0.15 * rng.standard_normal(len(g)),
            g,
            s=10,
            facecolors=[(*GROUP_COLORS[iG - 1], 0.4)],  # MarkerFaceAlpha 0.4
            edgecolors="k",
            linewidths=0.5,
            zorder=3,
        )

    ax.set_xticks(range(1, len(GROUP_NAMES) + 1))
    ax.set_xticklabels(GROUP_NAMES)
    ax.set_xlim(0.5, len(GROUP_NAMES) + 0.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # ---- Kruskal-Wallis + multiple comparisons ----
    present = [i for i, g in enumerate(groups_clean) if len(g)]  # indexes into GROUP_NAMES
    pairs = []
    if len(present) >= 2:
        try:
            chi2, df, p_kw, pairs_local = kruskalwallis_multcompare(
                [groups_clean[i] for i in present]
            )
            print(f"\n--- Kruskal-Wallis ({console_name}) ---")
            print(f"Chi^2 = {chi2:.4f} | df = {df:.0f} | p = {p_kw:.4g}")
            print(f"\nPairwise comparisons ({console_name}):")
            print("Group1\tGroup2\tp-value")
            for i, j, pv in pairs_local:
                g1, g2 = present[i], present[j]
                print(f"{GROUP_NAMES[g1]}\t{GROUP_NAMES[g2]}\t{pv:.4g}")
                pairs.append((g1 + 1, g2 + 1, pv))  # x positions of the boxes
        except ValueError as e:  # e.g. all values identical
            print(f"\n[{console_name}] statistics not possible: {e}")
    else:
        print(f"\n[{console_name}] fewer than 2 groups with data, no statistics.")

    # ---- Stars ----
    yMax = np.nanmax(values)
    yMin = np.nanmin(values)
    yRange = yMax - yMin
    if yRange > 0:
        ax.set_ylim(yMin, yMax + 0.25 * yRange)
        starY = yMax + 0.05 * yRange
        starSpacing = 0.07 * yRange
        for x1, x2, pv in pairs:
            if pv < 0.05:
                ax.plot([x1, x2], [starY, starY], "k", linewidth=1.5)
                ax.text(
                    (x1 + x2) / 2.0,
                    starY + 0.01 * yRange,
                    star_text(pv),
                    ha="center",
                    va="center",
                    fontsize=14,
                )
                starY += starSpacing

    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_file}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(0)  # fixed seed -> the jitter is the same every run

    out_group_dir = baseDir / SESSION_OUT_FOLDER
    out_group_dir.mkdir(parents=True, exist_ok=True)

    soValues = {l[0]: [] for l in LAYERS}
    groupLabels = {l[0]: [] for l in LAYERS}
    n_sessions = {g: 0 for g in GROUP_NAMES}
    skipped = []

    animalDirs = sorted(
        (d for d in baseDir.iterdir() if d.is_dir()), key=lambda d: d.name.lower()
    )

    # ---------- Load all sessions for each animal ----------
    for animalDir in animalDirs:
        grp = detect_group(animalDir.name)
        if grp is None:
            continue  # animal name does not match any group

        sessionDirs = sorted(
            (s for s in animalDir.iterdir() if s.is_dir()), key=lambda d: d.name.lower()
        )
        for sessionDir in sessionDirs:
            csdFile = sessionDir / INPUT_FOLDER / "rippleCSDs.mat"
            if not csdFile.exists():
                continue  # skip sessions without rippleCSDs

            try:
                S = sio.loadmat(csdFile, simplify_cells=True)["rippleCSDs"]

                nTime = layer_values(S, "rad").shape[0]  # 251
                idx = central_window_idx(nTime)

                # mean over the central window, one value per ripple (NaN stays NaN, like MATLAB)
                means = {}
                for key, *_ in LAYERS:
                    means[key] = np.mean(layer_values(S, key)[idx, :], axis=0)
            except Exception as e:
                print(f"Warning: could not use {csdFile}: {e}")
                skipped.append(f"{animalDir.name}/{sessionDir.name}")
                continue

            for key, m in means.items():
                soValues[key].append(m)
                groupLabels[key].extend([grp] * len(m))
            n_sessions[grp] += 1

            if SAVE_SESSION_TABLE:
                try:
                    out_dir = sessionDir / SESSION_OUT_FOLDER
                    out_dir.mkdir(exist_ok=True)
                    table = pd.DataFrame(
                        {"ripple": np.arange(1, len(means["rad"]) + 1), **means}
                    )
                    table.to_csv(out_dir / "CSD_central_means.csv", index=False)
                except ValueError:  # layers with different number of ripples
                    print(f"Warning: session table not saved for {sessionDir.name}")

    print(
        "\nSessions used: "
        + ", ".join(f"{g}={n_sessions[g]}" for g in GROUP_NAMES)
        + (f" | skipped: {len(skipped)}" if skipped else "")
    )
    for s in skipped:
        print(f"   SKIPPED: {s}")

    # ---------- Plot each layer ----------
    for key, console_name, ylabel, title, fname in LAYERS:
        vals = np.concatenate(soValues[key]) if soValues[key] else np.array([])
        plot_layer(
            vals,
            groupLabels[key],
            console_name,
            ylabel,
            title,
            out_group_dir / fname,
            rng,
        )


if __name__ == "__main__":
    main()