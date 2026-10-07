# plot_group_comparison.py
# Translation of plot_group_comparison.m + the design of the lab figures.
#
# plot_group_comparison(WT, NPC, EFV, ylabel, title)       one panel
# plot_feature_figure(title, panels, ...)                  several panels in one figure
#   - boxes + individual points
#   - one-way ANOVA (anova1) + post-hoc Tukey-Kramer (multcompare, alpha 0.05)
#   - bracket + stars (* p<0.05, ** p<0.01, *** p<0.001) for every significant pair
#   - STYLE = "lab": coloured filled boxes, white median, open circles, statistics under the panel
#
# It does NOT need the power script: it only needs three vectors of numbers per panel.
# Run this file to make the figures from RESULTSNatalia/features_final.csv:
#   power  -> one figure per band / layer / state, with 2 panels: Mean Power (top) and Peak Power (Max) (below)
#   CSD    -> one figure per layer, panels Mean CSD and Median CSD
#   ripple -> one figure with frequency, amplitude and power
#   figures      -> <baseDir>/RESULTSNatalia/group_comparison/<name>.png
#   all p-values -> <baseDir>/RESULTSNatalia/group_comparison/group_comparison_pvalues.csv
import re
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter
from scipy.stats import bartlett, f_oneway, kruskal, shapiro, tukey_hsd

# ================== SETTINGS ==================
baseDir = Path(r"D:\Violeta\ANALISIS")
TOTAL_OUT_FOLDER = "RESULTSNatalia"
FEATURES_FILE = "features_final.csv"
FIG_FOLDER = "group_comparison"

# Names of the figures to make (None = all). Examples: ["Alpha_PYR_Theta", "ripples", "CSD_PYR"]
ONLY_FIGURES = None

# STYLE = "lab"    -> design of the lab figures: coloured filled boxes, white median, open circles,
#                    ANOVA / Kruskal-Wallis text under the plot, groups in the order WT, NPC, EFV
# STYLE = "matlab" -> design of plot_group_comparison.m (blue boxes, red median, alphabetical order)
STYLE = "lab"
GROUP_COLORS = {"WT": (0.24, 0.60, 0.38), "NPC": (0.22, 0.45, 0.70), "EFV": (0.58, 0.40, 0.67)}
SHOW_GROUP_LABELS = True  # WT / NPC / EFV under the boxes (the lab figures only use the colours)

LAB = STYLE == "lab"
BOX_ORDER = ["WT", "NPC", "EFV"] if LAB else ["EFV", "NPC", "WT"]  # MATLAB categorical = alphabetical
# anova1 / multcompare number the groups in the order they appear in the data: WT, NPC, EFV.
ANOVA_ORDER = ["WT", "NPC", "EFV"]
# MATLAB uses c(i,1) and c(i,2) (numbers in ANOVA order) directly as x positions of the
# brackets, which does not coincide with the alphabetical boxplot order. False = the bracket is
# drawn over the two groups that are really compared (same p-values, only the position changes).
MATLAB_BRACKET_POSITIONS = False

BAND_NAMES = {"SlowGamma": "Slow Gamma", "MidGamma": "Mid Gamma", "FastGamma": "Fast Gamma"}


# ----------------------------------------------------------------------------
# Same box statistics as MATLAB boxplot (prctile quartiles, whiskers at 1.5 IQR)
# ----------------------------------------------------------------------------
def matlab_prctile(x_sorted, p):
    n = len(x_sorted)
    pos = np.asarray(p) / 100.0 * n + 0.5
    return np.interp(pos, np.arange(1, n + 1), x_sorted)


def box_stats(x):
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


def star_text(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    return "*"


# ----------------------------------------------------------------------------
# anova1 + multcompare (+ Kruskal-Wallis, normality and variance checks for the text)
# ----------------------------------------------------------------------------
def anova_tukey(groups_by_name):
    """
    groups_by_name: dict name -> 1-D array (NaN allowed, ignored like anova1 does).
    Returns (p_anova, pairs, stats) with pairs = [(name1, name2, p_tukey), ...] in the order of
    multcompare: (1,2), (1,3), (2,3) of the ANOVA group order.
    stats: F, df (error), kw_p, chi2, norm (Shapiro flags per group), var_p (Bartlett).
    """
    stats = {}
    names = [n for n in ANOVA_ORDER if n in groups_by_name]
    data = {n: np.asarray(groups_by_name[n], dtype=float) for n in names}
    data = {n: v[~np.isnan(v)] for n, v in data.items()}
    names = [n for n in names if len(data[n]) > 0]  # anova1 ignores empty groups
    if len(names) < 2:
        return np.nan, [], stats
    arrays = [data[n] for n in names]
    N = sum(len(a) for a in arrays)
    if N - len(names) <= 0:
        return np.nan, [], stats
    fo = f_oneway(*arrays)
    stats.update(F=float(fo.statistic), df=N - len(names))
    try:
        kw = kruskal(*arrays)
        stats.update(kw_p=float(kw.pvalue), chi2=float(kw.statistic))
    except ValueError:  # all values identical
        pass
    # group checks shown in the text: 1 = looks normal (Shapiro p > 0.05), 0 = not normal
    flags = []
    for n in ANOVA_ORDER:
        v = data.get(n, np.array([]))
        flags.append("-" if len(v) < 3 or np.ptp(v) == 0 else str(int(shapiro(v).pvalue > 0.05)))
    stats["norm"] = " ".join(flags)
    try:
        stats["var_p"] = float(bartlett(*arrays).pvalue)
    except ValueError:
        pass
    res = tukey_hsd(*arrays)  # Tukey-Kramer for unequal sizes, like multcompare
    pairs = [(names[i], names[j], float(res.pvalue[i, j])) for i, j in combinations(range(len(names)), 2)]
    return float(fo.pvalue), pairs, stats


def stats_text(p_anova, stats):
    lines = []
    if "norm" in stats:
        vp = stats.get("var_p", np.nan)
        lines.append(f"norm:{stats['norm']}; var(p)={vp:.2g}")
    if "F" in stats:
        lines.append(f"ANOVA p={p_anova:.3g}, F({stats['df']})={stats['F']:.5g}")
    if "kw_p" in stats:
        lines.append(f"KW p={stats['kw_p']:.3g}, Chi-sq={stats['chi2']:.5g}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# One panel
# ----------------------------------------------------------------------------
def draw_panel(ax, raw, ylabel, panel_title, rng):
    """raw: dict WT/NPC/EFV -> array. Returns (p_anova, pairs, stats)."""
    data = np.concatenate([raw["WT"], raw["NPC"], raw["EFV"]])  # data = [WT(:); NPC(:); EFV(:)]
    xpos = {name: i for i, name in enumerate(BOX_ORDER, start=1)}

    # ----- Boxes -----
    for name in BOX_ORDER:
        v = raw[name][~np.isnan(raw[name])]
        if not len(v):
            continue
        st, pos = box_stats(v), xpos[name]
        if LAB:  # filled box, thick white median, thin whisker line, no caps
            col = GROUP_COLORS[name]
            ax.plot([pos, pos], [st["whislo"], st["whishi"]], color=col, linewidth=0.8, zorder=1)
            ax.add_patch(plt.Rectangle((pos - 0.3, st["q1"]), 0.6, st["q3"] - st["q1"],
                                       facecolor=col, edgecolor="none", zorder=2))
            ax.plot([pos - 0.3, pos + 0.3], [st["med"]] * 2, color="w", linewidth=3,
                    solid_capstyle="butt", zorder=3)
        else:  # boxplot(data, groupCat, 'Symbol','')
            ax.bxp([st], positions=[pos], widths=0.5, showfliers=False,
                   boxprops=dict(color="b", linewidth=0.75), medianprops=dict(color="r", linewidth=0.75),
                   whiskerprops=dict(color="k", linewidth=0.75, linestyle="--"),
                   capprops=dict(color="k", linewidth=0.75))

    # ----- Individual points -----
    for name in BOX_ORDER:
        v = raw[name][~np.isnan(raw[name])]
        if not len(v):
            continue
        x = xpos[name] + 0.15 * rng.uniform(-1, 1, len(v))
        if LAB:  # open black circles
            ax.scatter(x, v, s=45, facecolors="none", edgecolors="k", linewidths=0.8, zorder=4)
        else:  # scatter(..., 15, 'k', 'filled', alpha 0.4)
            ax.scatter(x, v, s=15, c="k", alpha=0.4, linewidths=0, zorder=3)

    ax.set_xticks(range(1, len(BOX_ORDER) + 1))
    ax.set_xticklabels(BOX_ORDER if SHOW_GROUP_LABELS or not LAB else [""] * len(BOX_ORDER))
    ax.set_xlim(0.5, len(BOX_ORDER) + 0.5)
    ax.set_ylabel(ylabel, fontsize=11 if LAB else 12)
    ax.set_title(panel_title, fontsize=11 if LAB else 12, fontweight="bold" if LAB else "normal")
    ax.tick_params(labelsize=10 if LAB else 12)
    for side in ("top", "right"):  # box off
        ax.spines[side].set_visible(False)
    if LAB:
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((-3, 4))
        ax.yaxis.set_major_formatter(fmt)

    # ----- ANOVA + Tukey -----
    try:
        p_anova, pairs, stats = anova_tukey(raw)
    except ValueError as e:
        print(f"[{panel_title}] statistics not possible: {e}")
        p_anova, pairs, stats = np.nan, [], {}

    # ----- Brackets and stars -----
    if np.any(~np.isnan(data)):
        yMax, yMin = np.nanmax(data), np.nanmin(data)
        yRange = yMax - yMin
        anova_idx = {n: i for i, n in enumerate([n for n in ANOVA_ORDER if len(raw[n][~np.isnan(raw[n])])], 1)}
        top = yMax
        if LAB:  # bracket hanging down, stars on top (small)
            yStep, y = yRange * 0.07, yMax + yRange * 0.06
            tick = yRange * 0.02
        else:
            yStep, y = yRange * 0.08, yMax + yRange * 0.08
        for a, b, pv in pairs:
            if pv >= 0.05 or yRange <= 0:
                continue
            x1, x2 = (anova_idx[a], anova_idx[b]) if MATLAB_BRACKET_POSITIONS else (xpos[a], xpos[b])
            if LAB:
                ax.plot([x1, x1, x2, x2], [y - tick, y, y, y - tick], "k", linewidth=1.0)
                ax.text((x1 + x2) / 2.0, y - yRange * 0.005, star_text(pv), ha="center", va="bottom", fontsize=9)
                top, y = y + yRange * 0.05, y + yStep
            else:
                ax.plot([x1, x1, x2, x2], [y, y + yStep, y + yStep, y], "k", linewidth=1.5)
                ax.text((x1 + x2) / 2.0, y + yStep * 1.2, star_text(pv), ha="center", va="center", fontsize=14)
                top, y = y + yStep * 1.6, y + yStep * 1.8
        if top > yMax:
            ax.set_ylim(top=top + 0.03 * yRange)

    if LAB:
        txt = stats_text(p_anova, stats)
        if txt:
            ax.set_xlabel(txt, fontsize=9, labelpad=8)
    return p_anova, pairs, stats


def _raw(WT, NPC, EFV):
    return {k: np.asarray(v, dtype=float).ravel() for k, v in (("WT", WT), ("NPC", NPC), ("EFV", EFV))}


# ----------------------------------------------------------------------------
# The function of the .m (one panel)
# ----------------------------------------------------------------------------
def plot_group_comparison(WT, NPC, EFV, yLabelText, figTitle, out_file=None, rng=None):
    """Returns (figure, p_anova, pairs). Saves the figure when out_file is given."""
    rng = rng if rng is not None else np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(4.0, 4.8) if LAB else (5.6, 4.2))
    p_anova, pairs, _ = draw_panel(ax, _raw(WT, NPC, EFV), yLabelText, figTitle, rng)
    fig.tight_layout()
    if out_file is not None:
        fig.savefig(out_file, dpi=150)
        plt.close(fig)
    return fig, p_anova, pairs


# ----------------------------------------------------------------------------
# Several panels in one figure
# ----------------------------------------------------------------------------
def plot_feature_figure(title, panels, layout="column", out_file=None, rng=None):
    """
    panels: list of (panel_title, ylabel, {"WT":..., "NPC":..., "EFV":...}).
    layout: "column" (panels one under the other) or "row" (side by side).
    Returns the list of per-panel results (panel_title, p_anova, pairs, stats).
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    n = len(panels)
    if layout == "column":
        fig, axes = plt.subplots(n, 1, figsize=(4.0, 4.6 * n), squeeze=False)
    else:
        fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 4.8), squeeze=False)
    out = []
    for ax, (ptitle, ylabel, raw) in zip(axes.ravel(), panels):
        p_an, pairs, stats = draw_panel(ax, _raw(raw["WT"], raw["NPC"], raw["EFV"]), ylabel, ptitle, rng)
        out.append((ptitle, p_an, pairs, stats))
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    if out_file is not None:
        fig.savefig(out_file, dpi=150)
        plt.close(fig)
    return out


# ----------------------------------------------------------------------------
# Driver: figures from features_final.csv
# ----------------------------------------------------------------------------
def group_values(df, col):
    return {g: pd.to_numeric(df.loc[df["group"] == g, col], errors="coerce").to_numpy() for g in ("WT", "NPC", "EFV")}


def has_data(vals):
    return not all(np.all(np.isnan(v)) for v in vals.values())


def build_figures(df):
    """Returns a list of (file_name, figure_title, layout, [(feature, panel_title, ylabel, values)])."""
    cols = set(df.columns)
    figs = []

    # power: {Band}_{PYR|SLM}_{Theta|LIA}_{mean|max}
    pw = re.compile(r"^([A-Za-z]+)_(PYR|SLM)_(Theta|LIA)_mean$")
    for c in df.columns:
        m = pw.match(c)
        if not m:
            continue
        band, layer, state = m.groups()
        panels = []
        for suffix, ptitle in (("mean", "Mean Power"), ("max", "Peak Power (Max)")):
            col = f"{band}_{layer}_{state}_{suffix}"
            if col in cols:
                panels.append((col, ptitle, "dB", group_values(df, col)))
        figs.append((f"{band}_{layer}_{state}", f"{BAND_NAMES.get(band, band)} Power {layer} {state}",
                     "column", panels))

    # CSD: CSD_{SO|PYR|RAD|SLM}_{mean|median}
    for layer in ("SO", "PYR", "RAD", "SLM"):
        panels = []
        for suffix, ptitle in (("mean", "Mean CSD"), ("median", "Median CSD")):
            col = f"CSD_{layer}_{suffix}"
            if col in cols:
                panels.append((col, ptitle, "CSD (a.u.)", group_values(df, col)))
        if panels:
            figs.append((f"CSD_{layer}", f"CSD {layer}", "column", panels))

    # ripples: first three columns of propRipS, mean per session
    rip = [("ripple_frequency_mean", "Frequency", "Hz"),
           ("ripple_amplitude_mean", "Amplitude", "uV"),
           ("ripple_power_mean", "Power", "dB")]
    panels = [(c, t, y, group_values(df, c)) for c, t, y in rip if c in cols]
    if panels:
        figs.append(("ripples", "Ripple features (mean per session)", "row", panels))
    return figs


def main():
    f = baseDir / TOTAL_OUT_FOLDER / FEATURES_FILE
    df = pd.read_csv(f)
    out_dir = baseDir / TOTAL_OUT_FOLDER / FIG_FOLDER
    out_dir.mkdir(parents=True, exist_ok=True)

    figs = build_figures(df)
    if ONLY_FIGURES:
        figs = [x for x in figs if x[0] in ONLY_FIGURES]
    print(f"{len(df)} sessions, {len(figs)} figures")

    rng = np.random.default_rng(0)  # fixed seed -> same jitter every run
    rows = []
    for name, title, layout, panels in figs:
        panels = [p for p in panels if has_data(p[3])]
        if not panels:
            continue
        res = plot_feature_figure(title, [(pt, yl, v) for _, pt, yl, v in panels], layout,
                                  out_file=out_dir / f"{name}.png", rng=rng)
        for (feature, ptitle, _, vals), (_, p_an, pairs, stats) in zip(panels, res):
            row = {"figure": name, "feature": feature, "p_anova": p_an,
                   "F": stats.get("F", np.nan), "df": stats.get("df", np.nan),
                   "kw_p": stats.get("kw_p", np.nan), "chi2": stats.get("chi2", np.nan)}
            for g in ("WT", "NPC", "EFV"):
                v = vals[g][~np.isnan(vals[g])]
                row[f"n_{g}"] = len(v)
                row[f"mean_{g}"] = float(v.mean()) if len(v) else np.nan
            for a, b, pv in pairs:
                row[f"p_tukey_{a}_vs_{b}"] = pv
            rows.append(row)

    pd.DataFrame(rows).to_csv(out_dir / "group_comparison_pvalues.csv", index=False)
    print(f"Saved {len(figs)} figures and group_comparison_pvalues.csv in {out_dir}")


if __name__ == "__main__":
    main()