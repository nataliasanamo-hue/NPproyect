"""
Disease progression on the t-SNE map
====================================
Question: do the EARLY sessions of NPC (and EFV) mice look more like WT than the
LATE ones? Are sessions close on the map because they were recorded close in time?

Uses the same data and features as pca_analysis_new.py (keep it in the same folder).

What it makes (RESULTSNatalia/progression_selected/ or progression/):
  01_tsne_chronology.png
      The t-SNE map 4 times: all groups, then WT, NPC and EFV highlighted.
      Highlighted sessions are colored by WHEN they were recorded
      (dark = early -> yellow = late); the rest are light grey.
      Thin lines join the sessions of the same mouse in time order (its "trajectory").
  02_distance_to_WT.png
      For every session: how far it is from the average WT session, measured in the
      FULL feature space (all selected features, not the 2-D map, because distances
      in a t-SNE map are not reliable). One line per mouse, in time order.
      If NPC sessions move AWAY from WT over time -> progression.
      If EFV sessions stay close / move back towards WT -> treatment effect.
  03_groups_early_sessions.png
      WT / NPC / EFV colored by group (like before), but NPC and EFV only with the
      FIRST HALF of each mouse's sessions (EARLY_FRACTION), because later on both
      may become similar. Shown on the same map, and on a new t-SNE of only those.
  progression.xlsx
      every session with its time variables, t-SNE coordinates and distance to WT,
      plus a summary per group and per mouse.

Time is read from the session folder name (e.g. 2026-01-16_17-50-35). By default it
is RELATIVE to each mouse (0 = its first session, 1 = its last), so a mouse with
3 sessions and one with 9 both go from dark to yellow.
Sessions whose name has no date (e.g. "P-7") cannot be placed in time: they are
drawn in grey and listed in the output.
"""

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import pca_analysis_new as cfg   # same Excel, columns and feature selection

# =============================================================================
# SETTINGS
# =============================================================================
TIME_MEASURE = "relative"  # "relative": position WITHIN each mouse, 0 = its first session,
                           #   1 = its last (every mouse goes from dark to yellow, whether
                           #   it has 3 or 9 sessions)
                           # "order": session number of that mouse (1st, 2nd, 3rd...)
                           # "days":  days since that mouse's first recording
                           # "date":  calendar date (same scale for all mice)
PERPLEXITY = 15          # same value as one of the maps of nonlinear_embedding_new.py
RANDOM_STATE = 42        # same seed -> the same map you already saw
REUSE_MAP = True         # use the t-SNE coordinates already saved by nonlinear_embedding_new.py
DRAW_TRAJECTORIES = True # arrows joining the sessions of each mouse in time order
CMAP = "viridis"         # dark purple = early, yellow = late
OUTPUT_DIR = cfg.RESULTS_DIR / f"progression{getattr(cfg, 'RUN_TAG', '')}"
GROUPS = ["WT", "NPC", "EFV"]
MARKERS = {"WT": "o", "NPC": "s", "EFV": "^"}
# Figure 3 (groups using only EARLY sessions):
EARLY_GROUPS = ["NPC", "EFV"]   # groups reduced to their first sessions (WT keeps all)
EARLY_FRACTION = 0.5            # 0.5 = first half of each mouse's sessions
                                # (rounded up: 3 sessions -> first 2, 4 -> first 2, 5 -> first 3)
# =============================================================================


def parse_date(name):
    """Find a date anywhere in a session folder name (NaT if there is none)."""
    s = str(name)
    for pat, order in [(r"(\d{4})[-_.](\d{1,2})[-_.](\d{1,2})", "ymd"),
                       (r"(\d{1,2})[-_.](\d{1,2})[-_.](\d{4})", "dmy"),
                       (r"(20\d{2})(\d{2})(\d{2})", "ymd")]:
        m = re.search(pat, s)
        if m:
            a, b, c = (int(x) for x in m.groups())
            y, mo, d = (a, b, c) if order == "ymd" else (c, b, a)
            try:
                return pd.Timestamp(year=y, month=mo, day=d)
            except ValueError:
                pass
    return pd.NaT


def parse_datetime(name):
    """Date + time when available (to order two sessions recorded the same day)."""
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})", str(name))
    if m:
        return pd.Timestamp(*(int(x) for x in m.groups()))
    return parse_date(name)


def get_map(ids, X):
    """t-SNE coordinates: reuse the saved ones if possible, otherwise compute."""
    emb_file = cfg.RESULTS_DIR / f"embedding_results{getattr(cfg, 'RUN_TAG', '')}" / "embeddings.xlsx"
    cols = [f"tSNE_p{PERPLEXITY}_1", f"tSNE_p{PERPLEXITY}_2"]
    if REUSE_MAP and emb_file.exists():
        emb = pd.read_excel(emb_file)
        if all(c in emb for c in cols):
            key = ids["animal"].astype(str) + "|" + ids["session"].astype(str)
            ekey = emb["animal"].astype(str) + "|" + emb["session"].astype(str)
            Y = emb.set_index(ekey)[cols].reindex(key).to_numpy()
            if not np.isnan(Y).any():
                print(f"Using the t-SNE map saved in {emb_file.name} (perplexity {PERPLEXITY})")
                return Y
    print(f"Computing t-SNE (perplexity {PERPLEXITY})")
    return TSNE(n_components=2, perplexity=PERPLEXITY, init="pca", learning_rate="auto",
                random_state=RANDOM_STATE).fit_transform(X)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ids, feats = cfg.load_data(cfg.EXCEL_PATH, cfg.SHEET, cfg.ID_COLS)
    groups = ids[cfg.GROUP_COL].astype(str).to_numpy()
    animals = ids["animal"].astype(str).to_numpy()
    X = StandardScaler().fit_transform(feats.to_numpy())
    print(f"{len(ids)} sessions x {X.shape[1]} features")

    # ---------------- time variables ----------------
    dt = ids["session"].map(parse_datetime)
    date = dt.dt.normalize()
    first = date.groupby(ids["animal"]).transform("min")
    order = dt.groupby(ids["animal"]).rank(method="first")
    n_dated = order.groupby(ids["animal"]).transform("max")
    t = pd.DataFrame({
        "date": date,
        "order": order,
        "days": (date - first).dt.days,
        # 0 = first session of the mouse, 1 = last (NaN if the mouse has only 1 dated session)
        "relative": ((order - 1) / (n_dated - 1)).where(n_dated > 1),
    })
    no_date = ids.loc[dt.isna(), ["group", "animal", "session"]]
    if len(no_date):
        print(f"\n[i] {len(no_date)} session(s) without a date in the name (drawn in grey):")
        print(no_date.to_string(index=False))
    if TIME_MEASURE == "date":
        tval = (date - date.min()).dt.days.to_numpy(float)
        tlabel = f"days since {date.min():%Y-%m-%d}"
    else:
        tval = t[TIME_MEASURE].to_numpy(float)
        tlabel = {"order": "session number of the mouse",
                  "days": "days since the mouse's first recording",
                  "relative": "position in the mouse's recordings (0 = first, 1 = last)"}[TIME_MEASURE]
    has_t = ~np.isnan(tval)
    vmin, vmax = np.nanmin(tval), np.nanmax(tval)

    # ---------------- distance to WT (full feature space) ----------------
    wt_centre = X[groups == "WT"].mean(axis=0)
    dist_wt = np.linalg.norm(X - wt_centre, axis=1)

    Y = get_map(ids, X)

    # ---------------- figure 1: t-SNE colored by time ----------------
    panels = ["all"] + [g for g in GROUPS if g in set(groups)]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 5.2), squeeze=False,
                             layout="constrained")
    for ax, which in zip(axes[0], panels):
        hi = np.ones(len(ids), bool) if which == "all" else (groups == which)
        ax.scatter(Y[~hi, 0], Y[~hi, 1], s=22, color="#dddddd", edgecolor="none", zorder=1)
        if DRAW_TRAJECTORIES:
            for a in np.unique(animals[hi]):
                m = (animals == a) & has_t
                if m.sum() > 1:
                    o = np.argsort(tval[m])
                    px, py = Y[m, 0][o], Y[m, 1][o]
                    for i in range(len(px) - 1):          # one arrow per step in time
                        ax.annotate("", xy=(px[i + 1], py[i + 1]), xytext=(px[i], py[i]),
                                    arrowprops=dict(arrowstyle="-|>", color="gray", lw=0.7,
                                                    alpha=0.7, mutation_scale=9,
                                                    shrinkA=4, shrinkB=5), zorder=2)
        for g in GROUPS:
            m = hi & (groups == g)
            ax.scatter(Y[m & ~has_t, 0], Y[m & ~has_t, 1], s=40, marker=MARKERS[g],
                       color="lightgray", edgecolor="k", linewidth=0.4, zorder=3)
            sc = ax.scatter(Y[m & has_t, 0], Y[m & has_t, 1], s=46, marker=MARKERS[g],
                            c=tval[m & has_t], cmap=CMAP, vmin=vmin, vmax=vmax,
                            edgecolor="k", linewidth=0.5, zorder=4)
        ax.set_title("All groups" if which == "all" else f"{which} highlighted")
        ax.set_xticks([]); ax.set_yticks([])
    handles = [Line2D([], [], marker=MARKERS[g], ls="", color="w", markeredgecolor="k",
                      markersize=8, label=g) for g in GROUPS]
    axes[0, 0].legend(handles=handles, loc="best", fontsize=8, title="marker = group")
    fig.colorbar(sc, ax=axes, location="bottom", shrink=0.4, aspect=40, label=f"{tlabel}  (dark = early, yellow = late)")
    fig.suptitle(f"t-SNE (perplexity {PERPLEXITY}) colored by recording time"
                 + ("; arrows = from each session of a mouse to its next one" if DRAW_TRAJECTORIES else ""))
    fig.savefig(OUTPUT_DIR / "01_tsne_chronology.png", dpi=200)

    # ---------------- figure 2: distance to WT over time ----------------
    fig2, axes2 = plt.subplots(1, len(panels) - 1, figsize=(5 * (len(panels) - 1), 4.2),
                               squeeze=False, sharey=True)
    summary = []
    for ax, g in zip(axes2[0], panels[1:]):
        for a in np.unique(animals[groups == g]):
            m = (animals == a) & has_t
            if m.any():
                o = np.argsort(tval[m])
                ax.plot(tval[m][o], dist_wt[m][o], "-o", ms=4, lw=1, alpha=0.8, label=a)
        m = (groups == g) & has_t
        rho = spearmanr(tval[m], dist_wt[m]).correlation if m.sum() > 2 else np.nan
        # per mouse: does the distance go up or down with time?
        slopes = []
        for a in np.unique(animals[m]):
            ma = m & (animals == a)
            if ma.sum() >= 2 and np.ptp(tval[ma]) > 0:
                slopes.append(np.polyfit(tval[ma], dist_wt[ma], 1)[0])
        n_up = int(np.sum(np.array(slopes) > 0))
        summary.append({"group": g, "sessions_with_time": int(m.sum()),
                        "median_distance_to_WT": float(np.median(dist_wt[groups == g])),
                        "spearman_time_vs_distance": rho,
                        "mice_with_2+_sessions": len(slopes),
                        "mice_moving_away_from_WT": n_up,
                        "mice_moving_towards_WT": len(slopes) - n_up})
        ax.set_title(f"{g}: rho = {rho:.2f}, away from WT in {n_up}/{len(slopes)} mice")
        ax.set_xlabel(tlabel)
        ax.legend(fontsize=6, ncol=2)
    axes2[0, 0].set_ylabel("distance to the average WT session\n(all selected features)")
    fig2.suptitle("Do sessions move away from WT over time? (one line per mouse)")
    fig2.tight_layout()
    fig2.savefig(OUTPUT_DIR / "02_distance_to_WT.png", dpi=200)

    # ---------------- figure 3: groups with only the EARLY sessions ----------------
    n_ses = t["order"].groupby(ids["animal"]).transform("max").to_numpy(float)
    early_rank = np.ceil(n_ses * EARLY_FRACTION)
    is_reduced = np.isin(groups, EARLY_GROUPS)
    keep = ~is_reduced | (t["order"].to_numpy(float) <= early_rank)   # NaN order -> False
    dropped_no_date = is_reduced & ~has_t
    print(f"\nFIGURE 3: WT = all sessions; {', '.join(EARLY_GROUPS)} = first "
          f"{EARLY_FRACTION:.0%} of each mouse's sessions")
    print("  sessions kept per group: " +
          ", ".join(f"{g} {int((keep & (groups == g)).sum())}/{int((groups == g).sum())}" for g in GROUPS))
    if dropped_no_date.any():
        print(f"  [i] {int(dropped_no_date.sum())} {'/'.join(EARLY_GROUPS)} session(s) left out "
              "because their name has no date (cannot tell if they are early)")

    def sil(Z, lab):
        return silhouette_score(Z, lab) if len(np.unique(lab)) > 1 else np.nan

    Y_early = TSNE(n_components=2, perplexity=min(PERPLEXITY, keep.sum() - 1), init="pca",
                   learning_rate="auto", random_state=RANDOM_STATE).fit_transform(X[keep])
    fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5.2))
    views = [
        (axes3[0], Y, np.ones(len(ids), bool),
         f"ALL sessions (same map as before)\nsilhouette = {sil(X, groups):.2f}"),
        (axes3[1], Y, keep,
         f"Same map, only EARLY {'/'.join(EARLY_GROUPS)} sessions\n(grey = later sessions, hidden)"
         f"   silhouette = {sil(X[keep], groups[keep]):.2f}"),
        (axes3[2], Y_early, None,
         f"NEW t-SNE computed only with these sessions\nsilhouette = {sil(X[keep], groups[keep]):.2f}"),
    ]
    for ax, YY, show, title in views:
        if show is None:                         # new map: all its points are the kept ones
            gg = groups[keep]
            for gi, g in enumerate(GROUPS):
                m = gg == g
                ax.scatter(YY[m, 0], YY[m, 1], s=40, color=cfg.color_for(g, gi),
                           edgecolor="k", linewidth=0.5, label=g)
        else:
            ax.scatter(YY[~show, 0], YY[~show, 1], s=22, color="#e3e3e3", edgecolor="none")
            for gi, g in enumerate(GROUPS):
                m = show & (groups == g)
                ax.scatter(YY[m, 0], YY[m, 1], s=40, color=cfg.color_for(g, gi),
                           edgecolor="k", linewidth=0.5, label=g)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=8)
    fig3.suptitle(f"Groups using only the first {EARLY_FRACTION:.0%} of the {'/'.join(EARLY_GROUPS)} "
                  "sessions (WT: all).  Silhouette computed on all selected features")
    fig3.tight_layout()
    fig3.savefig(OUTPUT_DIR / "03_groups_early_sessions.png", dpi=200)

    # ---------------- report + Excel ----------------
    summary = pd.DataFrame(summary)
    print("\nSUMMARY (rho > 0 = later sessions are further from WT):")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\nNote: within WT, 'distance to WT' is just the normal spread of WT sessions;\n"
          "it is the reference for how far NPC / EFV are.")

    out = ids.copy()
    out["recording_date"] = date
    out["session_number_of_mouse"] = t["order"]
    out["days_since_first_recording"] = t["days"]
    out["relative_position_in_mouse"] = t["relative"]
    out["in_early_plot"] = keep
    out["distance_to_WT"] = dist_wt
    out[f"tSNE_p{PERPLEXITY}_1"], out[f"tSNE_p{PERPLEXITY}_2"] = Y[:, 0], Y[:, 1]
    per_mouse = (out.groupby(["group", "animal"])
                 .agg(sessions=("session", "count"),
                      first_date=("recording_date", "min"), last_date=("recording_date", "max"),
                      median_distance_to_WT=("distance_to_WT", "median"))
                 .reset_index())
    with pd.ExcelWriter(OUTPUT_DIR / "progression.xlsx") as w:
        out.to_excel(w, sheet_name="sessions", index=False)
        per_mouse.to_excel(w, sheet_name="per_mouse", index=False)
        summary.to_excel(w, sheet_name="summary", index=False)
    print(f"\nSaved in {OUTPUT_DIR}")
    plt.show()


if __name__ == "__main__":
    main()