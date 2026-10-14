"""
What makes the odd sessions odd?
================================
Reads your Excel odd_sessions.xlsx (one row per session, animals separated by
blank rows, ODD sessions highlighted with a fill colour, e.g. yellow) and compares
every odd session with the NORMAL sessions of the SAME mouse.

How "different" is measured
---------------------------
For each feature:
    deviation = (odd value - median of the normal sessions of that mouse) / typical spread
"typical spread" = how much normal sessions of the same mouse usually differ from
each other, pooled over all mice (so it also works for mice with only 1 normal
session, like NPC_30 and NPC_47). Read it like a z-score:
    |deviation| < 1  -> normal variation
    |deviation| > 2  -> clearly different ("failing")
    |deviation| > 4  -> very different
Power features are compared in log scale (their values are very skewed).
Channel columns (ch_*, config_*, ...) are reported as "how many channels away from
the usual channel of that mouse".
For CSD, sinks are negative and sources positive, so "higher" can mean "weaker";
the column failing_and_WEAKER counts sessions where the value got closer to 0.

Output (same folder as the Excel, subfolder odd_sessions_report/):
  01_deviation_heatmap.png    odd sessions x features, red = higher, blue = lower
  02_common_failing.png       how many odd sessions fail each feature, and in which direction
  odd_sessions_report.xlsx    all the numbers
The terminal shows the summary: common failing features + top features per session.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd

# =============================================================================
# SETTINGS
# =============================================================================
baseDir = Path(r"D:\Violeta\ANALISIS")
ODD_FILE = baseDir / "ProcesamientoNatalia" / "odd_sessions.xlsx"   # change if it is elsewhere
Z_FAIL = 2.0            # |deviation| above this = "failing"
N_TOP_PER_SESSION = 6   # features listed per odd session
# Used only if no highlighted rows are found in the Excel:
DEFAULT_ODD = {
    "EFV_218": ["2026-01-16_17-50-35"],
    "EFV_220": ["2026-01-13_14-12-19", "2026-01-13_15-18-23", "2026-01-15_18-37-42"],
    "EFV_277": ["2026-02-20_12-02-47"],
    "NPC_249": ["2026-02-04_16-01-16", "2026-02-05_16-14-02", "2026-02-05_16-40-43"],
    "NPC_29": ["P-7"], "NPC_30": ["P4-HI"], "NPC_47": ["P3-HD"],
}
CHANNEL_PREFIXES = ("ch_", "config_", "power_pyr_channel", "power_slm_channel")
SKIP_COLUMNS = ("power_channels_match", "config_matches", "config_theta_equals")
# =============================================================================

POWER_RE = re.compile(r"^(Delta|Theta|Alpha|SlowGamma|MidGamma|FastGamma|Ripple)_(PYR|SLM)_(Theta|LIA)_(mean|max)$",
                      re.IGNORECASE)


def find_file():
    if ODD_FILE.exists():
        return ODD_FILE
    hits = sorted(baseDir.rglob("odd_sessions*.xlsx"))
    if hits:
        print(f"[i] {ODD_FILE} not found; using {hits[0]}")
        return hits[0]
    raise FileNotFoundError(f"odd_sessions.xlsx not found (looked at {ODD_FILE} and inside {baseDir})")


def is_filled(cell):
    f = cell.fill
    if f is None or f.fill_type in (None, "none"):
        return False
    rgb = str(getattr(f.fgColor, "rgb", "") or "")
    return rgb not in ("", "00000000", "FFFFFFFF") or getattr(f.fgColor, "theme", None) is not None


def read_excel(path):
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb.worksheets[0]
    rows = list(ws.iter_rows())
    header = [str(c.value).strip() if c.value is not None else "" for c in rows[0]]
    data, odd_flags = [], []
    for row in rows[1:]:
        vals = [c.value for c in row]
        if all(v is None or str(v).strip() == "" for v in vals):
            continue                                  # blank separator row
        data.append(vals)
        odd_flags.append(any(is_filled(c) for c in row[:3]))   # look at animal/session/group cells
    df = pd.DataFrame(data, columns=header)
    df = df.loc[:, [c for c in df.columns if c]]
    df["animal"] = df["animal"].astype(str).str.strip()
    df["session"] = df["session"].astype(str).str.strip()
    odd = np.array(odd_flags)
    if not odd.any():
        print("[i] No highlighted rows found: using the list of 11 odd sessions written in the script")
        odd = np.array([s in DEFAULT_ODD.get(a, []) for a, s in zip(df["animal"], df["session"])])
    return df, odd


def kind_of(col):
    if col.startswith(CHANNEL_PREFIXES):
        return "channel"
    if col.startswith("n_") or col.endswith("_n"):
        return "count"
    if POWER_RE.match(col):
        return "power"
    if col.upper().startswith("CSD_"):
        return "CSD"
    if col.lower().startswith("ripple_"):
        return "ripple"
    return "other"


def main():
    path = find_file()
    out_dir = path.parent / "odd_sessions_report"
    out_dir.mkdir(exist_ok=True)
    df, odd = read_excel(path)
    print(f"{len(df)} sessions, {odd.sum()} odd, {df['animal'].nunique()} mice")

    cols = [c for c in df.columns if c not in ("animal", "session", "group")
            and not c.startswith(SKIP_COLUMNS)]
    num = df[cols].apply(pd.to_numeric, errors="coerce")
    num = num.loc[:, num.notna().sum() > 2]
    kinds = {c: kind_of(c) for c in num.columns}

    # compare in log scale for power features
    val = num.copy()
    logged = set()
    for c in val.columns:
        if kinds[c] == "power" and (val[c].dropna() > 0).all():
            val[c] = np.log10(val[c])
            logged.add(c)

    normal = ~odd
    animals = df["animal"].to_numpy()
    med = val[normal].groupby(animals[normal]).median()          # usual value of each mouse
    n_normal = pd.Series(animals[normal]).value_counts()
    resid = val[normal] - med.reindex(animals[normal]).to_numpy()
    # typical spread between normal sessions of the same mouse (robust, pooled over mice)
    spread = 1.4826 * resid.abs().median()
    spread = spread.where(spread > 0, resid.std())

    odd_idx = np.where(odd)[0]
    rows_dev, rows_ch = [], []
    for i in odd_idx:
        a = animals[i]
        if a not in med.index:
            print(f"[!] {a} has no normal session to compare with: skipped")
            continue
        base = med.loc[a]
        for c in val.columns:
            if pd.isna(val.iloc[i][c]) or pd.isna(base[c]):
                continue
            if kinds[c] == "channel":
                rows_ch.append({"animal": a, "session": df["session"].iloc[i], "column": c,
                                "odd_value": num.iloc[i][c], "usual_value": base[c],
                                "channels_away": num.iloc[i][c] - base[c]})
            elif kinds[c] != "count" and spread[c] > 0:
                raw_base = 10 ** base[c] if c in logged else base[c]
                rows_dev.append({"animal": a, "session": df["session"].iloc[i], "feature": c,
                                 "family": kinds[c], "odd_value": num.iloc[i][c],
                                 "usual_value_of_mouse": raw_base,
                                 "deviation": (val.iloc[i][c] - base[c]) / spread[c],
                                 "n_normal_sessions": int(n_normal.get(a, 0))})
            elif kinds[c] == "count":
                rows_dev.append({"animal": a, "session": df["session"].iloc[i], "feature": c,
                                 "family": "count", "odd_value": num.iloc[i][c],
                                 "usual_value_of_mouse": base[c],
                                 "deviation": (val.iloc[i][c] - base[c]) / spread[c] if spread[c] > 0 else np.nan,
                                 "n_normal_sessions": int(n_normal.get(a, 0))})
    dev = pd.DataFrame(rows_dev)
    chans = pd.DataFrame(rows_ch)
    # for CSD (sinks < 0, sources > 0) "higher/lower" is confusing: also say if it got WEAKER
    dev["closer_to_zero"] = dev["odd_value"].abs() < dev["usual_value_of_mouse"].abs()
    dev["label"] = dev["animal"] + "  " + dev["session"]
    n_odd = dev["label"].nunique()

    # ---------- common failing features ----------
    g = dev.groupby(["feature", "family"])["deviation"]
    common = pd.DataFrame({
        "odd_sessions_failing": g.apply(lambda d: int((d.abs() > Z_FAIL).sum())),
        "failing_LOW": g.apply(lambda d: int((d < -Z_FAIL).sum())),
        "failing_HIGH": g.apply(lambda d: int((d > Z_FAIL).sum())),
        "failing_and_WEAKER": dev.assign(f=(dev["deviation"].abs() > Z_FAIL) & dev["closer_to_zero"])
                                 .groupby(["feature", "family"])["f"].sum().astype(int),
        "median_deviation": g.median(),
        "n_odd_sessions": g.count(),
    }).reset_index()
    common["share_failing"] = common["odd_sessions_failing"] / common["n_odd_sessions"]
    common = common.sort_values(["odd_sessions_failing", "median_deviation"],
                                key=lambda s: s.abs() if s.name == "median_deviation" else s,
                                ascending=False).reset_index(drop=True)

    pd.set_option("display.width", 200)
    print(f"\nCOMMON FAILING FEATURES (|deviation| > {Z_FAIL} compared with the same mouse), "
          f"out of {n_odd} odd sessions:")
    show = common[common["odd_sessions_failing"] > 0].head(25)
    print(show[["feature", "family", "odd_sessions_failing", "failing_LOW", "failing_HIGH",
                "failing_and_WEAKER", "median_deviation"]].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("  LOW / HIGH = below / above the usual value of the mouse.")
    print("  WEAKER = closer to 0 than usual (for CSD: a sink less negative or a source less positive).")

    print("\nBY FAMILY (share of odd-session x feature pairs that fail):")
    fam = (dev.assign(fail=dev["deviation"].abs() > Z_FAIL)
           .groupby("family")["fail"].mean().sort_values(ascending=False))
    print((fam * 100).round(0).astype(int).astype(str).add(" %").to_string())

    if len(chans):
        print("\nCHANNELS: odd session vs the usual channel of that mouse (+ = deeper, - = more superficial)")
        tab = chans.pivot_table(index=["animal", "session"], columns="column", values="channels_away")
        print(tab.to_string(float_format=lambda v: f"{v:+.0f}" if pd.notna(v) else ""))
        moved = (tab.abs() >= 1).mean().sort_values(ascending=False)
        print("\nShare of odd sessions with the channel moved by >= 1 site:")
        print((moved * 100).round(0).astype(int).astype(str).add(" %").to_string())

    print(f"\nTOP {N_TOP_PER_SESSION} DIFFERENT FEATURES OF EACH ODD SESSION:")
    for lab, d in dev[dev["family"] != "count"].groupby("label", sort=False):
        top = d.reindex(d["deviation"].abs().sort_values(ascending=False).index).head(N_TOP_PER_SESSION)
        n_norm = int(d["n_normal_sessions"].iloc[0])
        txt = ", ".join(f"{r.feature} {r.deviation:+.1f}" for r in top.itertuples())
        print(f"  {lab} (vs {n_norm} normal session{'s' if n_norm != 1 else ''}): {txt}")

    # ---------- figure 1: heatmap ----------
    feat_order = [f for f in common["feature"] if f in set(dev[dev["family"] != "count"]["feature"])]
    H = (dev[dev["family"] != "count"].pivot_table(index="label", columns="feature", values="deviation")
         .reindex(columns=feat_order))
    fig, ax = plt.subplots(figsize=(max(8, 0.32 * H.shape[1] + 4), 0.45 * H.shape[0] + 3))
    im = ax.imshow(np.clip(H.to_numpy(float), -5, 5), cmap="RdBu_r", vmin=-5, vmax=5, aspect="auto")
    ax.set_xticks(range(H.shape[1]), H.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(H.shape[0]), H.index, fontsize=8)
    for r in range(H.shape[0]):
        for c in range(H.shape[1]):
            v = H.iloc[r, c]
            if pd.notna(v) and abs(v) > Z_FAIL:
                ax.text(c, r, "•", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="deviation from the same mouse (clipped ±5)")
    ax.set_title(f"Odd sessions vs the normal sessions of the same mouse "
                 f"(• = |deviation| > {Z_FAIL}); features sorted by how often they fail")
    fig.tight_layout(); fig.savefig(out_dir / "01_deviation_heatmap.png", dpi=200)

    # ---------- figure 2: common failing ----------
    top_c = common[(common["odd_sessions_failing"] > 0) & (common["family"] != "count")].head(20)[::-1]
    fig2, ax2 = plt.subplots(figsize=(8, 0.35 * len(top_c) + 2))
    ax2.barh(range(len(top_c)), -top_c["failing_LOW"], color="#2166ac", label="lower than usual")
    ax2.barh(range(len(top_c)), top_c["failing_HIGH"], color="#b2182b", label="higher than usual")
    ax2.axvline(0, color="k", lw=0.8)
    ax2.set_yticks(range(len(top_c)), top_c["feature"], fontsize=8)
    ax2.set_xlabel(f"number of odd sessions failing (out of {n_odd})   ← lower | higher →")
    ax2.set_xlim(-n_odd - 0.5, n_odd + 0.5)
    ax2.legend(fontsize=8, loc="lower right")
    ax2.set_title("Features that fail in most odd sessions, and in which direction")
    fig2.tight_layout(); fig2.savefig(out_dir / "02_common_failing.png", dpi=200)

    with pd.ExcelWriter(out_dir / "odd_sessions_report.xlsx") as w:
        common.to_excel(w, sheet_name="common_failing", index=False)
        dev.drop(columns="label").to_excel(w, sheet_name="deviation_per_session", index=False)
        if len(chans):
            chans.to_excel(w, sheet_name="channels", index=False)
        spread.rename("typical_spread_within_mouse").to_frame().to_excel(w, sheet_name="typical_spread")
    print(f"\nSaved in {out_dir}")
    plt.show()


if __name__ == "__main__":
    main()