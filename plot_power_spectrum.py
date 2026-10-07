# plot_power_spectrum.py
# Power spectrum figure by group (WT, NPC, EFV): mean +- SEM across sessions,
# 2 x 2 panels: PYR / SLM channel  x  Theta / LIA state (LIA dashed), 0-80 Hz, log y axis.
# This is the figure that save_power_spectrum_struct.m / save_power_spectrum_NPC_project.m draw with plotFill.
#
# READS only small files (it does NOT open psProfile, nothing is recomputed):
#   <baseDir>/RESULTSNatalia/features_final.csv                    sessions that are in the final table
#   <session>/ProcesamientoNatalia/mean_power_spectra.csv          made by save_power_spectrum_NPC_project.py
#       columns: freq_Hz, PYR_Theta, PYR_LIA, SLM_Theta, SLM_LIA (mean spectrum of the session)
# WRITES:
#   <baseDir>/RESULTSNatalia/group_comparison/power_spectrum_groups.png
#   <baseDir>/RESULTSNatalia/group_comparison/power_spectrum_groups_curves.csv  (mean, SEM, n per frequency)
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ================== SETTINGS ==================
baseDir = Path(r"D:\Violeta\ANALISIS")
SESSION_OUT_FOLDER = "ProcesamientoNatalia"
TOTAL_OUT_FOLDER = "RESULTSNatalia"
FIG_FOLDER = "group_comparison"
FEATURES_FILE = "features_final.csv"

GROUPS = ["WT", "NPC", "EFV"]
COLORS = {"WT": (0.24, 0.60, 0.38), "NPC": (0.22, 0.45, 0.70), "EFV": (0.58, 0.40, 0.67)}
XLIM = (0, 80)
PANELS = [  # (column of mean_power_spectra.csv, title, linestyle)
    ("PYR_Theta", "PYR Channel - Theta State", "-"),
    ("SLM_Theta", "SLM Channel - Theta State", "-"),
    ("PYR_LIA", "PYR Channel - LIA State", "--"),
    ("SLM_LIA", "SLM Channel - LIA State", "--"),
]


def load_spectra(sessions):
    """Returns freq, {column: {group: array (n_sessions, n_freq)}}."""
    freq = None
    spectra = {c: {g: [] for g in GROUPS} for c, _, _ in PANELS}
    for _, r in sessions.iterrows():
        f = baseDir / r["animal"] / r["session"] / SESSION_OUT_FOLDER / "mean_power_spectra.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        if freq is None:
            freq = d["freq_Hz"].to_numpy(float)
        elif len(d) != len(freq):
            print(f"Skipped (different frequencies): {r['animal']}/{r['session']}")
            continue
        for c, _, _ in PANELS:
            if c in d and r["group"] in GROUPS:
                spectra[c][r["group"]].append(d[c].to_numpy(float))
    spectra = {c: {g: (np.vstack(v) if v else np.empty((0, 0))) for g, v in gd.items()} for c, gd in spectra.items()}
    return freq, spectra


def mean_sem(M):
    """Mean and SEM across sessions (rows), ignoring NaN."""
    n = np.sum(~np.isnan(M), axis=0)
    mean = np.nanmean(M, axis=0)
    sd = np.nanstd(M, axis=0, ddof=1) if M.shape[0] > 1 else np.full(M.shape[1], np.nan)
    return mean, sd / np.sqrt(np.maximum(n, 1)), n


def main():
    sessions = pd.read_csv(baseDir / TOTAL_OUT_FOLDER / FEATURES_FILE)[["animal", "session", "group"]]
    freq, spectra = load_spectra(sessions)
    if freq is None:
        print("No mean_power_spectra.csv found.")
        return

    out_dir = baseDir / TOTAL_OUT_FOLDER / FIG_FOLDER
    out_dir.mkdir(parents=True, exist_ok=True)
    sel = (freq >= XLIM[0]) & (freq <= XLIM[1])

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4))
    curves = {"freq_Hz": freq}
    for ax, (col, title, ls) in zip([axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]], PANELS):
        for g in GROUPS:
            M = spectra[col][g]
            if M.size == 0:
                continue
            m, sem, n = mean_sem(M)
            curves[f"{col}_{g}_mean"], curves[f"{col}_{g}_sem"], curves[f"{col}_{g}_n"] = m, sem, n
            ax.fill_between(freq[sel], (m - sem)[sel], (m + sem)[sel], color=COLORS[g], alpha=0.4, linewidth=0)
            ax.plot(freq[sel], m[sel], color=COLORS[g], linestyle=ls, linewidth=1.2, label=g)
        ax.set_yscale("log")
        ax.set_xlim(*XLIM)
        ax.set_xlabel("Frequency (Hz)", fontsize=11)
        ax.set_ylabel("Power (dB)", fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
    axes[0, 0].legend(loc="upper right", fontsize=9, frameon=True, edgecolor="k", fancybox=False)
    fig.suptitle("Power Spectrum Group Comparison (Mean ± SEM)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "power_spectrum_groups.png", dpi=150)
    plt.close(fig)

    pd.DataFrame(curves).to_csv(out_dir / "power_spectrum_groups_curves.csv", index=False)
    print("Sessions per group (PYR Theta): " + ", ".join(f"{g}={spectra['PYR_Theta'][g].shape[0]}" for g in GROUPS))
    print(f"Saved: {out_dir / 'power_spectrum_groups.png'}")


if __name__ == "__main__":
    main()