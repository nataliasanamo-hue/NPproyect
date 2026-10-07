# save_power_spectrum_NPC_project.py
# Translation of save_power_spectrum_NPC_project.m -- the FEATURE part
# (sections 1, 2, 3, 5 and 6 of the .m). Section 4 (power spectrum figure with plotFill)
# and the boxplots with stars (groupStats) are NOT here yet.
#
# For every session it computes, for the PYR and SLM channels, during THETA and LIA:
#   mean and max power in each frequency band (Delta, Theta, Alpha, SlowGamma, MidGamma,
#   FastGamma, Ripple), plus the mean of the first three columns of propRipS per session.
#
# FOLDERS
#   reads only : <session>/analyset/{psProfile,thetaLIAAnalysis,configChannels}.mat  (coworkers' files)
#                <session>/ProcesamientoNatalia/propRipS.csv       (made by export_propRipS.m)
#   per session: <session>/ProcesamientoNatalia/power_features.csv
#                <session>/ProcesamientoNatalia/mean_power_spectra.csv
#   all sessions: <baseDir>/RESULTSNatalia/features_all_sessions.csv (+ .xlsx)
import traceback
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# ================== SETTINGS ==================
baseDir = Path(r"D:\Violeta\ANALISIS")
SESSION_OUT_FOLDER = "ProcesamientoNatalia"
TOTAL_OUT_FOLDER = "RESULTSNatalia"
# To test ONE session first, write its path here, e.g.
# ONLY_SESSION = r"D:\Violeta\ANALISIS\EFV_218\2026-01-12_18-02-06"
ONLY_SESSION = None

CONDITIONS = ["WT", "NPC", "EFV"]
FS = 2500  # MATLAB: fs = 2500
PS_FREQS = np.arange(1, 1000.5, 0.5)  # MATLAB: psFreqs = 1:0.5:1000 (1999 values)

# Same order as the MATLAB struct "bands" (both ends included)
BANDS = {
    "Delta": (0, 3),
    "Theta": (4, 12),
    "Alpha": (9, 16),
    "SlowGamma": (20, 40),
    "MidGamma": (40, 60),
    "FastGamma": (60, 80),
    "Ripple": (150, 250),
}
# (field of configChannels.ch, name)   -- 'theta' is the SLM channel in your config
LAYERS = [("pyr", "PYR"), ("theta", "SLM")]
STATES = ["Theta", "LIA"]

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ----------------------------------------------------------------------------
# Reading the v7.3 (HDF5) files.  MATLAB stores arrays with the axes reversed in HDF5,
# so a MATLAB matrix [a x b x c] arrives here as [c x b x a].
# ----------------------------------------------------------------------------
def h5_scalar(f, path):
    return float(np.squeeze(f[path][()]))


def events_matrix(f, path):
    """MATLAB [nEvents x 2] (start, end) -> numpy (nEvents, 2)."""
    a = np.asarray(f[path][()], dtype=float)
    if a.size == 0:
        return np.empty((0, 2))
    a = a.T
    if a.ndim == 1:
        a = a.reshape(1, -1)
    return a


def load_session_inputs(analyset):
    """Reads only what the .m uses. Returns a dict (nothing is written to analyset)."""
    with h5py.File(analyset / "configChannels.mat", "r") as f:
        ch = {name: int(h5_scalar(f, f"configChannels/ch/{name}")) for name, _ in LAYERS}

    with h5py.File(analyset / "thetaLIAAnalysis.mat", "r") as f:
        timesTheta = events_matrix(f, "thetaLIAAnalysis/timesTheta")
        timesLIA = events_matrix(f, "thetaLIAAnalysis/timesLIA")

    with h5py.File(analyset / "psProfile.mat", "r") as f:
        tWin = h5_scalar(f, "psProfile/params/tWin")
        dtWin = h5_scalar(f, "psProfile/params/dtWin")
        pS = f["psProfile/pS"]  # HDF5 shape (nCh, nWin, nFreq) = MATLAB (nFreq, nWin, nCh)
        n_ch, n_win, n_freq = pS.shape
        for name, _ in LAYERS:
            if not 1 <= ch[name] <= n_ch:
                raise IndexError(f"channel {ch[name]} ({name}) outside 1..{n_ch}")
        # only the 2 channels that are needed are read (the file is ~600 MB)
        slabs = {name: np.asarray(pS[ch[name] - 1, :, :], dtype=float) for name, _ in LAYERS}
        pSfreqs = (
            np.asarray(f["psProfile/pSfreqs"][()], dtype=float).ravel()
            if "pSfreqs" in f["psProfile"]
            else None
        )

    return dict(
        ch=ch,
        timesTheta=timesTheta,
        timesLIA=timesLIA,
        tWin=tWin,
        dtWin=dtWin,
        slabs=slabs,  # name -> (nWin, nFreq)
        pSfreqs=pSfreqs,
    )


# ----------------------------------------------------------------------------
# Calculations (same logic as the .m)
# ----------------------------------------------------------------------------
def windows_with_events(times, bins):
    """
    MATLAB: calcWindows = @(times) any((times(:,1) < pSbins(:,2)') & (times(:,2) > pSbins(:,1)'), 1)'
    A power-spectrum window is used if ANY event overlaps it.
    """
    if times.shape[0] == 0:
        return np.zeros(bins.shape[0], dtype=bool)
    return np.any(
        (times[:, 0:1] < bins[:, 1][None, :]) & (times[:, 1:2] > bins[:, 0][None, :]), axis=0
    )


def mean_spectrum(slab, mask):
    """MATLAB squeeze(mean(pS(:, mask, :), 2)) for one channel. No windows -> NaN."""
    if not mask.any():
        return np.full(slab.shape[1], np.nan)
    return slab[mask, :].mean(axis=0)


def band_mean_max(spectrum, lo, hi):
    idx = (PS_FREQS >= lo) & (PS_FREQS <= hi)
    seg = spectrum[idx]
    if seg.size == 0:
        return np.nan, np.nan
    mx = np.nan if np.all(np.isnan(seg)) else np.nanmax(seg)  # MATLAB max ignores NaN
    return float(np.mean(seg)), float(mx)  # MATLAB mean keeps NaN


def ripple_means(session_out):
    """
    MATLAB section 6: mean (omitnan) of columns 1, 2 and 3 of propRipS, per session.
    Needs propRipS.csv made by export_propRipS.m.
    """
    f = session_out / "propRipS.csv"
    if not f.exists():
        return {}, np.nan, []
    df = pd.read_csv(f)
    first3 = df.iloc[:, :3].apply(pd.to_numeric, errors="coerce")
    names = list(first3.columns)
    return {f"ripple_{c}_mean": float(first3[c].mean(skipna=True)) for c in names}, len(df), names


def process_session(session, animal, condition):
    inp = load_session_inputs(session / "analyset")

    wSize = inp["tWin"] / FS
    wStep = inp["dtWin"] / FS
    n_win, n_freq = next(iter(inp["slabs"].values())).shape
    if n_freq != len(PS_FREQS):
        raise ValueError(f"pS has {n_freq} frequencies, psFreqs has {len(PS_FREQS)}")
    if inp["pSfreqs"] is not None and not np.allclose(inp["pSfreqs"], PS_FREQS):
        print("   WARNING: psProfile.pSfreqs differs from 1:0.5:1000 (the .m uses 1:0.5:1000)")

    starts = np.arange(n_win) * wStep
    bins = np.column_stack([starts, starts + wSize])
    masks = {
        "Theta": windows_with_events(inp["timesTheta"], bins),
        "LIA": windows_with_events(inp["timesLIA"], bins),
    }

    # mean spectrum per state and layer
    spectra = {
        (name, state): mean_spectrum(inp["slabs"][field], masks[state])
        for field, name in LAYERS
        for state in STATES
    }

    session_out = session / SESSION_OUT_FOLDER
    session_out.mkdir(exist_ok=True)

    rip, n_rip, rip_cols = ripple_means(session_out)

    row = {
        "animal": animal,
        "session": session.name,
        "group": condition,
        "pyr_channel": inp["ch"]["pyr"],
        "slm_channel": inp["ch"]["theta"],
        "n_theta_windows": int(masks["Theta"].sum()),
        "n_LIA_windows": int(masks["LIA"].sum()),
        "n_ripples": n_rip,
        **rip,
    }
    # Same loop order as the .m: band > state > layer
    for band, (lo, hi) in BANDS.items():
        for state in STATES:
            for _, name in LAYERS:
                m, mx = band_mean_max(spectra[(name, state)], lo, hi)
                row[f"{band}_{name}_{state}_mean"] = m
                row[f"{band}_{name}_{state}_max"] = mx

    pd.DataFrame([row]).to_csv(session_out / "power_features.csv", index=False)
    pd.DataFrame(
        {"freq_Hz": PS_FREQS, **{f"{n}_{s}": spectra[(n, s)] for _, n in LAYERS for s in STATES}}
    ).to_csv(session_out / "mean_power_spectra.csv", index=False)
    return row, rip_cols


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def detect_condition(animal_name):
    """MATLAB: contains(animalName, c, 'IgnoreCase', true)"""
    for c in CONDITIONS:
        if c.lower() in animal_name.lower():
            return c
    return None


def list_sessions():
    if ONLY_SESSION:
        s = Path(ONLY_SESSION)
        return [(s.parent.name, detect_condition(s.parent.name) or "?", s)]
    out = []
    for animalDir in sorted((d for d in baseDir.iterdir() if d.is_dir()), key=lambda d: d.name.lower()):
        cond = detect_condition(animalDir.name)
        if cond is None:
            continue
        for sess in sorted((d for d in animalDir.iterdir() if d.is_dir()), key=lambda d: d.name.lower()):
            if (sess / "analyset").is_dir():  # MATLAB: if isfolder(analysetFolder)
                out.append((animalDir.name, cond, sess))
    return out


def main():
    sessions = list_sessions()
    print(f"{len(sessions)} sessions to process")
    rows, failed, shown_cols = [], [], False

    for i, (animal, cond, sess) in enumerate(sessions, start=1):
        print(f"[{i}/{len(sessions)}] {cond}  {animal}  {sess.name}")
        try:
            row, rip_cols = process_session(sess, animal, cond)
            rows.append(row)
            if not shown_cols:
                if rip_cols:
                    print(f"   propRipS columns used as 1,2,3: {rip_cols}")
                else:
                    print("   (no propRipS.csv yet: run export_propRipS.m to get the ripple features)")
                shown_cols = True
        except Exception as e:
            print(f"   Warning: {sess.name} failed: {type(e).__name__}: {e}")
            if ONLY_SESSION:
                traceback.print_exc()
            failed.append(f"{animal}/{sess.name}")

    if not rows:
        print("Nothing processed.")
        return

    table = pd.DataFrame(rows)
    out_dir = baseDir / TOTAL_OUT_FOLDER
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "features_all_sessions.csv", index=False)
    try:
        table.to_excel(out_dir / "features_all_sessions.xlsx", index=False)
    except Exception as e:
        print(f"(Excel not written: {e})")

    print("\nSessions per group: " + ", ".join(f"{g}={int((table.group == g).sum())}" for g in CONDITIONS))
    print(f"Saved: {out_dir / 'features_all_sessions.csv'}")
    if failed:
        print(f"{len(failed)} failed:")
        for f in failed:
            print("   ", f)


if __name__ == "__main__":
    main()