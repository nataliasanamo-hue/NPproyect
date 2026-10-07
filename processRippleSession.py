# processRippleSession.py
# Line-by-line translation of processRippleSession.m (MATLAB) to Python.
# Output goes to <session>/analyset_python so it never overwrites the MATLAB results.
import traceback
import warnings
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import scipy.io as sio
from scipy.signal import firls, filtfilt, find_peaks, savgol_filter

# Expected warnings on all-NaN slices (ripples that fall outside the window, etc.)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# MATLAB's default colormap is parula, which matplotlib does not ship.
# This rebuilds it by interpolating between its main colours
# (dark blue -> blue -> cyan -> green -> olive -> orange -> yellow).
_PARULA_ANCHORS = [
    (0.2422, 0.1504, 0.6603),
    (0.2810, 0.3228, 0.9579),
    (0.1786, 0.5289, 0.9682),
    (0.0689, 0.6948, 0.8394),
    (0.2161, 0.7843, 0.5923),
    (0.6720, 0.7793, 0.2227),
    (0.9970, 0.7659, 0.2199),
    (0.9769, 0.9839, 0.0805),
]
CMAP = LinearSegmentedColormap.from_list("parula", _PARULA_ANCHORS, N=256)
CLIM = (-8, 5)  # MATLAB: clim([-8 5])


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def matlab_round(x):
    """MATLAB round(): halves go away from zero (numpy/python round to even)."""
    return int(np.floor(x + 0.5)) if x >= 0 else int(-np.floor(-x + 0.5))


def bz_LoadBinary(full_dat_file, num_channels):
    """Loads int16 binary LFP into a double matrix [samples x channels]."""
    raw_data = np.fromfile(full_dat_file, dtype=np.int16)
    n_samples = len(raw_data) // num_channels
    # like bz_LoadBinary: drop an incomplete last sample instead of crashing
    raw_data = raw_data[: n_samples * num_channels]
    return raw_data.reshape((n_samples, num_channels)).astype(np.float64)


def to_scalar_int(val):
    """Safely convert numpy/MATLAB 0D or 1D single elements to int."""
    return int(np.squeeze(np.array(val)).item())


def load_mat_variable(file_path, var_name):
    """Loads a MATLAB variable supporting both v5/v7 and v7.3 (HDF5) formats."""
    file_path = Path(file_path)
    try:
        data = sio.loadmat(file_path, simplify_cells=True)
        return data[var_name]
    except Exception:
        with h5py.File(file_path, "r") as f:
            if var_name not in f:
                raise KeyError(f"Variable '{var_name}' not found in {file_path}")
            obj = f[var_name]

            if isinstance(obj, h5py.Dataset):
                return np.array(obj)

            def h5_to_dict(group):
                res = {}
                for k, v in group.items():
                    if isinstance(v, h5py.Dataset):
                        arr = np.array(v)
                        res[k] = arr.squeeze() if arr.size == 1 else arr
                    elif isinstance(v, h5py.Group):
                        res[k] = h5_to_dict(v)
                return res

            return h5_to_dict(obj)


def movmean_matlab(x, wlen):
    """
    MATLAB movmean(x, wlen) with the default 'Endpoints','shrink'.
      odd  wlen -> (wlen-1)/2 samples on each side
      even wlen -> wlen/2 samples before, wlen/2 - 1 after
    At the edges the window shrinks (it is NOT zero-padded like np.convolve).
    """
    wlen = matlab_round(wlen)
    if wlen <= 1:
        return x.copy()
    if wlen % 2 == 1:
        nb = na = (wlen - 1) // 2
    else:
        nb, na = wlen // 2, wlen // 2 - 1
    n = len(x)
    c = np.concatenate(([0.0], np.cumsum(x)))
    idx = np.arange(n)
    lo = np.maximum(idx - nb, 0)
    hi = np.minimum(idx + na, n - 1)
    return (c[hi + 1] - c[lo]) / (hi - lo + 1)


def layer_available(configChannels, area, shankNum):
    """MATLAB: isfield(ch, area) && ~all(isnan(ch.(area))) && ~isnan(ch.(area)(shankNum))"""
    try:
        vals = np.atleast_1d(
            np.array(configChannels["ch"][area], dtype=float).flatten()
        )
    except Exception:
        return False
    if vals.size == 0 or np.all(np.isnan(vals)):
        return False
    idx = int(shankNum) - 1
    return 0 <= idx < vals.size and not np.isnan(vals[idx])


# ----------------------------------------------------------------------------
# Main function
# ----------------------------------------------------------------------------
def processRippleSession(session_path, so_ch, pyr_ch, rad_ch, slm_ch):
    session_path = Path(session_path)
    analyset_path = session_path / "analyset"

    if not analyset_path.exists():
        raise FileNotFoundError(f"Analyset directory not found: {analyset_path}")

    try:
        rippleAnalysis = load_mat_variable(
            analyset_path / "rippleAnalysis.mat", "rippleAnalysis"
        )
        basicData = load_mat_variable(analyset_path / "basicData.mat", "basicData")
        configChannels = load_mat_variable(
            analyset_path / "configChannels.mat", "configChannels"
        )

        windowSize = 0.05
        spontMarginSizes = np.array([0.02, 0.02])
        slack = 0.008

        fs = float(np.squeeze(basicData["fsLFP"]).item())
        windowHalfSize = int(np.ceil(windowSize / 2.0 * fs))
        spontMarginSizes = np.array([matlab_round(v * fs) for v in spontMarginSizes])
        slack = matlab_round(slack * fs)

        rip_chs_flat = np.atleast_1d(
            np.array(configChannels["ch"]["ripple"]).flatten()
        )

        def ripple_channel_index(shankNum):
            """0-based column of dataLFP for configChannels.ch.ripple(shankNum)."""
            s_idx = to_scalar_int(shankNum) - 1
            if not (0 <= s_idx < len(rip_chs_flat)):
                # MATLAB would raise "index exceeds array bounds" here
                raise IndexError(
                    f"No ripple channel defined for shank {shankNum} "
                    f"(configChannels.ch.ripple has {len(rip_chs_flat)} entries)"
                )
            return to_scalar_int(rip_chs_flat[s_idx]) - 1

        # ---- filter design (firls, order N -> N+1 taps) -----------------------
        N = 20
        fStop1, fPass1, fPass2, fStop2 = 65.0, 70.0, 400.0, 422.0
        wStop1, wPass, wStop2 = 1.0, 1.0, 1.0

        nyq = fs / 2.0
        bRP = firls(
            N + 1,
            [0, fStop1 / nyq, fPass1 / nyq, fPass2 / nyq, fStop2 / nyq, 1.0],
            [0, 0, 1, 1, 0, 0],
            weight=[wStop1, wPass, wStop2],
        )
        # MATLAB filtfilt pads 3*(max(len(a),len(b))-1) samples; scipy's default is 3*max(...)
        padlen = 3 * (len(bRP) - 1)

        frameLength = 2 * int(np.floor(0.0334 * fs / 2.0)) + 1
        datFile = "LFP_downsampled.dat"

        chMap = np.array(configChannels["chMap"]).flatten()
        numChannels = int(np.max(chMap))

        probeType = to_scalar_int(configChannels["probeType"])

        if probeType == 8:
            rec_dirs = sorted(
                d
                for d in session_path.iterdir()
                if d.is_dir() and d.name.startswith("Record")
            )
            if not rec_dirs:
                raise FileNotFoundError(f"No Record* folder found in {session_path}")

            contPath = rec_dirs[0] / "experiment1" / "recording1" / "continuous"
            if not contPath.exists():
                raise FileNotFoundError(f"continuous folder not found in {contPath}")

            # MATLAB: Intan* first, and only if none, Acquisition*
            data_dirs = sorted(
                d for d in contPath.iterdir() if d.is_dir() and d.name.startswith("Intan")
            )
            if not data_dirs:
                data_dirs = sorted(
                    d
                    for d in contPath.iterdir()
                    if d.is_dir() and d.name.startswith("Acquisition")
                )
            if not data_dirs:
                raise FileNotFoundError(
                    f"No Intan* or Acquisition* folder found in {contPath}"
                )
            datPath = data_dirs[0]

        elif probeType == 1:
            datPath = session_path
        else:
            raise ValueError(f"Unknown probeType: {probeType}")

        fullDatFile = datPath / datFile
        if not fullDatFile.exists():
            raise FileNotFoundError(f"{fullDatFile} not found")

        dataLFP = bz_LoadBinary(fullDatFile, numChannels)

        # ---- filter signal and compute envelope -------------------------------
        shRipS = np.atleast_1d(np.array(rippleAnalysis["shRipS"]).flatten())
        unique_shanks = np.unique(shRipS)

        filtLFP = {}
        ripEnv = {}

        for shankNum in unique_shanks:
            key = to_scalar_int(shankNum)
            ch_idx = ripple_channel_index(shankNum)

            filtered = filtfilt(bRP, [1.0], dataLFP[:, ch_idx], padlen=padlen)
            filtLFP[key] = filtered

            envelope = filtered * filtered
            envelope = savgol_filter(envelope, frameLength, 4)
            # MATLAB: movmean(movmean(env, 0.0030*fs), 0.0065*fs)
            envelope = movmean_matlab(envelope, 0.0030 * fs)
            envelope = movmean_matlab(envelope, 0.0065 * fs)
            ripEnv[key] = envelope

        # ---- re-align ripples -------------------------------------------------
        iRipS = np.atleast_2d(np.array(rippleAnalysis["iRipS"]))
        if iRipS.shape[1] != 2 and iRipS.shape[0] == 2:
            iRipS = iRipS.T  # v7.3 files come transposed

        numRipples = int(iRipS.shape[0])  # MATLAB: height(iRipS)
        if len(shRipS) != numRipples:
            raise ValueError(
                f"iRipS has {numRipples} ripples but shRipS has {len(shRipS)} entries"
            )

        rawRipples = np.full((numRipples, 2 * windowHalfSize + 1), np.nan)
        filtRipples = np.full((numRipples, 2 * windowHalfSize + 1), np.nan)
        alignedRippleTimes = np.full((numRipples, 3), np.nan)

        RipStart = iRipS[:, 0]
        RipEnd = iRipS[:, 1]

        for rippleNum in range(numRipples):
            shankNum = to_scalar_int(shRipS[rippleNum])

            # iRipS is 1-based (MATLAB) -> 0-based
            r_start = to_scalar_int(RipStart[rippleNum]) - 1
            r_end = to_scalar_int(RipEnd[rippleNum]) - 1

            rippleIdx = np.arange(
                r_start - spontMarginSizes[0], r_end + spontMarginSizes[1] + 1
            )
            env = ripEnv[shankNum][rippleIdx]
            iMax = np.argmax(env)
            iMiddle = rippleIdx[0] + iMax

            # align to a minimum of the filtered LFP
            seg = -filtLFP[shankNum][iMiddle - slack : iMiddle + slack + 1]
            valleys, _ = find_peaks(seg)
            if len(valleys) == 0:
                raise ValueError(
                    f"Ripple {rippleNum + 1}: no local minimum found around the "
                    "envelope maximum (MATLAB would also fail here)"
                )
            iLowest = np.argmax(seg[valleys])
            iMinimum = iMiddle - slack + valleys[iLowest]

            windowIdx = np.arange(
                iMinimum - windowHalfSize, iMinimum + windowHalfSize + 1
            )

            ch_rip = ripple_channel_index(shankNum)
            rawRipples[rippleNum, :] = dataLFP[windowIdx, ch_rip]
            filtRipples[rippleNum, :] = filtLFP[shankNum][windowIdx]

            alignedRippleTimes[rippleNum, :] = [
                iMinimum - windowHalfSize,
                iMinimum,
                iMinimum + windowHalfSize,
            ]

        alignedRippleTimes = alignedRippleTimes.astype(int)  # 0-based sample indices

        # ---- CSD --------------------------------------------------------------
        windowSize_csd = 0.1
        windowHalfSize_csd = int(np.ceil(windowSize_csd / 2.0 * fs))

        if probeType == 8:
            shankNums = np.array([1])
            shankChs = np.arange(1, 17)
        else:  # probeType == 1
            shankNums = np.array([1])
            shankChs = np.arange(1, 16)

        chJump = 1
        previousShankChs = shankChs[: -chJump * 2]
        centralShankChs = shankChs[chJump:-chJump]
        nextShankChs = shankChs[chJump * 2 :]

        CSD = {}
        CSDStds = {}
        shMap = np.array(configChannels["shMap"]).flatten()

        for shankNum in shankNums:
            key = int(shankNum)
            firstShankCh = np.where(shMap == shankNum)[0][0] + 1  # 1-based

            previousChs = previousShankChs + firstShankCh - 1
            centralChs = centralShankChs + firstShankCh - 1
            nextChs = nextShankChs + firstShankCh - 1

            CSD[key] = (
                2 * dataLFP[:, centralChs - 1]
                - dataLFP[:, previousChs - 1]
                - dataLFP[:, nextChs - 1]
            )
            # MATLAB std() normalises by N-1 -> ddof=1 (numpy default is N)
            CSDStds[key] = np.std(CSD[key], axis=0, ddof=1)

        CSDs = np.full(
            (windowHalfSize_csd * 2 + 1, len(centralShankChs), numRipples), np.nan
        )

        for rippleNum in range(numRipples):
            rippleShank = to_scalar_int(shRipS[rippleNum])
            if rippleShank in CSD:
                center_t = alignedRippleTimes[rippleNum, 1]
                rippleIdx = np.arange(
                    center_t - windowHalfSize_csd, center_t + windowHalfSize_csd + 1
                )
                if rippleIdx[0] >= 0 and rippleIdx[-1] < CSD[rippleShank].shape[0]:
                    CSDs[:, :, rippleNum] = (
                        CSD[rippleShank][rippleIdx, :] / CSDStds[rippleShank]
                    )

        # ---- layer CSD values -------------------------------------------------
        def layer_csd(ch):
            idx = np.where(centralShankChs == ch)[0]
            if len(idx) == 0:
                return None
            return CSDs[:, idx[0], :]

        CSD_so = layer_csd(so_ch)
        CSD_pyr = layer_csd(pyr_ch)
        CSD_rad = layer_csd(rad_ch)
        CSD_slm = layer_csd(slm_ch)

        if CSD_rad is None or CSD_slm is None:
            raise ValueError(
                f"Rad ({rad_ch}) or SLM ({slm_ch}) channel not found in "
                f"centralShankChs ({centralShankChs})"
            )
        empty = np.full((CSDs.shape[0], numRipples), np.nan)
        if CSD_so is None:
            CSD_so = empty
        if CSD_pyr is None:
            CSD_pyr = empty

        def layer_struct(v):
            return {
                "values": v,
                "mean": np.mean(v, axis=0),  # MATLAB mean(.,1)
                "max": np.nanmax(v, axis=0),  # MATLAB max ignores NaN
                "min": np.nanmin(v, axis=0),
            }

        rippleCSDs = {
            "CSDs": CSDs,
            "previousShankChs": previousShankChs,
            "centralShankChs": centralShankChs,
            "nextShankChs": nextShankChs,
            "shankNums": shankNums,
            "t": np.arange(-windowHalfSize_csd, windowHalfSize_csd + 1) / fs,
            "windowHalfSize": windowHalfSize_csd,
            "so": layer_struct(CSD_so),
            "pyr": layer_struct(CSD_pyr),
            "rad": layer_struct(CSD_rad),
            "slm": layer_struct(CSD_slm),
        }

        output_dir = session_path / "analyset_python"
        output_dir.mkdir(parents=True, exist_ok=True)
        sio.savemat(output_dir / "rippleCSDs.mat", {"rippleCSDs": rippleCSDs})

        # ---- plot -------------------------------------------------------------
        t_ms = rippleCSDs["t"] * 1000.0
        pad = 10
        nSh = len(shankNums)
        fig, axes = plt.subplots(
            1, nSh, figsize=(min(6 * nSh, 19.2), 10.8), squeeze=False
        )

        for iShank, shankNum in enumerate(shankNums):
            ax = axes[0, iShank]
            shankNum = int(shankNum)

            allShankChs = np.union1d(
                np.union1d(previousShankChs, centralShankChs), nextShankChs
            )
            first_sh_idx = np.where(shMap == shankNum)[0][0]  # 0-based
            chs = allShankChs + first_sh_idx  # 1-based channel numbers

            shankRippleNums = np.where(shRipS == shankNum)[0]
            LFPs = np.full((len(t_ms), len(chs), len(shankRippleNums)), np.nan)

            for iShankRipple, shankRippleNum in enumerate(shankRippleNums):
                center_t = alignedRippleTimes[shankRippleNum, 1]
                rippleIdx = np.arange(
                    center_t - windowHalfSize_csd, center_t + windowHalfSize_csd + 1
                )
                if rippleIdx[0] >= 0 and rippleIdx[-1] < dataLFP.shape[0]:
                    LFPs[:, :, iShankRipple] = dataLFP[np.ix_(rippleIdx, chs - 1)]

            sel = shRipS == shankNum
            meanCSD = np.nanmean(CSDs[:, :, sel], axis=2)  # (time, central chs)

            # imagesc(t, y0:y1, meanCSD'): one pixel per channel, NO interpolation
            y0 = np.where(allShankChs == centralShankChs[0])[0][0] + 1
            y1 = np.where(allShankChs == centralShankChs[-1])[0][0] + 1
            im = ax.imshow(
                np.ma.masked_invalid(meanCSD.T),
                aspect="auto",
                extent=[t_ms[0], t_ms[-1], y1 + 0.5, y0 - 0.5],
                cmap=CMAP,
                vmin=CLIM[0],
                vmax=CLIM[1],
                interpolation="nearest",
                origin="upper",
            )
            fig.colorbar(im, ax=ax, pad=0.12)

            # LFP traces: -meanLFP / (max|meanLFP|/2) + iCh
            meanLFP = np.nanmean(LFPs, axis=2)
            normDivisor = np.nanmax(np.abs(meanLFP)) / 2.0
            for iCh in range(len(chs)):
                ax.plot(t_ms, -meanLFP[:, iCh] / normDivisor + (iCh + 1), "k", lw=0.8)

            ax.set_xlim(t_ms[0], t_ms[-1])
            ax.set_ylim(len(chs) + 2, -2)  # MATLAB: ylim([-2, n+2]) with YDir reverse

            areas = {"so": so_ch, "pyr": pyr_ch, "rad": rad_ch, "slm": slm_ch}
            for area, areaCh in areas.items():
                if layer_available(configChannels, area, shankNum) and not np.isnan(areaCh):
                    ax.text(
                        t_ms[0] - pad,
                        areaCh,
                        area.upper(),
                        ha="left",
                        va="center",
                        fontsize=9,
                    )

            ax.set_yticks(np.arange(1, len(chs) + 1))
            ax.set_title(f"shank {shankNum} ({len(shankRippleNums)} ripples)")
            ax.set_xlabel("Time (ms)")

            # right-hand axis with the real channel numbers
            ax2 = ax.twinx()
            ax2.set_ylim(ax.get_ylim())
            ax2.set_yticks(np.arange(1, len(chs) + 1))
            ax2.set_yticklabels(chs)

        sessionName = session_path.name
        fig.suptitle(sessionName)
        fig.tight_layout()
        fig.savefig(output_dir / f"{sessionName}MeanRippleCSD.png", dpi=150)
        plt.close(fig)

    except Exception:
        print(f"\n--- DETAILED ERROR TRACE FOR SESSION {session_path.name} ---")
        traceback.print_exc()
        print("-----------------------------------------------------------\n")
        raise
    