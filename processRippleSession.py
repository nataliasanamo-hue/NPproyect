# processRippleSession.py
import os
import sys
import traceback
import warnings
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import firls, filtfilt, find_peaks, savgol_filter
import scipy.io as sio

# Ignore expected NumPy slice/mean warnings on empty or NaN arrays
warnings.filterwarnings("ignore", category=RuntimeWarning)


def bz_LoadBinary(full_dat_file, num_channels):
    """Loads binary LFP data into double precision matrix [samples x channels]."""
    raw_data = np.fromfile(full_dat_file, dtype=np.int16)
    n_samples = len(raw_data) // num_channels
    return raw_data.reshape((n_samples, num_channels)).astype(np.float64)


def to_scalar_int(val):
    """Safely convert numpy/MATLAB 0D or 1D single elements to standard int."""
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


def safe_nanmean(arr, axis=0):
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.full(arr.shape[1] if arr.ndim > 1 else 1, np.nan)
    return np.nanmean(arr, axis=axis)


def safe_nanmax(arr, axis=0):
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.full(arr.shape[1] if arr.ndim > 1 else 1, np.nan)
    return np.nanmax(arr, axis=axis)


def safe_nanmin(arr, axis=0):
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.full(arr.shape[1] if arr.ndim > 1 else 1, np.nan)
    return np.nanmin(arr, axis=axis)


def processRippleSession(session_path, so_ch, pyr_ch, rad_ch, slm_ch):
    session_path = Path(session_path)
    analyset_path = session_path / "analyset"

    if not analyset_path.exists():
        raise FileNotFoundError(f"Analyset directory not found: {analyset_path}")

    try:
        rippleAnalysis = load_mat_variable(
            analyset_path / "rippleAnalysis.mat", "rippleAnalysis"
        )
        basicData = load_mat_variable(
            analyset_path / "basicData.mat", "basicData"
        )
        configChannels = load_mat_variable(
            analyset_path / "configChannels.mat", "configChannels"
        )

        windowSize = 0.05
        spontMarginSizes = np.array([0.02, 0.02])
        slack = 0.008

        fs = float(np.squeeze(basicData["fsLFP"]).item())
        windowHalfSize = int(np.ceil(windowSize / 2.0 * fs))
        spontMarginSizes = np.round(spontMarginSizes * fs).astype(int)
        slack = int(np.round(slack * fs))

        rip_chs_flat = np.atleast_1d(
            np.array(configChannels["ch"]["ripple"]).flatten()
        )

        filtLFP = {}
        ripEnv = {}

        N = 20
        fStop1 = 65.0
        fPass1 = 70.0
        fPass2 = 400.0
        fStop2 = 422.0
        wStop1 = 1.0
        wPass = 1.0
        wStop2 = 1.0

        nyq = fs / 2.0
        bRP = firls(
            N + 1,
            [0, fStop1 / nyq, fPass1 / nyq, fPass2 / nyq, fStop2 / nyq, 1.0],
            [0, 0, 1, 1, 0, 0],
            weight=[wStop1, wPass, wStop2],
        )

        frameLength = 2 * int(np.floor(0.0334 * fs / 2.0)) + 1
        datFile = "LFP_downsampled.dat"

        chMap = np.array(configChannels["chMap"]).flatten()
        numChannels = int(np.max(chMap))

        probeType = to_scalar_int(configChannels["probeType"])

        if probeType == 8:
            rec_dirs = [
                d
                for d in session_path.iterdir()
                if d.is_dir() and d.name.startswith("Record")
            ]
            if not rec_dirs:
                raise FileNotFoundError(f"No Record* folder found in {session_path}")

            recordPath = rec_dirs[0]
            contPath = recordPath / "experiment1" / "recording1" / "continuous"

            if not contPath.exists():
                raise FileNotFoundError(f"continuous folder not found in {contPath}")

            data_dirs = [
                d
                for d in contPath.iterdir()
                if d.is_dir()
                and (
                    d.name.startswith("Intan") or d.name.startswith("Acquisition")
                )
            ]
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

        shRipS = np.atleast_1d(np.array(rippleAnalysis["shRipS"]).flatten())
        unique_shanks = np.unique(shRipS)

        for shankNum in unique_shanks:
            s_idx = to_scalar_int(shankNum) - 1

            if 0 <= s_idx < len(rip_chs_flat):
                target_ch = to_scalar_int(rip_chs_flat[s_idx])
            else:
                target_ch = to_scalar_int(rip_chs_flat[0])

            ch_idx = max(0, min(target_ch - 1, dataLFP.shape[1] - 1))

            filtered = filtfilt(bRP, [1.0], dataLFP[:, ch_idx])
            filtLFP[shankNum] = filtered

            envelope = filtered * filtered
            sg = savgol_filter(envelope, frameLength, 4)

            m1 = int(round(0.0030 * fs))
            m2 = int(round(0.0065 * fs))
            smooth1 = (
                np.convolve(sg, np.ones(m1) / m1, mode="same")
                if m1 > 0
                else sg
            )
            smooth2 = (
                np.convolve(smooth1, np.ones(m2) / m2, mode="same")
                if m2 > 0
                else smooth1
            )

            ripEnv[shankNum] = smooth2

        # Re-align ripples to local trough minima
        iRipS = np.atleast_2d(np.array(rippleAnalysis["iRipS"]))
        if iRipS.shape[1] < 2 and iRipS.shape[0] >= 2:
            iRipS = iRipS.T

        numRipples = max(int(iRipS.shape[0]), len(shRipS))

        rawRipples = np.full((numRipples, 2 * windowHalfSize + 1), np.nan)
        filtRipples = np.full((numRipples, 2 * windowHalfSize + 1), np.nan)
        alignedRippleTimes = np.full((numRipples, 3), -1, dtype=int)

        RipStart = iRipS[:, 0] if iRipS.ndim > 1 else iRipS
        RipEnd = iRipS[:, 1] if iRipS.ndim > 1 else iRipS

        for rippleNum in range(min(len(RipStart), numRipples)):
            shankNum = shRipS[rippleNum] if rippleNum < len(shRipS) else shRipS[0]
            marginSizes = spontMarginSizes

            r_start = to_scalar_int(RipStart[rippleNum]) - 1
            r_end = to_scalar_int(RipEnd[rippleNum]) - 1

            rippleIdx = np.arange(
                r_start - marginSizes[0], r_end + marginSizes[1] + 1
            )
            env = ripEnv[shankNum][rippleIdx]
            iMax = np.argmax(env)
            iMiddle = rippleIdx[0] + iMax

            search_sig = -filtLFP[shankNum][iMiddle - slack : iMiddle + slack + 1]
            valleys, _ = find_peaks(search_sig)

            if len(valleys) > 0:
                iLowest = np.argmax(search_sig[valleys])
                iMinimum = iMiddle - slack + valleys[iLowest]
            else:
                iMinimum = iMiddle

            windowIdx = np.arange(
                iMinimum - windowHalfSize, iMinimum + windowHalfSize + 1
            )

            s_idx = to_scalar_int(shankNum) - 1
            if 0 <= s_idx < len(rip_chs_flat):
                target_ch = to_scalar_int(rip_chs_flat[s_idx])
            else:
                target_ch = to_scalar_int(rip_chs_flat[0])

            ch_rip = max(0, min(target_ch - 1, dataLFP.shape[1] - 1))
            rawRipples[rippleNum, :] = dataLFP[windowIdx, ch_rip]
            filtRipples[rippleNum, :] = filtLFP[shankNum][windowIdx]

            alignedRippleTimes[rippleNum, :] = [
                iMinimum - windowHalfSize,
                iMinimum,
                iMinimum + windowHalfSize,
            ]

        # CSD Calculation
        windowSize_csd = 0.1
        windowHalfSize_csd = int(np.ceil(windowSize_csd / 2.0 * fs))

        if probeType == 8:
            shankNums = np.array([1])
            shankChs = np.arange(1, 17)
        elif probeType == 1:
            shankNums = np.array([1])
            shankChs = np.arange(1, 16)

        previousShankChs = shankChs[:-2]
        centralShankChs = shankChs[1:-1]
        nextShankChs = shankChs[2:]

        CSD = {}
        CSDStds = {}

        shMap = np.array(configChannels["shMap"]).flatten()

        for shankNum in shankNums:
            firstShankCh = np.where(shMap == shankNum)[0][0] + 1

            previousChs = previousShankChs + firstShankCh - 1
            centralChs = centralShankChs + firstShankCh - 1
            nextChs = nextShankChs + firstShankCh - 1

            c_idx = centralChs - 1
            p_idx = previousChs - 1
            n_idx = nextChs - 1

            CSD[shankNum] = 2 * dataLFP[:, c_idx] - dataLFP[:, p_idx] - dataLFP[:, n_idx]
            CSDStds[shankNum] = np.std(CSD[shankNum], axis=0)

        CSDs = np.full(
            (windowHalfSize_csd * 2 + 1, len(centralShankChs), numRipples), np.nan
        )

        for rippleNum in range(numRipples):
            if rippleNum < len(shRipS):
                rippleShank = shRipS[rippleNum]
            else:
                continue

            if np.isin(rippleShank, shankNums):
                if rippleNum < alignedRippleTimes.shape[0]:
                    center_t = alignedRippleTimes[rippleNum, 1]
                else:
                    continue

                if center_t >= 0:
                    rippleIdx = np.arange(
                        center_t - windowHalfSize_csd, center_t + windowHalfSize_csd + 1
                    )
                    if rippleIdx[0] >= 0 and rippleIdx[-1] < CSD[rippleShank].shape[0]:
                        CSDs[:, :, rippleNum] = (
                            CSD[rippleShank][rippleIdx, :] / CSDStds[rippleShank]
                        )

        # Extract layer CSD values
        idx_so = np.where(centralShankChs == so_ch)[0]
        idx_pyr = np.where(centralShankChs == pyr_ch)[0]
        idx_rad = np.where(centralShankChs == rad_ch)[0]
        idx_slm = np.where(centralShankChs == slm_ch)[0]

        if len(idx_rad) == 0 or len(idx_slm) == 0:
            raise ValueError(
                f"Rad ({rad_ch}) or SLM ({slm_ch}) channel not found in centralShankChs ({centralShankChs})"
            )

        idx_so = idx_so[0]
        idx_pyr = idx_pyr[0]
        idx_rad = idx_rad[0]
        idx_slm = idx_slm[0]

        CSD_so = CSDs[:, idx_so, :]
        CSD_pyr = CSDs[:, idx_pyr, :]
        CSD_rad = CSDs[:, idx_rad, :]
        CSD_slm = CSDs[:, idx_slm, :]

        rippleCSDs = {
            "CSDs": CSDs,
            "previousShankChs": previousShankChs,
            "centralShankChs": centralShankChs,
            "nextShankChs": nextShankChs,
            "shankNums": shankNums,
            "t": np.arange(-windowHalfSize_csd, windowHalfSize_csd + 1) / fs,
            "windowHalfSize": windowHalfSize_csd,
            "so": {
                "values": CSD_so,
                "mean": safe_nanmean(CSD_so, axis=0),
                "max": safe_nanmax(CSD_so, axis=0),
                "min": safe_nanmin(CSD_so, axis=0),
            },
            "pyr": {
                "values": CSD_pyr,
                "mean": safe_nanmean(CSD_pyr, axis=0),
                "max": safe_nanmax(CSD_pyr, axis=0),
                "min": safe_nanmin(CSD_pyr, axis=0),
            },
            "rad": {
                "values": CSD_rad,
                "mean": safe_nanmean(CSD_rad, axis=0),
                "max": safe_nanmax(CSD_rad, axis=0),
                "min": safe_nanmin(CSD_rad, axis=0),
            },
            "slm": {
                "values": CSD_slm,
                "mean": safe_nanmean(CSD_slm, axis=0),
                "max": safe_nanmax(CSD_slm, axis=0),
                "min": safe_nanmin(CSD_slm, axis=0),
            },
        }

        output_dir = session_path / "analyset_python"
        output_dir.mkdir(parents=True, exist_ok=True)
        sio.savemat(output_dir / "rippleCSDs.mat", {"rippleCSDs": rippleCSDs})

        t_ms = rippleCSDs["t"] * 1000.0
        fig, ax = plt.subplots(figsize=(10, 8))

        for iShank, shankNum in enumerate(shankNums):
            shankChs = np.union1d(
                np.union1d(previousShankChs, centralShankChs), nextShankChs
            )
            first_sh_idx = np.where(shMap == shankNum)[0][0]
            chs = shankChs + first_sh_idx

            shankRippleIndices = np.where(shRipS == shankNum)[0]
            LFPs = np.full(
                (len(t_ms), len(chs), len(shankRippleIndices)), np.nan
            )

            for iShankRipple, shankRippleNum in enumerate(shankRippleIndices):
                if shankRippleNum < alignedRippleTimes.shape[0]:
                    center_t = alignedRippleTimes[shankRippleNum, 1]
                else:
                    continue

                if center_t >= 0:
                    rippleIdx = np.arange(
                        center_t - windowHalfSize_csd, center_t + windowHalfSize_csd + 1
                    )
                    if rippleIdx[0] >= 0 and rippleIdx[-1] < dataLFP.shape[0]:
                        LFPs[:, :, iShankRipple] = dataLFP[rippleIdx, :][:, chs - 1]

            valid_ripple_indices = np.where(shRipS == shankNum)[0]
            valid_ripple_indices = valid_ripple_indices[valid_ripple_indices < CSDs.shape[2]]

            if len(valid_ripple_indices) > 0:
                meanCSD = np.nanmean(CSDs[:, :, valid_ripple_indices], axis=2)
            else:
                meanCSD = np.zeros((len(t_ms), len(centralShankChs)))

            im = ax.imshow(
                meanCSD.T,
                aspect="auto",
                extent=[t_ms[0], t_ms[-1], len(centralShankChs), 1],
                cmap="jet",
                vmin=-8,
                vmax=5,
            )
            plt.colorbar(im, ax=ax)

            if LFPs.shape[2] > 0:
                meanLFP = safe_nanmean(LFPs, axis=2)
                max_val = np.nanmax(np.abs(meanLFP))
                
                if np.isnan(max_val) or max_val == 0:
                    normDivisor = 1.0
                else:
                    normDivisor = float(max_val) / 2.0

                for iCh in range(len(chs)):
                    ax.plot(t_ms, -meanLFP[:, iCh] / normDivisor + (iCh + 1), color="black")

            ax.set_xlim([t_ms[0], t_ms[-1]])
            ax.set_ylim([len(chs) + 2, -2])

            areas = ["so", "pyr", "rad", "slm"]
            area_chs = [so_ch, pyr_ch, rad_ch, slm_ch]

            pad = 10
            for name, areaCh in zip(areas, area_chs):
                ax.text(
                    t_ms[0] - pad,
                    areaCh,
                    name.upper(),
                    verticalalignment="center",
                )

            ax.set_title(f"shank {shankNum} ({len(shankRippleIndices)} ripples)")
            ax.set_xlabel("Time (ms)")

        sessionName = session_path.name
        plt.suptitle(sessionName)

        plt.tight_layout()
        plt.savefig(output_dir / f"{sessionName}MeanRippleCSD.png")
        plt.close()

    except Exception as e:
        print(f"\n--- DETAILED ERROR TRACE FOR SESSION {session_path.name} ---")
        traceback.print_exc()
        print("-----------------------------------------------------------\n")
        raise e