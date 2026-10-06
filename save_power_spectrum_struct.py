# Takes time-series LFP data and computes the power spectrum using Welch's method, 
# then saves the results in a MATLAB-compatible .mat file.

from pathlib import Path
import numpy as np
import scipy.io as sio
from scipy.signal import welch

# ==========================================
# 1. FIXED POWER SPECTRUM COMPUTATION
# ==========================================
def compute_power_spectrum_session(lfp_data, fs=2500, max_freq=80):
    """Compute Welch PSD per channel matching MATLAB pwelch (1 Hz bins)."""
    if lfp_data is None or lfp_data.size == 0:
        return np.linspace(0, max_freq, max_freq + 1), None

    if lfp_data.ndim == 1:
        lfp_data = lfp_data[np.newaxis, :]  # Shape: (1, time)

    # Ensure shape is (channels, time)
    if lfp_data.shape[0] > lfp_data.shape[1]:
        lfp_data = lfp_data.T

    n_channels = lfp_data.shape[0]
    target_freqs = np.linspace(0, max_freq, max_freq + 1)  # 81 points (0 to 80 Hz)
    
    ps_list = []
    for ch in range(n_channels):
        # nperseg=fs guarantees 1 Hz bin resolution
        freqs, psd = welch(lfp_data[ch, :], fs=fs, nperseg=fs, noverlap=fs // 2)
        # Interpolate cleanly to 0-80 Hz integer bins
        psd_80 = np.interp(target_freqs, freqs, psd)
        ps_list.append(psd_80)

    return target_freqs, np.array(ps_list)  # Shape: (channels, 81)

# ==========================================
# 2. STRUCT EXPORT FUNCTION (KEEP THIS!)
# ==========================================
def save_power_spectrum_struct(session_path, lfp_theta=None, lfp_lia=None, fs=2500, out_file=None):
    """Calculate power spectrum structures and save output to analyset_python/."""
    session_path = Path(session_path)
    
    if out_file is None:
        analyset_dir = session_path / "analyset_python"
        analyset_dir.mkdir(parents=True, exist_ok=True)
        out_file = analyset_dir / "psProfile_python.mat"
    else:
        out_file = Path(out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)

    target_freqs = np.linspace(0, 80, 81)
    ps_theta = None
    ps_lia = None

    if lfp_theta is not None and lfp_theta.size > 0:
        _, ps_theta = compute_power_spectrum_session(lfp_theta, fs=fs, max_freq=80)

    if lfp_lia is not None and lfp_lia.size > 0:
        _, ps_lia = compute_power_spectrum_session(lfp_lia, fs=fs, max_freq=80)

    # Do not cross-contaminate theta and lia if one is missing
    ps_profile_dict = {
        "psProfile": {
            "pS": ps_theta if ps_theta is not None else ps_lia,
            "pSTheta": ps_theta if ps_theta is not None else np.array([]),
            "pSLIA": ps_lia if ps_lia is not None else np.array([]),
            "f": target_freqs,
        }
    }

    sio.savemat(out_file, ps_profile_dict)
    return out_file