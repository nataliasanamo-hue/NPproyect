# Takes time-series LFP data and computes the power spectrum using Welch's method, 
# then saves the results in a MATLAB-compatible .mat file.

from pathlib import Path
import numpy as np
import scipy.io as sio
from scipy.signal import welch

# ==========================================
# 1. FIXED POWER SPECTRUM COMPUTATION
# ==========================================
def compute_power_spectrum_session(lfp_data, fs=2500, nperseg=1024, max_freq=80):
    """Compute Welch Power Spectral Density per channel up to max_freq Hz."""
    if lfp_data is None or lfp_data.size == 0:
        return np.linspace(0, max_freq, 81), None

    if lfp_data.ndim == 1:
        lfp_data = lfp_data[:, np.newaxis]

    # Ensure shape is (channels, time)
    if lfp_data.shape[0] > lfp_data.shape[1]:
        lfp_data = lfp_data.T

    n_channels, _ = lfp_data.shape
    ps_list = []

    for ch in range(n_channels):
        freqs, psd = welch(lfp_data[ch, :], fs=fs, nperseg=nperseg)
        freq_mask = freqs <= max_freq
        ps_list.append(psd[freq_mask])
        valid_freqs = freqs[freq_mask]

    ps_matrix = np.array(ps_list)  # (channels x frequencies)
    return valid_freqs, ps_matrix

# ==========================================
# 2. STRUCT EXPORT FUNCTION (KEEP THIS!)
# ==========================================
def save_power_spectrum_struct(session_path, lfp_theta=None, lfp_lia=None, fs=2500, out_file=None):
    """Calculate power spectrum structures and save output for downstream plotting."""
    session_path = Path(session_path)
    
    if out_file is None:
        analyset_dir = session_path / "analyset"
        analyset_dir.mkdir(parents=True, exist_ok=True)
        out_file = analyset_dir / "psProfile.mat"
    else:
        out_file = Path(out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)

    freqs = np.linspace(0, 80, 81)
    ps_theta = None
    ps_lia = None

    if lfp_theta is not None:
        freqs, ps_theta = compute_power_spectrum_session(lfp_theta, fs=fs, max_freq=80)

    if lfp_lia is not None:
        freqs, ps_lia = compute_power_spectrum_session(lfp_lia, fs=fs, max_freq=80)

    # Structure matching expected MATLAB / Python psProfile fields
    ps_profile_dict = {
        "psProfile": {
            "pS": ps_theta if ps_theta is not None else ps_lia,
            "pSTheta": ps_theta if ps_theta is not None else np.array([]),
            "pSLIA": ps_lia if ps_lia is not None else np.array([]),
            "f": freqs,
        }
    }

    sio.savemat(out_file, ps_profile_dict)
    return out_file