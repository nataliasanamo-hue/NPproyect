# Loads CSD data and extracts time and CSD array, averages and inverts sign. 

import h5py
import numpy as np
import scipy.io as sio


def compute_ripples_CSD(csd_file_path, window_pct=0.2):
    """Loads the precomputed 'rippleCSDs.mat' file, averages across ripples,

    applies sign inversion (-1), and extracts the mean vertical profile per
    channel.
    """
    # 1. Load the .mat file with support for MATLAB v5/v7 and v7.3 (HDF5)
    try:
        mat_csd = sio.loadmat(csd_file_path)
        ripple_struct = mat_csd["rippleCSDs"][0, 0]
        time_vector = ripple_struct["t"].flatten()  # Time array in ms
        csd_raw = ripple_struct["CSDs"]  # 3D Matrix (time, channels, ripples)
    except Exception:
        with h5py.File(csd_file_path, "r") as f:
            ripple_struct = f["rippleCSDs"]
            time_vector = np.array(ripple_struct["t"]).flatten()
            csd_raw = np.array(ripple_struct["CSDs"])
            # h5py reads inverted dimensions (ripples, channels, time)
            if csd_raw.shape[0] != len(time_vector) and csd_raw.shape[2] == len(
                time_vector
            ):
                csd_raw = np.transpose(csd_raw, (2, 1, 0))

    # 2. Average across ripples (axis 2), multiply by -1, and transpose (.T)
    # From (time, channels, ripples) -> (time, channels) -> [channels, time]
    csd_matrix = -1 * np.mean(csd_raw, axis=2).T

    # 3. Extract dynamic dimensions
    num_channels, num_samples = csd_matrix.shape

    # 4. Define the central ripple window
    center_idx = num_samples // 2
    window_half_width = int(num_samples * window_pct)
    csd_window = csd_matrix[
        :, center_idx - window_half_width : center_idx + window_half_width
    ]

    # 5. Compute average per channel for the vertical profile
    mean_csd_per_channel = np.mean(csd_window, axis=1)

    return {
        "csd_matrix": csd_matrix,
        "mean_csd_per_channel": mean_csd_per_channel,
        "time_vector": time_vector,
        "num_channels": num_channels,
        "num_samples": num_samples,
    }