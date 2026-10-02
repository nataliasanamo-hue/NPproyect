# Master project script: batch processes all animal and session folders,
# calls save_power_spectrum_struct, and aggregates group power spectra.

# save_power_spectrum_NPC_project.py
from pathlib import Path
import numpy as np
import scipy.io as sio
import h5py
from save_power_spectrum_struct import save_power_spectrum_struct

# Setup paths & parameters
working_dir = Path(r"D:\Violeta\ANALISIS")
conditions = ["WT", "NPC", "EFV"]
folders_to_ignore = {".", "..", "RESULTS", "RESULTSNatalia", "ProcesamientoNatalia"}
fs = 2500  # Sampling rate

# Robust loader function defined locally to prevent ImportError
def load_mat_robust(file_path):
    """Load MATLAB files cleanly for both v5/v7 and v7.3 HDF5 formats."""
    file_path = Path(file_path)
    try:
        return sio.loadmat(file_path, simplify_cells=True)
    except Exception:
        data_dict = {}
        with h5py.File(file_path, "r") as f:
            for k in f.keys():
                if not k.startswith("#"):
                    val = np.array(f[k])
                    if val.size == 1:
                        val = val.item()
                    data_dict[k] = val
        return data_dict

# Helper to slice LFP based on time intervals from thetaLIAAnalysis.mat
def extract_state_lfp(lfp_matrix, time_intervals, fs=2500):
    if time_intervals is None or np.size(time_intervals) == 0:
        return None
    
    time_intervals = np.array(time_intervals)
    if time_intervals.ndim == 1:
        time_intervals = time_intervals.reshape(-1, 2)
        
    extracted_chunks = []
    for interval in time_intervals:
        start_idx = int(interval[0] * fs) if interval[0] < lfp_matrix.shape[0] / fs else int(interval[0])
        end_idx = int(interval[1] * fs) if interval[1] < lfp_matrix.shape[0] / fs else int(interval[1])
        
        start_idx = max(0, start_idx)
        end_idx = min(lfp_matrix.shape[0], end_idx)
        
        if end_idx > start_idx:
            extracted_chunks.append(lfp_matrix[start_idx:end_idx, :])
            
    if len(extracted_chunks) > 0:
        return np.vstack(extracted_chunks)
    return None

# Discover sessions
sessions = []
for animal_folder in [f for f in working_dir.iterdir() if f.is_dir() and f.name not in folders_to_ignore]:
    matched_cond = next((c for c in conditions if c.lower() in animal_folder.name.lower()), None)
    if matched_cond:
        for sf in [s for s in animal_folder.iterdir() if s.is_dir() and s.name not in folders_to_ignore]:
            sessions.append({"session": sf, "animal": animal_folder.name, "condition": matched_cond})

print(f"[INFO] Discovered {len(sessions)} sessions across WT, NPC, and EFV.")

# Process sessions with State Segmentation
processed_count = 0

for sess in sessions:
    sess_path = sess["session"]
    lfp_file = sess_path / "LFP_downsampled.dat"
    theta_lia_file = sess_path / "analyset" / "thetaLIAAnalysis.mat"
    
    # Save output in analyset_python folder
    out_dir = sess_path / "analyset_python"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "psProfile_python.mat"
    
    if lfp_file.exists():
        try:
            raw_lfp = np.fromfile(lfp_file, dtype=np.int16)
            num_channels = 16
            
            if raw_lfp.size % num_channels == 0:
                lfp_matrix = raw_lfp.reshape((-1, num_channels))
            else:
                remainder = raw_lfp.size % num_channels
                lfp_matrix = raw_lfp[:-remainder].reshape((-1, num_channels))

            lfp_theta = None
            lfp_lia = None

            # Load state time windows from MATLAB thetaLIAAnalysis.mat
            if theta_lia_file.exists():
                t_data = load_mat_robust(theta_lia_file)
                times_theta = t_data.get("timesTheta", t_data.get("times_theta", None))
                times_lia = t_data.get("timesLIA", t_data.get("times_lia", None))

                lfp_theta = extract_state_lfp(lfp_matrix, times_theta, fs=fs)
                lfp_lia = extract_state_lfp(lfp_matrix, times_lia, fs=fs)

            # Fallback if state timestamps are missing
            if lfp_theta is None: lfp_theta = lfp_matrix
            if lfp_lia is None: lfp_lia = lfp_matrix

            save_power_spectrum_struct(
                session_path=sess_path,
                lfp_theta=lfp_theta,
                lfp_lia=lfp_lia,
                fs=fs,
                out_file=out_file
            )
            processed_count += 1
            print(f"[OK] Segmented PSD saved to {out_file}")

        except Exception as e:
            print(f"[WARNING] Skipping {sess_path.name}: {e}")

print(f"\n[OK] Successfully processed {processed_count} sessions into analyset_python!")