# main runner:loops through all mouse/session, compiles all the metrics and 
# generates group-level comparison figures.

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import seaborn as sns
import h5py

# ==========================================
# ROBUST MATLAB v7.3 / HDF5 LOADER FUNCTION
# ==========================================
def load_mat_robust(file_path):
    """Loads a MATLAB .mat file supporting both v5/v7 (SciPy) and v7.3 (h5py HDF5)."""
    file_path = Path(file_path)
    try:
        # Try SciPy first (MATLAB v5 / v7.0)
        return sio.loadmat(file_path, simplify_cells=True)
    except Exception:
        # Fallback to h5py for MATLAB v7.3 (HDF5 format)
        data_dict = {}
        with h5py.File(file_path, "r") as f:
            for k in f.keys():
                if not k.startswith("#"):  # Ignore metadata like #refs#
                    val = np.array(f[k])
                    # Squeeze single dimensions if necessary
                    if val.size == 1:
                        val = val.item()
                    data_dict[k] = val
        return data_dict

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
working_dir = Path(r"D:\Violeta\ANALISIS")  # Adjust drive letter if needed

results_dir = working_dir / "RESULTSNatalia"
results_dir.mkdir(parents=True, exist_ok=True)

conditions = ["WT", "NPC", "EFV"]
group_colors = {
    "WT": np.array([76, 153, 102]) / 256.0,
    "NPC": np.array([70, 120, 180]) / 256.0,
    "EFV": np.array([150, 110, 170]) / 256.0,
}

psFreqs = np.linspace(0, 80, 81)

bands = {
    "Delta": [0, 3],
    "Theta": [4, 12],
    "Alpha": [9, 16],
    "SlowGamma": [20, 40],
    "MidGamma": [40, 60],
    "FastGamma": [60, 80],
    "Ripple": [150, 250],
}

# Ignore non-animal folders when scanning
folders_to_ignore = {
    ".",
    "..",
    "RESULTS",
    "RESULTSNatalia",
    "ProcesamientoNatalia",
}

# ==========================================
# 2. SESSION DISCOVERY
# ==========================================
sessions = []

for animal_folder in [
    f
    for f in working_dir.iterdir()
    if f.is_dir() and f.name not in folders_to_ignore
]:
    animal_name = animal_folder.name

    matched_cond = None
    for cond in conditions:
        if cond.lower() in animal_name.lower():
            matched_cond = cond
            break

    if matched_cond is not None:
        for sess_folder in [
            sf
            for sf in animal_folder.iterdir()
            if sf.is_dir() and sf.name not in folders_to_ignore
        ]:
            sessions.append({
                "session": sess_folder,
                "animal": animal_name,
                "condition": matched_cond,
            })

print(
    f"Identified {len(sessions)} recording sessions across WT, NPC, and EFV groups."
)
print(f"Group plots will be saved to: {results_dir}")

# ==========================================
# HELPER FUNCTION: UNWRAP NESTED MAT ARRAYS
# ==========================================
def unwrap_mat_val(val, max_depth=5):
    """Safely unwrap deeply nested MATLAB arrays/structs without getting stuck in infinite loops."""
    if val is None:
        return None
    
    val_arr = np.array(val)
    depth = 0
    
    while val_arr.dtype == object and val_arr.size > 0 and depth < max_depth:
        # Handle 0-dimensional object arrays (scalars wrapped in array)
        if val_arr.ndim == 0:
            val_arr = np.array(val_arr.item())
        else:
            val_arr = np.array(val_arr.flat[0])
        depth += 1

    # Extract scalar if array has size 1
    if val_arr.size == 1 and val_arr.dtype != object:
        val_arr = val_arr.item()

    return val_arr

# ==========================================
# SECTION 3: ROBUST DATA LOADING & EXTRACTION
# ==========================================
analysisResults = {cond: [] for cond in conditions}

for currentCond in conditions:
    condSess = [s for s in sessions if s["condition"] == currentCond]
    print(f"Processing condition: {currentCond}...")

    for sess in condSess:
        sessPath = sess["session"]
        
        # 1. Target output saved by save_power_spectrum_NPC_project.py
        analysetPythonFolder = sessPath / "analyset_python"
        ps_file = analysetPythonFolder / "psProfile_python.mat"

        # Fallback to legacy analyset folder
        if not ps_file.exists():
            ps_file = sessPath / "analyset" / "psProfile.mat"

        if ps_file.exists():
            try:
                mat_contents = load_mat_robust(ps_file)
                
                # Unwrap top-level dictionary / struct
                ps_struct = mat_contents.get("psProfile", mat_contents)
                
                # Drills past 1x1 SciPy object wrappers
                while isinstance(ps_struct, np.ndarray) and ps_struct.dtype == object and ps_struct.size > 0:
                    ps_struct = ps_struct.flat[0]

                ps_theta = None
                ps_lia = None

                # Extract pSTheta and pSLIA arrays
                if isinstance(ps_struct, dict):
                    ps_theta = ps_struct.get("pSTheta", ps_struct.get("pS", None))
                    ps_lia = ps_struct.get("pSLIA", ps_struct.get("pS", None))
                elif hasattr(ps_struct, "pSTheta"):
                    ps_theta = getattr(ps_struct, "pSTheta")
                    ps_lia = getattr(ps_struct, "pSLIA", ps_theta)

                ps_theta = unwrap_mat_val(ps_theta)
                ps_lia = unwrap_mat_val(ps_lia)

                # 2. Extract channel configurations if present
                config_file = sessPath / "analyset" / "configChannels.mat"
                ch_struct = None
                if config_file.exists():
                    cfg = load_mat_robust(config_file)
                    ch_raw = cfg.get("configChannels", cfg) if isinstance(cfg, dict) else cfg
                    ch_unwrapped = unwrap_mat_val(ch_raw)
                    if isinstance(ch_unwrapped, dict):
                        ch_struct = ch_unwrapped.get("ch", ch_unwrapped)

                if ps_theta is not None and np.size(ps_theta) > 0:
                    analysisResults[currentCond].append({
                        "pSTheta": ps_theta,
                        "pSLIA": ps_lia,
                        "animal": sess["animal"],
                        "channelInfo": ch_struct,
                    })
            except Exception as e:
                print(f"[WARNING] Skipping {sessPath.name}: {e}")

for cond in conditions:
    print(f"  -> Successfully loaded {len(analysisResults[cond])} valid sessions for {cond}.")

print("[OK] Data extraction complete!")

# ==========================================
# SECTION 4: POWER SPECTRUM OVERVIEW PLOT (0-80 Hz)
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle(
    "Power Spectrum Group Comparison (Mean ± SEM)",
    fontsize=14,
    fontweight="bold",
)

layers = ["pyr", "theta"]
layer_names = ["PYR", "SLM"]
states = ["pSTheta", "pSLIA"]
state_names = ["Theta", "LIA"]

for iEst in range(2):
    for iCap in range(2):
        ax = axes[iEst, iCap]

        for cond in conditions:
            res_list = analysisResults[cond]
            currentData = []

            for res in res_list:
                ch_info = res.get("channelInfo", None)
                ps_data = res.get(states[iEst], None)

                if ps_data is not None:
                    ps_arr = np.array(ps_data)
                    ch_idx = 0  # Default channel index fallback

                    # Extract target layer channel safely
                    if ch_info is not None:
                        try:
                            ch_val = None
                            if isinstance(ch_info, dict):
                                ch_val = ch_info.get(layers[iCap], None)
                            elif hasattr(ch_info, layers[iCap]):
                                ch_val = getattr(ch_info, layers[iCap])

                            if ch_val is not None:
                                unwrapped_ch = unwrap_mat_val(ch_val)
                                raw_ch = int(
                                    unwrapped_ch.item()
                                    if unwrapped_ch.size == 1
                                    else unwrapped_ch[0]
                                )
                                # Convert 1-based indexing (MATLAB) to 0-based indexing (Python)
                                ch_idx = raw_ch - 1 if raw_ch > 0 else raw_ch
                        except Exception:
                            ch_idx = 0

                    # Extract frequency trace based on array dimensions
                    try:
                        if ps_arr.ndim == 1:
                            currentData.append(ps_arr)
                        elif ps_arr.ndim == 2:
                            # Detect whether shape is (Channels x Frequencies) or (Frecuencies x Channels)
                            if ps_arr.shape[0] < ps_arr.shape[1]:
                                safe_idx = min(ch_idx, ps_arr.shape[0] - 1)
                                currentData.append(ps_arr[safe_idx, :])
                            else:
                                safe_idx = min(ch_idx, ps_arr.shape[1] - 1)
                                currentData.append(ps_arr[:, safe_idx])
                    except Exception:
                        continue

            # Plot group Mean and SEM (Standard Error of the Mean)
            if len(currentData) > 0:
                currentData = np.array(currentData)
                mean_signal = np.nanmean(currentData, axis=0)

                n_samples = currentData.shape[0]
                sem_signal = (
                    np.nanstd(currentData, axis=0) / np.sqrt(n_samples)
                    if n_samples > 1
                    else np.zeros_like(mean_signal)
                )

                line_style = "-" if iEst == 0 else "--"
                color = group_colors[cond]

                freq_axis = psFreqs[: len(mean_signal)]
                ax.plot(
                    freq_axis,
                    mean_signal,
                    label=f"{cond} (n={n_samples})",
                    color=color,
                    linestyle=line_style,
                    linewidth=1.8,
                )
                ax.fill_between(
                    freq_axis,
                    mean_signal - sem_signal,
                    mean_signal + sem_signal,
                    color=color,
                    alpha=0.2,
                )

        ax.set_xlim([0, 80])
        ax.set_yscale("log")
        ax.grid(True, which="both", ls=":", alpha=0.6)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power (dB)")
        ax.set_title(f"{layer_names[iCap]} Channel - {state_names[iEst]} State")

        if iEst == 0 and iCap == 0:
            ax.legend(loc="upper right")

plt.tight_layout()
output_fig_path = results_dir / "PowerSpectrum_Overview.png"
plt.savefig(output_fig_path, dpi=300)
plt.show()
print(f"[OK] Figure saved to: {output_fig_path}")