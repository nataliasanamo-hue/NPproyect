# Load LFP data and rippleAnalysis for one session, extract and process CSD data and render the plots.
# Extract summary metrics and return them in a dictionary for that sesion.
# Extract layer channel from channels excel

from pathlib import Path
import numpy as np
import pandas as pd
from compute_ripples_CSD import compute_ripples_CSD
from Plot_CSD_results import Plot_CSD_results


def processRippleSession(
    session_path, channels_table, show_plot=True, csd_window_pct=0.2
):
    """Direct translation of processRippleSession.m.

    Matches animal and session names against the channels.xlsx table to retrieve
    layer channels (so, pyr, rad, slm), runs compute_ripples_CSD, and plots the
    results.
    """
    session_path = Path(session_path)

    # 1. Extract animal and session names to match channels.xlsx
    # Path structure: .../ANALISIS/animalName/sessionName
    animal_name = session_path.parent.name
    session_name = session_path.name

    # 2. Look up matching row in channels.xlsx
    matched_rows = channels_table[
        (channels_table["animal"].astype(str) == str(animal_name))
        & (channels_table["session"].astype(str) == str(session_name))
    ]

    if matched_rows.empty:
        raise ValueError(
            f"No matching row in channels.xlsx for Animal: {animal_name}, Session: {session_name}"
        )

    # Extract 1-based channel indices from Excel
    row = matched_rows.iloc[0]
    so_ch = int(row["so"])
    pyr_ch = int(row["pyr"])
    rad_ch = int(row["rad"])
    slm_ch = int(row["slm"])

    # 3. Locate rippleCSDs.mat
    csd_file = session_path / "rippleCSDs.mat"
    if not csd_file.exists():
        csd_file = session_path / "analyset" / "rippleCSDs.mat"

    if not csd_file.exists():
        raise FileNotFoundError(
            f"rippleCSDs.mat not found in {session_path} or {session_path / 'analyset'}"
        )

    # 4. Compute CSD metrics
    csd_results = compute_ripples_CSD(csd_file, window_pct=csd_window_pct)

    # 5. Extract CSD values for each layer (converting 1-based MATLAB channels to 0-based Python indices)
    mean_csd = csd_results["mean_csd_per_channel"]

    layer_values = {
        "so_val": (
            float(mean_csd[so_ch - 1])
            if 0 <= so_ch - 1 < len(mean_csd)
            else np.nan
        ),
        "pyr_val": (
            float(mean_csd[pyr_ch - 1])
            if 0 <= pyr_ch - 1 < len(mean_csd)
            else np.nan
        ),
        "rad_val": (
            float(mean_csd[rad_ch - 1])
            if 0 <= rad_ch - 1 < len(mean_csd)
            else np.nan
        ),
        "slm_val": (
            float(mean_csd[slm_ch - 1])
            if 0 <= slm_ch - 1 < len(mean_csd)
            else np.nan
        ),
    }

    # 6. Plot CSD results if enabled
    if show_plot:
        session_title = f"{animal_name} - {session_name} | Pyr Ch: {pyr_ch}"
        Plot_CSD_results(csd_results, session_title=session_title)

    # 7. Return session summary dictionary
    return {
        "animal": animal_name,
        "session": session_name,
        "so_ch": so_ch,
        "pyr_ch": pyr_ch,
        "rad_ch": rad_ch,
        "slm_ch": slm_ch,
        "so_csd": layer_values["so_val"],
        "pyr_csd": layer_values["pyr_val"],
        "rad_csd": layer_values["rad_val"],
        "slm_csd": layer_values["slm_val"],
        "num_channels": csd_results["num_channels"],
    }