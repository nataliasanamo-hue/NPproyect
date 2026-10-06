# compute_ripples_CSD.py
import os
from pathlib import Path
import numpy as np
import pandas as pd
from processRippleSession import processRippleSession

baseDir = Path(r"D:\Violeta\ANALISIS")
os.chdir(baseDir)

# ---- Load Excel ONCE ----
channelsTable = pd.read_excel("channelsNatalia.xlsx")

# Standardize column names
channelsTable.columns = channelsTable.columns.str.strip().str.lower()

# Clean strings
channelsTable["animal"] = channelsTable["animal"].astype(str).str.strip()
channelsTable["session"] = (
    channelsTable["session"].astype(str).str.strip().str.replace(" ", "_")
)

animalDirs = [d for d in baseDir.iterdir() if d.is_dir()]

for animalDir in animalDirs:
    print(f"\nProcessing animal: {animalDir.name}")

    sessionDirs = [s for s in animalDir.iterdir() if s.is_dir()]

    for sessionDir in sessionDirs:
        print(f"   Session: {sessionDir.name}")

        try:
            animalName = str(animalDir.name).strip()
            sessionName = str(sessionDir.name).strip().replace(" ", "_")

            # Match animal & session ignoring hyphens/underscores
            idxRow = channelsTable[
                (channelsTable["animal"] == animalName)
                & (
                    channelsTable["session"].str.replace("-", "_")
                    == sessionName.replace("-", "_")
                )
            ]

            if idxRow.empty:
                raise ValueError(
                    f"No matching row in channelsNatalia.xlsx for {animalName} / {sessionName}"
                )

            row = idxRow.iloc[0]

            # Safely extract scalar integers using np.squeeze().item()
            so_ch = int(np.squeeze(row["so"]).item())
            pyr_ch = int(np.squeeze(row["pyr"]).item())
            rad_ch = int(np.squeeze(row["rad"]).item())
            slm_ch = int(np.squeeze(row["slm"]).item())

            # Call processing function
            processRippleSession(
                str(sessionDir), so_ch, pyr_ch, rad_ch, slm_ch
            )

        except Exception as ME:
            print(f"Warning: Error in {sessionDir.name}: {ME}")