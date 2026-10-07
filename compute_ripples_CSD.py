# compute_ripples_CSD.py
# Python translation of compute_ripples_CSD.m
# Loops over every animal / session in ANALISIS, looks up the layer channels
# (so, pyr, rad, slm) in the channels Excel and calls processRippleSession.
import os
from pathlib import Path

import numpy as np
import pandas as pd

from processRippleSession import processRippleSession

# MATLAB used E:\Violeta\ANALISIS and channels.xlsx. Change here if your paths differ.
baseDir = Path(r"D:\Violeta\ANALISIS")
CHANNELS_FILE = "channelsNatalia.xlsx"


def clean_id(v):
    """
    Text version of an Excel cell, like MATLAB's string().
    Excel numbers come in as floats (12 -> 12.0), so 12.0 must become "12", not "12.0".
    """
    if isinstance(v, (float, np.floating)) and not np.isnan(v) and float(v).is_integer():
        return str(int(v))
    return str(v).strip()


def norm_session(v):
    """Session key: ignores spaces vs '_' vs '-' differences."""
    return clean_id(v).replace(" ", "_").replace("-", "_")


def get_channel(row, col):
    """
    Channel number from the Excel row. A blank cell stays NaN (MATLAB passes the NaN on
    and processRippleSession decides what to do with it) instead of crashing here.
    """
    v = row[col]
    if pd.isna(v):
        return np.nan
    if float(v) != int(v):
        raise ValueError(f"Channel '{col}' is not a whole number: {v}")
    return int(v)


def main():
    # MATLAB: cd(baseDir)
    os.chdir(baseDir)

    # ---- Load Excel ONCE ----
    channelsTable = pd.read_excel(CHANNELS_FILE)
    channelsTable.columns = channelsTable.columns.str.strip().str.lower()
    channelsTable["animal_key"] = channelsTable["animal"].map(clean_id)
    channelsTable["session_key"] = channelsTable["session"].map(norm_session)

    # dir() in MATLAB returns folders sorted by name
    animalDirs = sorted(
        (d for d in baseDir.iterdir() if d.is_dir()), key=lambda d: d.name.lower()
    )

    failed = []
    n_ok = 0

    for animalDir in animalDirs:
        print(f"\nProcessing animal: {animalDir.name}")

        sessionDirs = sorted(
            (s for s in animalDir.iterdir() if s.is_dir()), key=lambda d: d.name.lower()
        )

        for sessionDir in sessionDirs:
            print(f"   Session: {sessionDir.name}")

            try:
                animalName = clean_id(animalDir.name)
                sessionName = norm_session(sessionDir.name)

                # ---- Look up channels in the Excel ----
                idxRow = channelsTable[
                    (channelsTable["animal_key"] == animalName)
                    & (channelsTable["session_key"] == sessionName)
                ]

                if idxRow.empty:
                    raise ValueError(
                        f"No matching row in {CHANNELS_FILE} for "
                        f"{animalDir.name} / {sessionDir.name}"
                    )
                if len(idxRow) > 1:
                    # MATLAB would get a vector of channels here and break later on
                    raise ValueError(
                        f"{len(idxRow)} rows in {CHANNELS_FILE} match "
                        f"{animalDir.name} / {sessionDir.name}; expected exactly one"
                    )

                row = idxRow.iloc[0]
                so_ch = get_channel(row, "so")
                pyr_ch = get_channel(row, "pyr")
                rad_ch = get_channel(row, "rad")
                slm_ch = get_channel(row, "slm")

                # ---- Call the function passing the channels ----
                processRippleSession(str(sessionDir), so_ch, pyr_ch, rad_ch, slm_ch)
                n_ok += 1

            except Exception as ME:
                # MATLAB: warning('Error in %s: %s', ...) and continue with the next session
                print(f"Warning: Error in {sessionDir.name}: {ME}")
                failed.append(f"{animalDir.name}/{sessionDir.name}")

    print(f"\nDone: {n_ok} sessions processed, {len(failed)} failed.")
    for f in failed:
        print(f"   FAILED: {f}")


if __name__ == "__main__":
    main()