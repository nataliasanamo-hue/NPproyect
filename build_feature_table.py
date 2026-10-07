# build_feature_table.py
# Builds the big Excel (one row per session) by JOINING small files. It never opens the
# 600 MB psProfile.mat files, so it runs in seconds and can be repeated as often as needed.
#
# REFERENCE CHANNELS: channels.xlsx (original, from the coworkers) is the reference configuration (ch_so, ch_pyr, ch_rad, ch_slm).
# The channels of the coworkers' configChannels.mat are kept as config_* only for comparison.
#
# READS (nothing is modified):
#   <baseDir>/channelsNatalia.xlsx                           channel of each layer (REFERENCE)
#   <session>/analyset/configChannels.mat                    coworkers' channels (tiny file)
#   <session>/ProcesamientoNatalia/CSD_central_means.csv     made by Plot_CSD_results.py
#   <session>/ProcesamientoNatalia/power_features.csv        made by save_power_spectrum_NPC_project.py
# WRITES:
#   <baseDir>/RESULTSNatalia/features_final.xlsx  (+ .csv)
#     sheet "features"           one row per session
#     sheet "channel_mismatches" every layer where the Excel and another source disagree
from pathlib import Path

import re

import numpy as np
import pandas as pd
import scipy.io as sio

# ================== SETTINGS ==================
baseDir = Path(r"D:\Violeta\ANALISIS")
SESSION_OUT_FOLDER = "ProcesamientoNatalia"
TOTAL_OUT_FOLDER = "RESULTSNatalia"
CHANNELS_FILE = "channelsNatalia.xlsx"  # same channels as channels.xlsx, only the session names of 14 sessions fixed

# Which sessions stay in the table:
#   "any" = remove only sessions with NO data at all (no power and no CSD)
#   "all" = keep only sessions that have BOTH power and CSD (the most complete table for ML)
KEEP_RULE = "any"
# True = also remove sessions that have no reference channels (not found in the channels Excel)
REQUIRE_CHANNELS = True

CONDITIONS = ["WT", "NPC", "EFV"]
CH_FIELDS = ["so", "pyr", "rad", "slm", "ripple", "theta", "DG"]  # configChannels.ch
LAYERS = ["so", "pyr", "rad", "slm"]  # layers defined in the Excel


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def detect_condition(animal_name):
    """Same rule as save_power_spectrum_NPC_project.m: contains(name, c, 'IgnoreCase', true)."""
    for c in CONDITIONS:
        if c.lower() in animal_name.lower():
            return c
    return None


def clean_id(v):
    """12.0 -> '12' (Excel numbers arrive as floats), text is stripped."""
    if isinstance(v, (float, np.floating)) and not np.isnan(v) and float(v).is_integer():
        return str(int(v))
    return str(v).strip()


def norm_session(v):
    """Ignores spaces / '_' / '-' differences (same rule as compute_ripples_CSD.py)."""
    # lower case, and only letters/digits are compared: 'NPC_260114-171627' == 'npc 260114_171627'
    return re.sub(r"[^0-9a-z]", "", clean_id(v).lower())


def read_ch_fields(path):
    """configChannels.ch.<field> for every field in CH_FIELDS (NaN when missing)."""
    out = {f: np.nan for f in CH_FIELDS}
    try:  # v7.3 (HDF5) file, like your coworkers' ones
        import h5py

        with h5py.File(path, "r") as f:
            grp = f["configChannels/ch"]
            for name in CH_FIELDS:
                if name in grp:
                    try:
                        out[name] = float(np.squeeze(grp[name][()]))
                    except (TypeError, ValueError):
                        pass  # empty or not numeric
        return out
    except Exception:
        pass
    # older (v5) file
    ch = sio.loadmat(path, simplify_cells=True)["configChannels"]["ch"]
    for name in CH_FIELDS:
        try:
            out[name] = float(np.squeeze(ch[name]))
        except (KeyError, TypeError, ValueError):
            pass
    return out


def load_channels_excel():
    f = baseDir / CHANNELS_FILE
    if not f.exists():
        print(f"WARNING: {f} not found: the reference channel columns will be empty")
        return None
    df = pd.read_excel(f)
    df.columns = df.columns.str.strip().str.lower()
    df["animal_key"] = df["animal"].map(lambda v: norm_session(v))
    df["session_key"] = df["session"].map(norm_session)
    return df


def excel_channels(df, animal, session):
    out = {f"ch_{k}": np.nan for k in LAYERS}
    if df is None:
        return out
    rows = df[(df["animal_key"] == norm_session(animal)) & (df["session_key"] == norm_session(session))]
    if len(rows) != 1:  # none, or ambiguous
        return out
    r = rows.iloc[0]
    for k in LAYERS:
        if k in r and not pd.isna(r[k]):
            out[f"ch_{k}"] = float(r[k])
    return out


def csd_summary(session_out):
    """Per-session summary of the per-ripple central CSD means saved by Plot_CSD_results.py."""
    f = session_out / "CSD_central_means.csv"
    if not f.exists():
        return {}
    df = pd.read_csv(f)
    out = {"n_csd_ripples": len(df)}
    for k in LAYERS:
        v = pd.to_numeric(df[k], errors="coerce").dropna() if k in df else pd.Series(dtype=float)
        K = k.upper()
        out[f"CSD_{K}_mean"] = float(v.mean()) if len(v) else np.nan
        out[f"CSD_{K}_median"] = float(v.median()) if len(v) else np.nan
        out[f"CSD_{K}_std"] = float(v.std()) if len(v) > 1 else np.nan
        out[f"CSD_{K}_n"] = len(v)
    return out


def power_row(session_out):
    f = session_out / "power_features.csv"
    if not f.exists():
        return {}
    df = pd.read_csv(f)
    if len(df) != 1:
        return {}
    d = df.iloc[0].to_dict()
    for k in ("animal", "session", "group"):
        d.pop(k, None)
    return d


def differs(a, b):
    """True/False when both values exist, None when one is missing."""
    if pd.isna(a) or pd.isna(b):
        return None
    return a != b


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    chan_xlsx = load_channels_excel()
    rows, mismatches = [], []

    animals = sorted((d for d in baseDir.iterdir() if d.is_dir()), key=lambda d: d.name.lower())
    for animalDir in animals:
        cond = detect_condition(animalDir.name)
        if cond is None:
            continue
        for sess in sorted((d for d in animalDir.iterdir() if d.is_dir()), key=lambda d: d.name.lower()):
            if not (sess / "analyset").is_dir():
                continue
            out = sess / SESSION_OUT_FOLDER
            who = dict(animal=animalDir.name, session=sess.name)

            row = {"animal": animalDir.name, "session": sess.name, "group": cond}

            # --- REFERENCE channels (Excel) ---
            row.update(excel_channels(chan_xlsx, animalDir.name, sess.name))

            # --- coworkers' channels, for comparison ---
            try:
                cfg = read_ch_fields(sess / "analyset" / "configChannels.mat")
            except Exception as e:
                print(f"Warning: configChannels of {animalDir.name}/{sess.name}: {e}")
                cfg = {f: np.nan for f in CH_FIELDS}
            row.update({f"config_{k}": v for k, v in cfg.items()})

            pw = power_row(out)
            used_pyr = pw.pop("pyr_channel", np.nan)  # channels the power script really used
            used_slm = pw.pop("slm_channel", np.nan)  # (ch.pyr and ch.theta of configChannels)
            row["power_pyr_channel_used"] = used_pyr
            row["power_slm_channel_used"] = used_slm

            # --- checks against the Excel ---
            bad_pyr = differs(used_pyr, row["ch_pyr"])
            bad_slm = differs(used_slm, row["ch_slm"])
            if bad_pyr:
                mismatches.append({**who, "layer": "PYR", "excel": row["ch_pyr"], "other": used_pyr,
                                   "other_is": "channel used for the power spectrum"})
            if bad_slm:
                mismatches.append({**who, "layer": "SLM", "excel": row["ch_slm"], "other": used_slm,
                                   "other_is": "channel used for the power spectrum (ch.theta)"})
            checks = [c for c in (bad_pyr, bad_slm) if c is not None]
            row["power_channels_match_excel"] = np.nan if not checks else (not any(checks))

            cfg_checks = []
            for k in LAYERS:
                d = differs(cfg[k], row[f"ch_{k}"])
                if d is not None:
                    cfg_checks.append(d)
                    if d:
                        mismatches.append({**who, "layer": k.upper(), "excel": row[f"ch_{k}"], "other": cfg[k],
                                           "other_is": f"configChannels ch.{k}"})
            row["config_matches_excel"] = np.nan if not cfg_checks else (not any(cfg_checks))
            row["config_theta_equals_config_slm"] = (
                np.nan if pd.isna(cfg["theta"]) or pd.isna(cfg["slm"]) else cfg["theta"] == cfg["slm"]
            )

            # --- the other blocks ---
            csd = csd_summary(out)
            row.update({k: pw[k] for k in ("n_theta_windows", "n_LIA_windows", "n_ripples") if k in pw})
            row["n_csd_ripples"] = csd.pop("n_csd_ripples", np.nan)
            row.update({k: v for k, v in pw.items() if k.startswith("ripple_")})
            row.update(csd)
            row.update({k: v for k, v in pw.items()
                        if k not in ("n_theta_windows", "n_LIA_windows", "n_ripples")
                        and not k.startswith("ripple_")})

            row["_has_power"] = bool(pw)
            row["_has_csd"] = not pd.isna(row["n_csd_ripples"])
            rows.append(row)

    if not rows:
        print("No sessions found.")
        return

    table = pd.DataFrame(rows)
    hp, hc = table.pop("_has_power"), table.pop("_has_csd")
    keep = (hp & hc) if KEEP_RULE == "all" else (hp | hc)
    out_dir = baseDir / TOTAL_OUT_FOLDER
    out_dir.mkdir(parents=True, exist_ok=True)
    has_ch = ~table[[f"ch_{k}" for k in LAYERS]].isna().all(axis=1)
    if REQUIRE_CHANNELS:
        keep = keep & has_ch
    removed = table.loc[~keep, ["animal", "session", "group"]].assign(
        has_power=hp[~keep], has_csd=hc[~keep], has_channels=has_ch[~keep])
    if len(removed):
        removed.to_csv(out_dir / "removed_sessions.csv", index=False)
        print(f"Removed {len(removed)} sessions without enough data (KEEP_RULE='{KEEP_RULE}'): see removed_sessions.csv")
    table = table[keep].reset_index(drop=True)
    has_power, has_csd = int(hp[keep].sum()), int(hc[keep].sum())

    out_dir = baseDir / TOTAL_OUT_FOLDER
    out_dir.mkdir(parents=True, exist_ok=True)
    FLAGS = ["power_channels_match_excel", "config_matches_excel", "config_theta_equals_config_slm"]
    table_out = table.drop(columns=FLAGS)  # checks are only printed / kept in 'channel_mismatches'
    table_out.to_csv(out_dir / "features_final.csv", index=False)
    mism = pd.DataFrame(mismatches)
    if len(mism):
        mism.to_csv(out_dir / "channel_mismatches.csv", index=False)
    # sessions whose name was not found in the channels file
    miss = table.loc[table[[f"ch_{k}" for k in LAYERS]].isna().all(axis=1), ["animal", "session"]]
    if len(miss):
        miss = miss.copy()
        miss["sessions_of_this_animal_in_channels_file"] = [
            " | ".join(chan_xlsx.loc[chan_xlsx["animal_key"] == norm_session(a), "session"].astype(str))
            if chan_xlsx is not None else ""
            for a in miss["animal"]
        ]
        miss.to_csv(out_dir / "sessions_not_in_channels_file.csv", index=False)
    try:
        with pd.ExcelWriter(out_dir / "features_final.xlsx", engine="openpyxl") as w:
            table_out.to_excel(w, sheet_name="features", index=False)
            if len(mism):
                mism.to_excel(w, sheet_name="channel_mismatches", index=False)
    except Exception as e:
        print(f"(Excel not written: {e}; the CSV was saved)")

    print(f"\n{len(table)} sessions: " + ", ".join(f"{g}={int((table.group == g).sum())}" for g in CONDITIONS))
    print(f"with power features: {has_power} | with CSD: {has_csd}")
    print(f"sessions without reference channels (not in {CHANNELS_FILE}): "
          f"{int(table[[f'ch_{k}' for k in LAYERS]].isna().all(axis=1).sum())}")

    ok = table["power_channels_match_excel"].dropna()
    if len(ok):
        n_bad = int((~ok.astype(bool)).sum())
        print(f"POWER CHANNELS vs Excel: same channels in {len(ok) - n_bad} of {len(ok)} sessions"
              + (f"  -> {n_bad} sessions used other channels (see sheet 'channel_mismatches')" if n_bad else ""))
    both = table.dropna(subset=["power_pyr_channel_used", "power_slm_channel_used", "ch_pyr", "ch_slm"])
    if len(both):
        d_pyr = both["power_pyr_channel_used"] - both["ch_pyr"]
        d_slm = both["power_slm_channel_used"] - both["ch_slm"]
        print(f"OFFSET check (power channel minus Excel channel), PYR: {d_pyr.value_counts().head(4).to_dict()}")
        print(f"OFFSET check (power channel minus Excel channel), SLM: {d_slm.value_counts().head(4).to_dict()}")
    cm = table["config_matches_excel"].dropna()
    if len(cm):
        print(f"configChannels vs Excel (so, pyr, rad, slm): same in {int(cm.astype(bool).sum())} of {len(cm)} sessions")
    same = table["config_theta_equals_config_slm"].dropna()
    if len(same):
        print(f"configChannels: ch.theta equals ch.slm in {int(same.astype(bool).sum())} of {len(same)} sessions")
    print(f"Saved: {out_dir / 'features_final.xlsx'}")


if __name__ == "__main__":
    main()