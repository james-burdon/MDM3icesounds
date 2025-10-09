import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ========== 1) CONFIG ==========
BASE_DIR = r"C:\Users\mayue\Desktop\2020_oct_to_december_sea_ice[1]\csv_out"
OUT_DIR  = Path(BASE_DIR) / "_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Elephant Island location and ROI size (degrees)
EI_LAT, EI_LON = -61.1, -55.0
ROI_DLAT, ROI_DLON = 2.0, 3.0

# Mapping from ice concentration to one-way surface loss L_s (dB)
LS_LMAX, LS_GAMMA = 6.0, 1.5

# Figure settings
BINS = 40
DPI  = 160

# ========== 2) HELPERS ==========

def load_all_csvs(base_dir: str) -> pd.DataFrame:
    base = Path(base_dir)
    if not base.exists():
        raise RuntimeError(f"BASE_DIR does not exist: {base_dir}")

    files = list(base.rglob("*.csv"))
    print(f"[DEBUG] BASE_DIR = {base.resolve()}")
    print(f"[DEBUG] CSV files found = {len(files)}")
    if not files:
        print("[DEBUG] Children under BASE_DIR:")
        for p in base.iterdir():
            print("  -", p)
        raise RuntimeError("No CSV files found. Check BASE_DIR or file extensions.")

    frames = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"[WARN] Failed to read {fp}: {e}")
            continue

        cols = {c.lower(): c for c in df.columns}

        def pick(keys: str):
            for k in cols:
                if k in keys or any(tok in k for tok in keys.split("|")):
                    return cols[k]
            return None

        c_time = pick("time|date|datetime|valid")
        c_lat  = pick("lat|latitude|grid_latitude|y")
        c_lon  = pick("lon|longitude|grid_longitude|x")
        c_val  = pick("ice_conc|sic|sea_ice|sea_ice_concentration|"
                      "sea_ice_area_fraction|concentration|cdr_seaice|cdr_sea_ice")

        if c_val is None:
            num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
            if num_cols:
                c_val = num_cols[-1]
        if c_val is None:
            print(f"[WARN] Skip {fp} (no numeric value-like column).")
            continue

        rename = {}
        if c_time and c_time != "time": rename[c_time] = "time"
        if c_lat  and c_lat  != "lat" : rename[c_lat]  = "lat"
        if c_lon  and c_lon  != "lon" : rename[c_lon]  = "lon"
        if c_val  and c_val  != "ice_conc": rename[c_val] = "ice_conc"
        df = df.rename(columns=rename)

        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
        else:
            df["time"] = pd.NaT

        keep = [c for c in ["time", "lat", "lon", "ice_conc"] if c in df.columns]
        df = df[keep].copy()
        df["source_file"] = fp.name
        frames.append(df)

    if not frames:
        raise RuntimeError("CSV files were found, but none had suitable columns.")
    return pd.concat(frames, ignore_index=True)

def pick_roi(d: pd.DataFrame, lat0=EI_LAT, lon0=EI_LON, dlat=ROI_DLAT, dlon=ROI_DLON):
    if not all(c in d.columns for c in ["lat", "lon"]):
        return d.iloc[0:0]
    m = (
        (d["lat"] >= lat0 - dlat) & (d["lat"] <= lat0 + dlat) &
        (d["lon"] >= lon0 - dlon) & (d["lon"] <= lon0 + dlon)
    )
    return d.loc[m].copy()

def map_Ls(conc, Lmax=LS_LMAX, gamma=LS_GAMMA):
    c = np.asarray(conc) / 100.0
    return Lmax * (c ** gamma)

def save_histogram(data, xlabel, title, out_path, bins=BINS, dpi=DPI):
    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=bins)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()

# ========== 3) LOAD, CLEAN, ANALYZE ==========

# 3.1 Load
df = load_all_csvs(BASE_DIR)

# 3.2 Cleaning: drop NaN (keep zeros), normalize units to [0,100]
x = pd.to_numeric(df["ice_conc"], errors="coerce")
mask_ok = ~x.isna()
df = df.loc[mask_ok].copy()
x = x.loc[mask_ok]

q99 = x.quantile(0.99)
if q99 <= 1.5:
    x = x * 100.0

x = x.clip(0, 100)
df["ice_conc"] = x.values

# 3.3 Map to surface loss L_s (MUST be before any L_s plots or df_ge15 that uses L_s)
df["L_s"] = map_Ls(df["ice_conc"])

# 3.4 Global stats
global_stats = {
    "n_rows": int(len(df)),
    "time_start": str(df["time"].min()) if "time" in df.columns else None,
    "time_end":   str(df["time"].max()) if "time" in df.columns else None,
    "lat_min": float(df["lat"].min()) if "lat" in df.columns else None,
    "lat_max": float(df["lat"].max()) if "lat" in df.columns else None,
    "lon_min": float(df["lon"].min()) if "lon" in df.columns else None,
    "lon_max": float(df["lon"].max()) if "lon" in df.columns else None,
    "conc_min": float(df["ice_conc"].min()),
    "conc_max": float(df["ice_conc"].max()),
    "conc_mean": float(df["ice_conc"].mean()),
    "conc_median": float(df["ice_conc"].median()),
    "conc_p95": float(df["ice_conc"].quantile(0.95)),
    "share_zero": float((df["ice_conc"] == 0).mean()),
}
(OUT_DIR / "global_stats.json").write_text(json.dumps(global_stats, indent=2), encoding="utf-8")

# --- Global histograms ---
save_histogram(
    data=df["ice_conc"].values,
    xlabel="ice_conc (0–100)",
    title="Global histogram of sea-ice concentration",
    out_path=OUT_DIR / "hist_global.png"
)

df_nz = df[df["ice_conc"] > 0]
if len(df_nz):
    save_histogram(
        data=df_nz["ice_conc"].values,
        xlabel="ice_conc (0–100), zeros excluded",
        title="Global histogram of sea-ice concentration (no zeros)",
        out_path=OUT_DIR / "hist_global_nozero.png"
    )

df_ge15 = df[df["ice_conc"] >= 15]
if len(df_ge15):
    save_histogram(
        data=df_ge15["ice_conc"].values,
        xlabel="ice_conc (0–100), >=15% only",
        title="Sea-ice concentration (>=15%)",
        out_path=OUT_DIR / "hist_global_ge15.png"
    )

# 3.5 ROI near Elephant Island
roi = pick_roi(df)
roi_stats = {
    "roi_count": int(len(roi)),
    "conc_min": float(roi["ice_conc"].min()) if len(roi) else None,
    "conc_max": float(roi["ice_conc"].max()) if len(roi) else None,
    "conc_mean": float(roi["ice_conc"].mean()) if len(roi) else None,
    "conc_median": float(roi["ice_conc"].median()) if len(roi) else None,
    "conc_p95": float(roi["ice_conc"].quantile(0.95)) if len(roi) else None,
    "share_zero": float((roi["ice_conc"] == 0).mean()) if len(roi) else None
}
(OUT_DIR / "roi_stats_elephant_island.json").write_text(json.dumps(roi_stats, indent=2),
                                                      encoding="utf-8")

if len(roi) and roi["time"].notna().any():
    ts = (
        roi.dropna(subset=["time"])
           .groupby(pd.Grouper(key="time", freq="D"))["ice_conc"]
           .mean()
           .reset_index()
    )
    ts.to_csv(OUT_DIR / "ts_roi_daily_mean.csv", index=False, encoding="utf-8")
    plt.figure(figsize=(7, 3.5))
    plt.plot(ts["time"], ts["ice_conc"])
    plt.xlabel("Date")
    plt.ylabel("Mean ice_conc (ROI)")
    plt.title("Elephant Island — daily mean sea-ice concentration")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "ts_roi_daily_mean.png", dpi=DPI)
    plt.close()

# 3.6 Export tidy table
df[["time", "lat", "lon", "ice_conc", "L_s", "source_file"]].to_csv(
    OUT_DIR / "sea_ice_tidy.csv", index=False, encoding="utf-8"
)

# --- L_s histograms (after L_s is mapped) ---
save_histogram(
    data=df["L_s"].values,
    xlabel="L_s (dB one-way)",
    title="Surface loss L_s mapped from ice concentration",
    out_path=OUT_DIR / "hist_Ls.png"
)

df_Ls_nz = df[df["L_s"] > 0]
if len(df_Ls_nz):
    save_histogram(
        data=df_Ls_nz["L_s"].values,
        xlabel="L_s (dB one-way), zeros excluded",
        title="Surface loss L_s mapped from ice concentration (no zeros)",
        out_path=OUT_DIR / "hist_Ls_nozero.png"
    )

# IMPORTANT: df_ge15 was defined from df after L_s mapping, so it has L_s now.
if len(df_ge15):
    save_histogram(
        data=df_ge15["L_s"].values,
        xlabel="L_s (dB one-way), >=15% only",
        title="Surface loss L_s (>=15%)",
        out_path=OUT_DIR / "hist_Ls_ge15.png"
    )

print("Done. Outputs written to:", OUT_DIR.resolve())
