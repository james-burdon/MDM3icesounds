# pip install netCDF4
from netCDF4 import Dataset, num2date
import numpy as np
import csv

nc_path = r"air.mon.mean.nc"
t_idx = 859  # time slice to export

def to_datetime_label(time_var, idx):
    try:
        units = time_var.units
        cal = getattr(time_var, "calendar", "standard")
        return num2date(time_var[idx], units=units, calendar=cal).isoformat()
    except Exception:
        return str(time_var[idx])

with Dataset(nc_path, mode="r") as ds:
    # --- required coords/vars ---
    lat = ds.variables["lat"][:]           # (n_lat,)
    lon = ds.variables["lon"][:]           # (n_lon,)
    time = ds.variables["time"]            # (n_time,)
    air  = ds.variables["air"]             # shape: (time, [level], lat, lon)

    # --- sanity checks ---
    n_time = len(time[:])
    if not (0 <= t_idx < n_time):
        raise IndexError(f"t_idx={t_idx} out of range (0..{n_time-1}).")

    dims = air.dimensions  # e.g. ('time','lat','lon') or ('time','level','lat','lon')
    # Build slice for selected time (and optionally first level if present)
    if "time" not in dims:
        raise ValueError(f"'time' not in air.dimensions: {dims}")

    # figure out indices per dimension
    idx_tuple = []
    for d in dims:
        if d == "time":
            idx_tuple.append(t_idx)
        elif d in ("level", "lev", "zlev", "plev"):
            # pick the first level by default; change here if you need a specific level
            idx_tuple.append(0)
        else:
            idx_tuple.append(slice(None))

    # slice once (masked array with auto scale/offset applied by netCDF4)
    air_sel = air[tuple(idx_tuple)]  # shape: (lat, lon) after squeezing
    air_sel = np.squeeze(air_sel)    # ensure 2D
    if air_sel.ndim != 2:
        raise ValueError(f"Expected 2D after selecting time/level, got shape {air_sel.shape}.")

    n_lat, n_lon = air_sel.shape
    if n_lat != len(lat) or n_lon != len(lon):
        raise ValueError("Shape mismatch between 'air' and coordinate arrays.")

    time_label = to_datetime_label(time, t_idx)

    # --- write CSV ---
    with open("air.csv", mode="w", newline="") as f:
        w = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        w.writerow(["time", "lat", "lon", "air"])

        # vectorized row writing by iterating rows once
        for i in range(n_lat):
            print(f"row {i+1} of {n_lat}")
            # fill masked values with NaN row-wise to avoid per-element Python calls
            row_vals = np.ma.filled(air_sel[i, :], np.nan).astype(float)
            for j in range(n_lon):
                w.writerow([time_label, float(lat[i]), float(lon[j]), row_vals[j]])
