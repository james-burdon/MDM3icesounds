# pip install netCDF4
from netCDF4 import Dataset, num2date
import numpy as np
import csv

# File path (raw string to avoid backslash escapes)
nc_path = r"C:\Users\jburd\Desktop\MDM3ice\data\SEAICE_GLO_SEAICE_L4_REP_OBSERVATIONS_011_009\OSISAF-GLO-SEAICE_CONC_TIMESERIES-SH-LA-OBS_202003\2020\10\ice_conc_sh_ease2-250_cdr-v3p0_202010011200.nc"

t_idx = 0  # usually only one timestep

def to_datetime_label(time_var, idx):
    try:
        units = time_var.units
        cal = getattr(time_var, "calendar", "standard")
        return num2date(time_var[idx], units=units, calendar=cal).isoformat()
    except Exception:
        return str(time_var[idx])

with Dataset(nc_path, mode="r") as ds:
    # variables
    lat = ds.variables["lat"][:]  # shape (432, 432)
    lon = ds.variables["lon"][:]  # shape (432, 432)
    time = ds.variables["time"]
    ice_conc = ds.variables["ice_conc"]

    # sanity check on time index
    n_time = len(time[:])
    if not (0 <= t_idx < n_time):
        raise IndexError(f"t_idx={t_idx} out of range (0..{n_time-1})")

    # select time slice (squeeze removes extra dimensions)
    ice_sel = np.squeeze(ice_conc[t_idx, :, :])
    time_label = to_datetime_label(time, t_idx)

    # flatten 2D fields for export
    lat_flat = lat.flatten()
    lon_flat = lon.flatten()
    ice_flat = np.ma.filled(ice_sel, np.nan).flatten()

    # write CSV
    with open("ice_conc.csv", mode="w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time", "lat", "lon", "ice_conc"])
        rows = zip([time_label]*len(lat_flat), lat_flat, lon_flat, ice_flat)
        w.writerows(rows)

print("✅ Exported ice_conc.csv successfully!")
