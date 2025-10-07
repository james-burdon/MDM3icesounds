import os
import csv
import numpy as np
from pathlib import Path
from netCDF4 import Dataset, num2date

ROOT = r"C:\Users\mayue\Desktop\2020_oct_to_december_sea_ice[1]"
OUT_ROOT = os.path.join(ROOT, "csv_out")

# ----------------- Helpers -----------------
def find_coord_var(ds, candidates):
    names = {name.lower(): name for name in ds.variables.keys()}
    for cand in candidates:
        if cand.lower() in names:
            return ds.variables[names[cand.lower()]]
    for vname, var in ds.variables.items():
        std = getattr(var, "standard_name", "").lower()
        if std in [c.lower() for c in candidates]:
            return var
    return None

def guess_time(ds):
    return find_coord_var(ds, ["time"])

def guess_lat_1d(ds):
    var = find_coord_var(ds, ["lat", "latitude", "y", "grid_latitude"])
    if var is not None and getattr(var, "ndim", 0) == 1:
        return var
    return None

def guess_lon_1d(ds):
    var = find_coord_var(ds, ["lon", "longitude", "x", "grid_longitude"])
    if var is not None and getattr(var, "ndim", 0) == 1:
        return var
    return None

def guess_lat_2d(ds):
    for k, v in ds.variables.items():
        if k.lower() in ["lat", "latitude"] and getattr(v, "ndim", 0) == 2:
            return v
        std = getattr(v, "standard_name", "").lower()
        if std == "latitude" and getattr(v, "ndim", 0) == 2:
            return v
    return None

def guess_lon_2d(ds):
    for k, v in ds.variables.items():
        if k.lower() in ["lon", "longitude"] and getattr(v, "ndim", 0) == 2:
            return v
        std = getattr(v, "standard_name", "").lower()
        if std == "longitude" and getattr(v, "ndim", 0) == 2:
            return v
    return None

def is_numeric(var):
    return np.issubdtype(var.dtype, np.number)

def guess_main_data_var(ds):
    preferred = [
        "ice", "sic", "sea_ice", "sea_ice_concentration",
        "sea_ice_area_fraction", "seaice_conc", "concentration",
        "cdr_seaice_conc", "cdr_sea_ice_concentration"
    ]
    axis_like = {"lat","latitude","lon","longitude","x","y","grid_latitude","grid_longitude"}
    def ok(var): return is_numeric(var) and len(var.dimensions) >= 2

    # 1) preferred names
    for name, var in ds.variables.items():
        if name.lower() in preferred and ok(var):
            dims_low = set(d.lower() for d in var.dimensions)
            if dims_low & axis_like:
                return var
    # 2) CF attributes hint
    for name, var in ds.variables.items():
        if not ok(var): continue
        std = getattr(var, "standard_name", "").lower()
        lng = getattr(var, "long_name", "").lower()
        if any(k in std for k in ["sea_ice","ice"]) or any(k in lng for k in ["sea ice","ice"]):
            return var
    # 3) fallback: first numeric >=2D with geo-like dims
    for name, var in ds.variables.items():
        if ok(var):
            dims_low = set(d.lower() for d in var.dimensions)
            if dims_low & axis_like:
                return var
    raise RuntimeError("No suitable 2D+ numeric data variable with geo-like dims found.")

def to_time_label(time_var, t_idx):
    try:
        units = time_var.units
        cal = getattr(time_var, "calendar", "standard")
        dt = num2date(time_var[t_idx], units=units, calendar=cal)
        return dt.isoformat()
    except Exception:
        try:
            return str(time_var[t_idx])
        except Exception:
            return ""  # some files have no time at all

def first_level_index(dims):
    for d in dims:
        if d.lower() in ("level","lev","plev","zlev"):
            return True, 0
    return False, None

def quick_header(ds, limit=10):
    keys = list(ds.variables.keys())
    out = []
    for k in keys[:limit]:
        v = ds.variables[k]
        out.append(f"  - {k}: shape={getattr(v, 'shape', '?')} dims={getattr(v, 'dimensions', '?')}")
    return "\n".join(out)

# ----------------- Core -----------------
def convert_nc_to_csv(nc_path, out_dir):
    with Dataset(nc_path, mode="r") as ds:
        time_var = guess_time(ds)
        data_var = guess_main_data_var(ds)

        lat1d = guess_lat_1d(ds)
        lon1d = guess_lon_1d(ds)
        lat2d = lon2d = None
        if lat1d is None or lon1d is None:
            lat2d = guess_lat_2d(ds)
            lon2d = guess_lon_2d(ds)

        has_time = time_var is not None
        n_time = len(time_var[:]) if has_time else 1

        dims = data_var.dimensions
        has_level, level_idx = first_level_index(dims)

        Path(out_dir).mkdir(parents=True, exist_ok=True)

        for t in range(n_time):
            # Build slice
            idx = []
            for d in dims:
                dl = d.lower()
                if dl == "time":
                    idx.append(t)
                elif dl in ("level","lev","plev","zlev"):
                    idx.append(level_idx if has_level else 0)
                else:
                    idx.append(slice(None))

            arr = np.squeeze(data_var[tuple(idx)])
            if arr.ndim < 2:
                raise ValueError(f"Expected 2D+ after indexing. Got {arr.shape} for dims {dims}")
            if arr.ndim > 2:
                arr = np.squeeze(arr[0, ...])   # (band, y, x) → take first band
            if arr.ndim != 2:
                raise ValueError(f"Still not 2D grid. Got {arr.shape}")

            arr2 = np.ma.filled(arr, np.nan).astype(float)
            ny, nx = arr2.shape

            t_label = to_time_label(time_var, t) if has_time else ""
            base = Path(nc_path).stem
            safe_time = t_label.replace(":", "-") if t_label else f"t{t:03d}"
            out_csv = os.path.join(out_dir, f"{base}_{safe_time}.csv")

            with open(out_csv, "w", newline="") as f:
                w = csv.writer(f)

                # Case A: strict 1D lat/lon
                if (lat1d is not None and lon1d is not None and
                    getattr(lat1d, "ndim", 0) == 1 and getattr(lon1d, "ndim", 0) == 1 and
                    len(lat1d[:]) == ny and len(lon1d[:]) == nx):

                    w.writerow(["time", "lat", "lon", data_var.name])
                    lat_vals = np.asarray(lat1d[:], dtype=float)
                    lon_vals = np.asarray(lon1d[:], dtype=float)
                    for i in range(ny):
                        row_vals = arr2[i, :]
                        for j in range(nx):
                            w.writerow([t_label, lat_vals[i], lon_vals[j], row_vals[j]])

                # Case B: 2D lat/lon grids
                elif (lat2d is not None and lon2d is not None and
                      getattr(lat2d, "ndim", 0) == 2 and getattr(lon2d, "ndim", 0) == 2 and
                      lat2d.shape == arr2.shape and lon2d.shape == arr2.shape):

                    w.writerow(["time", "lat", "lon", data_var.name])
                    lat_grid = np.ma.filled(lat2d[:], np.nan).astype(float)
                    lon_grid = np.ma.filled(lon2d[:], np.nan).astype(float)
                    for i in range(ny):
                        row_vals = arr2[i, :]
                        for j in range(nx):
                            w.writerow([t_label, lat_grid[i, j], lon_grid[i, j], row_vals[j]])

                # Case C: no lat/lon → write grid indices
                else:
                    w.writerow(["time", "i", "j", data_var.name])
                    for i in range(ny):
                        row_vals = arr2[i, :]
                        for j in range(nx):
                            w.writerow([t_label, i, j, row_vals[j]])

            print(f"✓ Wrote {out_csv}")

# ----------------- Walk & run -----------------
def main():
    for root, _, files in os.walk(ROOT):
        rel = os.path.relpath(root, ROOT)
        out_dir = os.path.join(OUT_ROOT, rel)
        for fn in files:
            if fn.lower().endswith((".nc", ".nc4", ".cdf")):
                nc_path = os.path.join(root, fn)
                try:
                    convert_nc_to_csv(nc_path, out_dir)
                except Exception as e:
                    # Quick header to help diagnose special files
                    try:
                        with Dataset(nc_path) as ds:
                            hdr = quick_header(ds)
                    except Exception:
                        hdr = "(failed to open for header dump)"
                    print(f"⚠️ Failed on {nc_path}: {e}\n{hdr}")

if __name__ == "__main__":
    main()
