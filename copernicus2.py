import copernicusmarine
import copernicusmarine as cm



cm.get(
    dataset_id="cmems_mod_glo_phy_my_0.25deg_P1M-m",   # ✅ monthly reanalysis (full-depth)
    output_directory="./data",
    filter="*2020_01*",            # ✅ correct pattern for MDS filenames
    overwrite=True
)


