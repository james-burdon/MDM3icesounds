import copernicusmarine
import copernicusmarine as cm

cm.get(
    dataset_id="OSISAF-GLO-SEAICE_CONC_TIMESERIES-SH-LA-OBS",
    output_directory="./data",
    filter="*2020[1-5]*",      # February 2020
    overwrite=True
)