import copernicusmarine
import copernicusmarine as cm
import copernicusmarine as cm
import copernicusmarine as cm

# try:
#     cm.describe("SEAICE_GLO_SEAICE_L4_REP_OBSERVATIONS_011_009")
#     print("Dataset exists!")
# except Exception as e:
#     print("Error:", e)

print(copernicusmarine.__version__)
cm.get(
    dataset_id="OSISAF-GLO-SEAICE_CONC_TIMESERIES-SH-LA-OBS",
    output_directory="./data",
    overwrite=True
)
