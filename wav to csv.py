import os, sys
from scipy.io import wavfile
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


input_dir = "./wavfile"       
output_dir = "./csv_output"   
num_threads = 4               

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Find all .wav files in input_dir
wav_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".wav")]

if not wav_files:
    print(f"No .wav files found in {input_dir}")
    sys.exit()

print(f"Found {len(wav_files)} .wav files in {input_dir}")
print(f"Using {num_threads} threads...\n")

def save_with_header(filepath, data, column_names):
    """Save a numpy array to CSV with an index and column names"""
    indices = np.arange(len(data)).reshape(-1, 1)
    data_with_index = np.hstack((indices, data))
    header = ",".join(["Index"] + column_names)
    np.savetxt(filepath, data_with_index, delimiter=",", fmt="%d", header=header, comments='')

def convert_wav_to_csv(filename):
    """Convert one WAV file"""
    in_path = os.path.join(input_dir, filename)
    basename = os.path.splitext(filename)[0]

    try:
        samrate, data = wavfile.read(in_path)

        if data.ndim == 1:
            out_file = os.path.join(output_dir, f"{basename}_mono.csv")
            save_with_header(out_file, data.reshape(-1, 1), ["Amplitude"])
            msg = f"[Mono] {filename} converted successfully."

        elif data.ndim == 2 and data.shape[1] == 2:
            right_file = os.path.join(output_dir, f"{basename}_stereo_R.csv")
            left_file  = os.path.join(output_dir, f"{basename}_stereo_L.csv")
            save_with_header(right_file, data[:, 0].reshape(-1, 1), ["R"])
            save_with_header(left_file,  data[:, 1].reshape(-1, 1), ["L"])
            msg = f"[Stereo] {filename} converted successfully."

        else:
            columns = [f"Ch{i+1}" for i in range(data.shape[1])]
            out_file = os.path.join(output_dir, f"{basename}_multi.csv")
            save_with_header(out_file, data, columns)
            msg = f"[Multi-channel] {filename} converted successfully."

    except Exception as e:
        msg = f"[Error] {filename}: {e}"

    return msg

# Run conversions in parallel
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = [executor.submit(convert_wav_to_csv, f) for f in wav_files]
    for future in as_completed(futures):
        print(future.result())

print("\nAll conversions completed successfully.")
