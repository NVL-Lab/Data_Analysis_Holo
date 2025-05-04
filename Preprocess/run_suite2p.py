import os
import sys
import json
import numpy as np
from pathlib import Path
from suite2p.run_s2p import run_s2p

def get_limited_tiff_files(folder_path, limit):
    """Returns a limited list of TIFF file paths from the given folder."""
    folder_path = Path(folder_path)
    
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"Error: The folder '{folder_path}' does not exist or is not a directory.")
        return []

    tiff_files = sorted([str(file) for file in folder_path.iterdir() if file.suffix.lower() == ".tif"])
    
    return tiff_files[:limit] 

# Ensure batch and output folder lists exist
batch_list_file = "batch_list.json"
output_folder_file = "output_folders.json"

if not os.path.exists(batch_list_file) or not os.path.exists(output_folder_file):
    print("Error: Missing batch_list.json or output_folder.json. Run prepare_data.py first.")
    sys.exit(1)

# Retrieve SLURM_ARRAY_TASK_ID
SLURM_ARRAY_TASK_ID = int(os.getenv('SLURM_ARRAY_TASK_ID', -1))

# Validate SLURM array job index
if SLURM_ARRAY_TASK_ID == -1:
    print("Error: SLURM_ARRAY_TASK_ID not set. Run this script via an array job.")
    sys.exit(1)

# Load batch list
with open(batch_list_file, "r") as f:
    valid_batches = json.load(f)

if SLURM_ARRAY_TASK_ID >= len(valid_batches):
    print(f"Error: SLURM_ARRAY_TASK_ID {SLURM_ARRAY_TASK_ID} is out of range.")
    sys.exit(1)

# Load output folder mapping
with open(output_folder_file, "r") as f:
    output_folders = json.load(f)

# Get the current batch and output folder
data_path = valid_batches[SLURM_ARRAY_TASK_ID]
output_folder = Path(output_folders[str(SLURM_ARRAY_TASK_ID)])  
tiff_limits = [2600,27000,75600,75600]
limited_tiff_list = []
for folder, limit in zip(data_path, tiff_limits):
        limited_tiff_list.extend(get_limited_tiff_files(folder, limit))
# Create output folder if it doesn’t exist
output_folder.mkdir(parents=True, exist_ok=True)

# Load the ops file
default_path = "/data/project/nvl_lab/HoloBMI/default_ops.npy"
aux_ops = np.load(Path(default_path), allow_pickle=True)
ops = aux_ops.take(0)  # Extract the first item from the loaded numpy array
ops['data_path'] = data_path
ops['tiff_list'] = limited_tiff_list
# Update database paths
db = {
    'data_path': data_path,
    'save_path0': str(output_folder),
    'fast_disk': str(output_folder)
}

# Run Suite2p
ops_after_1st_pass = run_s2p(ops, db)

# Save the resulting ops
np.save(output_folder / 'ops_after_1st_pass.npy', ops_after_1st_pass, allow_pickle=True)

print(f"Processing complete for batch {SLURM_ARRAY_TASK_ID}. Results saved to {output_folder}.")
