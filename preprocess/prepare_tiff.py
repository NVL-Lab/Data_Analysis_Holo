import os
import pandas as pd
from pathlib import Path
import json
from dataframe_sessions import get_sessions_df

df = get_sessions_df('hE2_rew')
# Define TIFF limits for each category
tiff_limits = {
    "Holostim_seq_im": 2600,
    "Baseline_im": 27000,
    "Pretrain_im": 76000,
    "BMI_im": 76000
}

def get_limited_tiff_files(folder_path, limit):
    """
    Returns a list of TIFF file paths up to the specified limit.
    
    Args:
        folder_path (str or Path): The directory containing TIFF files.
        limit (int): The maximum number of TIFF files to return.
    
    Returns:
        list: A list of full paths to the limited TIFF files.
    """
    folder_path = Path(folder_path)

    if not folder_path.exists() or not folder_path.is_dir():
        return None  # Invalid path

    # Get all .tif files and apply limit
    tiff_files = sorted([file for file in folder_path.iterdir() if file.suffix == ".tif"])[:limit]
    print(len(tiff_files))
    return folder_path if len(tiff_files) > 0 else None

# Lists to store valid batches and output folders
batches = []
output_folders = {}

for index, row in df.iterrows():
    session_path = row["session_path"]
    
    # Construct paths with TIFF limits applied
    batch = [
        get_limited_tiff_files(f"/data/project/nvl_lab/HoloBMI/Raw/{session_path}/im/{row['Holostim_seq_im']}", tiff_limits["Holostim_seq_im"]) if pd.notna(row['Holostim_seq_im']) else None,
        get_limited_tiff_files(f"/data/Raw/{session_path}/im/{row['Baseline_im']}", tiff_limits["Baseline_im"]) if pd.notna(row['Baseline_im']) else None,
        get_limited_tiff_files(f"/data/Raw/{session_path}/im/{row['Pretrain_im']}", tiff_limits["Pretrain_im"]) if pd.notna(row['Pretrain_im']) else None,
        get_limited_tiff_files(f"/dataRaw/{session_path}/im/{row['BMI_im']}", tiff_limits["BMI_im"]) if pd.notna(row['BMI_im']) else None
    ]
    if len([path for path in batch if path]) == 4:
        batches.append(batch)
        output_folders[len(batches) - 1] = f"/data/project/nvl_lab/HoloBMI/Processed/{session_path}"

batch_list_file = "batch_list.json"
with open(batch_list_file, "w") as f:
    json.dump(batches, f)
print(f"Saved {len(batches)} valid batches to {batch_list_file}.")