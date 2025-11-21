import json
import pandas as pd
from dataframe_sessions import get_sessions_df

# Load the experiment session dataframe
df = get_sessions_df('hE2_rew')

# Prepare valid batches and corresponding output folders
batches = []
output_folders = {}

for index, row in df.iterrows():
    session_path = row["session_path"]

    # Construct paths
    batch = [
        f"/data/project/nvl_lab/HoloBMI/Raw/{session_path}/im/{row['Holostim_seq_im']}" if pd.notna(row['Holostim_seq_im']) else None,
        f"/data/project/nvl_lab/HoloBMI/Raw/{session_path}/im/{row['Baseline_im']}" if pd.notna(row['Baseline_im']) else None,
        f"/data/project/nvl_lab/HoloBMI/Raw/{session_path}/im/{row['Pretrain_im']}" if pd.notna(row['Pretrain_im']) else None,
        f"/data/project/nvl_lab/HoloBMI/Raw/{session_path}/im/{row['BMI_im']}" if pd.notna(row['BMI_im']) else None
    ]

    # Ensure batch contains exactly 4 valid paths
    if len([path for path in batch if path]) == 4:
        batches.append(batch)
        output_folders[len(batches) - 1] = f"/data/project/nvl_lab/HoloBMI/Processed/{session_path}"  
        
# Save valid batches and output folders
batch_list_file = "batch_list.json"
output_folders_file = "output_folders.json"

with open(batch_list_file, "w") as f:
    json.dump(batches, f)

with open(output_folders_file, "w") as f:
    json.dump(output_folders, f)

print(f"Saved {len(batches)} valid batches to {batch_list_file}.")
print(f"Saved corresponding output folders to {output_folders_file}.")



