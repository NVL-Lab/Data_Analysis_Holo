__author__ = 'Nuria'

# __author__ = ('Nuria', 'John Doe')

import pandas as pd
import numpy as np
import subprocess

from pathlib import Path
from typing import Tuple, Optional

from preprocess.preprocess_suite2p import process_1_session_suite2p_offline

def process_single_session(row_index: int, df_path: str, folder_save: str, folder_raw: str, default_path: Path, frame_rate:float):
    # df: /home/sgurgua4/Documents/project/nvl_lab/Data_Analysis_Holo/holobmi_df.parquet
    # default_path: path to default rois
    # folder_save: /data/project/nvl_lab/HoloBMI/
    # folder_raw: /data/project/nvl_lab/HoloBMI/Raw
    # frame_rate: 29.752

    df = pd.read_parquet(df_path)
    row = df.iloc[row_index]

    folder_processed_experiment = Path(folder_save).resolve(strict=True) / 'process' / row['session_path']
    folder_processed_experiment.mkdir(parents=True, exist_ok=True)
    # May not need
    #if not folder_suite2p.exists():
        #folder_suite2p.mkdir(parents=True, exist_ok=True)

    # Stores image and voltage recording paths
    folder_raw_experiment = Path(folder_raw).resolve(strict=True) / row['session_path'] / 'im'
    folder_im_paths = [str(folder_raw_experiment / row['Holostim_seq_im']),
                       str(folder_raw_experiment / row['Baseline_im']),
                       str(folder_raw_experiment / row['Pretrain_im']),
                       str(folder_raw_experiment / row['BMI_im'])]
    voltage_rec_paths = [str(folder_raw_experiment / row['Holostim_seq_im'] / row['Holostim_seq_im_voltage_file']),
                         str(folder_raw_experiment / row['Baseline_im'] / row['Baseline_im_voltage_file']),
                         str(folder_raw_experiment / row['Pretrain_im'] / row['Pretrain_im_voltage_file']),
                         str(folder_raw_experiment / row['BMI_im'] / row['BMI_im_voltage_file'])]

    size_recordings = []
    for folder in folder_im_paths:
        size_recordings.append(len(list(Path(folder).glob(f'*.tif'))))

    process_1_session_suite2p_offline(default_path, folder_processed_experiment, folder_im_paths, voltage_rec_paths, size_recordings,
                                      frame_rate)

def run_all_suite2p(df_path: str, folder_save: str, folder_raw: str, default_path: Path, frame_rate: float, slurm: bool=False):
    """ function to run and process all experiments with suite2p"""

    row_count = pd.read_parquet(df_path).shape[0]
    if slurm:
        script_dir = 'preprocess_slurm.sh'
        output = subprocess.check_output(['sbatch', script_dir, row_count, df_path, folder_save, folder_raw, default_path, frame_rate], text=True)
        print('SLURM response: ', output)
    else:
        for i in range(row_count):
            process_single_session(i, df_path, folder_save, folder_raw, default_path, frame_rate)
        print('Done')
