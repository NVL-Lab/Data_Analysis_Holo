__author__ = ('Nuria', 'Saul')

import pandas as pd
import numpy as np

from pathlib import Path
from typing import Tuple, Optional

from preprocess.preprocess_suite2p import process_1_session_suite2p_offline
from preprocess.preprocess_suite2p_v1 import process_single_session

def run_all_suite2p_local(df: pd.DataFrame, default_path: Path, folder_save: Path, folder_raw: Path, frame_rate:float):
    """ function to run and process all experiments with suite2p locally"""
    folder_save = Path(folder_save)
    for index, row in df.iterrows():
        folder_process = folder_save / 'process'
        folder_raw_experiment = Path(folder_raw) / row['session_path']
        folder_processed_experiment = Path(folder_process) / row['session_path']
        folder_suite2p = folder_processed_experiment / 'suite2p' / 'plane0'
        if not Path(folder_suite2p).exists():
            Path(folder_suite2p).mkdir(parents=True, exist_ok=True)
        folder_im_paths = [str(folder_raw_experiment / 'im' / row['Holostim_seq_im']),
                           str(folder_raw_experiment / 'im' / row['Baseline_im']),
                           str(folder_raw_experiment / 'im' / row['Pretrain_im']),
                           str(folder_raw_experiment / 'im' / row['BMI_im'])]
        voltage_rec_paths = [str(folder_raw_experiment / 'im' / row['Holostim_seq_im'] /
                                 row['Holostim_seq_im_voltage_file']),
                             str(folder_raw_experiment / 'im' / row['Baseline_im'] / row['Baseline_im_voltage_file']),
                             str(folder_raw_experiment / 'im' / row['Pretrain_im'] / row['Pretrain_im_voltage_file']),
                             str(folder_raw_experiment / 'im' / row['BMI_im'] / row['BMI_im_voltage_file'])]
        size_recordings = []
        for folder in folder_im_paths:
            size_recordings.append(len(list(Path(folder).glob(f'*.tif'))))
        process_1_session_suite2p_offline(default_path, folder_suite2p, folder_im_paths, voltage_rec_paths, size_recordings, frame_rate)

def run_suite2p_local(row_index: int, df_path: str, folder_save: str, folder_raw: str, frame_rate:float, default_path: str = ''):
    """ function to run and process an experiment with suite2p locally"""

    if not Path(default_path).is_file():
        default_path = ''

    df = pd.read_parquet(df_path).reset_index(drop=True)
    row = df.iloc[row_index]

    suite2p_save_path = Path(folder_save).resolve(strict=True) / 'processed_suite2p' / row['session_path']
    suite2p_save_path.mkdir(parents=True, exist_ok=True)

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

    process_single_session(folder_im_paths, voltage_rec_paths, size_recordings, frame_rate, suite2p_save_path, default_path)
