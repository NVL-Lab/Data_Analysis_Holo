__author__ = ('Saul', 'Nuria')

import pandas as pd
import numpy as np
import argparse

from pathlib import Path
from typing import Tuple, Optional

from preprocess_suite2p_v1 import process_1_session_suite2p_offline
import subprocess

def get_filters() -> dict:
    return {
        'session_date': '190930',
        'mice_name': ['NVI13','NVI16']
    }

def run_suite2p_local(row_index: int, df_path: str, folder_save: str, folder_raw: str, frame_rate:float, default_path: str = ''):
    # df: /home/sgurgua4/Documents/project/nvl_lab/Data_Analysis_Holo/holobmi_df.parquet
    # default_path: path to default rois
    # folder_save: /data/project/nvl_lab/HoloBMI/
    # folder_raw: /data/project/nvl_lab/HoloBMI/Raw
    # frame_rate: 29.752

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

    process_1_session_suite2p_offline(folder_im_paths, voltage_rec_paths, size_recordings, frame_rate, suite2p_save_path, default_path)

def run_suite2p_df_batch(filter_indexes: list, df_dir: str, folder_save: str, folder_raw: str, frame_rate: float, slurm_file_dir: str = '', default_path: str = ''):
    """ function to run and process experiments with suite2p"""

    if slurm_file_dir:
        output = subprocess.check_output(['sbatch', slurm_file_dir, df_dir, folder_save, folder_raw, str(frame_rate), default_path, *map(str, filter_indexes)], text=True)
        print('SLURM response: ', output)
    else:
        for i in filter_indexes:
            run_suite2p_local(i, df_dir, folder_save, folder_raw, default_path, frame_rate)
        print('Local run complete')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run suite2p')
    parser.add_argument('row_index', type=int, help='row of dataframe')
    parser.add_argument('df_dir', type=Path, help='path to dataframe')
    parser.add_argument('folder_save', type=Path, help='path to save folder')
    parser.add_argument('folder_raw', type=Path, help='path to raw folder')
    parser.add_argument('frame_rate', type=float, help='frame rate at which recording was gathered')
    parser.add_argument('slurm_file_dir', type=Path, help='path to default folder')
    parser.add_argument('default_path', type=Path, nargs='?', default='', help='path to default folder')

    args = parser.parse_args()

    df = pd.read_parquet(args.df_dir).reset_index(drop=True)
    filters = get_filters()

    if args.row_index == -1:
        for label in filters:
            if isinstance(filters[label], list):
                df = df[df[label].isin(filters[label])]
            elif isinstance(filters[label], str):
                df = df[df[label].eq(filters[label])]
            else:
                raise TypeError('Column values are of incorrect type')

        filter_indexes = df.index
        print(df)
        print(filter_indexes)

        run_suite2p_df_batch(filter_indexes, args.df_dir, args.folder_save, args.folder_raw, args.frame_rate, args.slurm_file_dir, args.default_path)
    else:
        run_suite2p_local(args.row_index, args.df_dir, args.folder_save, args.folder_raw, args.frame_rate, args.default_path)
   
    # python preprocess/run_preprocess_sessions2.py -1 /home/sgurgua4/Documents/project/nvl_lab/Data_Analysis_Holo/holobmi_df.parquet /home/sgurgua4/Downloads /data/project/nvl_lab/HoloBMI/Raw 29.752 /home/sgurgua4/Documents/project/nvl_lab/Data_Analysis_Holo/preprocess/preprocess_slurm.sh
