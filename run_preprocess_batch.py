__author__ = ('Saul', 'Nuria')

import pandas as pd
import numpy as np
import argparse

from pathlib import Path
from typing import Tuple, Optional

from preprocess.preprocess_suite2p_v1 import process_single_session
from preprocess.run_preprocess_sessions import run_suite2p_local
import subprocess

def get_filters() -> dict:
    return {
        'session_date': '191005',
        #'mice_name': ['NVI13','NVI16']
    }

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

        df_flagged = df[df.filter(like='Flag').notna().any(axis=1)]
        print('**The following datasets are flagged**')
        print(df_flagged)

        # Uncomment to not run flagged datasets
        #df = df[df.filter(like='Flag').isna().all(axis=1)]
        filter_indexes = df.index
        print('**The following datasets will be ran**')
        print(df)
        print(filter_indexes)

        run_suite2p_df_batch(filter_indexes, args.df_dir, args.folder_save, args.folder_raw, args.frame_rate, args.slurm_file_dir, args.default_path)
    else:
        run_suite2p_local(args.row_index, args.df_dir, args.folder_save, args.folder_raw, args.frame_rate, args.default_path)
   
    # python run_preprocess_batch.py -1 /home/sgurgua4/Documents/project/nvl_lab/Data_Analysis_Holo/holobmi_df.parquet /home/sgurgua4/Downloads /data/project/nvl_lab/HoloBMI/Raw 29.752 /home/sgurgua4/Documents/project/nvl_lab/Data_Analysis_Holo/utils/preprocess_slurm.sh
