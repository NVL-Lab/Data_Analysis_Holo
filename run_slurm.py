__author__ = 'Saul'

import argparse
import subprocess

from utils.holo_df_tools import get_data_indexes
from utils.params import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run slurm [suite2p, nwb]')
    parser.add_argument('input_type', type=int, help='row of dataframe')
    args = parser.parse_args()

    # change reading df to get_sessions_df
    if args.input_type == 0:
        params = get_suite2p_params()
        indexes = get_data_indexes(params['df_dir'], 'suite2p')
    elif args.input_type == 1:
        params = get_nwb_params()
        indexes = get_data_indexes(params['df_dir'], 'nwb')
    output = subprocess.check_output(['sbatch', params['slurm_file_dir'], *map(str, indexes)], text=True)
        
    print('SLURM response: ', output)
