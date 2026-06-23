__author__ = 'Saul'


def get_suite2p_params() -> dict:
    return {
        'df_dir': '/home/sgurgua4/Documents/project/nvl_lab/Data_Analysis_Holo/utils/holobmi_df/holobmi_df.parquet',
        'folder_save': '/gpfs/scratch/sgurgua4', # /home/sgurgua4/Documents/project/nvl_lab
        'folder_raw': '/data/project/nvl_lab/HoloBMI/Raw',
        'frame_rate': 29.752,
        'default_path': '',
        'slurm_file_dir': '/home/sgurgua4/Documents/project/nvl_lab/Data_Analysis_Holo/utils/slurm/suite2p_preprocess_slurm.sh'
    }

def get_nwb_params() -> dict:
    return {
        'df_dir': '/home/sgurgua4/Documents/project/nvl_lab/Data_Analysis_Holo/utils/holobmi_df/holobmi_df.parquet',
        'folder_save': '/home/sgurgua4/Documents/project/nvl_lab',
        'folder_raw': '/data/project/nvl_lab/HoloBMI/Raw',
        'behavior_folder_raw': '/data/project/nvl_lab/HoloBMI/Behavior',
        'slurm_file_dir': '/home/sgurgua4/Documents/project/nvl_lab/Data_Analysis_Holo/utils/slurm/nwb_preprocess_slurm.sh'
    }
