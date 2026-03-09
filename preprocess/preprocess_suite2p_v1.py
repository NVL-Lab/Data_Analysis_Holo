__author__ = ('Saul', 'Nuria')

import pandas as pd
import numpy as np

from pathlib import Path
from typing import List, Tuple, Optional
import suite2p

from utils.suite2p_v1_config import *
import preprocess.syncronize_voltage_rec as svr
#import syncronize_voltage_rec as svr
from preprocess.preprocess_suite2p import obtain_bad_frames_from_voltage_rec

def get_settings(default_settings_dir, settings = suite2p.default_settings()) -> dict:
    """
        Gives appropriate settings
    """
    if default_settings_dir:
        settings = np.load(Path(default_settings_dir) / 'default_settings.npy', allow_pickle=True)
    else:
        settings = get_suite2p_holo_settings()

    return settings


def process_single_session(im_dirs: List[str], voltage_rec_dirs: List[str], size_recordings: List[int], frame_rate: float, suite2p_save_path: Path, default_settings_dir: str = ''):
    """
        Function to process suite2p offline with a tone of info needed
            :param im_dirs: list of paths where the images to be concatenated are
            :param voltage_rec_dirs: list of paths of the voltage recordings needed. same size as folder_im_paths
            :param size_recordings: list of the sizes of the recordings
            :param frame_rate: frame rate of the recording
            :param suite2p_save_path: path where to store the result of suite2p
            :param default_settings_dir: path where the default ops are stored
    """

    if len(im_dirs) != len(size_recordings) | len(im_dirs) != len(voltage_rec_paths):
        raise ValueError('The sizes of the list need to be all the same')

    bad_frames, bad_frames_bool = obtain_bad_frames_from_voltage_rec(voltage_rec_dirs, frame_rate, size_recordings)
    db = get_suite2p_holo_db(im_dirs, suite2p_save_path, bad_frames, bad_frames_bool)
    settings = get_settings(default_settings_dir)

    suite2p.run_s2p(db, settings)
