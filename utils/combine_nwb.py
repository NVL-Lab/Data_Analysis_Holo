__author__ = 'Nuria'

import numpy as np
import pandas as pd

from typing import Tuple
from pathlib import Path
from pynwb import NWBHDF5IO, TimeSeries, ogen

from utils.analysis_configuration import AnalysisConfiguration as aconf
from utils.analysis_constants import AnalysisConstants as act


def list_nwb_paths(folder_nwb: Path, row: pd.Series) -> list:
    """ Function to combine in a list all NWB files that would normally belong to the same session
    :param folder_nwb: folder were all the nwb files are stored
    :param row: row: row series from the dataframe with all the animals / sessions
    :return: list of nwbfiles_path"""

    return [f'{folder_nwb / row.mice_name / row.mice_name}_{row.session_date}_holostim_seq.nwb',
            f'{folder_nwb / row.mice_name / row.mice_name}_{row.session_date}_baseline.nwb',
            f'{folder_nwb / row.mice_name / row.mice_name}_{row.session_date}_pretrain.nwb',
            f'{folder_nwb / row.mice_name / row.mice_name}_{row.session_date}_bmi.nwb']


def combine_indices_nwb(nwb_filenames: list[str], attribute: str) -> Tuple[np.array, int]:
    """ function to return the indices of a session based on the two photon frames
    This function is used to put together all nwb files of a session and obtain the timestamps
    of the whole session (understood as all the nwb_filenames list) together
    :param nwb_filenames: list of filenames and the order how to concatenate them
    :param attribute: the attribute to concatenate
    return: a np array with the indices of all nwb_filenames combined, related to the two photon frames
    """
    indices = [[]]
    len_recording = 0
    for path_filename in nwb_filenames:
        io = NWBHDF5IO(Path(path_filename), mode='r')
        nwbfile = io.read()
        if attribute in nwbfile.acquisition.keys():
            indices.append(nwbfile.acquisition[attribute].timestamps + len_recording)
        len_recording += nwbfile.acquisition['TwoPhotonSeries'].data.shape[0]
        io.close()
    return np.concatenate(indices), len_recording
