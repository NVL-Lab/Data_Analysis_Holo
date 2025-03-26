__author__ = 'Nuria'
# __author__ = ("Nuria", "John Doe")
# constants to be used on analysis (for offline processing)

import posixpath
from dataclasses import dataclass
from typing import Tuple


def population_directory(aux_dir: posixpath) -> str:
    return posixpath.join(aux_dir, "population")

def name_parquet(aux_name: str) -> str:
    return f'{aux_name}.parquet'


@dataclass
class AnalysisConstants:
    """  Class containing various constants for analysis, such as str for filenames """

    # recording constants
    calibration_frames: int = 27000  # in frames
    seq_holo_frames: int = 2600  # in frames
    bmi_frames: int = 75600 # in frames
    percentil_threshold: Tuple = (0.1, 99.9)
    index_before_pretrain = seq_holo_frames + calibration_frames
    index_after_pretrain = seq_holo_frames + calibration_frames + bmi_frames
    total_size = seq_holo_frames + calibration_frames + 2 * bmi_frames

    dff_win = 10  # number of frames to smooth dff
    len_calibration = 15  # length calibration in minutes
    experiment_types = ['hE2_rew', 'hE2_norew', 'hE3_rew', 'randrew', 'hE2_rew_fb', 'randrew_fb']

    # stim removal


    # preprocess
    mice = ['NVI12', 'NVI13', 'NVI16', 'NVI17', 'NVI20', 'NVI21', 'NVI22']