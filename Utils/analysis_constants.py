__author__ = 'Nuria'
# __author__ = ("Nuria", "John Doe")
# constants to be used on analysis (for offline processing)

import posixpath
from dataclasses import dataclass


def population_directory(aux_dir: posixpath) -> str:
    return posixpath.join(aux_dir, "population")

def name_parquet(aux_name: str) -> str:
    return f'{aux_name}.parquet'


@dataclass
class AnalysisConstants:
    """  Class containing various constants for analysis, such as str for filenames """
    var_sweep = 'sweep'
    var_tuned = 'tuned'
    var_error = 'error'
    var_count = 'count'
    var_bins = 'bins'
    var_slope = 'slope'
    # from Prairie
    framerate = 29.752  # framerate of acquisition of images
    calibration_frames = 27000  # number of frames during calibration
    dff_win = 10  # number of frames to smooth dff
    len_calibration = 15  # length calibration in minutes
    lag_stim = 7e-3  # seconds from triggering daq with matlab to turning on light
    experiment_types = ['hE2_rew', 'hE2_norew', 'hE3_rew', 'randrew', 'hE2_rew_fb', 'randrew_fb']
    # stim removal
    height_stim_artifact = 3  # height of the stim artifact in std
    # preprocess
    mice = ['m13', 'm15', 'm16', 'm18', 'm21', 'm22', 'm23', 'm25', 'm26', 'm27', 'm28', 'm29']