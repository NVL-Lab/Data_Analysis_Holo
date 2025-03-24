__author__ = 'Nuria'
# __author__ = ("Nuria", "John Doe")
# constants to be used on analysis (for offline processing)

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class AnalysisConfiguration:
    """
    Class containing various configuration parameters for analysis. Reasonable defaults are
    provided.
    """

    # dirs
    local_dir = Path("C:/Users/nuria/DATA/Analysis/")  # None
    experiment_dir = Path("F:/data")

    # pre-process
    filter_size: int = 500  # in frames
    height_stim_artifact = 10
    calibration_frames: int = 27000  # in frames
    seq_holo_frames: int = 2700  # in frames
    pretrain_frames: int = 75600 # in frames
    percentil_threshold: Tuple = (0.1, 99.9)
    index_before_pretrain = seq_holo_frames + calibration_frames
    index_after_pretrain = seq_holo_frames + calibration_frames + pretrain_frames