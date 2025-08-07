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
    local_dir = Path('C:/Users/nuria/DATA/Analysis/')  # None
    experiment_dir = Path('F:/data')


    # pre-process
    filter_size: int = 500  # in frames
    height_stim_artifact = 10

