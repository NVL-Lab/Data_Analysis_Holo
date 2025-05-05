__author__ = 'Nuria'
# __author__ = ("Nuria", "John Doe")
# quick script to load all packages to work in ipython


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
from matplotlib import interactive
from scipy import signal
from scipy.signal import find_peaks
from scipy.io import loadmat
from typing import Tuple, Optional

from pynwb import NWBHDF5IO, TimeSeries, ogen
interactive(True)



from pynwb import NWBHDF5IO, TimeSeries, ogen
from preprocess import syncronize_voltage_rec as svr