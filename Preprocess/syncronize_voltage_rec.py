__author__ = 'Nuria'
# __author__ = ("Nuria", "John Doe")

# make sure to be in environment with pynwb

from pathlib import Path
from typing import Tuple

from scipy.signal import find_peaks
from scipy.io import loadmat

import pandas as pd
import numpy as np


def obtain_peaks_voltage(voltage_recording: Path, frame_rate: float, size_of_recording: int) \
        -> Tuple[np.array, np.array, np.array]:
    """ Funtion to obtain the peaks of the voltage recording file
    :param voltage_recording: the path to the voltage recording file
        Input_1 = Trigger for each frame of the microscope recording
        Input_5 = Trigger for reward
        Input_6 = Trigger for a holographic stim
        Input_7 = Trigger for each frame evaluated by the BMI
    :param frame_rate: the frame rate of the 2p recording
    :param size_of_recording: the size of the 2p data
    :return: the indices of the peaks of the voltage """

    df_voltage = pd.read_csv(voltage_recording)
    peaks_I1, _ = find_peaks(df_voltage[' Input 1'][:int(size_of_recording / frame_rate * 1000)], height=2, distance=15)
    peaks_I5, _ = find_peaks(df_voltage[' Input 5'][:int(size_of_recording / frame_rate * 1000)], height=3,
                             distance=1000)
    peaks_I6, _ = find_peaks(df_voltage[' Input 6'][:int(size_of_recording / frame_rate * 1000)], height=0.1,
                             distance=1000)
    peaks_I7, _ = find_peaks(df_voltage[' Input 7'][:int(size_of_recording / frame_rate * 1000)], height=2, distance=15)

    if peaks_I1.shape[0] != size_of_recording:
        raise ValueError(f'We found more frame triggers {peaks_I1.shape[0]} '
                         f'than the size of the recording {size_of_recording}')

    indices_for_5 = np.searchsorted(peaks_I1, peaks_I5) - 1
    indices_for_5 = np.maximum(indices_for_5, 0).astype('int')
    indices_for_6 = np.searchsorted(peaks_I1, peaks_I6) - 1
    indices_for_6 = np.maximum(indices_for_6, 0).astype('int')
    indices_for_7 = np.searchsorted(peaks_I1, peaks_I7) - 1
    indices_for_7 = np.maximum(indices_for_7, 0).astype('int')

    return indices_for_5, indices_for_6, indices_for_7

