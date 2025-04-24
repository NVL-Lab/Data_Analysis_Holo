__author__ = ('Nuria', 'Saul')

# make sure to be in environment with pynwb

from pathlib import Path
from typing import Tuple

from scipy.signal import find_peaks
from scipy.io import loadmat

import pandas as pd
import numpy as np


def obtain_peaks_voltage(voltage_recording: Path, frame_rate: float, size_of_recording: int, limit_size: bool = False) \
        -> Tuple[list, np.array, np.array, np.array, np.array, np.array, np.array, np.array, np.array, list]:
    """ 
        Obtains the peaks of the voltage recording file
            :param voltage_recording: the path to the voltage recording file
                Input_1 = Trigger for each frame of the microscope recording
                Input_5 = Trigger for reward
                Input_6 = Trigger for a holographic stim
                Input_7 = Trigger for each frame evaluated by the BMI
            :param frame_rate: the frame rate of the 2p recording
            :param size_of_recording: the size of the 2p data
            :param limit_size: whether to limit the size of the voltage recording to the last reward trigger
                               which seems to be 100% reliable across all experiments that has it.
            :return: the peaks of each channel from the voltage recording files
    """

    df_voltage = pd.read_csv(voltage_recording)
    comments = []

    # peaks_I0,I2,I3,and I4 are most likely from behavioral data
    peaks_I0, _ = find_peaks(df_voltage[' Input 0'][:int(size_of_recording / frame_rate * 1000)], height=0.6,
                             prominence=0.6, distance=25)
    peaks_I1, _ = find_peaks(df_voltage[' Input 1'][:int(size_of_recording / frame_rate * 1000)], prominence=0.05,
                             distance=15)
    peaks_I2, _ = find_peaks(df_voltage[' Input 2'][:int(size_of_recording / frame_rate * 1000)], height=0.2,
                             distance=30)
    peaks_I3, _ = find_peaks(df_voltage[' Input 3'][:int(size_of_recording / frame_rate * 1000)], height=0.2,
                             prominence=2, distance=30)
    peaks_I4, _ = find_peaks(df_voltage[' Input 4'][:int(size_of_recording / frame_rate * 1000)], prominence=0.05)

    if peaks_I1.shape[0] > size_of_recording:
        comments.append(f'We found more frame triggers {peaks_I1.shape[0]} '
                        f'than the size of the recording {size_of_recording}')
        peaks_I1 = peaks_I1[:size_of_recording]
        # raise Warning(comments)
    else:
        comments.append(f'Triggers for image frames: {peaks_I1.shape[0]} found successfully ')

    peaks_I4, _ = find_peaks(df_voltage[' Input 4'][:int(size_of_recording / frame_rate * 1000)], prominence=0.05)
    peaks_I5, _ = find_peaks(df_voltage[' Input 5'][:int(size_of_recording / frame_rate * 1000)], prominence=0.05,
                             distance=100)

    if not limit_size or peaks_I5.size == 0:
        peaks_I6, _ = find_peaks(df_voltage[' Input 6'][:int(size_of_recording / frame_rate * 1000)], prominence=0.05,
                                 distance=100)
        peaks_I7, _ = find_peaks(df_voltage[' Input 7'][:int(size_of_recording / frame_rate * 1000)], prominence=0.05,
                                 distance=15, width=(0, 45))
    else:
        peaks_I6, _ = find_peaks(df_voltage[' Input 6'][:peaks_I5[-1]], prominence=0.05,
                                 distance=100)
        peaks_I7, _ = find_peaks(df_voltage[' Input 7'][:peaks_I5[-1]], prominence=0.05,
                                 distance=15, width=(0, 45))

    return (list(df_voltage.keys()), peaks_I0, peaks_I1, peaks_I2, peaks_I3, peaks_I4, peaks_I5, peaks_I6, peaks_I7,
            comments)


def obtain_indices_per_peaks(peaks_a: np.array, peaks_b: np.array) -> np.array:
    """ Function to obtain the indices of peaks B closely aligned to the previous peak from A"""
    indices_peak = np.searchsorted(peaks_a, peaks_b) - 1
    indices_peak = np.maximum(indices_peak, 0).astype('int')
    return indices_peak
