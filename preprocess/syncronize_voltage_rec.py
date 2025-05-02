__author__ = ('Nuria', 'Saul')

# make sure to be in environment with pynwb

from pathlib import Path
from typing import Tuple

from scipy.signal import find_peaks

import pandas as pd
import numpy as np

def obtain_peaks_voltage(voltage_recording: Path, frame_rate: float, size_of_recording: int, limit_size: bool = False) \
        -> Tuple[list, np.array, np.array, np.array, np.array, np.array, np.array, np.array, np.array, np.array, np.array, np.array, np.array, list]:
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
            :return: the indices of the peaks of the voltage 
    """

    df_voltage = pd.read_csv(voltage_recording)
    comments = []

    # peaks_I0,I2,I3,and I4 are most likely from behavioral data
    peaks_i0, _ = find_peaks(df_voltage[' Input 0'][:int(size_of_recording / frame_rate * 1000)], height=0.6, prominence=0.6, distance=25)
    peaks_i1, _ = find_peaks(df_voltage[' Input 1'][:int(size_of_recording / frame_rate * 1000)], prominence=0.05, distance=15)
    peaks_i2, _ = find_peaks(df_voltage[' Input 2'][:int(size_of_recording / frame_rate * 1000)], height=0.2, distance=30)
    peaks_i3, _ = find_peaks(df_voltage[' Input 3'][:int(size_of_recording / frame_rate * 1000)], height=0.2, prominence=2, distance=30)
    peaks_i4, _ = find_peaks(df_voltage[' Input 4'][:int(size_of_recording / frame_rate * 1000)], prominence=0.05)

    if peaks_i1.shape[0] > size_of_recording:
        comments.append(f'We found more frame triggers {peaks_i1.shape[0]} '
                        f'than the size of the recording {size_of_recording}')
        peaks_i1 = peaks_i1[:size_of_recording]
        #raise Warning(comments)
    else:
        comments.append(f'Triggers for image frames: {peaks_i1.shape[0]} found successfully ')

    peaks_i4, _ = find_peaks(df_voltage[' Input 4'][:int(size_of_recording / frame_rate * 1000)], prominence=0.05)
    peaks_i5, _ = find_peaks(df_voltage[' Input 5'][:int(size_of_recording / frame_rate * 1000)], prominence=0.05,
                             distance=100)

    if not limit_size or peaks_i5.size == 0 :
        peaks_i6, _ = find_peaks(df_voltage[' Input 6'][:int(size_of_recording / frame_rate * 1000)], prominence=0.05,
                                 distance=100)
        peaks_i7, _ = find_peaks(df_voltage[' Input 7'][:int(size_of_recording / frame_rate * 1000)], prominence=0.05,
                                  distance=15, width=(0, 45))
    else:
        peaks_i6, _ = find_peaks(df_voltage[' Input 6'][:peaks_i5[-1]], prominence=0.05,
                                 distance=100)
        peaks_i7, _ = find_peaks(df_voltage[' Input 7'][:peaks_i5[-1]], prominence=0.05,
                                 distance=15, width=(0, 45))

    indices_for_4 = np.searchsorted(peaks_i1, peaks_i4) - 1
    indices_for_4 = np.maximum(indices_for_4, 0).astype('int')
    indices_for_5 = np.searchsorted(peaks_i1, peaks_i5) - 1
    indices_for_5 = np.maximum(indices_for_5, 0).astype('int')
    indices_for_6 = np.searchsorted(peaks_i1, peaks_i6) - 1
    indices_for_6 = np.maximum(indices_for_6, 0).astype('int')
    indices_for_7 = np.searchsorted(peaks_i1, peaks_i7) - 1
    indices_for_7 = np.maximum(indices_for_7, 0).astype('int')

    return df_voltage.keys(), peaks_i0, peaks_i1, peaks_i2, peaks_i3, peaks_i4, peaks_i5, peaks_i6, peaks_i7, indices_for_4, indices_for_5, indices_for_6, indices_for_7, comments


def obtain_indices_per_peaks(peaks_a, peaks_b) -> np.array:
    """ Function to obtain the indices of peaks_b closely aligned to the previous peak from peaks_a """
    indices_peak = np.searchsorted(peaks_a, peaks_b) - 1
    indices_peak = np.maximum(indices_peak, 0).astype('int')
    return indices_peak