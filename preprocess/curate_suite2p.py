__author__ = 'Nuria'

# __author__ = ('Nuria', 'John Doe')

import os
import shutil
import copy
import math

import numpy as np
import pandas as pd
import numpy as np
import scipy.io as sio

from pathlib import Path
from typing import Tuple, Optional
from scipy import signal

from utils.analysis_configuration import AnalysisConfiguration as aconf
from utils.analysis_constants import AnalysisConstants as act
from utils.combine_nwb import combine_indices_nwb


def refine_classifier(folder_suite2p: Path, dn_bool: bool = True):
    """ function to refine the suite2p classifier """
    neurons = np.load(Path(folder_suite2p) / 'stat.npy', allow_pickle=True)
    is_cell = np.load(Path(folder_suite2p) / 'iscell.npy')
    is_cell_new = copy.deepcopy(is_cell)
    snr_val = snr_neuron(folder_suite2p)
    stable_neuron = stability_neuron(folder_suite2p, init=act.calibration_frames)
    for nn, neuron in enumerate(neurons):
        if neuron['skew'] > 10 or neuron['skew'] < 0.4 or neuron['compact'] > 1.4 or \
                neuron['footprint'] == 0 or neuron['footprint'] == 3 or neuron['npix'] < 80 or \
                snr_val[nn] < aconf.snr_min or ~stable_neuron[nn]:
            is_cell_new[nn, :] = [0, 0]
    if dn_bool:
        aux_dn = np.load(Path(folder_suite2p) / 'direct_neurons.npy', allow_pickle=True)
        direct_neurons_info = aux_dn.take(0)
        direct_neurons = direct_neurons_info['E1'] + direct_neurons_info['E2']
        direct_neurons.sort()
        for dn in direct_neurons:
            is_cell_new[dn, :] = [1, 1]
    np.save(Path(folder_suite2p) / 'iscell.npy', is_cell_new)


def snr_neuron(folder_suite2p: Path) -> np.array:
    """
    function to find snr of a cell
    :param folder_suite2p: folder where the files are stored
    :return: array with the snr of each neuron
    """
    Fneu = np.load(Path(folder_suite2p) / 'Fneu.npy')
    F_raw = np.load(Path(folder_suite2p) / 'F.npy')
    power_signal_all = np.nanmean(np.square(F_raw), 1)
    power_noise_all = np.nanmean(np.square(Fneu), 1)

    # Calculate the SNR
    snr = 10 * np.log10(power_signal_all / power_noise_all)

    # Access the PlaneSegmentation object
    plane_segmentation = nwbfile.processing['ophys'] \
        .data_interfaces['ImageSegmentation'] \
        .plane_segmentations['PlaneSegmentation']

    # Add the 'snr' column
    plane_segmentation.add_column(
        name='snr',
        description='Signal-to-noise ratio for each ROI',
        data=snr_data
    )

    # Write the changes back to the file
    io.write(nwbfile)

    return snr


def stability_neuron(folder_suite2p: Path, init: int = 0, end: Optional[int] = None,
                     low_pass_std: float = 1) -> np.array:
    """
    function to obtain the stability of all the neurons in F_raw given by changes on mean and low_pass std
    :param folder_suite2p: folder where the files are stored
    :param init: initial frame to consider
    :param end: last frame to consider
    :param low_pass_std: the threshold to consider for the low pass check
    :return: array of bools to show stability of each neuron
    """
    F_raw = np.load(Path(folder_suite2p) / 'F.npy')
    if end is None:
        end = F_raw.shape[1]
    try:
        bad_frames_dict = np.load(folder_suite2p / 'bad_frames_dict.npy', allow_pickle=True).take(0)
        bad_frames_bool = bad_frames_dict['bad_frames_bool'][init:end]
    except FileNotFoundError:
        bad_frames_bool = np.zeros(F_raw.shape[1], dtype=bool)[init:end]
    F_to_analyze = F_raw[:, init:end]
    F_to_analyze = F_to_analyze[:, ~bad_frames_bool]
    arr_stab = np.zeros(F_to_analyze.shape[0], dtype=bool)
    for i in np.arange(F_to_analyze.shape[0]):
        arr_stab[i] = check_arr_stability(F_to_analyze[i, :]) and \
                      np.std(low_pass_arr(F_to_analyze[i, :])) < low_pass_std
    return arr_stab

