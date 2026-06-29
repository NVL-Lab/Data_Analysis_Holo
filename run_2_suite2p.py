import os
import shutil
import copy
import math

import numpy as np
import pandas as pd
import numpy as np
import scipy.io as sio

from pathlib import Path
from typing import Tuple
from scipy import signal

from suite2p.run_s2p import run_s2p


def obtain_bad_frames_from_fneu(fneu_old: np.array) -> Tuple[np.array, np.array, np.array, np.array, np.array, bool]:
    """ Function to obtain the frames of stim that need to go """
    conv_win = np.ones(AnalysisConfiguration.filter_size)
    window = int(AnalysisConfiguration.filter_size/2)
    aux_bad_frames = []
    st = np.zeros(fneu_old.shape[0])
    aux_fden = np.zeros(fneu_old.shape)
    for neu in np.arange(fneu_old.shape[0]):
    # Fmean = np.nanmean(fneu_old, 0)
        Fconv = signal.fftconvolve(fneu_old[neu,:], conv_win/conv_win.shape, 'valid')
        xx = np.arange(window, Fconv.shape[0]+window)
        poly = np.polyfit(xx, Fconv, 1)
        aux_f = np.zeros(fneu_old.shape[1])
        aux_f[:window] = np.polyval(poly, np.arange(window))
        aux_f[Fconv.shape[0]+window-1:] = np.polyval(poly, np.arange(Fconv.shape[0]-window,Fconv.shape[0]))
        aux_f[window:Fconv.shape[0]+window] = Fconv
        F_denoised = fneu_old[neu,:]-aux_f
        aux_fden[neu, :] = F_denoised
        st[neu] = ((np.percentile(F_denoised[AnalysisConfiguration.index_before_pretrain:
                                           AnalysisConfiguration.index_after_pretrain],
                                AnalysisConfiguration.percentil_threshold)[1]/
                   np.nanstd(F_denoised[AnalysisConfiguration.index_before_pretrain:
                                        AnalysisConfiguration.index_after_pretrain]))/
                   np.nanstd(F_denoised[AnalysisConfiguration.index_after_pretrain:]))
    top_index = np.argsort(st)[::-1]
    Fmean = np.nanmean(aux_fden[top_index[:10],:],0)
    Fmean[Fmean< AnalysisConfiguration.height_stim_artifact * np.nanstd(Fmean[AnalysisConfiguration.index_before_pretrain:
                                                                              AnalysisConfiguration.index_after_pretrain])] = 0
    bad_frames_index = np.where(Fmean > 0)[0]
    bad_frames_index.sort()
    frames_include = np.setdiff1d(np.arange(fneu_old.shape[1]), bad_frames_index)
    bad_frames_bool = np.zeros(fneu_old.shape[1], dtype=bool)
    bad_frames_bool[bad_frames_index] = 1
    stim_index, stim_time_bool = obtain_stim_time(bad_frames_bool)
    if np.sum(stim_index<AnalysisConstants.calibration_frames) > 0:
        sanity_check = True
    else:
        sanity_check = False
    return bad_frames_index, bad_frames_bool, frames_include, stim_index, stim_time_bool, sanity_check


def refine_classifier(folder_suite2p: Path, dn_bool: bool = True):
    """ function to refine the suite2p classifier """
    neurons = np.load(Path(folder_suite2p) / "stat.npy", allow_pickle=True)
    is_cell = np.load(Path(folder_suite2p) / "iscell.npy")
    is_cell_new = copy.deepcopy(is_cell)
    snr_val = snr_neuron(folder_suite2p)
    stable_neuron = stability_neuron(folder_suite2p, init=AnalysisConstants.calibration_frames)
    for nn, neuron in enumerate(neurons):
        if neuron['skew'] > 10 or neuron['skew'] < 0.4 or neuron['compact'] > 1.4 or \
                neuron['footprint'] == 0 or neuron['footprint'] == 3 or neuron['npix'] < 80 or \
                snr_val[nn] < AnalysisConfiguration.snr_min or ~stable_neuron[nn]:
            is_cell_new[nn, :] = [0, 0]
    if dn_bool:
        aux_dn = np.load(Path(folder_suite2p) / "direct_neurons.npy", allow_pickle=True)
        direct_neurons_info = aux_dn.take(0)
        direct_neurons = direct_neurons_info["E1"] + direct_neurons_info["E2"]
        direct_neurons.sort()
        for dn in direct_neurons:
            is_cell_new[dn, :] = [1, 1]
    np.save(Path(folder_suite2p) / "iscell.npy", is_cell_new)

def snr_neuron(folder_suite2p: Path) -> np.array:
    """
    function to find snr of a cell
    :param folder_suite2p: folder where the files are stored
    :return: array with the snr of each neuron
    """
    Fneu = np.load(Path(folder_suite2p) / "Fneu.npy")
    F_raw = np.load(Path(folder_suite2p) / "F.npy")
    power_signal_all = np.nanmean(np.square(F_raw), 1)
    power_noise_all = np.nanmean(np.square(Fneu), 1)

    # Calculate the SNR
    snr = 10 * np.log10(power_signal_all / power_noise_all)
    return snr


def obtain_stim_time(bad_frames_bool: np.array) -> Tuple[np.array, np.array]:
    """ function that reports the time of stim (by returning the first frame of each stim) """
    stim_index = np.insert(np.diff(bad_frames_bool.astype(int)), 0, 0)
    stim_index[stim_index < 1] = 0
    return np.where(stim_index)[0], stim_index.astype(bool)


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
    F_raw = np.load(Path(folder_suite2p) / "F.npy")
    if end is None:
        end = F_raw.shape[1]
    try:
        bad_frames_dict = np.load(folder_suite2p / "bad_frames_dict.npy", allow_pickle=True).take(0)
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
    percentil_threshold: float = [0.1, 99.9]
    index_before_pretrain = seq_holo_frames + calibration_frames
    index_after_pretrain = seq_holo_frames + calibration_frames + pretrain_frames
