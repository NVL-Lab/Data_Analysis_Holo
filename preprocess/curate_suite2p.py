__author__ = ('Nuria', 'Saul')

import os
import shutil
import copy
import math

import numpy as np
import pandas as pd
import scipy.io as sio

from pathlib import Path
from typing import Tuple, Optional
from scipy import signal

from utils.analysis_configuration import AnalysisConfiguration as aconf
from utils.analysis_constants import AnalysisConstants as act
from utils.combine_nwb import combine_indices_nwb

import matplotlib.pyplot as plt
from scipy.io import loadmat

def refine_classifier(folder_suite2p: Path, dn_bl: bool = True):
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

def get_neuron_indexes(es_idxs: list, mat_bin_cells: np.ndarray, s2p_indexes: list, s2p_stat: np.ndarray) -> Tuple[list, list]:
    """
        function that matches neuron indexes from mat files to suite2p detected neurons
        :param es_idxs: indexes of ensemble neurons
        :param mat_bin_cells: 3d array with binarized masks of each ensembele neuron
        :param s2p_indexes: indexes of suite2p detected neurons
        :param s2p_stat: stat data of suite2p detected neurons
    """
    idxs = [None] * len(es_idxs)
    jaccs = [None] * len(es_idxs)
    k = 0
    for i in es_idxs:
        coords = set(zip(np.where(mat_bin_cells[i] != 0)[1], np.where(mat_bin_cells[i] != 0)[0]))
        jacc_temp = 0
        s2p_idx = None

        for j in s2p_indexes:
            s2p_coords = set(zip(s2p_stat[j]['xpix'], s2p_stat[j]['ypix']))
            inter = coords & s2p_coords
            jacc = len(inter) / len(coords | s2p_coords)
            if jacc > jacc_temp:
                jacc_temp = jacc
                s2p_idx = j

        idxs[k] = s2p_idx
        jaccs[k] = jacc_temp
        k += 1

    return idxs, jaccs

def compare_neurons(processed_data_path:Path, raw_data_path:Path, dataset_path: Path) -> None:
    """
        function that helps compare holobmi neurons and suite2p detected neurons
        :param processed_data_path: folder where the suite2p processed files are stored
        :param raw_data_path: folder where the raw holobmi files are stored
        :param dataset_path: folder where the dataset is stored <session_date>/<mouse_id>/<day>
    """

    # loads suite2p processed data
    data_path = Path(processed_data_path) / dataset_path / 'suite2p/plane0'
    regos = np.load(data_path / 'reg_outputs.npy', allow_pickle=True).item()
    iscell = np.load(data_path / 'iscell.npy', allow_pickle=True)
    stat = np.load(data_path / 'stat.npy', allow_pickle=True)

    # loads mat data from raw directory
    mat_path = Path(raw_data_path) / dataset_path
    roi_mat = loadmat(mat_path / 'roi_data.mat')
    bmi_mat = loadmat([f for f in list(mat_path.iterdir()) if f.match('BMI_online*.mat')][-1])

    # Gets roi masks from mat files
    roi_mat_data = roi_mat['roi_data'][0,0]
    roi_bin_cell = roi_mat_data['roi_bin_cell'][0]
    bdata = bmi_mat['bData'][0,0]
    e1_mat = bdata['E1_base'][0]
    e2_mat = bdata['E2_base'][0]
    e1 = e1_mat-1
    e2 = e2_mat-1
    aux_r = [roi_bin_cell[i] for i in e1]
    aux_b = [roi_bin_cell[i] for i in e2]
    r = np.sum(aux_r, axis=0) / 2
    b = np.sum(aux_b, axis=0) / 2

    emean_image = regos['meanImgE']
    # Creates masks for suite2p cells and non-cells
    roi_mask = np.zeros((emean_image.shape[0], emean_image.shape[1]), dtype=np.float32)  # (Lx, Ly)
    cell_indexes = np.where(iscell[:, 0] == 1)[0].tolist()
    for i in cell_indexes:
        j = i+1
        roi_mask[stat[i]['ypix'], stat[i]['xpix']] = j  # + stat[i]['lam']

    roi_mask_false = roi_mask.copy()
    false_cell_indexes = np.where(iscell[:, 0] == 0)[0].tolist()
    for i in false_cell_indexes:
        j = i+1
        roi_mask_false[stat[i]['ypix'], stat[i]['xpix']] = j  # + stat[i]['lam']

    es_idxs = [*e1, *e2] # combines ensemble neuron indexes from mat files
    # gets indexes and jaccard scores for suite2p cell and non-cell detections
    idxs, jaccs = get_neuron_indexes(es_idxs, roi_bin_cell, cell_indexes, stat)
    idxs_false, jaccs_false = get_neuron_indexes(es_idxs, roi_bin_cell, false_cell_indexes, stat)

    # creates masks for ensemble neuron matches from suite2p cells and non-cells
    e1_mask = np.where(np.isin(roi_mask, [x + 1 for x in idxs[:len(e1)] if x is not None]), roi_mask, 0)
    e2_mask = np.where(np.isin(roi_mask, [x + 1 for x in idxs[len(e1):] if x is not None]), roi_mask, 0)
    e1_mask_false = np.where(np.isin(roi_mask_false, [x + 1 for x in idxs_false[:len(e1)] if x is not None]), roi_mask_false, 0)
    e2_mask_false = np.where(np.isin(roi_mask_false, [x + 1 for x in idxs_false[len(e1):] if x is not None]), roi_mask_false, 0)

    print(f'e1: {e1_mat}; jaccard: {jaccs[:len(e1)]}; jaccard_false: {jaccs_false[:len(e1)]}')
    print(f'e2: {e2_mat}; jaccard: {jaccs[len(e1):]}; jaccard_false: {jaccs_false[len(e1):]}')

    fig, axes = plt.subplots(2, 2)
    axes[0, 0].imshow(np.stack([r, emean_image, e1_mask], axis=-1), cmap='bone')
    axes[0, 0].set_title(f'e1: {e1_mat}; s2p: {idxs[:len(e1)]}')
    axes[1, 0].imshow(np.stack([b, emean_image, e2_mask], axis=-1), cmap='bone')
    axes[1, 0].set_title(f'e2: {e2_mat}; s2p: {idxs[len(e1):]}')
    axes[0, 1].imshow(np.stack([r, emean_image, e1_mask_false], axis=-1), cmap='bone')
    axes[0, 1].set_title(f'e1: {e1_mat}; s2p_false: {idxs_false[:len(e1)]}')
    axes[1, 1].imshow(np.stack([b, emean_image, e2_mask_false], axis=-1), cmap='bone')
    axes[1, 1].set_title(f'e2: {e2_mat}; s2p_false: {idxs_false[len(e1):]}')
    plt.show()
    plt.close()