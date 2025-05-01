__author__ = 'Nuria'

# __author__ = ("Nuria", "John Doe")

import pandas as pd
import numpy as np

from pathlib import Path
from typing import Tuple, Optional

from suite2p.run_s2p import run_s2p

from preprocess import syncronize_voltage_rec as svr


def obtain_bad_frames_from_voltage_rec(voltage_rec_paths: list[str], frame_rate: float,
                                       size_recordings: list[int]) -> Tuple[np.array, np.array]:
    """ function to return the bad_frames of a session based on the voltage recording file
    inside the folders given as folder_im_paths.
    This function is used to put together all voltage recording of a session and obtain the bad_frames
    of the whole session (by combining the voltage_recordings files)
    :param voltage_rec_paths: list of voltage_recording_paths to open
    :param frame_rate: frame rate of the recording
    :param size_recordings: the size of the recordings
    return: a np array with the indices of all voltage_recording_paths combined, related to the two photon frames
    """
    indices = [[]]
    len_recording = 0
    for voltage_recording, recording_size in zip(voltage_rec_paths, size_recordings):
        _, _, peaks_I1, _, _, _, _, peaks_I6, _, _ = (
            svr.obtain_peaks_voltage(voltage_recording, frame_rate, recording_size))
        indices_for_6 = svr.obtain_indices_per_peaks(peaks_I1, peaks_I6)
        if len(indices_for_6) > 0:
            if indices_for_6[0] < 5:
                indices_for_6 = indices_for_6[1:]
        indices.append(indices_for_6 + len_recording)
        len_recording += recording_size
    stim_index = np.concatenate(indices)
    bad_frames_index = np.unique(np.concatenate([stim_index - 1, stim_index, stim_index + 1])).astype(int)
    bad_frames_bool = np.zeros(len_recording, dtype=bool)
    if len(bad_frames_index) > 0:
        bad_frames_index.sort()
        bad_frames_bool[bad_frames_index] = 1
    return bad_frames_index, bad_frames_bool


def prepare_ops_1st_pass(default_path: Path, ops_path: Path, bad_frames: np.array = np.empty(0)) -> dict:
    """ Function to modify the default ops file before 1st pass"""
    aux_ops = np.load(Path(default_path) / "default_ops.npy", allow_pickle=True)
    ops = aux_ops.take(0)
    # ops['nwb_series'] = 'TwoPhotonSeries'
    # ops['nwb_file'] = nwb_filepath
    if len(bad_frames) > 0:
        ops['badframes'] = bad_frames
    np.save(ops_path, ops, allow_pickle=True)
    return ops


def process_1_session_suite2p_offline(default_path: Path, folder_suite2p: Path, folder_im_paths: list[str],
                                      voltage_rec_paths: list[str], size_recordings: list[int],
                                      frame_rate: float):
    """ Function to process suite2p offline with a tone of info needed
    :param default_path: path where the default ops are stored
    :param folder_suite2p: path where to store the result of suite2p
    :param folder_im_paths: list of paths where the images to be concatenated are
    :param voltage_rec_paths: list of paths of the voltage recordings needed. same size as folder_im_paths
    :param size_recordings: list of the sizes of the recordings
    :param frame_rate: frame rate of the recording"""

    if len(folder_im_paths) != len(size_recordings) | len(folder_im_paths) != len(voltage_rec_paths):
        raise ValueError("The sizes of the list need to be all the same")

    db = {
        'data_path': folder_im_paths,
        'save_path0': str(folder_suite2p),
        'fast_disk': str(folder_suite2p)
    }
    bad_frames, _ = obtain_bad_frames_from_voltage_rec(voltage_rec_paths, frame_rate, size_recordings)
    np.save(Path(folder_im_paths[0]) / 'bad_frames.npy', bad_frames)
    ops_1st_pass = prepare_ops_1st_pass(default_path, folder_suite2p / 'ops_before_1st.npy', bad_frames)
    ops_after_1st_pass = run_s2p(ops_1st_pass, db)
    np.save(folder_suite2p / 'ops_after_1st_pass.npy', ops_after_1st_pass, allow_pickle=True)
