__author__ = ('Saul', 'Nuria')

import pandas as pd
import numpy as np

from pathlib import Path
from typing import List, Tuple, Optional

import suite2p

import syncronize_voltage_rec as svr

def obtain_bad_frames_from_voltage_rec(voltage_rec_paths: List[str], frame_rate: float, size_recordings: List[int]) -> Tuple[np.array, np.array]:
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
        bad_frames_bool[bad_frames_index] = True

    return bad_frames_index, bad_frames_bool


def prepare_db(im_dirs, suite2p_save_path, bad_frames, bad_frames_bool):
    """
        Modifies the default db before run
    """
    db = suite2p.default_db()

    db['data_path'] = im_dirs
    db['keep_movie_raw'] = True
    db['save_path0'] = str(suite2p_save_path)
    db['fast_disk'] = db['save_path0']
    
    if bad_frames_bool.any():
        np.save(Path(db['data_path'][0]) / 'bad_frames.npy', bad_frames)
        db['bad_frames'] = bad_frames

    return db

def prepare_settings(settings = suite2p.default_settings()) -> dict:
    """
        Modifies the default settings before run
    """
   
    # General settings
    settings['torch_device'] = 'cuda'
    settings['tau'] = 1.5
    settings['fs'] = 29.752 # HoloBMI 
    settings['diameter'] = [22., 22.]
    
    # Pipeline steps to run
    settings['run']['do_registration'] = True
    settings['run']['do_regmetrics'] = True
    settings['run']['do_detection'] = True
    settings['run']['do_deconvolution'] = True
    settings['run']['multiplane_parallel'] = False

    # File input/output settings
    settings['io']['save_NWB'] = True
    settings['io']['save_ops_orig'] = True

    # Registration settings
    settings['registration']['do_bidiphase'] = True
    settings['registration']['bidiphase'] = 0 # Offset default
    settings['registration']['smooth_sigma'] = 1.
    settings['registration']['two_step_registration'] = True 

    # ROI Detection settings
    settings['detection']['algorithm'] = 'sparsery' #['sparsery', 'sourcery', 'cellpose']
    settings['detection']['sparsery_settings']['spatial_scale'] = 0 # defaut
    settings['detection']['highpass_time'] = 100 # default
    settings['detection']['threshold_scaling'] = .7 

    # Cell classification
    settings['classification']['use_builtin_classifier'] = True
    settings['classification']['preclassify'] = 0.5

    return settings


def process_1_session_suite2p_offline(im_dirs: List[str], voltage_rec_dirs: List[str], size_recordings: List[int], frame_rate: float, suite2p_save_path: Path, default_settings_dir: str = ''):
    """
        Function to process suite2p offline with a tone of info needed
            :param im_dirs: list of paths where the images to be concatenated are
            :param voltage_rec_dirs: list of paths of the voltage recordings needed. same size as folder_im_paths
            :param size_recordings: list of the sizes of the recordings
            :param frame_rate: frame rate of the recording
            :param suite2p_save_path: path where to store the result of suite2p
            :param default_settings_dir: path where the default ops are stored
    """

    if len(im_dirs) != len(size_recordings) | len(im_dirs) != len(voltage_rec_paths):
        raise ValueError('The sizes of the list need to be all the same')

    bad_frames, bad_frames_bool = obtain_bad_frames_from_voltage_rec(voltage_rec_dirs, frame_rate, size_recordings)
    db = prepare_db(im_dirs, suite2p_save_path, bad_frames, bad_frames_bool)
    
    if default_settings_dir:
        settings = np.load(Path(default_settings_dir) / 'default_settings.npy', allow_pickle=True)
        settings = prepare_settings(settings)
    else:
        settings = prepare_settings()

    suite2p.run_s2p(db, settings)
