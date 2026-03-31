__author__ = 'Saul'

import numpy as np
from pathlib import Path
import suite2p

def get_suite2p_holo_db(im_dirs, suite2p_save_path, bad_frames, bad_frames_bool):
    """  
        Database configurations for suite2p (v1.0.0.1 onwards)
    """
    
    db = suite2p.default_db()

    # Database
    db['data_path'] = im_dirs
    db['keep_movie_raw'] = True # False
    db['save_path0'] = str(suite2p_save_path)
    db['fast_disk'] = db['save_path0']

    if bad_frames_bool.any():
        np.save(Path(db['data_path'][0]) / 'bad_frames.npy', bad_frames)
        db['bad_frames'] = bad_frames

    return db


def get_suite2p_holo_settings():
    """  
        Settings configurations for suite2p (v1.0.0.1 onwards)
    """
    
    settings = suite2p.default_settings()

    # General settings
    settings['torch_device'] = 'cuda'
    settings['tau'] = 1.5 # 1.
    settings['fs'] = 29.752 # HoloBMI 
    
    # Pipeline steps to run (All are default)
    settings['run']['do_registration'] = 1
    settings['run']['do_regmetrics'] = True
    settings['run']['do_detection'] = True
    settings['run']['do_deconvolution'] = True
    settings['run']['multiplane_parallel'] = False

    # File input/output settings
    settings['io']['save_NWB'] = True # False
    settings['io']['save_ops_orig'] = True

    # Registration settings
    settings['registration']['do_bidiphase'] = True # False
    settings['registration']['bidiphase'] = 0
    settings['registration']['smooth_sigma'] = 1.15 #1.
    settings['registration']['two_step_registration'] = True # db['keep_movie_raw'] needs to be True

    # ROI Detection settings
    settings['detection']['algorithm'] = 'sparsery'
    settings['detection']['highpass_time'] = 100
    settings['detection']['threshold_scaling'] = .7 # 1.
    settings['detection']['sparsery_settings']['spatial_scale'] = 0

    # Cell classification
    settings['classification']['preclassify'] = 0.5 #1.

    return settings

def get_suite2p_holo_settings2():
    """  
        Settings configurations for suite2p (v1.0.0.1 onwards)
    """
    
    settings = suite2p.default_settings()

    # General settings
    settings['torch_device'] = 'cuda'
    settings['tau'] = 1.5 # 1.
    settings['fs'] = 29.752 # HoloBMI 
    #settings['diameter'] = [22., 22.] # [12., 12.]
    
    # Pipeline steps to run (All are default)
    settings['run']['do_registration'] = 1
    settings['run']['do_regmetrics'] = True
    settings['run']['do_detection'] = True
    settings['run']['do_deconvolution'] = True
    settings['run']['multiplane_parallel'] = False

    # File input/output settings
    settings['io']['save_NWB'] = True # False
    settings['io']['save_ops_orig'] = True

    # Registration settings
    settings['registration']['do_bidiphase'] = True # False
    settings['registration']['bidiphase'] = 0
    settings['registration']['smooth_sigma'] = 1.15 #1.
    settings['registration']['two_step_registration'] = True # db['keep_movie_raw'] needs to be True

    # ROI Detection settings
    settings['detection']['algorithm'] = 'sparsery'
    settings['detection']['highpass_time'] = 100
    settings['detection']['threshold_scaling'] = .7 # 1.
    settings['detection']['sparsery_settings']['spatial_scale'] = 0

    # Cell classification
    settings['classification']['preclassify'] = 0.5 #1.

    return settings

def get_suite2p_holo_settings1():
    """  
        Settings configurations for suite2p (v1.0.0.1 onwards)
    """
    
    settings = suite2p.default_settings()

    # General settings
    settings['torch_device'] = 'cuda'
    #settings['tau'] = 1.5 # 1.
    settings['fs'] = 29.752 # HoloBMI 
    #settings['diameter'] = [22., 22.] # [12., 12.]
    
    # Pipeline steps to run (All are default)
    settings['run']['do_registration'] = 1
    settings['run']['do_regmetrics'] = True
    settings['run']['do_detection'] = True
    settings['run']['do_deconvolution'] = True
    settings['run']['multiplane_parallel'] = False

    # File input/output settings
    settings['io']['save_NWB'] = True # False
    settings['io']['save_ops_orig'] = True

    # Registration settings
    settings['registration']['do_bidiphase'] = True # False
    settings['registration']['bidiphase'] = 0
    #settings['registration']['smooth_sigma'] = 1. # 1.15
    settings['registration']['two_step_registration'] = True # db['keep_movie_raw'] needs to be True

    # ROI Detection settings
    settings['detection']['algorithm'] = 'sparsery'
    settings['detection']['highpass_time'] = 100
    #settings['detection']['threshold_scaling'] = .7 # 1.
    settings['detection']['sparsery_settings']['spatial_scale'] = 0

    # Cell classification
    #settings['classification']['preclassify'] = 0.5 #1.

    return settings
 
def get_suite2p_holo_settings0():
    """  
        Settings configurations for suite2p (v1.0.0.1 onwards)
    """
    
    settings = suite2p.default_settings()

    # General settings
    settings['torch_device'] = 'cuda'
    #settings['tau'] = 1.5 # 1.
    settings['fs'] = 29.752 # HoloBMI 
    #settings['diameter'] = [22., 22.] # [12., 12.]
    
    # Pipeline steps to run (All are default)
    settings['run']['do_registration'] = 1
    settings['run']['do_regmetrics'] = True
    settings['run']['do_detection'] = True
    settings['run']['do_deconvolution'] = True
    settings['run']['multiplane_parallel'] = False

    # File input/output settings
    settings['io']['save_NWB'] = True # False
    settings['io']['save_ops_orig'] = True

    # Registration settings
    settings['registration']['do_bidiphase'] = True # False
    settings['registration']['bidiphase'] = 0
    #settings['registration']['smooth_sigma'] = 1. # 1.15
    settings['registration']['two_step_registration'] = True # db['keep_movie_raw'] needs to be True

    # ROI Detection settings
    settings['detection']['algorithm'] = 'sparsery'
    settings['detection']['highpass_time'] = 100
    #settings['detection']['threshold_scaling'] = .7 # 1.
    settings['detection']['sparsery_settings']['spatial_scale'] = 0

    # Cell classification
    #settings['classification']['use_builtin_classifier'] = True #False
    settings['classification']['preclassify'] = 0.5 #1.

    return settings
