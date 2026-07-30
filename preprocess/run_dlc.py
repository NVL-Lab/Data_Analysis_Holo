import deeplabcut as dlc
from typing import Optional
from pathlib import Path

def get_dlc_settings():
    project_settings = {
        'project': '191005_NVI12_D10',
        'experimenter': 'Nuria',
        'videos': [
            '/data/project/nvl_lab/HoloBMI/Behavior/191003/NVI13/base/video_2019-10-03T16_03_57_corrected.avi'
        ],
        'working_directory': '/home/sgurgua4/Downloads/dlc_test',
        'copy_videos': False, # False (default)
        'video_type': 'avi',
        'multianimal': False, # False (default)
        'individuals': None   # None (default); relevant only if multianimal is True
    }

    frame_settings = {
        # manual extraction does not involve the rest of the options
        'extract': {
            'mode': 'automatic',        # [automatic (default), manual]
            'algo': 'kmeans',           # [kmeans (default), uniform]
            'crop': False,              # False (default)
            'userfeedback': True,       # True (default)
            'cluster_step': 1,          # 1 (default)
            'cluster_resizewidth': 30,  # 30 (default)
            'cluster_color': False,     # False (default)
            'opencv': True,             # True (default)
            'slider_width': 25,         # 25 (default)
            'config3d': None,           # None (default)
            'extracted_cam': 0,         # 0 (default)
            'videos_list': None         # None (default)
        },
        'label': {
            'image_folder': None # None (default)
        },
        'check': {
            'Labels': ['+', '.', 'x'],  # '+' (default)
            'scale': 1,                 # 1 (default)
            'dpi': 100,                 # 100 (default)
            'draw_skeleton': True,      # True (default)
            'visualizeindviduals': True # '+' (default)
        }
    }

    training_data_settings = {
        'new': {
            'num_shuffles': 1,          # 1 (default)
            'Shuffles': None,           # None (default)
            'windows2linux': False,     # False (default)
            'userfeedback': True,       # True (default)
            'trainIndices': None,       # None (default)
            'testIndices': None,        # None (default)
            'net_type': None,           # None (default); for tensorflow and pytorch
            'detector_type': None,      # None (default); only for pyTorch engine - ssdlite default
            'augmenter_type': None,     # None (default); for tensorflow and pytorch
            'posecfg_template': None,   # None (default); only for tensorflow
            'superanimal_name': '',     # '' (default); only for ternsorflow
            'weight_init': None,        #'WeightInitialization | None' = None, # None (default); only for pytorch
            'engine': None,             #'Engine | None' = None, # None (default); for tenserflow and pythorch - deeplabcut.compat.DEFAULT_ENGINE default
            'ctd_conditions': None      #'int | str | Path | tuple[int, str] | tuple[int, int] | None' = None # None (default); for only for pytorch and must be specified if net_type=ctd...
        },
        'existing': {
            'from_shuffle': -1,         # int (no default); create multiple training datasets to benchmark the performance of different training settings
            'from_trainsetindex': 0,    # 0 (default)
            'num_shuffles': 1,          # 1 (default)
            'shuffles': None,           # None (default)
            'userfeedback': True,       # True (default)
            'net_type': None,           # None (default)
            'detector_type': None,      # None (default)
            'augmenter_type': None,     # None (default)
            'ctd_conditions': None,     # None (default)
            'posecfg_template': None,   # None (default)
            'superanimal_name': '',     # '' (default)
            'weight_init': None,        # None (default)
            'engine': None              # None (default)
        }
    }

    network_settings = {
        'train': {
            'shuffle': 1,                           # 1 (default)
            'trainingsetindex': 0,                  # 0 (default)
            'max_snapshots_to_keep': None,          # None (default)
            'displayiters': None,                   # None (default)
            'saveiters': None,                      # None (default)
            'maxiters': None,                       # None (default)
            'epochs': None,                         # None (default)
            'save_epochs': None,                    # None (default)
            'allow_growth': True,                   # True (default)
            'gputouse': None,                       # None (default)
            'autotune': False,                      # False (default)
            'keepdeconvweights': True,              # True (default)
            'modelprefix': '',                      # '' (default)
            'superanimal_name': '',                 # '' (default)
            'superanimal_transfer_learning': False, # False (default)
            'engine': None,                         # None (default)
            'device': None,                         # None (default)
            'snapshot_path': None,                  # None (default)
            'detector_path': None,                  # None (default)
            'batch_size': None,                     # None (default)
            'detector_batch_size': None,            # None (default)
            'detector_epochs': None,                # None (default)
            'detector_save_epochs': None,           # None (default)
            'pose_threshold': 0.1,                  # 0.1 (default)
            'pytorch_cfg_updates': None             # None (default)
        },
        'evaluate': {
            'Shuffles': (1,),                   #'Iterable[int]' = (1,) - default
            'trainingsetindex': 0,              #'int | str' = 0, # 0 (default)
            'plotting': False,                  # 'bool | str' = False, # False (default)
            'show_errors': True,                # True (default)
            'comparisonbodyparts': 'all',       #'str | list[str]' = 'all', # 'all' (default)
            'gputouse': None,                   #'str | None' = None, # None (default)
            'rescale': False,                   # False (default)
            'modelprefix': '',                  # '' (default)
            'per_keypoint_evaluation': False,   # False (default)
            'snapshots_to_evaluate': None,      #'list[str] | None' = None, # None (default)
            'pcutoff': None,                    #'float | list[float] | dict[str, float] | None' = None, # None (default)
            'engine': None                      # None (default)
            #** torch_kwargs
        }
    }

    save_maps_settings = {
        'shuffle': 1,                       # 1 (default)
        'trainingsetindex': 0,              # 0 (default)
        'comparisonbodyparts': 'all',       #'str | list[str]' = 'all' - default
        'extract_paf': True,                # True (default)
        'all_paf_in_one': True,             # True (default)
        'gputouse': None,                   #'int | None' = None - default
        'device': None,                     #'str | None' = None - default
        'rescale': False,                   #'bool' = False - default
        'Indices': None,                    #'list[int] | None' = None - default
        'modelprefix': '',                  # '' (default)
        'dest_folder': None,                #'str' = None - default
        'snapshot_index': None,             #'int | str | None' = None - default
        'detector_snapshot_index': None,    #'int | str | None' = None - default
        'engine': None                      # None (default)
    }

    analysis_settings = {
        'analyze': {
            'videotype': '',            #'str' = ''
            'shuffle': 1,               #'int' = 1
            'trainingsetindex': 0,      #'int' = 0,
            'gputouse': None,           #'str | None' = None,
            'save_as_csv': False,       #'bool' = False,
            'in_random_order': True,    #'bool' = True,
            'destfolder': None,         #'str | None' = None,
            'batchsize': None,          #'int' = None,
            'cropping': None,           #'list[int] | None' = None,
            'TFGPUinference': True,     #'bool' = True,
            'dynamic': (False, 0.5, 10),#'tuple[bool, float, int]' = (False, 0.5, 10),
            'modelprefix': '',          #'str' = '',
            'robust_nframes': False,    #'bool' = False,
            'allow_growth': False,      #'bool' = False,
            'use_shelve': False,        #'bool' = False,
            'auto_track': True,         #'bool' = True,
            'n_tracks': None,           #'int | None' = None,
            'animal_names': None,       #'list[str] | None' = None,
            'calibrate': False,         #'bool' = False,
            'identity_only': False,     #'bool' = False,
            'use_openvino': None,       #'str | None' = None,
            'engine': None              #'Engine | None' = None,
            #** torch_kwargs
        },
        'filter': {
            'videotype': 'avi',     # '', default
            'shuffle': 1,           # 1, default
            'trainingsetindex': 0,  # 0, default
            'filtertype': 'median', # 'median' (default)
            'windowlength': 5,      # 5, default
            'p_bound': 0.001,       # 0.001, default
            'ARdegree': 3,          # 3, default
            'MAdegree': 1,          # 1, default
            'alpha': 0.01,          # 0.01, default
            'save_as_csv': True,    # True, default
            'destfolder': None,     # None, default
            'modelprefix': '',      # '', default
            'track_method': '',     # '', default
            'return_data': False,   # False, default
            #** kwargs
        },
        'create_label': {
            'videotype': 'avi',             #'str' = '',
            'shuffle': 1,                   #'int' = 1,
            'trainingsetindex': 0,          #'int' = 0,
            'filtered': False,              #'bool' = False,
            'fastmode': True,               #'bool' = True,
            'save_frames': False,           #'bool' = False,
            'keypoints_only': False,        #'bool' = False,
            'Frames2plot': None,            #'list[int] | None' = None,
            'displayedbodyparts': 'all',    #'list[str] | str' = 'all',
            'displayedindividuals': 'all',  #'list[str] | str' = 'all',
            'codec': 'mp4v',                #'str' = 'mp4v',
            'outputframerate': None,        # 'int | None' = None,
            'destfolder': None,             #'Path | str | None' = None,
            'draw_skeleton': False,         #'bool' = False,
            'trailpoints': 0,               #'int' = 0,
            'displaycropped': False,        #'bool' = False,
            'color_by': 'bodypart',         #str' = 'bodypart',
            'modelprefix': '',              #'str' = '',
            'init_weights': '',             #'str' = '',
            'track_method': '',             #'str' = '',
            'superanimal_name': '',         #'str' = '',
            'pcutoff': None,                #'float | None' = None,
            'skeleton': [],                 #'list' = [],
            'skeleton_color': 'white',      #'str' = 'white',
            'dotsize': 8,                   #'int' = 8,
            'colormap': 'rainbow',          #'str' = 'rainbow',
            'alphavalue': 0.05,             #'float' = 0.5,
            'overwrite': False,             #'bool' = False,
            'confidence_to_alpha': False,   #'Union[bool, Callable[[float], float]]' = False,
            'plot_bboxes': True,            #'bool' = True,
            'bboxes_pcutoff': None,         #'float | None' = None,
            #** kwargs
        },
        'plot': {
            'videotype': 'avi',             # '' (default)
            'shuffle': 1,                   # 1 (default)
            'trainingsetindex': 0,          # 0 (default)
            'filtered': False,              # False (default)
            'displayedbodyparts': 'all',    # 'all' (default)
            'displayedindividuals': 'all',  #'all' (default)
            'showfigures': False,           # False (default)
            'destfolder': None,             # None (default)
            'modelprefix': '',              # '' (default)
            'imagetype': '.png',            # '.png' (default)
            'resolution': 100,              # 100 (default)
            'linewidth': 1.0,               # 1.0 (default)
            'track_method': '',             # '' (default)
            'pcutoff': None                 #'float | None' = None,
            #** kwargs
        },
        'skeleton': {
            'videotype': 'avi',     # '' (default)
            'shuffle': 1,           # 1 (default)
            'trainingsetindex': 0,  # 0 (default)
            'filtered': False,      # False (default)
            'save_as_csv': False,   # False (default)
            'destfolder': None,     # None (default)
            'modelprefix': '',      # '' (default)
            'track_method': '',     # '' (default)
            'return_data': False,   # False (default)
            #** kwargs
        }
    }

    refinement_settings = {
        'extract_outliers': {
            'videotype': 'avi',             # '' (default)
            'shuffle': 1,                   # 1 (default)
            'trainingsetindex': 0,          # 0 (default)
            'outlieralgorithm': 'jump',     # 'jump' (default); should look at manual after for visual inspection
            'frames2use': None,             # None (default)
            'comparisonbodyparts': 'all',   # 'all' (default)
            'epsilon': 20,                  # 20 (default)
            'p_bound': 0.01,                # 0.01 (default)
            'ARdegree': 3,                  # 3 (default)
            'MAdegree': 1,                  # 1 (default)
            'alpha': 0.01,                  # 0.01 (default)
            'extractionalgorithm': 'kmeans',# 'kmeans' (default)
            'automatic': False,             # False (default)
            'cluster_resizewidth': 30,      # 30 (default)
            'cluster_color': False,         # False (default)
            'opencv': True,                 # True (default)
            'savelabeled': False,           # False (default)
            'copy_videos': False,           # False (default)
            'destfolder': None,             # None (default)
            'modelprefix': '',              # '' (default)
            'track_method': '',             # '' (default)
            #** kwargs
        },
        'refine_labels': {
            'image_folder': None #'str | None' = None
        },
        'merge_datasets': {
            'forceiterate': None # None (default)
        },
    }

    return project_settings, frame_settings, training_data_settings, network_settings, save_maps_settings, analysis_settings, refinement_settings

def main(project_settings: dict, frame_settings: dict, training_data_settings: dict, network_settings: dict, save_maps_settings:dict, analysis_settings: dict, refinement_settings: dict):

    # =====================================================
    # Phase 1: Project Setup
    # =====================================================

    # project creation
    config_path = dlc.create_new_project(**project_settings)

    print('Config file:', config_path)

    # EDIT config.yaml

    # =====================================================
    # Phase 2: Data preparation
    # =====================================================

    dlc.extract_frames(config_path, **frame_settings['extract'])
    dlc.label_frames(config_path, **frame_settings['label'])
    dlc.check_labels(config_path, **frame_settings['check'])

    # =====================================================
    # Phase 3: Training and Evaluation
    # =====================================================

    # pytorch_config.yaml in train and pose_cfg.yaml in test for PyTorch models
    # pose_cfg.yaml in train and test for TensorFlow models
    # read on Data Augmentation
    dlc.create_training_dataset(config_path, **training_data_settings['new'])

    # Added in version 3.0.0: You can now create new shuffles using the same train/test split as existing shuffle
    if training_data_settings['existing']['from_shuffle'] != -1:
        dlc.create_training_dataset_from_existing_split(config_path, **training_data_settings['existing'])

    dlc.train_network(config_path, **network_settings['train'])
    dlc.evaluate_network(config_path, **network_settings['evaluate'])

    # cont
    # You can also plot the scoremaps, locref layers, and PAFs
    save_maps = True
    if save_maps:
        dlc.extract_save_all_maps(config_path, **save_maps_settings)

    # =====================================================
    # Phase 4: Analysis
    # =====================================================

    test_videos = [
        r"/data/new_videos/session1.mp4",
        r"/data/new_videos/session2.mp4",
    ]

    dlc.analyze_videos(config_path, test_videos, **analysis_settings['analyze'])
    dlc.filterpredictions(config_path, test_videos, **analysis_settings['filter'])
    dlc.create_labeled_video(config_path, test_videos, **analysis_settings['create_label'])
    dlc.plot_trajectories(config_path, test_videos, **analysis_settings['plot'])
    dlc.analyzeskeleton(config_path, test_videos, **analysis_settings['skeleton'])

    # =====================================================
    # Phase 4: Refinement (optional)
    # =====================================================
    dlc.extract_outlier_frames(config_path, **refinement_settings['extract_outliers'])
    # do manual after?
    dlc.refine_labels(config_path, **refinement_settings['refine_labels'])
    dlc.merge_datasets(config_path, **refinement_settings['merge_datasets'])

    print('Finished!')

if __name__ == '__main__':
    main()