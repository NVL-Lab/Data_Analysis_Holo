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
        'shuffle': 1,
        'trainingsetindex': 0,
        'comparisonbodyparts': 'all', #'str | list[str]' = 'all',
        'extract_paf': True,
        'all_paf_in_one': True,
        'gputouse': None, #'int | None' = None,
        'device': None, #'str | None' = None,
        'rescale': None, #'bool' = False,
        'Indices': None, #'list[int] | None' = None,
        'modelprefix': '',
        'dest_folder': None, #'str' = None,
        'snapshot_index': None, #'int | str | None' = None,
        'detector_snapshot_index': None, #'int | str | None' = None,
        'engine': None
    }

    analysis_settings = {
        'analyze': {},
        'filter': {},
        'plot': {},
        'create_label': {},
        'skeleton': {}
    }

    refinement_settings = {
        'extract_outliers': {},
        'refine_labels': {},
        'merge_datasets': {},
    }

    return project_settings, frame_settings, training_data_settings, network_settings, save_maps_settings, analysis_settings, refinement_settings

def main(project_settings: dict, frame_settings: dict, network_settings: dict, analysis_settings: dict, refinement_settings: dict):

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

    dlc.extract_frames(
        config_path,
        **frame_settings['extract']
    )

    dlc.label_frames(
        config_path,
        **frame_settings['label']
    )

    dlc.check_labels(
        config_path,
        **frame_settings['check']
    )

    # =====================================================
    # Phase 3: Training and Evaluation
    # =====================================================

    # pytorch_config.yaml in train and pose_cfg.yaml in test for PyTorch models
    # pose_cfg.yaml in train and test for TensorFlow models
    # read on Data Augmentation
    dlc.create_training_dataset(
        config_path,
        **training_data_settings['new']
    )

    # Added in version 3.0.0: You can now create new shuffles using the same train/test split as existing shuffle
    if training_data_settings['existing']['from_shuffle'] != -1:
        dlc.create_training_dataset_from_existing_split(
            config_path,
            **training_data_settings['existing']
        )

    dlc.train_network(
        config_path,
        **network_settings['train']
    )

    dlc.evaluate_network(
        config_path,
        **network_settings['evaluate']
    )

    # cont
    # You can also plot the scoremaps, locref layers, and PAFs
    #dlc.extract_save_all_maps(config_path, shuffle=shuffle, Indices=[0, 5])

    # =====================================================
    # Phase 4: Analysis
    # =====================================================

    test_videos = [
        r"/data/new_videos/session1.mp4",
        r"/data/new_videos/session2.mp4",
    ]

    deeplabcut.analyze_videos(
        config_path,
        test_videos,
        save_as_csv=True,
    )

    deeplabcut.filterpredictions(
        config_path,
        test_videos,
    )

    deeplabcut.create_labeled_video(
        config_path,
        test_videos,
    )

    deeplabcut.plot_trajectories(
        config_path,
        test_videos,
    )

    # =====================================================
    # Phase 4: Refinement (optional)
    # =====================================================
    #deeplabcut.extract_outlier_frames(config_path, ["videofile_path"])
    #deeplabcut.extract_outlier_frames(config_path, ["videofile_path"], outlieralgorithm="manual")
    #deeplabcut.refine_labels(config_path)
    #deeplabcut.merge_datasets(config_path)

    print('Finished!')

if __name__ == '__main__':
    main()