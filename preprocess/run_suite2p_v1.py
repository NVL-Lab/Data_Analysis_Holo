__author__ = 'Saul'

import sys
import numpy as np
import suite2p
from pathlib import Path
from utils.suite2p_v1_config import get_suite2p_holo_db
import logging
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    data_path = sys.argv[1]
    save_path = sys.argv[2]

    Path(save_path).mkdir(parents=True, exist_ok=True)
    db = get_suite2p_holo_db([data_path], save_path, [], np.array([False]))
    settings = suite2p.default_settings()

    # General settings
    settings['torch_device'] = 'cuda'
    settings['fs'] = 38.6

    # File input/output settings
    settings['io']['save_ops_orig'] = True

    # Registration settings
    settings['registration']['do_bidiphase'] = True  # False
    settings['registration']['two_step_registration'] = True  # db['keep_movie_raw'] needs to be True

    # Cell classification
    #settings['classification']['preclassify'] = 0.5  # 1.

    # Data dimensions
    f_reg = np.load(data_path).astype(np.float32)
    #settings['Ly'] = rec.shape[1]
    #settings['Lx'] = rec.shape[2]

    # Needed for the extraction and addition of ROIs
    #print('Binarizing data...')
    #f_input = suite2p.io.BinaryFile(Ly=rec.shape[1], Lx=rec.shape[2], filename=data_path)  # reads in data from npy
    #_ = suite2p.io.BinaryFile(Ly=rec.shape[1], Lx=rec.shape[2], filename=save_path / 'data.bin',
    #                          n_frames=rec.shape[0])  # writes data into a bin file
    #print('Registering movie...')

    #print('Detecting ROIs...')
    #ops, stat = suite2p.detection_wrapper(f_reg=image, ops=ops, classfile=suite2p.classification.builtin_classfile)
    #print('Classifying cells...')
    #iscell = suite2p.classification.classify(stat, suite2p.classification.builtin_classfile)
    db['input_format'] = 'npy'
    np.save(f'{save_path}/db.npy', db)
    np.save(f'{save_path}/settings.npy', settings)
    suite2p.pipeline(save_path, f_reg)

    #suite2p.run_s2p(db, settings)
