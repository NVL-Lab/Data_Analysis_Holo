__author__ = 'Saul'

import sys
import numpy as np
import suite2p
from utils.suite2p_v1_config import get_suite2p_holo_db

if __name__ == '__main__':
    data_path = sys.argv[1]
    save_path = sys.argv[2]

    db = get_suite2p_holo_db([data_path], save_path, [], False)
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
    settings['classification']['preclassify'] = 0.5  # 1.

    suite2p.run_s2p(db, settings)