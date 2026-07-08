__author__ = 'Saul'

import sys
import numpy as np
import importlib
import suite2p
from pathlib import Path
from utils.suite2p_v1_config import get_suite2p_holo_db
import contextlib
import os

if __name__ == '__main__':
    s2p = importlib.import_module("suite2p.run_s2p")
    data_path = sys.argv[1]
    save_path = sys.argv[2]

    Path(save_path).mkdir(parents=True, exist_ok=True)
    db = get_suite2p_holo_db([data_path], save_path, [], np.array([False]))
    db = suite2p.io.init_dbs(db)[0] # dirs are made here
    settings = suite2p.default_settings()

    # General settings
    settings['torch_device'] = 'cuda'
    settings['fs'] = 38.6

    # File input/output settings
    settings['io']['save_ops_orig'] = True

    # Registration settings
    #settings['registration']['do_bidiphase'] = True  # False
    #settings['registration']['two_step_registration'] = True  # db['keep_movie_raw'] needs to be True

    # Cell classification
    #settings['classification']['preclassify'] = 0.5  # 1.

    # Data dimensions
    f_npy = np.load(data_path)#.astype(np.float32)
    suite2p.io.BinaryFile.convert_numpy_file_to_suite2p_binary(data_path, db['raw_file'])
    # Need by the GUI
    #db['reg_file'] = f'{save_path}/data.bin'
    #db['save_path'] = save_path
    db['input_format'] = 'bin'
    db['nframes'] = f_npy.shape[0]
    db['Ly'] = f_npy.shape[1]
    db['Lx'] = f_npy.shape[2]
    twoc = db['nchannels'] > 1
    reg_file_chan2 = db["reg_file_chan2"] if twoc else None
    raw_file_chan2 = db.get("raw_file_chan2", None) if twoc else None
    #db['raw_file'] = data_path
    raw = db["keep_movie_raw"] and os.path.isfile(db['raw_file'])
    badframes0 = np.zeros(db["nframes"], "bool")
    device = s2p._assign_torch_device(settings["torch_device"])
    run_registration = s2p._check_run_registration(settings, db)

    np.save(f'{db["save_path0"]}/{db["save_folder"]}/db.npy', db)
    np.save(f'{db["save_path0"]}/{db["save_folder"]}/settings.npy', settings)

    s2p.logger_setup(save_path)
    null = contextlib.nullcontext()
    with suite2p.io.BinaryFile(Ly=db['Ly'], Lx=db['Lx'], filename=db['raw_file'], n_frames=db['nframes'], write=False) \
            if raw else null as f_raw, \
        suite2p.io.BinaryFile(Ly=db['Ly'], Lx=db['Lx'], filename=db['reg_file'], n_frames=db['nframes'], write=True) as f_reg, \
        suite2p.io.BinaryFile(Ly=db['Ly'], Lx=db['Lx'], filename=raw_file_chan2, n_frames=n_frames, write=False) \
            if raw and twoc else null as f_raw_chan2, \
        suite2p.io.BinaryFile(Ly=db['Ly'], Lx=db['Lx'], filename=reg_file_chan2, n_frames=n_frames, write=True) \
            if twoc else null as f_reg_chan2:

        outputs = suite2p.pipeline(db['save_path'], f_reg, f_raw, f_reg_chan2, f_raw_chan2,
                   run_registration, settings, badframes=badframes0, stat=None, device=device, Zstack=None)

    #suite2p.run_s2p(db, settings)
