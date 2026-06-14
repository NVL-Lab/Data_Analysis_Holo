
import collections
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import re
from preprocess.session_paths import _hE2_rew,_hE2_norew,_hE2_rew_fb,_hE3_rew,_No_Reward_Pretrain,_randrew,_randrew_fb,_BMI
from utils.analysis_constants import AnalysisConstants as act
from get_session_paths import get_session_paths

# def get_all_sessions() -> pd.DataFrame:
#     """ function to get a df with all sessions"""
#     df_Holostim = pd.DataFrame(index=np.concatenate(list(_hE2_rew.values())))
#     df_Holostim['experiment_type'] = 'Holostim'
    
#     df_HoloE3 = pd.DataFrame(index=np.concatenate(list(_HoloE3.values())))
#     df_HoloE3['experiment_type'] ='HoloE3'

#     df_HoloE2 = pd.DataFrame(index=np.concatenate(list(_HoloE2.values())))
#     df_HoloE2['experiment_type'] ='HoloE2'

#     df_Random_Reward = pd.DataFrame(index=np.concatenate(list(_Random_Reward.values())))
#     df_Random_Reward['experiment_type'] ='Random_Reward'


#     list_experiments = [df_Holostim,df_HoloE3,df_HoloE2, df_Random_Reward]
#     df_experiments = pd.concat(list_experiments)
#     # print(df_experiments.sort_index().reset_index())
#     return df_experiments.sort_index().reset_index()

# get_all_sessions()
# print(_Holostim.items())

# print(collections.defaultdict(list))



def get_all_sessions(folder_save: str = '') -> pd.DataFrame:
    """ function to put all the experiment types in a dataframe"""
    df = pd.DataFrame()
    experiment_types = ['hE2_rew', 'hE2_norew', 'hE3_rew', 'randrew']
    for experiment in experiment_types:
        df_new = get_sessions_df(experiment)
        df = pd.concat([df, df_new], ignore_index=True)
    if folder_save:
        df.to_csv(folder_save / 'df_holobmi.csv', index=False)
    return df   

def get_sessions_df(experiment_type: str) -> pd.DataFrame:
    experiment_dict = {
        'hE2_rew': _hE2_rew,
        'hE2_norew': _hE2_norew,
        'hE3_rew': _hE3_rew,
        'randrew': _randrew,
        'randrew_fb': _randrew_fb,
        'hE2_rew_fb': _hE2_rew_fb,
        'No_Reward_Pretrain': _No_Reward_Pretrain,
        'BMI': _BMI
    }
  
    if experiment_type is None:
        return get_all_sessions()
    elif experiment_type not in experiment_dict:
        raise ValueError(f'Invalid experiment type: {experiment_type}. Choose from {", ".join(experiment_dict.keys())}')

    ret = collections.defaultdict(list)
    folder_raw = Path('/data/project/nvl_lab/HoloBMI/Raw')

    for mice_name, sessions in experiment_dict[experiment_type].items():
        for day_index, session_path in enumerate(sessions):
            session_date, mice_name, day_init = session_path.split('/')
            ret['mice_name'].append(str(mice_name))
            ret['session_date'].append(str(int(float(session_date)))) 
            ret['day_init'].append(day_init)
            ret['experiment_type'].append(str(experiment_type))
            ret['session_path'].append(str(session_path))
            ret['day_index'].append(str(int(float(day_index))))

            dir_files = folder_raw / session_path
            category_map = {
                'baseline': 'baseline_im',
                'BMI': 'bmi_im',
                'holostim_seq': 'holostim_seq_im'  # Updated naming
            }
            tiff_limits = {
                "holostim_seq": act.seq_holo_frames,
                "baseline": act.calibration_frames,
                "BMI": act.bmi_frames,
                "pretrain": act.bmi_frames  
            }
            # Initialize data columns with None, Flags with False
            for category in category_map.values():
                if f'flag_session_more_tif_{category}' not in ret:
                    ret[f'flag_session_more_tif_{category}'] = []
                ret[category].append(None)
                ret[f'{category}_voltage_file'].append(None)  # Updated voltage naming
                ret[f'flag_{category}'].append(False)  # Flagging for multiple files
                ret[f'flag_session_more_tif_{category}'].append(False)
                ret[f'flag_{category}_voltage_file'].append(False)


            ret['pretrain_im'].append(None)
            ret['flag_pretrain_im'].append(False)
            ret['pretrain_im_voltage_file'].append(None)
            ret['flag_session_more_tif_pretrain'].append(False)
            ret[f'flag_pretrain_im_voltage_file'].append(False)
            # Iterate over session directory
            for file_name in os.listdir(dir_files):
                file_path = dir_files / file_name

                # Look for im folders
                if file_name.startswith('im'):
                    for im_subfolder in os.listdir(file_path):
                        im_subfolder_path = file_path / im_subfolder
                        im_subfolders = os.listdir(file_path)  # Get all subfolders first
                        for key, category in category_map.items():
                            
                            if not any(key in subfolder for subfolder in im_subfolders):
                                    ret[f'flag_{category}'][-1] = True
                            if im_subfolder.startswith(key):
                                if not os.listdir(im_subfolder_path):
                                    ret[f'flag_{category}'][-1] = True
                                    ret[f'flag_session_more_tif_{category}'][-1] = True
                                    continue 
                                im_files = [subfolder for subfolder in os.listdir(im_subfolder_path) if subfolder.startswith(f"{key}_")]
                                parsed_folder_name = f"{key}/{im_files[0]}" if im_files else None
                                # Check if multiple files exist
                                if len(im_files) > 1 or not im_files:
                                    ret[f'flag_{category}'][-1] = True  # Flag session
                                    ret[category][-1] = None  # No single file assigned
                                    ret[f'flag_session_more_tif_{category}'][-1] = False

                                elif len(im_files) == 1:
                                    ret[category][-1] = parsed_folder_name  
                                    ret[f'flag_{category}'][-1] = False  
                                    parsed_folder = im_subfolder_path / im_files[0]
                                    num_tiff_files = sum(1 for file in os.listdir(parsed_folder) if file.endswith(".tif"))

                                    ret[f'flag_session_more_tif_{category}'][-1] = True if num_tiff_files > tiff_limits[key] else False
                                else:
                                    ret[category][-1] = None
                                    ret[f'flag_{category}'][-1] = False  # No flag needed
                                    ret[f'flag_session_more_tif_{category}'][-1] = False 

                                # Look for voltage recording file inside the correct subfolder
                                if im_files:
                                    full_subfolder_name = im_files[0]
                                    voltage_file_pattern = f"{full_subfolder_name}_Cycle00001_VoltageRecording_001.csv"
                                    voltage_file_path = im_subfolder_path / full_subfolder_name / voltage_file_pattern

                                    if voltage_file_path.exists():
                                        ret[f'{category}_voltage_file'][-1] = voltage_file_pattern
                                        ret[f'flag_{category}_voltage_file'][-1] = False
                                    else:
                                        ret[f'{category}_voltage_file'][-1] = None
                                        ret[f'flag_{category}_voltage_file'][-1] = True
                        

                        # Checking if "_pretrain" is missing in any subfolder names
                        if not any("_pretrain" in subfolder for subfolder in im_subfolders):
                            ret['flag_pretrain_im'][-1] = True                    

                        if "_pretrain" in im_subfolder:
                                if not os.listdir(im_subfolder_path):
                                    ret['flag_pretrain_im'][-1] = True
                                    ret['flag_session_more_tif_pretrain'][-1] = True
                                    continue 
                                im_files = [subfolder for subfolder in os.listdir(im_subfolder_path) if "_pretrain" in subfolder]
                                parsed_folder_name = f"{im_subfolder}/{im_files[0]}" if im_files else None

                                if len(im_files) > 1 or not im_files:
                                    ret['flag_pretrain_im'][-1] = True
                                    ret['pretrain_im'][-1] = None
                                    ret['flag_session_more_tif_pretrain'][-1] = False

                                elif len(im_files) == 1:
                                    ret['pretrain_im'][-1] = parsed_folder_name
                                    ret['flag_pretrain_im'][-1] = False
                                    parsed_folder = im_subfolder_path / im_files[0]
                                    num_tiff_files = sum(1 for file in os.listdir(parsed_folder) if file.endswith(".tif"))
                                    if num_tiff_files > tiff_limits['pretrain']:
                                        ret['flag_session_more_tif_pretrain'][-1] = True
                                    else:
                                        ret['flag_session_more_tif_pretrain'][-1] = False

                                else:
                                    ret['pretrain_im'][-1] = None
                                    ret['flag_pretrain_im'][-1] = False   
                                    ret['flag_session_more_tif_pretrain'][-1] = False

                                if im_files:
                                    full_subfolder_name = im_files[0]
                                    pretrain_voltage_file_pattern = f"{full_subfolder_name}_Cycle00001_VoltageRecording_001.csv"
                                    pretrain_voltage_file_path = im_subfolder_path / full_subfolder_name / pretrain_voltage_file_pattern
                                    
                                    if pretrain_voltage_file_path.exists():
                                        ret['pretrain_im_voltage_file'][-1] = pretrain_voltage_file_pattern
                                        ret[f'flag_pretrain_im_voltage_file'][-1] = False
                                    else:
                                        ret['pretrain_im_voltage_file'][-1] = None
                                        ret[f'flag_pretrain_im_voltage_file'][-1] = True


            # Additional file processing
            roi_mat_files = []
            target_files = []
            bmitarget_files = []
            bmi_files = []
            baseline_files=[]
            holostim_seq_files =[]
            mainProt_files=[]

            for file_name in os.listdir(dir_files):
                if file_name.startswith('roi'):
                    roi_mat_files.append(file_name)
                elif file_name[:5] == 'BOT_c':
                    ret['bot_candidates'].append(str(file_name))
                elif file_name[:5] == 'BOT_e':
                    ret['bot_ensemble'].append(str(file_name))
                elif file_name[:5] == 'GPL_c':
                    ret['gpl_candidates'].append(str(file_name))
                elif file_name[:5] == 'GPL_e':
                    ret['gpl_ensemble'].append(str(file_name))
                elif file_name[:5] == 'XML_c':
                    ret['xml_candidates'].append(str(file_name))
                elif file_name[:5] == 'XML_e':
                    ret['xml_ensemble'].append(str(file_name))
                elif file_name.startswith('holostim_seq') and file_name.endswith('.mat'):
                    # ret['Holostim_seq_mat_file'].append(file_name)
                    holostim_seq_files.append(str(file_name))
                elif file_name.startswith('workspace'):
                    ret['workspace_mat_file'].append(str(file_name))
                elif file_name.startswith('holoMask'):
                    ret['holomask_gpl_file'].append(str(file_name))
                elif file_name.startswith('mainProt'):
                    # ret['MainProt_file'].append(str(file_name))
                    mainProt_files.append(str(file_name))
                elif file_name.startswith('seq_single'):
                    ret['xml_holostim_seq'].append(str(file_name))
                elif file_name.startswith('strcMask'):
                    ret['strc_mask_mat_file'].append(str(file_name))
                elif file_name.startswith('BMI_target_info'):
                    bmitarget_files.append(str(file_name))
                elif file_name.startswith('BaselineOnline') and file_name.endswith('.mat'):
                    baseline_files.append(str(file_name))
                elif file_name.startswith('target_calibration'):
                    target_files.append(str(file_name))
                elif file_name.startswith('BMI_online'):
                    match = re.search(r'BMI_online(\d{6}T\d{6})', file_name)
                    if match:
                        bmi_files.append((file_name, datetime.strptime(match.group(1), "%y%m%dT%H%M%S")))

            ret['roi_mat_file'].extend(roi_mat_files)
            ret['bmi_target_mat_file'].append(bmitarget_files[0] if len(bmitarget_files) == 1 else None)
            ret['baseline_mat_file'].append(baseline_files[0] if len(baseline_files) == 1 else None)
            ret['flag_baseline_online'].append(True if len(baseline_files) > 1 else False)
            ret['flag_bmi_target'].append(True if len(bmitarget_files) > 1 else False)
            ret['target_calibration_mat_file'].append(target_files[0] if len(target_files) == 1 else None)
            ret['flag_target_calibration'].append(True if len(target_files) > 1 else False)
            ret['holostim_seq_mat_file'].append(holostim_seq_files[0] if len(holostim_seq_files) == 1 else None)
            if not holostim_seq_files or not bmi_files or not baseline_files:  
                ret['flag_experiment_mat_file'].append(True) 
            else:
                ret['flag_experiment_mat_file'].append(False)
            ret['flag_holostim_seq_mat_file'].append(True if len(holostim_seq_files) > 1 else False)
            ret['main_prot_mat_file'].append(mainProt_files[0] if len(mainProt_files) == 1 else None)
            ret['flag_main_prot_mat_file'].append(True if len(mainProt_files) > 1 else False)
            
            if len(bmi_files) == 2:
                bmi_files.sort(key=lambda x: x[1])
                ret['pretrain_mat_file'].append(bmi_files[0][0])
                ret['bmi_mat_file'].append(bmi_files[1][0])
                ret['flag_bmi'].append(False)
            else:
                ret['pretrain_mat_file'].append(None)
                ret['bmi_mat_file'].append(None)
                ret['flag_bmi'].append(True if bmi_files else False)

    # Normalize DataFrame column lengths
    df = pd.DataFrame.from_dict(ret, orient='index').transpose()
    df = df.loc[:, list(ret.keys())]  # Ensure column order matches append order
    return df[list(df.columns[:6]) + sorted(df.columns[6:], key=str.casefold)]


def get_sessions(experiment_type: str = None) -> pd.DataFrame:
    raw_path = Path('/data/project/nvl_lab/HoloBMI/Raw')
    matches = raw_path.glob("[0-9][0-9][0-9][0-9][0-9][0-9]/NVI*/D*")
    session_paths = [p for p in matches if p.is_dir()]

    if experiment_type in act.experiment_types:
        experiment_session_paths = get_session_paths()[experiment_type]
        extra_session_paths = set(experiment_session_paths) - set(session_paths)
        if len(extra_session_paths) > 0:
            print(extra_session_paths)
            exit()
        session_paths = experiment_session_paths
    else:
        print('Doing all paths')
        #raise ValueError(f'Invalid experiment type: {experiment_type} Choose from {", ".join(act.experiment_types)}')
    
    frame_count = {
        'baseline': act.calibration_frames,
        'BMI': act.bmi_frames,
        'holostim_seq': act.seq_holo_frames,
        'pretrain': act.bmi_frames # HoloVTA and VTA
    }

    sessions = []
    for session_path in [session_paths[0]]:
        session_date, mouse_id, day_index = session_path.parts[-3:]

        row = {
            'mouse_id': mouse_id,
            'session_date': session_date,
            'day_index': day_index,
            'experiment_type': experiment_type,
            'session_path': session_path,
            'has_error': False,
            'has_warning': False
        }

        # imaging
        abs_session_path = raw_path / session_path
        parent_im_path = abs_session_path / 'im'

        if not parent_im_path.exists():
            print(f'no im: {parent_im_path}')
            exit()

        im_files = list(parent_im_path.iterdir())
        for experiment in frame_count:
            im_path_matches = [f for f in im_files if f.match(f'*{experiment}/{experiment}_{session_date}T*')]

            if len(im_path_matches) != 1:
                print(f'no im: {experiment}/{experiment}_{session_date}T*')
                exit()

            im_path = im_path_matches[0]
            im_path_files = list(im_path.iterdir())
            tiff_count = sum(1 for f in im_path_files if f.is_file() and f.suffix.lower() == '.tif')

            if tiff_count > frame_count[experiment]:
                print(tiff_count)
                print(frame_count[experiment])
                print('too many tiff')
                exit()
            elif tiff_count == 0:
                print(f'no tiff')
                exit()

            volt_matches = [f for f in im_path_files if f.match(f'{im_path.name}_Cycle00001_VoltageRecording_001.csv')]

            if len(volt_matches) == 1:
                row[f'{experiment.lower()}_voltage_file'] = volt_matches[0]
            else:
                print(f'no voltage recording')

        '''
        for experiment in frame_count:
            #im_matches = [f for f in im_files if f.match(pattern)]
            expt_path = next(parent_im_path.glob(f'*{experiment}*'), None)
            expt_name = expt_path.name   

            im_path = next(Path(parent_im_path / expt_name).glob(f'{expt_name}_{session_date}T*'), None)
            print(im_path)
            tiff_count = 0

            if im_path:
                tiff_count = sum(1 for f in im_path.iterdir() if f.is_file() and f.suffix.lower() == '.tif')
            else:
                print(f'no im: {im_path}')
                exit()

            if tiff_count > frame_count[experiment]:
                print('too many tiff')
            elif tiff_count == 0:
                print(f'no tiff: {im_path}')
                exit()

            volt_path = next(im_path.glob(f'{im_path.name}_Cycle00001_VoltageRecording_001.csv'), None)

            if volt_path:
                row[f'{experiment.lower()}_voltage_file'] = volt_path
            else:
                print(f'no voltage recording: {volt_path}')
        '''
        files = list(abs_session_path.iterdir())
        file_patterns = {
            'roi_mat_file': '*roi',
            'bot_candidates': '*BOT_c',
            'bot_ensemble': '*BOT_e',
            'gpl_candidates': '*GPL_c',
            'gpl_ensemble': '*GPL_e',
            'xml_candidates': '*XML_c',
            'xml_ensemble': '*XML_e',
            'workspace_mat_file': '*workspace',
            'holomask_gpl_file': '*holoMask',
            'xml_holostim_seq': '*seq_single',
            'strc_mask_mat_file': '*strcMask',
            'main_prot_mat_file': '*mainProt', # Flag
            'bmi_target_mat_file': '*BMI_target_info',  # Flag
            'holostim_seq_mat_file': '*holostim_seq*.mat', # Flag
            'baseline_mat_file': '*BaselineOnline*.mat', # Flag
            'target_calibration_mat_file': '*target_calibration', # Flag
            '_mat_file': '*BMI_online*T*.mat', # Flag
        }

        for file_name, pattern in file_patterns.items():
            matches = [f for f in files if f.match(pattern)]
            match_count = len(matches)
            if match_count > 1:
                if file_name == '_mat_file':
                    if match_count == 2:
                        matches = sorted(matches)
                        row[f'pretrain{file_name}'] = matches[0]
                        row[f'bmi_mat{file_name}'] = matches[1]
                        row[f'flag_bmi'] = False
                    else:
                        print('incorrect bmi_online files')
                        row[f'flag_bmi'] = True
                else:
                    row[f'flag_{file_name}'] = True
            elif match_count == 1:
                row[f'flag_{file_name}'] = False
            else:
                print('no matches')
                row[f'flag_{file_name}'] = True


        '''
        # Check for flags
        row['roi_mat_file'] = next(abs_session_path.glob('roi*'), None)
        row['bot_candidates'] = next(abs_session_path.glob('BOT_c*'), None)
        row['bot_ensemble'] = next(abs_session_path.glob('BOT_e*'), None)
        row['gpl_candidates'] = next(abs_session_path.glob('GPL_c*'), None)
        row['gpl_ensemble'] = next(abs_session_path.glob('GPL_e*'), None)
        row['xml_candidates'] = next(abs_session_path.glob('XML_c*'), None)
        row['xml_ensemble'] = next(abs_session_path.glob('XML_e*'), None)
        row['holostim_seq_mat_file'] = next(abs_session_path.glob('holostim_seq*.mat'), None)
        row['workspace_mat_file'] = next(abs_session_path.glob('workspace*'), None)
        row['holomask_gpl_file'] = next(abs_session_path.glob('holoMask*'), None)
        row['main_prot_mat_file'] = next(abs_session_path.glob('mainProt*'), None)
        row['xml_holostim_seq'] = next(abs_session_path.glob('seq_single*'), None)
        row['strc_mask_mat_file'] = next(abs_session_path.glob('strcMask*'), None)
        row['bmi_target_mat_file'] = next(abs_session_path.glob('BMI_target_info*'), None)
        row['baseline_mat_file'] = next(abs_session_path.glob('BaselineOnline*.mat'), None)
        row['target_calibration_mat_file'] = next(abs_session_path.glob('target_calibration*'), None)
        bmi_files = sorted((abs_session_path.glob('BMI_online*T*.mat')))
        if len(bmi_files) == 2:
            row['pretrain_mat_file'] = bmi_files[0]
            row['bmi_mat_file'] = bmi_files[1]
        else:
            print('no bmi')
        '''

        sessions.append(row)
    df_sessions = pd.DataFrame(sessions)

    return df_sessions
