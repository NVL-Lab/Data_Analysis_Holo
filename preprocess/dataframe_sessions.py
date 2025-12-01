
import collections
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import re
from session_paths import _hE2_rew,_hE2_norew,_hE2_rew_fb,_hE3_rew,_No_Reward_Pretrain,_randrew,_randrew_fb,_BMI
from utils.analysis_constants import AnalysisConstants as act

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



def get_all_sessions(folder_save: Path) -> pd.DataFrame:
    """ function to put all the experiment types in a dataframe"""
    df = pd.DataFrame()
    experiment_types = {'hE2_rew', 'hE2_norew', 'hE3_rew', 'randrew'}
    for experiment in experiment_types:
        df_new = get_sessions_df(experiment)
        df = pd.concat([df, df_new], ignore_index=True)
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
    
    if experiment_type not in experiment_dict:
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
                'baseline': 'Baseline_im',
                'BMI': 'BMI_im',
                'holostim_seq': 'Holostim_seq_im'  # Updated naming
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




