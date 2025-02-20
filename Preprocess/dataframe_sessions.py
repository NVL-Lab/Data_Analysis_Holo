
import collections
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import re
# from data.py import _Holostim, _HoloE3, _HoloE2, _Random_Reward

_Holostim = {
    'NVI12': [
        '190930/NVI12/D5',
        '191001/NVI12/D6',
        '191003/NVI12/D8',
        '191005/NVI12/D10',
        '191007/NVI12/D12',
        '191009/NVI12/D14',
        '191011/NVI12/D16',
        '191013/NVI12/D18',
        '191031/NVI12/D31',
        '191102/NVI12/D33',
        '191104/NVI12/D35',
        '191025/NVI12/D25',
        '191015/NVI12/D20',
        '191017/NVI12/D22',
        '191016/NVI12/D21',
    ],
    'NVI13': [
        '190930/NVI13/D5',
        '191001/NVI13/D6',
        '191003/NVI13/D8',
        '191005/NVI13/D10',
        '191007/NVI13/D12',
        '191009/NVI13/D14',
        '191011/NVI13/D16',
        '191013/NVI13/D18',
        '191031/NVI13/D31',
        '191102/NVI13/D33',
        '191104/NVI13/D35',
        '191025/NVI13/D25',
        '191015/NVI13/D20',
        '191017/NVI13/D22',
        '191016/NVI13/D21',
        
    ],
    'NVI16': [
        '190930/NVI16/D5',
        '191001/NVI16/D6',
        '191003/NVI16/D8',
        '191005/NVI16/D10',
        '191007/NVI16/D12',
        '191009/NVI16/D14',
        '191011/NVI16/D16',
        '191013/NVI16/D18',
        '191031/NVI16/D31',
        '191102/NVI16/D33',
        '191104/NVI16/D35',
        '191025/NVI16/D25',
        '191015/NVI16/D20',
        '191017/NVI16/D22',
        '191016/NVI16/D21',
        
    ],
    'NVI17': [
        '191106/NVI17/D02',
        '191108/NVI17/D04',
        '191110/NVI17/D06',
        '191112/NVI17/D08',
        '191114/NVI17/D10',
        '191116/NVI17/D12',
        '191118/NVI17/D14',
        '191120/NVI17/D16',
        '191122/NVI17/D18',
        
    ],
    'NVI20': [
        '191106/NVI20/D02',
        '191108/NVI20/D04',
        '191110/NVI20/D06',
        '191112/NVI20/D08',
        '191114/NVI20/D10',
        '191116/NVI20/D12',
        '191118/NVI20/D14',
        '191120/NVI20/D16',
        '191122/NVI20/D18',
        
    ],
    'NVI21': [
        '191106/NVI21/D02',
        '191108/NVI21/D04',
        '191110/NVI21/D06',
        '191112/NVI21/D08',
        
    ],
    'NVI22': [
        '191106/NVI22/D02',
        '191108/NVI22/D04',
        '191110/NVI22/D06',
        '191112/NVI22/D08',
        '191114/NVI22/D10',
        '191116/NVI22/D12',
        '191118/NVI22/D14',
        '191120/NVI22/D16',
        '191122/NVI22/D18',
        
    ]
}

_HoloE3 = {
   'NVI12': [
        '191004/NVI12/D9',
        '191026/NVI12/D26',
        '191028/NVI12/D28',
        
    ],
    'NVI13': [
        '191004/NVI13/D9',
        '191026/NVI13/D26',
        '191028/NVI13/D28',
        
    ],
    'NVI16': [
        '191004/NVI16/D9',
        '191026/NVI16/D26',
        '191028/NVI16/D28',
    ],
    'NVI17': [
        '191109/NVI17/D05',
        '191115/NVI17/D11',
        '191121/NVI17/D17',
        
    ],
    'NVI20': [
        '191109/NVI20/D05',
        '191115/NVI20/D11',
        '191121/NVI20/D17',
        
    ],
    'NVI21': [
        '191109/NVI21/D05',
        
    ],
    'NVI22': [
        '191109/NVI22/D05',
        '191115/NVI22/D11',
        '191121/NVI22/D17',
    ]
}
_HoloE2 = {
    'NVI12': [
        '191008/NVI12/D13',
        '191012/NVI12/D17',
        '191027/NVI12/D27',
        
    ],
    'NVI13': [
        '191008/NVI13/D13',
        '191012/NVI13/D17',
        '191027/NVI13/D27',
        
    ],
    'NVI16': [
        '191008/NVI16/D13',
        '191012/NVI16/D17',
        '191027/NVI16/D27',
    ],
    'NVI17': [
        '191107/NVI17/D03',
        '191113/NVI17/D09',
        '191119/NVI17/D15',
        
    ],
    'NVI20': [
        '191107/NVI20/D03',
        '191113/NVI20/D09',
        '191119/NVI20/D15',
        
    ],
    'NVI21': [
        '191107/NVI21/D03',
        
    ],
    'NVI22': [
        '191107/NVI22/D03',
        '191113/NVI22/D09',
        '191119/NVI22/D15',
        
    ]
}
_Random_Reward = {
    'NVI12': [
        '191006/NVI12/D11',
        '191010/NVI12/D15',
        '191014/NVI12/D19',
        '191024/NVI12/D24',
        '191030/NVI12/D30',
        '191101/NVI12/D32',
        '191103/NVI12/D34',
        '191018/NVI12/D23',
    ],
    'NVI13': [
        '191006/NVI13/D11',
        '191010/NVI13/D15',
        '191014/NVI13/D19',
        '191024/NVI13/D24',
        '191030/NVI13/D30',
        '191101/NVI13/D32',
        '191103/NVI13/D34',
        '191018/NVI13/D23',
        
    ],
    'NVI16': [
        '191006/NVI16/D11',
        '191010/NVI16/D15',
        '191014/NVI16/D19',
        '191024/NVI16/D24',
        '191030/NVI16/D30',
        '191101/NVI16/D32',
        '191103/NVI16/D34',
        '191018/NVI16/D23',
        
    ],
    'NVI17': [
        '191105/NVI17/D01',
        '191111/NVI17/D07',
        '191117/NVI17/D13',
        '191123/NVI17/D19',
        
    ],
    'NVI20': [
        '191105/NVI20/D01',
        '191111/NVI20/D07',
        '191117/NVI20/D13',
        '191123/NVI20/D19',
    ],
    'NVI21': [
        '191105/NVI21/D01',
        '191111/NVI21/D07',
        '191123/NVI21/D19',
        
    ],
    'NVI22': [
        '191105/NVI22/D01',
        '191111/NVI22/D07',
        '191117/NVI22/D13',
        '191123/NVI22/D19',
        
    ]
}
_Feedback_BMI_Random_Reward = {
    
    'NVI17': [
        '191124/NVI17/D20',
        '191126/NVI17/D22',
        '191128/NVI17/D24',
        
    ],
    'NVI20': [
        '191124/NVI20/D20',
        '191126/NVI20/D22',
        '191128/NVI20/D24',
        
    ],
    'NVI21': [
        
    ],
    'NVI22': [
        '191124/NVI22/D20',
        '191126/NVI22/D22',
        '191128/NVI22/D24',

    ]
}
_Feedback_BMI_Holostim = {
    
    'NVI17': [
        '191125/NVI17/D21',
        '191127/NVI17/D23',
        '191129/NVI17/D25',
        
    ],
    'NVI20': [
        '191125/NVI20/D21',
        '191127/NVI20/D23',
        '191129/NVI17/D25',
        
    ],
    'NVI21': [
        
    ],
    'NVI22': [
        '191125/NVI22/D21',
        '191127/NVI22/D23',
        '191129/NVI17/D25',

    ]
}
_No_Reward_Pretrain = {
    
    'NVI17': [
        '191211/NVI17/D26',
        '191213/NVI17/D28',
        
    ],
    'NVI20': [
        '191211/NVI20/D26',
        '191213/NVI20/D28',
        
    ],
    'NVI21': [
        
    ],
    'NVI22': [
        '191211/NVI22/D26',
    ]
}
_BMI = {
    
    'NVI17': [
        '191212/NVI17/D27',
        
    ],
    'NVI20': [
        '191212/NVI20/D27',
        
    ],
    'NVI21': [
        
    ],
    'NVI22': [
        '191212/NVI22/D27',
    ]
}

def get_all_sessions() -> pd.DataFrame:
    """ function to get a df with all sessions"""
    df_Holostim = pd.DataFrame(index=np.concatenate(list(_Holostim.values())))
    df_Holostim['experiment_type'] = 'Holostim'
    
    df_HoloE3 = pd.DataFrame(index=np.concatenate(list(_HoloE3.values())))
    df_HoloE3['experiment_type'] ='HoloE3'

    df_HoloE2 = pd.DataFrame(index=np.concatenate(list(_HoloE2.values())))
    df_HoloE2['experiment_type'] ='HoloE2'

    df_Random_Reward = pd.DataFrame(index=np.concatenate(list(_Random_Reward.values())))
    df_Random_Reward['experiment_type'] ='Random_Reward'


    list_experiments = [df_Holostim,df_HoloE3,df_HoloE2, df_Random_Reward]
    df_experiments = pd.concat(list_experiments)
    # print(df_experiments.sort_index().reset_index())
    return df_experiments.sort_index().reset_index()

# get_all_sessions()
# print(_Holostim.items())

# print(collections.defaultdict(list))



def get_sessions_df(experiment_type: str) -> pd.DataFrame:
    experiment_dict = {
        'Holostim': _Holostim,
        'HoloE2': _HoloE2,
        'HoloE3': _HoloE3,
        'Random_Reward': _Random_Reward
    }
    
    if experiment_type not in experiment_dict:
        raise ValueError(f'Invalid experiment type: {experiment_type}. Choose from Holostim, HoloE2, HoloE3, Random_Reward')

    ret = collections.defaultdict(list)
    folder_raw = Path('/data/project/nvl_lab/HoloBMI/Raw')

    for mice_name, sessions in experiment_dict[experiment_type].items():
        for day_index, session_path in enumerate(sessions):
            session_date, mice_name, day_init = session_path.split('/')
            ret['mice_name'].append(mice_name)
            ret['session_date'].append(session_date)
            ret['day_init'].append(day_init)
            ret['experiment_type'].append(experiment_type)
            ret['session_path'].append(session_path)
            ret['day_index'].append(day_index)

            dir_files = folder_raw / session_path
            category_map = {
                'baseline': 'Baseline_im',
                'BMI': 'BMI_im',
                'HoloVTA_pretrain': 'HoloVTApretrain_im',
                'holostim': 'Holostim_im'
            }

            # Initialize columns with None
            for category in category_map.values():
                ret[category].append(None)
                ret[f'Voltage_{category}'].append(None)

            # Iterate over session directory
            for file_name in os.listdir(dir_files):
                file_path = dir_files / file_name

                # Look for im folders
                if file_name.startswith('im'):
                    for im_subfolder in os.listdir(file_path):
                        im_subfolder_path = file_path / im_subfolder

                        for key, category in category_map.items():
                            if im_subfolder.startswith(key):
                                full_subfolder_name = None
                                for subfolder in os.listdir(im_subfolder_path):
                                    if subfolder.startswith(f"{key}_"):
                                        full_subfolder_name = subfolder
                                        break

                                if full_subfolder_name:
                                    ret[category][-1] = full_subfolder_name

                                    # Look for voltage recording file inside the correct subfolder
                                    voltage_file_pattern = f"{full_subfolder_name}_Cycle00001_VoltageRecording_001.csv"
                                    voltage_file_path = im_subfolder_path / full_subfolder_name / voltage_file_pattern

                                    if voltage_file_path.exists():
                                        ret[f'Voltage_{category}'][-1] = voltage_file_pattern

            # Additional file processing
            roi_files = []
            target_files = []
            bmitarget_files = []
            bmi_files = []

            for file_name in os.listdir(dir_files):
                if file_name.startswith('roi'):
                    roi_files.append(file_name)

                elif file_name[:3] in {'BOT', 'GPL', 'XML'}:
                    ret[file_name[:5].replace('_', '')].append(file_name)

                elif file_name.startswith(('holostim_seq', 'Baseline')) and file_name.endswith('.mat'):
                    ret['holostim' if file_name.startswith('holostim_seq') else 'BaselineOnline'].append(file_name)

                elif file_name.startswith(('workspace', 'holoMask', 'mainProt', 'seq_single', 'strcMask')):
                    ret[file_name.split('_')[0]].append(file_name)

                elif file_name.startswith('BMI_target_info'):
                    bmitarget_files.append(file_name)

                elif file_name.startswith('target_calibration'):
                    target_files.append(file_name)

                elif file_name.startswith('BMI_online'):
                    match = re.search(r'BMI_online(\d{6}T\d{6})', file_name)
                    if match:
                        bmi_files.append((file_name, datetime.strptime(match.group(1), "%y%m%dT%H%M%S")))

            ret['roi'].extend(roi_files)
            ret['BMI_target'].append(bmitarget_files[0] if len(bmitarget_files) == 1 else None)
            ret['Flag_BMITarget'].append(session_path if len(bmitarget_files) > 1 else None)
            ret['target_calibration'].append(target_files[0] if len(target_files) == 1 else None)
            ret['Flag_target_calibration'].append(session_path if len(target_files) > 1 else None)

            if len(bmi_files) == 2:
                bmi_files.sort(key=lambda x: x[1])
                ret['Pretrain_BMI'].append(bmi_files[0][0])
                ret['BMI'].append(bmi_files[1][0])
                ret['Flag_BMI'].append(None)
            else:
                ret['Pretrain_BMI'].append(None)
                ret['BMI'].append(None)
                ret['Flag_BMI'].append(session_path if bmi_files else None)

    # Normalize DataFrame column lengths
    max_len = max(len(v) for v in ret.values())
    for key in ret:
        ret[key].extend([None] * (max_len - len(ret[key])))

    return pd.DataFrame(ret)












df = get_sessions_df('Holostim')
print(df.head(2))
