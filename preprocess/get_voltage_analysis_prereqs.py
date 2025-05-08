__author__ = 'Saul'

from pathlib import Path
import pandas as pd

def get_data_df(df_dir) -> dict:
    """
        Load voltage data from csv files into pandas dataframes by reading from dataframe.

        Parameters:
            df_dir: directory to the location of dataframe

        Returns:
            test_info: dictionary containing dataframes with voltage data.
    """
    raw_data_dir = '/data/project/nvl_lab/HoloBMI/Raw/'

    # Dataset filtering
    df_path = Path(df_dir)
    df_data = pd.read_parquet(df_path)
    df_flagged = df_data.filter(like='Flag')
    flag_sessions = pd.unique(df_flagged.values.ravel())[1:] #First session is None
    df_data_filtered = df_data[~df_data['session_path'].isin(flag_sessions)]
    df_data_filtered = df_data_filtered.sort_values(by='session_path', ascending=True)
    df_data_filtered = df_data_filtered.reset_index(drop=True)

    print('All experiments:')
    print(df_data_filtered['session_path'])

    a1 = input('Which experiment? ')
    if a1 != 'all':
        try:
            df_data_filtered = df_data_filtered[df_data_filtered['session_path'] == a1]
        except KeyError as e:
            print(f'Error: Some indices not found - {e}')
            exit(1)

    df_data_filtered['indexes'] = df_data_filtered['session_path']
    df_data_filtered = df_data_filtered.set_index('indexes')
    sessions = df_data_filtered.index.tolist()
    df_data_filtered['session_path'] = raw_data_dir + df_data_filtered['session_path'].astype(str)
    df_data_filtered['session_path'] = df_data_filtered['session_path'].map(Path)

    test_info = {}
    for session in sessions:
        exp_path = df_data_filtered.loc[session, 'session_path']
        exp_im_path = exp_path / 'im'
        print(session)
        print(exp_path)
        test_info[session] = {'mats': [exp_path/df_data_filtered.loc[session, 'Holostim_seq_mat_file'], 
                exp_path/df_data_filtered.loc[session, 'Baseline_mat_file'],
                exp_path/df_data_filtered.loc[session, 'Pretrain_mat_file'],
                exp_path/df_data_filtered.loc[session, 'BMI_mat_file']]}
        im_dirs = [df_data_filtered.loc[session, 'Holostim_seq_im'], df_data_filtered.loc[session, 'Baseline_im'], df_data_filtered.loc[session, 'Pretrain_im'], df_data_filtered.loc[session, 'BMI_im']]
        volt_files = [df_data_filtered.loc[session, 'Holostim_seq_im_voltage_file'], df_data_filtered.loc[session, 'Baseline_im_voltage_file'], df_data_filtered.loc[session, 'Pretrain_im_voltage_file'], df_data_filtered.loc[session, 'BMI_im_voltage_file']]
        for i in range(len(im_dirs)):
            print(session)
            print(im_dirs[i])
            exp = im_dirs[i].split('/')[0]
            im_path = exp_im_path / im_dirs[i]
            tif_count = len(list(im_path.glob(f'{exp}*.tif')))
            volt_file = im_path / volt_files[i]
            test_info[session][exp] = [tif_count, volt_file]
    return test_info

def get_data_rec(raw_data_dir) -> dict:
    """
        Load voltage data from csv files into pandas dataframes by recursively iterating through dataset storage.
        
        Parameters:
            raw_data_dir: directory to the location of raw holo bmi data
                           ex. /data/project/nvl_lab/HoloBMI/Raw/

        Returns:
            test_info: dictionary containing DataFrames with voltage data.
    """

    # voltage path: "recordings path"/im/[bl,pretrain,..]/[name]/[name]_Voltage
    data_path = Path(raw_data_dir)
    im_path =  data_path / 'im'
    
    # Dataset filtering
    mice_info = pd.read_parquet('/data/project/nvl_lab/HoloBMI/mice_info.parquet')
    flag_cols = mice_info.filter(like='Flag')
    flag_sessions = pd.unique(flag_cols.values.ravel())[1:] #First session is None
    mice_info_filtered = mice_info[~mice_info['session_path'].isin(flag_sessions)]
    good_sessions = mice_info_filtered['session_path'].tolist() 

    # User interaction for data extraction
    if im_path.exists()and im_path.is_dir():
        print('Specific directory given')
    else:
        data_dates = sorted([d.name for d in data_path.iterdir() if d.is_dir()])[:-3] # Last three are not regular datasets
        data_paths = {}
        for date in data_dates:
            date_path = data_path / date
            data_paths[date] = [d for d in date_path.glob("NVI??/D*")] # Each mouse has one day experiment
        print("All dates: ")
        print(data_dates)
        date_input = input('Which date? ')
        if date_input in data_dates:
            q1 = input('Multiple mice? ')
            if q1 == 'n':
                print(data_paths[date_input])
                mouse_idx = int(input('Choose: '))
                if 0 <= mouse_idx < len(data_paths[date_input]):
                    datasets = [data_paths[date_input][mouse_idx]]
                else:
                    print('Invalid input')
                    exit(1)
            else:
                datasets = data_paths[date_input]           
        elif date_input == 'all':
            datasets = [ds for sublist in data_paths.values() for ds in sublist]
        else: 
            print('Invalid input')
            exit(0)
    
    # Gathering variable and voltage files
    test_info = {}
    for ds_path in datasets:
        ds_name = f'{ds_path.parts[-3]}/{ds_path.parts[-2]}/{ds_path.parts[-1]}'
        print(ds_path)

        holo_mat = [d for d in ds_path.glob('holostim_seq*.mat')][0]
        online_data = sorted([d for d in ds_path.glob('*nline*.mat')])
        pre_mat = online_data[0]
        bmi_mat = online_data[1]
        base_mat = online_data[2]
        
        # Voltage files
        test_names = [d.name for d in ds_path.glob("im/*") if not d.name.startswith('.')]
        test_info[f'{ds_path.parts[-3]}/{ds_path.parts[-2]}/{ds_path.parts[-1]}'] = {'mats': [holo_mat, base_mat, pre_mat, bmi_mat]}
        for name in test_names: # na
            print(name)
            volt_file = [d for d in ds_path.glob(f'im/{name}/{name}*/*.csv')]
            tif_count  = len(list(ds_path.glob(f'im/{name}/{name}*/{name}*.tif')))
            #Normal tif counts: 'holostim_seq'=2600, 'baseline'=27000, 'pretrain'=75600, 'bmi'=75600
            test_info[ds_name][name] = [tif_count, volt_file[0]] # test_info[NVImouse][mats:[4]; baseline:[tif_count, volt_dir]; ...]

    return test_info
