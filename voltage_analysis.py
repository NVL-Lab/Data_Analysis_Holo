__author__ = 'Saul'

import sys
from scipy.io import loadmat
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from preprocess.analyze_inputs import get_vars
from preprocess.syncronize_voltage_rec import obtain_peaks_voltage
from preprocess.get_data import get_data_df, get_data_rec

if __name__ == '__main__':
    args = sys.argv[1:]  # For read_df
    read_df = False      # Reading of dataframe for raw data retrieval
    show_all_res = False # Output all results rather
    to_csv = False       # Coversion of results to csv
    to_plot = False      # Plotting

    # Method for acquiring raw data
    if read_df:
        # ~/project/nvl_lab/holo_bmi/Data_Analysis_Holo/wholescale_analysis/files/holobmi_df.parquet
        info = get_data_df(args[0])
    else:
        # /data/project/nvl_lab/HoloBMI/Raw/
        info = get_data_rec(args[0])

    # Checks wether there is data
    if not info:
        print('Data was filtered out')
        exit(1)

    # Results template 
    input_data_temp = {
        'Variable': [], 
        'Peaks': [], 
        'Min_Diff': [], 
        'Sugg_Input': [], 
        'Voltage_File': [], 
        'Experiment': [],
        'Limit_Size': [],
        'Prev_Min_Diff': []
        }
    
    fr = 29.988635806461144 # Pairie frame rate
    # Main pipeline for computing voltage peaks and input designation
    for expt in info: # <date>/<mouse>/<Day>
        for test in info[expt]: # 'holostim_seq', 'baseline', 'pretrain', 'bmi'
            print(f'Processing {expt}:{test}...')
            if test == 'mats':
                holo_data = loadmat(info[expt][test][0])
                base_data = loadmat(info[expt][test][1])
                pre_var_data = loadmat(info[expt][test][2])
                bmi_var_data = loadmat(info[expt][test][3])
                continue
            peak_info = obtain_peaks_voltage(info[expt][test][1], fr, info[expt][test][0]) # Computes voltage trigger peaks
            trigg_peaks = list(peak_info[1:9])
            triggs = [len(peaks) for peaks in trigg_peaks]
            # Determines variable and peak count correspondance or do-over (Limiting size method)
            input_data = get_vars(test, np.array(triggs), expt, peak_info[0], info, holo_data, base_data, pre_var_data, bmi_var_data, input_data_temp) 
            # A tuple of data means do-over
            if isinstance(input_data, tuple):
                prev_bmi_diff = input_data[2]  # Previous bmi minimum difference
                prev_holo_diff = input_data[3] # Previous holo minimum difference
                bmi_size = input_data[4]       # Size limit to determine new values
                input_data = input_data_temp   # Allows for previously computed data to be ignored for new
                print('Limiting size of Input 7...') # Size limit applies to input 7
                peak_info = obtain_peaks_voltage(info[expt][test][1], fr, info[expt][test][0], True) # Redo with new size
                trigg_peaks = list(peak_info[1:9])
                triggs = [len(peaks) for peaks in trigg_peaks]
                # Redo with new size, with previous values for comparison
                input_data = get_vars(test, np.array(triggs), expt, peak_info[0], info, holo_data, base_data, pre_var_data, bmi_var_data, input_data_temp, True, prev_bmi_diff, prev_holo_diff, bmi_size)
            print(triggs)

    input_data = pd.DataFrame(input_data)
    if show_all_res:
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
    print(input_data)

    # Prints variable suggestion percentages per input voltages and minimum difference
    var_pcts = input_data.groupby('Variable')['Sugg_Input'].value_counts(normalize=True)
    print(var_pcts)
    var_pcts = input_data.groupby('Variable')['Min_Diff'].value_counts(normalize=True)
    print(var_pcts)

    if to_csv:
        input_data.to_csv('Suggested_Inputs.csv', index=False) # Results in /data/project/nvl_lab/HoloBMI/Raw/
    
    if to_plot:
        plt.figure(figsize=(8, 5))
        sns.barplot(data=input_data, x='Voltage_File', y='Sugg_Input', hue='Variable', errorbar=None, dodge=True)
        plt.xlabel('Voltage File')
        plt.ylabel('Sugg_Input Count')
        plt.title('Frequency of Sugg_Input by Voltage File and Variable')
        plt.legend(title='Variable')
        plt.show()
