__author__ = 'Saul'

import sys
from scipy.io import loadmat
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from utils.analyze_vars import get_vars, get_voltages
from utils.get_data import get_data_df, get_data_rec

if __name__ == '__main__':
    args = sys.argv[1:]
    to_plot = False

    # FUNCTION
    info = get_data_df(args[0])

    if not info:
        print('Data was filtered out')
        exit(1)

    # Table column initialization                         
    input_data_temp = {
        'Variable': [], 
        'Peaks': [], 
        'Min_Diff': [], 
        'Sugg_Input': [], 
        'Voltage_File': [], 
        'Experiment': []
        }

    for expt in info:
        for test in info[expt]:
            limit_size = False
            #triggs = np.array([])
            print(f'Processing {test}...')
            if test == 'mats':
                holo_data = loadmat(info[expt][test][0])
                base_data = loadmat(info[expt][test][1])
                pre_var_data = loadmat(info[expt][test][2])
                bmi_var_data = loadmat(info[expt][test][3])
                continue
            peak_info = get_voltages(info[expt][test][1], 29.988635806461144, info[expt][test][0], limit_size)
            trigg_peaks = list(peak_info[1:9])
            triggs = [len(peaks) for peaks in trigg_peaks]
            input_data = get_vars(test, np.array(triggs), expt, peak_info[0], info, holo_data, base_data, pre_var_data, bmi_var_data, input_data_temp, limit_size)
            if input_data == 'limit size':
                input_data = input_data_temp 
                limit_size = True
                print('Limiting size of Input 7...')
                peak_info = get_voltages(info[expt][test][1], 29.988635806461144, info[expt][test][0], limit_size)
                trigg_peaks = list(peak_info[1:9])
                triggs = [len(peaks) for peaks in trigg_peaks]
                input_data = get_vars(test, np.array(triggs), expt, peak_info[0], info, holo_data, base_data, pre_var_data, bmi_var_data, input_data_temp, limit_size)
            print(triggs)

    input_data = pd.DataFrame(input_data)
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_columns', None)
    print(input_data)
    var_pcts = input_data.groupby('Variable')['Sugg_Input'].value_counts(normalize=True)
    print(var_pcts)

    var_pcts = input_data.groupby('Variable')['Min_Diff'].value_counts(normalize=True)
    print(var_pcts)

    #input_data.to_csv('results/Suggested_Inputs.csv', index=False)
    
    '''
    if q1 == 'n':
        # holo, baseline, pre, bmi
        for var in set(input_data['Variable']):
            var_df = input_data[input_data['Variable'] == var]
            plt.plot(var_df['Voltage_File'], var_df['Sugg_Input'], '-o', label=var)
        plt.xlabel('Voltage File')
        plt.ylabel('Input')
        plt.title(f'Input Suggestion for Voltage Files')
        plt.legend()
        plt.show()
    else:
    '''
    plt.figure(figsize=(8, 5))
    sns.barplot(data=input_data, x='Voltage_File', y='Sugg_Input', hue='Variable', errorbar=None, dodge=True)
    plt.xlabel('Voltage File')
    plt.ylabel('Sugg_Input Count')
    plt.title('Frequency of Sugg_Input by Voltage File and Variable')
    plt.legend(title='Variable')
    plt.show()
