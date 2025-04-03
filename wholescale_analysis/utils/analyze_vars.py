__author__ = 'Saul'

import pandas as pd
from pathlib import Path
from typing import Tuple
from scipy.signal import find_peaks
import numpy as np
import math

def get_voltages(voltage_recording: Path, frame_rate: float, size_of_recording: int, limit_size: bool = False) \
                -> Tuple[list, np.array, np.array, np.array, np.array, np.array, np.array, np.array, np.array, np.array, np.array, np.array, list]:
    """ 
        Funtion to obtain the peaks of the voltage recording file
            :param voltage_recording: the path to the voltage recording file
                Input_1 = Trigger for each frame of the microscope recording
                Input_5 = Trigger for reward
                Input_6 = Trigger for a holographic stim
                Input_7 = Trigger for each frame evaluated by the BMI
            :param frame_rate: the frame rate of the 2p recording
            :param size_of_recording: the size of the 2p data
            :param limit_size: whether to limit the size of the voltage recording to the last reward trigger
                               which seems to be 100% reliable across all experiments that has it.
            :return: the indices of the peaks of the voltage 
    """

    df_voltage = pd.read_csv(voltage_recording)
    comments = []
    peaks_I0, _ = find_peaks(df_voltage[' Input 0'][:int(size_of_recording / frame_rate * 1000)], height=0.6, prominence=0.6, distance=25)
    peaks_I1, _ = find_peaks(df_voltage[' Input 1'][:int(size_of_recording / frame_rate * 1000)], prominence=0.05, distance=15)
    peaks_I2, _ = find_peaks(df_voltage[' Input 2'][:int(size_of_recording / frame_rate * 1000)], height=0.2, distance=30)
    peaks_I3, _ = find_peaks(df_voltage[' Input 3'][:int(size_of_recording / frame_rate * 1000)], height=0.2, prominence=2, distance=30)
    peaks_I4, _ = find_peaks(df_voltage[' Input 4'][:int(size_of_recording / frame_rate * 1000)], height=0.2, prominence=3, distance=30)
    if peaks_I1.shape[0] > size_of_recording:
        comments.append(f'We found more frame triggers {peaks_I1.shape[0]} '
                        f'than the size of the recording {size_of_recording}')
        peaks_I1 = peaks_I1[:size_of_recording]
        print(comments)
        #raise Warning(comments)
    else:
        comments.append(f'Triggers for image frames: {peaks_I1.shape[0]} found successfully ')

    peaks_I5, _ = find_peaks(df_voltage[' Input 5'][:int(size_of_recording / frame_rate * 1000)], prominence=0.05,
                             distance=1000)

    if not limit_size or peaks_I5.size == 0 :
        peaks_I6, _ = find_peaks(df_voltage[' Input 6'][:int(size_of_recording / frame_rate * 1000)], prominence=0.05,
                                 distance=1000)
        peaks_I7, _ = find_peaks(df_voltage[' Input 7'][:int(size_of_recording / frame_rate * 1000)], prominence=0.05,
                                  distance=15, width=(0, 45))
        #peaks_I7, _ = find_peaks(df_voltage[' Input 7'][:int(size_of_recording / frame_rate * 1000)], height=2, distance=15)
    else:
        peaks_I6, _ = find_peaks(df_voltage[' Input 6'][:peaks_I5[-1]], prominence=0.05,
                                 distance=1000)
        peaks_I7, _ = find_peaks(df_voltage[' Input 7'][:peaks_I5[-1]], prominence=0.05,
                                 distance=15, width=(0, 45))

    indices_for_5 = np.searchsorted(peaks_I1, peaks_I5) - 1
    indices_for_5 = np.maximum(indices_for_5, 0).astype('int')
    indices_for_6 = np.searchsorted(peaks_I1, peaks_I6) - 1
    indices_for_6 = np.maximum(indices_for_6, 0).astype('int')
    indices_for_7 = np.searchsorted(peaks_I1, peaks_I7) - 1
    indices_for_7 = np.maximum(indices_for_7, 0).astype('int')

    return df_voltage.keys(), peaks_I0, peaks_I1, peaks_I2, peaks_I3, peaks_I4, peaks_I5, peaks_I6, peaks_I7, indices_for_5, indices_for_6, indices_for_7, comments

'''
            volt_data = pd.read_csv(info[expt][test][1])
            for col in volt_data.keys()[1:]:
                fr = 29.988635806461144
                if col == ' Input 0': # Could be counter (for pre and bmi)
                    #locs, _ = find_peaks(volt_data[col], height=0.6, prominence=0.6, distance=25)
                    locs, _ = find_peaks(volt_data[col][:int(info[expt][test][0] / fr * 1000)], height=0.6, prominence=0.6,distance=25)
                elif col == ' Input 1': # Seems to be the frame counter
                    #locs, _ = find_peaks(volt_data[col], height=0.2, distance=30)
                    #locs, _ = find_peaks(volt_data[col][:int(info[expt][test][0] / fr * 1000)], height=2, distance=15)
                    locs, _ = find_peaks(volt_data[col][:int(info[expt][test][0] / fr * 1000)], height=2.5, distance=30)
                elif col == ' Input 2': 
                    #locs, _ = find_peaks(volt_data[col], height=0.2, distance=30)
                    locs, _ = find_peaks(volt_data[col][:int(info[expt][test][0] / fr * 1000)], height=0.2, distance=30)
                elif col == ' Input 3': 
                    #locs, _ = find_peaks(volt_data[col], height=0.2, prominence=2, distance=30)
                    locs, _ = find_peaks(volt_data[col][:int(info[expt][test][0] / fr * 1000)], height=0.2, prominence=2, distance=30)
                elif col == ' Input 4': # More precise than 2 or 3 (Could be STC for pre if 3000 apart)
                    #locs, _ = find_peaks(volt_data[col], height=0.2, prominence=3, distance=30)
                    locs, _ = find_peaks(volt_data[col][:int(info[expt][test][0] / fr * 1000)], height=0.2, prominence=3, distance=30)
                elif col == ' Input 5': # Seems to be specific for pretrain and bmi (reward)
                    #locs, _ = find_peaks(volt_data[col], height=0.1,  distance=30)
                    #locs, _ = find_peaks(volt_data[col][:int(info[expt][test][0] / fr * 1000)], height=3, distance=1000)
                    locs, _ = find_peaks(volt_data[col][:int(info[expt][test][0] / fr * 1000)], height=3, prominence=0.1, distance=1000)
                elif col == ' Input 6': # Seems to be specific for pretrain and holostim (holo)
                    locs, _ = find_peaks(volt_data[col][:int(info[expt][test][0] / fr * 1000)], height=0.1, prominence = 0.1, distance=1000)
                    # Remove intial artifact
                elif col == ' Input 7': # Seems to be the amount of tiff frames
                    #locs, _ = find_peaks(volt_data[col], height=0.2, distance=30)
                    #locs, _ = find_peaks(volt_data[col][:int(info[expt][test][0] / fr * 1000)], height=2, distance=15)
                    locs, _ = find_peaks(volt_data[col][:int(info[expt][test][0] / fr * 1000)], height=2, distance=15)
                triggs = np.append(triggs, len(locs))
                if to_plot and test == 'BMI' and col == ' Input 7':
                    plt.plot(volt_data[col], label='Signal')
                    plt.plot(locs, volt_data[col][locs], 'rv', label='Peaks')  # Red dots for peaks
                    plt.xlabel('Time(ms)')
                    plt.ylabel('Voltage')
                    plt.title(f'{test}:{col}')
                    plt.legend()
                    plt.show()
                    print('done')

                if col == ' Input 7':
                    print(triggs)
                    # FUNCTION
                    input_data = get_vars(test, triggs, expt, volt_data, info, holo_data, base_data, pre_var_data, bmi_var_data, input_data)
    
'''

def get_vars(test, triggs, expt, volt_keys, info, holo_data, base_data, pre_var_data, bmi_var_data, input_data, limit_size, prev_bmi_diff=None, prev_holo_diff=None) -> dict:
    # Frames from voltage files
    volt_diffs = triggs - info[expt][test][0]
    volt_idx = np.argmin(abs(volt_diffs)) # Does not take into account equal values
    volt_diff = volt_diffs[volt_idx]

    no_reward = False
    no_holo = False
    # Each experiment (in order)
    if test == 'holostim_seq':
        reward_idx = 5
        reward_diff = 0
        no_reward = True

        # BMI is the amount stims in one neuron
        bmi_neuron = holo_data['holoActivity'][0]
        bmi_count = len([value for value in bmi_neuron if not math.isnan(value)])
        bmi_diffs = triggs - bmi_count
        bmi_idx = np.argmin(abs(bmi_diffs)) 
        bmi_diff = bmi_diffs[bmi_idx]

        # Holo stims are the same as amount of neurons
        holo_count = base_data['baseActivity'].shape[0]
        holo_diffs = triggs - holo_count
        holo_idx = np.argmin(abs(holo_diffs))
        holo_diff = holo_diffs[holo_idx]
    
    elif test == 'baseline':
        reward_idx = 5
        reward_diff = 0
        no_reward = True

        # BMI is the amount stims in one neuron
        bmi_neuron = base_data['baseActivity'][0]
        bmi_count = len([value for value in bmi_neuron if not math.isnan(value)])
        bmi_diffs = triggs - bmi_count
        bmi_idx = np.argmin(abs(bmi_diffs)) 
        bmi_diff = bmi_diffs[bmi_idx]

        holo_idx = 6
        holo_diff = 0
        no_holo = True 

    elif test == 'HoloVTA_pretrain' or test == 'VTA_pretrain' or test == 'Holo_pretrain':
        reward_diffs = triggs - int(pre_var_data['data']['holoTargetCounter'])
        reward_idx = np.argmin(abs(reward_diffs))
        reward_diff = reward_diffs[reward_idx]

        bmi_diffs = triggs - int(pre_var_data['data']['frame'])
        #bmi_neuron = np.array(pre_var_data['data']['bmiAct'], dtype=np.float64)#[0]
        #bmi_count = len([value for value in bmi_neuron if not math.isnan(value)])
        #print(int(pre_var_data['data']['frame']))
        #print(bmi_count)
        #print(bmi_neuron[:, :np.where(~np.isnan(bmi_neuron).all(axis=0))[0][-1]].T)
        #exit()
        bmi_idx = np.argmin(abs(bmi_diffs))
        bmi_diff = bmi_diffs[bmi_idx]

        holo_count = len(pre_var_data['data']['vectorHolo'][0, 0].flatten().tolist())
        holo_diffs = triggs - holo_count
        holo_idx = np.argmin(abs(holo_diffs))
        holo_diff = holo_diffs[holo_idx]

    elif test == 'BMI':
        reward_diffs = triggs - int(bmi_var_data['data']['selfTargetCounter'])
        reward_idx = np.argmin(abs(reward_diffs))
        reward_diff = reward_diffs[reward_idx]

        # Typically has one more than the actual number of tiff images
        # Remove peaks whose width are greater than 15
        #   Inside of properties with width=(None, None)
        #   Try width = (0, 20)
        # Number of tiff frames should always be 75600
        # Synchronize input 1 and 7 and take into account
        bmi_diffs = triggs - int(bmi_var_data['data']['frame'])
        #bmi_diffs = triggs - int(bmi_var_data['data']['frame'])
        bmi_idx = np.argmin(abs(bmi_diffs))
        bmi_diff = bmi_diffs[bmi_idx]

        holo_idx = 6
        holo_diff = 0
        no_holo = True 

    if abs(bmi_diff) > 3 and not limit_size:
        print(bmi_diff)
        print(holo_diff)
        return bmi_diff, holo_diff

    volt_vars = ['frames', 'reward', 'bmi', 'holo']
    # Data Appending
    append_data(int(volt_keys[volt_idx+1].split(' ')[2]), triggs[volt_idx], volt_diff, volt_vars[0], test, expt, input_data, False, None)
    if no_reward:
        append_data(int(volt_keys[reward_idx+1].split(' ')[2]), reward_diff, reward_diff, volt_vars[1], test, expt, input_data, False, None)
    else:
        append_data(int(volt_keys[reward_idx+1].split(' ')[2]), triggs[reward_idx], reward_diff, volt_vars[1], test, expt, input_data, False, None)

    append_data(int(volt_keys[bmi_idx+1].split(' ')[2]), triggs[bmi_idx], bmi_diff, volt_vars[2], test, expt, input_data, limit_size, prev_bmi_diff)

    if no_holo:
        append_data(int(volt_keys[holo_idx+1].split(' ')[2]), holo_diff, holo_diff, volt_vars[3], test, expt, input_data, limit_size, prev_holo_diff)
    else:
        append_data(int(volt_keys[holo_idx+1].split(' ')[2]), triggs[holo_idx], holo_diff, volt_vars[3], test, expt, input_data, limit_size, prev_holo_diff)

    return input_data

def append_data(i, p, d, v, f, e, input_data, l, pd) -> dict:
    # Why input_data
    input_data['Variable'].append(v)
    input_data['Peaks'].append(p)
    input_data['Min_Diff'].append(d)
    input_data['Sugg_Input'].append(i)
    input_data['Voltage_File'].append(f)
    input_data['Experiment'].append(e)
    input_data['Limit_Size'].append(l)
    input_data['Prev_Min_Diff'].append(pd)
