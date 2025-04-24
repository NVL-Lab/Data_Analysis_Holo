__author__ = 'Saul'

import numpy as np
import math

def get_vars(test: str, triggs: np.array, expt: str, volt_keys: list, info: dict, holo_data: list, base_data: list, pre_var_data: list, bmi_var_data: list, input_data: dict, limit_size: bool = False, prev_bmi_diff: int = None, prev_holo_diff: int = None, last_rew_index: int =None) -> dict:
    '''
        Determines what variable corresponds to voltage input
        Parameters:
            test: corresponding experiment
            triggs: triggers for each input
            expt: date/mouse/day of experiment
            volt_keys: input names ex. ' Input 0'
            info: raw data of voltages
            holo_data: raw holo activity
            base_data: raw baseline activity
            pre_var_data: pretrain data
            bmi_var_data: bmi data
            input_data: template for results to be added to
            limit_size: wether size will be limited
                prev_bmi_diff: minimum bmi difference before limiting size
                prev_holo_diff: minimum holo difference before limiting size
                last_rew_index: actual size limit
        Return:
            Dictionary with data or tuple with data needed for recall for size limiting
    '''
    
    # Frames from voltage files
    volt_diffs = triggs - info[expt][test][0]
    volt_idx = np.argmin(abs(volt_diffs))
    volt_diff = volt_diffs[volt_idx]

    no_reward = False
    no_holo = False
    # Each experiment (in order)
    if test == 'holostim_seq':
        reward_idx = 5
        reward_diff = 0
        no_reward = True

        # Holo stims are the same as amount of neurons
        holo_count = base_data['baseActivity'].shape[0]
        holo_diffs = triggs - holo_count
        holo_idx = np.argmin(abs(holo_diffs))
        holo_diff = holo_diffs[holo_idx]

        # BMI is the amount stims in one neuron
        bmi_neuron = holo_data['holoActivity'][0]
        bmi_count = len([value for value in bmi_neuron if not math.isnan(value)])
        bmi_diffs = triggs - bmi_count
        bmi_idx = np.argmin(abs(bmi_diffs)) 

        # Handles nstance of having repeating minimum value
        if bmi_idx != 7:
            bmi_diffs[bmi_idx] = abs(bmi_diffs[7])+1
            bmi_idx = np.argmin(abs(bmi_diffs))
        bmi_diff = bmi_diffs[bmi_idx]
    
    elif test == 'baseline':
        reward_idx = 5
        reward_diff = 0
        no_reward = True

        holo_idx = 6
        holo_diff = 0
        no_holo = True 

        bmi_neuron = base_data['baseActivity'][0]
        bmi_count = len([value for value in bmi_neuron if not math.isnan(value)])
        bmi_diffs = triggs - bmi_count
        bmi_idx = np.argmin(abs(bmi_diffs)) 
        bmi_diff = bmi_diffs[bmi_idx]

    elif test == 'HoloVTA_pretrain' or test == 'VTA_pretrain' or test == 'Holo_pretrain':
        reward_diffs = triggs - int(pre_var_data['data']['holoTargetCounter'])
        reward_idx = np.argmin(abs(reward_diffs))
        reward_diff = reward_diffs[reward_idx]

        holo_count = pre_var_data['data']['holoDelivery'][0][0].sum()
        holo_diffs = triggs - holo_count
        holo_idx = np.argmin(abs(holo_diffs))
        holo_diff = holo_diffs[holo_idx]

        if limit_size:
            bmi_count = last_rew_index
        else:
            bmi_data = pre_var_data['data']['bmiAct'][0][0]
            bmi_count = len(bmi_data[:, :np.where(~np.isnan(bmi_data).all(axis=0))[0][-1]].T)
        # Frame value is incorrect: bmi_diffs = triggs - int(pre_var_data['data']['frame'])
        bmi_diffs = triggs - bmi_count
        bmi_idx = np.argmin(abs(bmi_diffs))
        bmi_diff = bmi_diffs[bmi_idx]

        # Determines to limit size
        if abs(bmi_diff) > 4 and not limit_size:
            self_reward_indexes = np.where(pre_var_data['data']['selfVTA'][0][0]==1)[1]
            holo_reward_indexes = np.where(pre_var_data['data']['holoVTA'][0][0]==1)[1]
            if len(self_reward_indexes) != 0:
                last_reward_index = self_reward_indexes[-1]
            elif len(holo_reward_indexes) != 0:
                last_reward_index = holo_reward_indexes[-1]
            return volt_diff, reward_diff, bmi_diff, holo_diff, last_reward_index 


    elif test == 'BMI':
        reward_diffs = triggs - int(bmi_var_data['data']['selfTargetCounter'])
        reward_idx = np.argmin(abs(reward_diffs))
        reward_diff = reward_diffs[reward_idx]

        holo_idx = 6
        holo_diff = 0
        no_holo = True 

        if limit_size:
            bmi_count = last_rew_index
        else:
            bmi_data = bmi_var_data['data']['bmiAct'][0][0]
            bmi_count = len(bmi_data[:, :np.where(~np.isnan(bmi_data).all(axis=0))[0][-1]].T)
        # Frame value is incorrect: bmi_diffs = triggs - int(bmi_var_data['data']['frame'])
        bmi_diffs = triggs - bmi_count
        bmi_idx = np.argmin(abs(bmi_diffs))
        bmi_diff = bmi_diffs[bmi_idx]

        # Determines to limit size
        if abs(bmi_diff) > 4 and not limit_size:
            self_reward_indexes = np.where(bmi_var_data['data']['selfVTA'][0][0]==1)[1]
            holo_reward_indexes = np.where(bmi_var_data['data']['holoVTA'][0][0]==1)[1]
            if len(self_reward_indexes) != 0:
                last_reward_index = self_reward_indexes[-1]
            elif len(holo_reward_indexes) != 0:
                last_reward_index = holo_reward_indexes[-1]
            return volt_diff, reward_diff, bmi_diff, holo_diff, last_reward_index 

    # Data Appending
    append_data(input_data, int(volt_keys[volt_idx+1].split(' ')[2]), triggs[volt_idx], volt_diff, 'frames', test, expt, False, None)
    if no_reward:
        append_data(input_data, int(volt_keys[reward_idx+1].split(' ')[2]), reward_diff, reward_diff, 'reward', test, expt, False, None)
    else:
        append_data(input_data, int(volt_keys[reward_idx+1].split(' ')[2]), triggs[reward_idx], reward_diff, 'reward', test, expt, False, None)

    append_data(input_data, int(volt_keys[bmi_idx+1].split(' ')[2]), triggs[bmi_idx], bmi_diff, 'bmi', test, expt, limit_size, prev_bmi_diff)

    if no_holo:
        append_data(input_data, int(volt_keys[holo_idx+1].split(' ')[2]), holo_diff, holo_diff, 'holo', test, expt, limit_size, prev_holo_diff)
    else:
        append_data(input_data, int(volt_keys[holo_idx+1].split(' ')[2]), triggs[holo_idx], holo_diff, 'holo', test, expt, limit_size, prev_holo_diff)

    return input_data

def append_data(input_data: dict, i: int, p: int, d: int, v: str, f: str, e: str, l: bool, pd: int) -> dict:
    '''
        Appends data
        Parameters:
            Corresponding to dictionary key 
    '''
    
    input_data['Variable'].append(v)
    input_data['Peaks'].append(p)
    input_data['Min_Diff'].append(d)
    input_data['Sugg_Input'].append(i)
    input_data['Voltage_File'].append(f)
    input_data['Experiment'].append(e)
    input_data['Limit_Size'].append(l)
    input_data['Prev_Min_Diff'].append(pd)
