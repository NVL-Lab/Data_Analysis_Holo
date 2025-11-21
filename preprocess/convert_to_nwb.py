__author__ = 'Nuria'

# __author__ = ('Nuria', 'John Doe')

# make sure to be in environment with pynwb and neuroconv installed
# In this case we use:
# https://neuroconv.readthedocs.io/en/main/conversion_examples_gallery/imaging/brukertiff.html
# it also needs pip install roiextractors
# make sure the ndx-templates for CaBMI and Holographic_stim are installed
# (see repositories in lab's github)

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from pynwb.device import Device
from typing import Optional, Tuple

from ndx_holostim import LightSource, SpatialLightModulator
from ndx_holostim import PatternedOptogeneticSeries, SpiralScanning, PatternedOptogeneticStimulusSite
from ndx_cabmi import Parameters_BMI, ROI_metadata, Calibration_metadata, CaBMISeries

from scipy.io import loadmat
from pynwb import NWBHDF5IO, TimeSeries, ogen
from pathlib import Path
from zoneinfo import ZoneInfo
from neuroconv.converters import BrukerTiffSinglePlaneConverter

from preprocess import dataframe_sessions as ds
from preprocess import syncronize_voltage_rec as svr
from utils.analysis_configuration import AnalysisConfiguration as aconf
from utils.analysis_constants import AnalysisConstants as act


def convert_bruker_images_to_nwb(folder_path: Path, microscope: Device, nwbfile_path: str):
    """ function to convert a bruker tiff file recording to nwb
    :param folder_path: path to the folder containing the tiff files
    :param nwbfile_path: path where we want to store the nwb file"""
    # convert data to nwb
    converter = BrukerTiffSinglePlaneConverter(folder_path=folder_path)
    metadata = converter.get_metadata()
    session_start_time = metadata['NWBFile']['session_start_time']
    tzinfo = ZoneInfo('US/Eastern')
    metadata['NWBFile'].update(
        session_start_time=session_start_time.replace(tzinfo=tzinfo))  # TODO this doesn't seem to work properly
    metadata['Ophys']['Device'][0]['description'] = microscope.description
    metadata['Ophys']['ImagingPlane'][0]['excitation_lambda'] = 920.0
    converter.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata)


def retrieve_holographic_point_data(tree_xml: ET.Element, tree_gpl: ET.Element) \
        -> Tuple[list, list, list, list, list, list, list, list]:
    """ Function to retrieve from the xml and gpl files the holographic metadata needed
    :param tree_xml: The xml tree for the xml file
    :param tree_gpl: The gpl tree for the gpl file
    :return: A tuple containing the holographic metadata"""
    # obtain the holographic metadata and store it
    troot_xml = tree_xml.getroot()
    index = []
    delay = []
    duration = []
    spiral_rev = []
    for elem in troot_xml.findall('PVMarkPointElement'):
        index.append(float(elem.find('PVGalvoPointElement').get('Indices')))
        delay.append(float(elem.find('PVGalvoPointElement').get('InterPointDelay')))
        duration.append(float(elem.find('PVGalvoPointElement').get('Duration')))
        spiral_rev.append(int(elem.find('PVGalvoPointElement').get('SpiralRevolutions')))
    troot_gpl = tree_gpl.getroot()
    spiral_size = []
    loc_x = []
    loc_y = []
    loc_z = []
    for elem in troot_gpl.findall('PVGalvoPoint'):
        spiral_size.append(float(elem.get('SpiralSize')))
        loc_x.append(float(elem.get('X')))
        loc_y.append(float(elem.get('Y')))
        loc_z.append(float(elem.get('Z')))
    if len(index) != len(loc_x):
        raise ValueError('The gpl and xml files have different size of holographic stimulation')
    return index, delay, duration, spiral_rev, spiral_size, loc_x, loc_y, loc_z


def convert_all_experiments_to_nwb(folder_raw: Path, experiment_type: Optional[str]=None):
    """function to convert all experiments within a experiment type to nwb
    :param folder_raw: folder where all the raw files are located
    :param experiment_type: the type of experiment to process if None all are done"""
    # get the dataframe with all the session in experiment type.
    df_sessions = ds.get_sessions_df(experiment_type)
    folder_nwb = folder_raw.parents[0] / 'nwb'

    microscope = Device(
        name='Chameleon',
        description='Ti:Sapphire Laser',
        manufacturer='Coherent',
        model_number='',
        model_name='',
        serial_number=''
    )

    holo_light_source = LightSource(
        name='Monaco',
        description='high-power femtosecond laser',
        manufacturer='Coherent',
        stimulation_wavelength=1035.,
        filter_description=''
    )

    holo_spatial_light_modulator = SpatialLightModulator(
        name='No SLM',
        model='stimulation was sequential',
        resolution=0.  # float
    )

    for index, row in df_sessions.iterrows():
        # TODO: Add behavior!
        # If any flag represents the session can not be process normally
        flag_cols = df_sessions.filter(like='Flag').columns
        if row[flag_cols].any():
            continue

        folder_nwb_mice = folder_nwb / row.mice_name
        if not Path(folder_nwb_mice).exists():
            Path(folder_nwb_mice).mkdir(parents=True, exist_ok=True)

        # =====================================================================================================
        #                   Holographic sequential
        # =====================================================================================================

        print('Holo_sequential: initiating and creating the nwbfile')
        # convert data and open file
        folder_holo_seq_im = Path(folder_raw) / row.session_path / 'im' / row.holostim_seq_im
        nwbfile_holograph_seq_path = f'{folder_nwb_mice / row.mice_name}_{row.session_date}_holostim_seq.nwb'
        convert_bruker_images_to_nwb(folder_holo_seq_im, microscope, nwbfile_holograph_seq_path)
        io_holographic_seq = NWBHDF5IO(nwbfile_holograph_seq_path, mode='a')
        nwbfile_holographic_seq = io_holographic_seq.read()
        frame_rate = nwbfile_holographic_seq.acquisition['TwoPhotonSeries'].rate
        size_of_recording = nwbfile_holographic_seq.acquisition['TwoPhotonSeries'].data.shape[0]

        # add the devices for holographic stim
        nwbfile_holographic_seq.add_device(microscope)
        nwbfile_holographic_seq.add_device(holo_light_source)
        nwbfile_holographic_seq.add_device(holo_spatial_light_modulator)

        print('Holo_sequential: retrieving online data')
        # Save the neural data that was store in the mat file
        holostim_seq_data = loadmat(folder_raw / row.session_path / row.holostim_seq_mat_file)['holoActivity']
        # select holodata that is not nan and transpose to have time x neurons
        holostim_seq_data = holostim_seq_data[:, ~np.isnan(np.sum(holostim_seq_data, 0))].T
        voltage_recording = folder_holo_seq_im / row.holostim_seq_im_voltage_file
        # retrieve the peaks from the voltage recording
        _, _, peaks_I1, _, _, _, _, peaks_I6, peaks_I7, comments_holo_neural = (
            svr.obtain_peaks_voltage(voltage_recording, frame_rate, size_of_recording))
        holo_indices_for_6 = svr.obtain_indices_per_peaks(peaks_I1, peaks_I6)
        holo_indices_for_7 = svr.obtain_indices_per_peaks(peaks_I1, peaks_I7)
        # compare indices from voltage rec to online acquisition
        if holo_indices_for_7.shape[0] < holostim_seq_data.shape[0]:
            holostim_seq_data = holostim_seq_data[:holo_indices_for_7.shape[0], :]
            comments_holo_neural.append(
                'Holostim_seq data has more items than triggers were obtained from the voltage file')
        elif holo_indices_for_7.shape[0] > holostim_seq_data.shape[0]:
            holo_indices_for_7 = holo_indices_for_7[:holostim_seq_data.shape[0]]
            comments_holo_neural.append(
                'Holostim_seq data has less items than triggers were obtained from the voltage file')
        else:
            comments_holo_neural.append('conversion worked correctly')

        print('Holo_sequential: neural data')
        # store the neural data
        holo_online_neural_data = TimeSeries(
            name='Online_neural_activity',
            description=(f'neural data obtained online from {holostim_seq_data.shape[1]} neurons while'
                         f' recording for the sequential stimulation of all initial neurons'),
            data=holostim_seq_data,
            timestamps=holo_indices_for_7.astype('float64'),
            unit='frames',
            comments=''.join(comments_holo_neural)
        )
        nwbfile_holographic_seq.add_acquisition(holo_online_neural_data)

        print('Holo_sequential: holographic stim data')
        # obtain the holographic metadata and store it
        tree_xml = ET.parse(folder_raw / row.session_path / row.xml_holostim_seq)
        tree_gpl = ET.parse(folder_raw / row.session_path / row.holomask_gpl_file)
        stim_point, delay, duration, spiral_rev, spiral_size, loc_x, loc_y, loc_z = (
            retrieve_holographic_point_data(tree_xml, tree_gpl))

        # compare indices from voltage rec to online recording
        if len(stim_point) != len(holo_indices_for_6) or len(stim_point) != holostim_seq_data.shape[1]:
            comments_holo_stim = 'The number of stims locations/times is not consistent with data retrieved'
        else:
            comments_holo_stim = 'All stimulus locations/times correctly retrieved from experimental data'

        # creating and storing optogenetic stimulus site, stimulation pattern (spiral scanning, unique for every neuron),
        # and patterned series (also, unique for every neuron)
        holo_stim_site = PatternedOptogeneticStimulusSite(
            name='Holographic_sequential_location',
            device=holo_light_source,
            description='Sequential stimulation of all the neurons selected as initial ROIs',
            excitation_lambda=1035.,  # nm
            location='motor cortex',
            effector='ChRmine'  # ligth effector protein, not required
        )

        nwbfile_holographic_seq.add_ogen_site(holo_stim_site)
        for pt in np.arange(len(stim_point)):
            holo_stim_pattern = SpiralScanning(
                # spiral scanning parameters should be correct
                name='Single cell stim on neuron ' + str(stim_point[pt]),
                description='One cell at a time stimulation',
                duration=duration[pt],
                number_of_stimulus_presentation=1,
                inter_stimulus_interval=delay[pt],
                diameter=spiral_size[pt],
                height=loc_z[pt],
                number_of_revolutions=spiral_rev[pt]
            )
            nwbfile_holographic_seq.add_lab_meta_data(holo_stim_pattern)

            holo_seq_series = PatternedOptogeneticSeries(
                name='Single cell stim on neuron ' + str(stim_point[pt]),
                rate=0.,  # float
                unit='',  # usually watts
                description='One cell at a time stimulation',
                site=holo_stim_site,
                device=microscope,
                light_source=holo_light_source,
                spatial_light_modulator=holo_spatial_light_modulator,
                stimulus_pattern=holo_stim_pattern,
                center_rois=np.expand_dims(np.array([loc_x[pt], loc_y[pt], spiral_size[pt]]), axis=0)
            )
            nwbfile_holographic_seq.add_acquisition(holo_seq_series)

        # store the voltage_rec data
        holo_stim_times = TimeSeries(
            name='Stim_times',
            description=(f'sequential stim times, information is in timestamps, data is ones'),
            data=np.ones(holo_indices_for_6.shape[0]),
            timestamps=holo_indices_for_6.astype('float64'),
            unit='frames',
            comments=''.join(comments_holo_stim)
        )
        nwbfile_holographic_seq.add_acquisition(holo_stim_times)

        # write and close the nwb file
        io_holographic_seq.write(nwbfile_holographic_seq)
        io_holographic_seq.close()

        print('Holo_sequential: done')

        # =====================================================================================================
        #                   Baseline
        # =====================================================================================================

        print('Baseline: initiating and creating the nwbfile')
        # convert data and open file
        folder_baseline_im = Path(folder_raw) / row.session_path / 'im' / row.baseline_im
        nwbfile_baseline_path = f'{folder_nwb_mice / row.mice_name}_{row.session_date}_baseline.nwb'
        convert_bruker_images_to_nwb(folder_baseline_im, nwbfile_baseline_path)
        io_baseline = NWBHDF5IO(nwbfile_baseline_path, mode='a')
        nwbfile_baseline = io_baseline.read()
        frame_rate = nwbfile_baseline.acquisition['TwoPhotonSeries'].rate
        size_of_recording = nwbfile_baseline.acquisition['TwoPhotonSeries'].data.shape[0]

        nwbfile_baseline.add_device(microscope)

        print('Baseline: retrieving online data')
        baseline_data = loadmat(folder_raw / row.session_path / row.baseline_mat_file)['baseActivity']
        baseline_data = baseline_data[:, ~np.isnan(np.sum(baseline_data, 0))].T
        voltage_recording = folder_baseline_im / row.baseline_im_voltage_file
        _, _, peaks_I1, _, _, peaks_I4, _, _, peaks_I7, comments_baseline = (
            svr.obtain_peaks_voltage(voltage_recording, frame_rate, size_of_recording))
        baseline_indices_for_4 = svr.obtain_indices_per_peaks(peaks_I1, peaks_I4)
        baseline_indices_for_7 = svr.obtain_indices_per_peaks(peaks_I1, peaks_I7)

        if baseline_indices_for_7.shape[0] < baseline_data.shape[0]:
            baseline_data = baseline_data[:baseline_indices_for_7.shape[0], :]
            comments_baseline.append(
                'Baseline data has more items than triggers were obtained from the voltage file')
        elif baseline_indices_for_7.shape[0] > baseline_data.shape[0]:
            baseline_indices_for_7 = baseline_indices_for_7[:baseline_data.shape[0]]
            comments_baseline.append(
                'Baseline data has less items than triggers were obtained from the voltage file')
        else:
            comments_baseline.append('conversion worked correctly')

        print('Baseline: neural data')
        baseline_online_neural_data = TimeSeries(
            name='Online_neural_activity',
            description=(f'neural data obtained online from {baseline_data.shape[1]} neurons while'
                         f' recording for the calibration of the BMI'),
            data=baseline_data,
            timestamps=baseline_indices_for_7.astype('float64'),
            unit='frames',
            comments=''.join(comments_baseline)
        )
        nwbfile_baseline.add_acquisition(baseline_online_neural_data)

        io_baseline.write(nwbfile_baseline)
        io_baseline.close()
        print('Baseline: done')

        # =====================================================================================================
        #                   Pretrain
        # =====================================================================================================

        print('Pretrain: initiating and creating the nwbfile')
        # convert data and open file
        folder_pretrain_im = Path(folder_raw) / row.session_path / 'im' / row.pretrain_im
        nwbfile_pretrain_path = f'{folder_nwb_mice / row.mice_name}_{row.session_date}_pretrain.nwb'
        convert_bruker_images_to_nwb(folder_pretrain_im, nwbfile_pretrain_path)
        io_pretrain = NWBHDF5IO(nwbfile_pretrain_path, mode='a')
        nwbfile_pretrain = io_pretrain.read()
        frame_rate = nwbfile_pretrain.acquisition['TwoPhotonSeries'].rate
        size_of_recording = nwbfile_pretrain.acquisition['TwoPhotonSeries'].data.shape[0]

        nwbfile_pretrain.add_device(microscope)

        print('Pretrain: retrieving online data')
        pretrain_mat = loadmat(folder_raw / row.session_path / row.pretrain_mat_file)['data']
        pretrain_data = pretrain_mat['bmiAct'].item()
        pretrain_data = pretrain_data[:, :np.where(~np.isnan(pretrain_data).all(axis=0))[0][-1]].T
        pretrain_calibration_mat = loadmat(folder_raw / row.session_path / row.target_calibration_mat_file)
        pretrain_rois_mat = loadmat(folder_raw / row.session_path / row.roi_mat_file)['roi_data']
        ensemble_indices = pretrain_calibration_mat['E_base_sel'].flatten()
        voltage_recording = folder_pretrain_im / row.pretrain_im_voltage_file
        _, _, peaks_I1, _, _, peaks_I4, peaks_I5, peaks_I6, peaks_I7, comments_pretrain = (
            svr.obtain_peaks_voltage(voltage_recording, frame_rate, size_of_recording))
        pretrain_indices_for_4 = svr.obtain_indices_per_peaks(peaks_I1, peaks_I4)
        pretrain_indices_for_5 = svr.obtain_indices_per_peaks(peaks_I1, peaks_I5)
        pretrain_indices_for_6 = svr.obtain_indices_per_peaks(peaks_I1, peaks_I6)
        pretrain_indices_for_7 = svr.obtain_indices_per_peaks(peaks_I1, peaks_I7)

        # initialize stims, will only change if the experiment is 'h'
        all_stims = 0

        if pretrain_indices_for_7.shape[0] < pretrain_data.shape[0]:
            pretrain_data = pretrain_data[:pretrain_indices_for_7.shape[0], :]
            comments_pretrain.append(
                'Pretrain data has more items than triggers were obtained from the voltage file')
        elif pretrain_indices_for_7.shape[0] > pretrain_data.shape[0]:
            pretrain_indices_for_7 = pretrain_indices_for_7[:pretrain_data.shape[0]]
            comments_pretrain.append(
                'Pretrain data has less items than triggers were obtained from the voltage file')
        else:
            comments_pretrain.append('conversion worked correctly')

        print('Pretrain: neural data')
        pretrain_online_neural_data = TimeSeries(
            name='Online_neural_activity',
            description=(f'neural data obtained online from {pretrain_data.shape[1]} neurons while'
                         f' performing the pretrain of the BMI'),
            data=pretrain_data,
            timestamps=pretrain_indices_for_7.astype('float64'),
            unit='frames',
            comments=''.join(comments_pretrain)
        )
        nwbfile_pretrain.add_acquisition(pretrain_online_neural_data)

        # if pretrain includes holo-stimulation the file name starts with an h
        if row.experiment_type[0] == 'h':
            # during pretrain there was a spurious artifact recorded as trigger at the beginning of recording
            if pretrain_indices_for_6[0] < 10:
                pretrain_indices_for_6 = pretrain_indices_for_6[1:]

            all_stims = pretrain_mat['schedHoloCounter'].item().item()
            if all_stims != len(pretrain_indices_for_6):
                comments_pretrain_stim = 'The number of stims recorded online is not consistent with data retrieved'
            else:
                comments_pretrain_stim = 'All stims correctly retrieved from experimental data'

            # store the voltage_rec stim times
            pretrain_stim_times = TimeSeries(
                name='Stim_times',
                description=(f'All ensemble neurons stim times, information is in timestamps, data is ones'),
                data=np.ones(pretrain_indices_for_6.shape[0]),
                timestamps=pretrain_indices_for_6.astype('float64'),
                unit='frames',
                comments=''.join(comments_pretrain_stim)
            )
            nwbfile_pretrain.add_acquisition(pretrain_stim_times)

            # if there was stim, add the devices
            nwbfile_pretrain.add_device(holo_light_source)
            nwbfile_pretrain.add_device(holo_spatial_light_modulator)

            print('Pretrain: holographic stim data')
            tree_xml = ET.parse(folder_raw / row.session_path / row.xml_ensemble)
            tree_gpl = ET.parse(folder_raw / row.session_path / row.gpl_ensemble)
            stim_point, delay, duration, spiral_rev, spiral_size, loc_x, loc_y, loc_z = (
                retrieve_holographic_point_data(tree_xml, tree_gpl))

            # creating and storing optogenetic stimulus site, stimulation pattern (spiral scanning, unique for every neuron),
            # and patterned series (also, unique for every neuron)
            pretrain_stim_site = PatternedOptogeneticStimulusSite(
                name='Holographic_location',
                device=holo_light_source,
                description='Sequential/semi-simultaneous stimulation of ' + row.experiment_type[1:3] + ' neurons',
                excitation_lambda=1035.,  # nm
                location='motor cortex',
                effector='ChRmine'  # ligth effector protein, not required
            )
            nwbfile_pretrain.add_ogen_site(pretrain_stim_site)

            for pt in np.arange(len(stim_point)):
                pretrain_stim_pattern = SpiralScanning(
                    # spiral scanning parameters should be correct
                    name='Single cell stim on neuron ' + str(stim_point[pt]),
                    description='One cell at a time stimulation',
                    duration=duration[pt],
                    number_of_stimulus_presentation=1,
                    inter_stimulus_interval=delay[pt],
                    diameter=spiral_size[pt],
                    height=loc_z[pt],
                    number_of_revolutions=spiral_rev[pt]
                )
                nwbfile_pretrain.add_lab_meta_data(pretrain_stim_pattern)

                pretrain_stim_series = PatternedOptogeneticSeries(
                    name='Single cell stim on neuron ' + str(stim_point[pt]),
                    rate=0.,  # float
                    unit='',  # usually watts
                    description='One cell at a time stimulation',
                    site=pretrain_stim_site,
                    device=microscope,
                    light_source=holo_light_source,
                    spatial_light_modulator=holo_spatial_light_modulator,
                    stimulus_pattern=pretrain_stim_pattern,
                    center_rois=np.expand_dims(np.array([loc_x[pt], loc_y[pt], spiral_size[pt]]), axis=0)
                )
                nwbfile_pretrain.add_acquisition(pretrain_stim_series)

        print('Pretrain: BMI calibration/Parameters/Results')
        # retrieve the calibration metadata
        pretrain_calibration = Calibration_metadata(
            name='Calibration_metadata',
            description='Information needed to calibrate the BMI',
            feedback_flag=False,  # bool
            ensemble_indexes=ensemble_indices,
            decoder=pretrain_calibration_mat['decoder'].flatten(),
            target=pretrain_calibration_mat['T'].flatten(),
            ensemble_mean=pretrain_calibration_mat['n_mean'].flatten(),
            ensemble_sd=pretrain_calibration_mat['n_std'].flatten()
        )
        nwbfile_pretrain.add_lab_meta_data(pretrain_calibration)

        # retrieve the parameters used in the BMI
        pretrain_parameters = Parameters_BMI(
            name='BMI_parameters',
            description='Parameters used to run the BMI',
            back_to_baseline_frames=pretrain_calibration_mat['back2BaseFramesThresh'].item(),
            prefix_window_frames=pretrain_calibration_mat['prefix_win'].item(),
            dff_baseline_window_frames=pretrain_calibration_mat['f0_win'].item(),
            smooth_window_frames=pretrain_calibration_mat['dff_win'].item(),
            cursor_zscore_bool=False,
            relaxation_window_frames=0,
            timeout_window_frames=0,
            back_to_baseline_threshold=pretrain_calibration_mat['b2base_thresh'].flatten(),
            conditions_rule=np.array(['cursor >= c1', 'The mean of E1 <= c2', 'At least 3 out of 4 E2 neurons >= c3'],
                                     dtype='str'),
            conditions_target=np.array([pretrain_calibration_mat['T'].item(), pretrain_calibration_mat['E1_mean'].item(),
                                        pretrain_calibration_mat['E2_subord_mean'].flatten()], dtype='object'),
            frames_per_reward_range=pretrain_calibration_mat['frames_per_reward_range'].flatten().astype(int)
        )
        nwbfile_pretrain.add_lab_meta_data(pretrain_parameters)

        # retrieve the BMI results
        pretrain_results_series = CaBMISeries(
            name='Pretrain_online_results',
            description='Time series results of the CaBMI experiment',
            experiment_type=row.experiment_type,
            self_hit_counter=pretrain_mat['selfTargetCounter'].item().item(),
            stim_hit_counter=pretrain_mat['holoTargetCounter'].item().item(),
            self_reward_counter=pretrain_mat['selfTargetVTACounter'].item().item(),
            stim_reward_counter=pretrain_mat['holoTargetVTACounter'].item().item(),
            scheduled_stim_counter=all_stims,
            scheduled_reward_counter=pretrain_mat['schedVTACounter'].item().item(),
            trial_counter=pretrain_mat['trialCounter'].item().item(),
            number_of_hits=pretrain_mat['selfTargetCounter'].item().item()
                           + pretrain_mat['holoTargetCounter'].item().item(),
            last_frame=pretrain_mat['frame'].item().item(),
            target=pretrain_calibration_mat['T'].flatten(),
            cursor=pretrain_mat['cursor'].item().flatten(),
            raw_activity=pretrain_data,
            baseline_vector=pretrain_mat['baseVector'].item(),
            self_hits=pretrain_mat['selfHits'].item().flatten(),
            stim_hits=pretrain_mat['holoHits'].item().flatten(),
            self_reward=pretrain_mat['selfVTA'].item().flatten(),
            stim_reward=pretrain_mat['holoVTA'].item().flatten(),
            stim_delivery=pretrain_mat['holoDelivery'].item().flatten(),
            trial_start=pretrain_mat['trialStart'].item().flatten(),
            time_vector=pretrain_mat['timeVector'].item().flatten(),
            scheduled_stim=pretrain_mat['vectorHolo'].item().flatten(),
            scheduled_reward=pretrain_mat['vectorVTA'].item().flatten(),
        )
        nwbfile_pretrain.add_acquisition(pretrain_results_series)

        # retrieve the ROI data
        pretrain_roi = ROI_metadata(
            name='ROI_metadata',
            description='Location of the ROIs, in a binary 2D array',
            image_mask_roi=np.stack(
                pretrain_rois_mat['roi_bin_cell'].item().flatten())[ensemble_indices, :, :][:, :, :, np.newaxis])
        nwbfile_pretrain.add_acquisition(pretrain_roi)

        # retrieve the rewards timing
        if 'no' not in row.experiment_type.lower():
            print('Pretrain: Reward')
            all_rewards = (pretrain_mat['selfTargetVTACounter'].item().item() +
                           pretrain_mat['holoTargetVTACounter'].item().item() +
                           pretrain_mat['schedVTACounter'].item().item())
            if all_rewards != len(pretrain_indices_for_5):
                comments_pretrain_reward = 'The number of rewards recorded online is not consistent with data retrieved'
            else:
                comments_pretrain_reward = 'All rewards correctly retrieved from experimental data'

            pretrain_reward_times = TimeSeries(
                name='Reward_times',
                description=(f'reward times, information is in timestamps, data is ones'),
                data=np.ones(pretrain_indices_for_5.shape[0]),
                timestamps=pretrain_indices_for_5.astype('float64'),
                unit='frames',
                comments=''.join(comments_pretrain_reward)
            )
            nwbfile_pretrain.add_acquisition(pretrain_reward_times)

        io_pretrain.write(nwbfile_pretrain)
        io_pretrain.close()
        print('Pretrain: done')

        # =====================================================================================================
        #                   BMI
        # =====================================================================================================

        print('BMI: initiating and creating the nwbfile')
        # convert data and open file
        folder_bmi_im = Path(folder_raw) / row.session_path / 'im' / row.bmi_im
        nwbfile_bmi_path = f'{folder_nwb_mice / row.mice_name}_{row.session_date}_bmi.nwb'
        convert_bruker_images_to_nwb(folder_bmi_im, nwbfile_bmi_path)
        io_bmi = NWBHDF5IO(nwbfile_bmi_path, mode='a')
        nwbfile_bmi = io_bmi.read()
        frame_rate = nwbfile_bmi.acquisition['TwoPhotonSeries'].rate
        size_of_recording = nwbfile_bmi.acquisition['TwoPhotonSeries'].data.shape[0]

        nwbfile_bmi.add_device(microscope)

        print('BMI: retrieving online data')
        bmi_mat = loadmat(folder_raw / row.session_path / row.bmi_mat_file)['data']
        bmi_data = bmi_mat['bmiAct'].item()
        bmi_data = bmi_data[:, :np.where(~np.isnan(bmi_data).all(axis=0))[0][-1]].T
        voltage_recording = folder_bmi_im / row.bmi_im_voltage_file
        _, _, peaks_I1, _, _, peaks_I4, peaks_I5, _, peaks_I7, comments_bmi = (
            svr.obtain_peaks_voltage(voltage_recording, frame_rate, size_of_recording))
        bmi_indices_for_4 = svr.obtain_indices_per_peaks(peaks_I1, peaks_I4)
        bmi_indices_for_5 = svr.obtain_indices_per_peaks(peaks_I1, peaks_I5)
        bmi_indices_for_7 = svr.obtain_indices_per_peaks(peaks_I1, peaks_I7)

        if bmi_indices_for_7.shape[0] < bmi_data.shape[0]:
            bmi_data = bmi_data[:bmi_indices_for_7.shape[0], :]
            comments_bmi.append(
                'BMI data has more items than triggers were obtained from the voltage file')
            raise Warning(comments_bmi)
        elif bmi_indices_for_7.shape[0] > bmi_data.shape[0]:
            bmi_indices_for_7 = bmi_indices_for_7[:bmi_data.shape[0]]
            comments_bmi.append(
                'BMI data has less items than triggers were obtained from the voltage file')
            raise Warning(comments_bmi)
        else:
            comments_bmi.append('conversion worked correctly')

        print('BMI: neural data')
        online_neural_data = TimeSeries(
            name='Online_neural_activity',
            description=(f'neural data obtained online from {bmi_data.shape[1]} neurons while'
                         f' performing the BMI'),
            data=bmi_data,
            timestamps=bmi_indices_for_7.astype('float64'),
            unit='frames',
        )
        nwbfile_bmi.add_acquisition(online_neural_data)

        print('BMI: BMI calibration/Parameters/Results')

        # add the calibration/parameters which are the same as the ones in pretrain
        nwbfile_bmi.add_lab_meta_data(pretrain_parameters)
        nwbfile_bmi.add_lab_meta_data(pretrain_calibration)

        # creating and storing BMI related data
        if 'fb' in row.experiment_type.lower():
            experiment_bmi = 'audio_feedback'
            cursor_audio = bmi_mat['fb_freq'].item().flatten()
            if pretrain_calibration_mat['fb_settings']['target_low_freq'].item().item():
                freq = pretrain_calibration_mat['fb_settings']['freq_min'].item().flatten()
            else:
                freq = pretrain_calibration_mat['fb_settings']['freq_max'].item().flatten()

            bmi_calibration = Calibration_metadata(
                name='Calibration_feedback',
                description='Calibration needed to run auditory feedback during the BMI part of the experiment',
                feedback_flag=True,  # bool
                feedback_target=freq,  # 1D array, dtype float, dims number of audio targets
            )
            nwbfile_bmi.add_lab_meta_data(bmi_calibration)
        else:
            experiment_bmi = 'no_audio_feedback'
            cursor_audio = np.array([])

        # retrieve timeseries data from BMI
        bmi_series = CaBMISeries(
            name='BMI_online_results',
            description='Time series results of the CaBMI experiment',
            experiment_type=experiment_bmi,
            self_hit_counter=bmi_mat['selfTargetCounter'].item().item(),
            self_reward_counter=bmi_mat['selfTargetVTACounter'].item().item(),
            trial_counter=bmi_mat['trialCounter'].item().item(),
            number_of_hits=bmi_mat['selfTargetCounter'].item().item(),
            last_frame=bmi_mat['frame'].item().item(),
            target=pretrain_calibration_mat['T'].flatten(),
            cursor=bmi_mat['cursor'].item().flatten(),
            cursor_audio=cursor_audio,
            raw_activity=bmi_data,
            baseline_vector=bmi_mat['baseVector'].item(),
            self_hits=bmi_mat['selfHits'].item().flatten(),
            self_reward=bmi_mat['selfVTA'].item().flatten(),
            trial_start=bmi_mat['trialStart'].item().flatten(),
            time_vector=bmi_mat['timeVector'].item().flatten(),
        )
        nwbfile_bmi.add_acquisition(bmi_series)

        # retrieve the rewards timing
        print('BMI: Reward')
        if bmi_mat['selfTargetVTACounter'].item().item() != len(bmi_indices_for_5):
            comments_bmi_reward = 'The number of rewards recorded online is not consistent with data retrieved'
        else:
            comments_bmi_reward = 'All rewards correctly retrieved from experimental data'

        bmi_reward_times = TimeSeries(
            name='Reward_times',
            description=(f'reward times, information is in timestamps, data is ones'),
            data=np.ones(bmi_indices_for_5.shape[0]),
            timestamps=bmi_indices_for_5.astype('float64'),
            unit='frames',
            comments=''.join(comments_bmi_reward)
        )
        nwbfile_bmi.add_acquisition(bmi_reward_times)

        nwbfile_bmi.add_acquisition(pretrain_roi)

        io_bmi.write(nwbfile_bmi)
        io_bmi.close()
        print('BMI: done')
