__author__ = 'Nuria'

# __author__ = ("Nuria", "John Doe")

# make sure to be in environment with pynwb and neuroconv installed

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from pynwb.device import Device

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


# you need to install neuroconv converter first. In this case we use:
# https://neuroconv.readthedocs.io/en/main/conversion_examples_gallery/imaging/brukertiff.html


def convert_bruker_images_to_nwb(folder_path: Path, nwbfile_path: str):
    """ function to convert a bruker tiff file recording to nwb
    :param folder_path: path to the folder containing the tiff files
    :param nwbfile_path: path where we want to store the nwb file"""
    # convert data to nwb
    converter = BrukerTiffSinglePlaneConverter(folder_path=folder_path)
    metadata = converter.get_metadata()
    session_start_time = metadata["NWBFile"]["session_start_time"]
    tzinfo = ZoneInfo("US/Eastern")
    metadata["NWBFile"].update(
        session_start_time=session_start_time.replace(tzinfo=tzinfo))  # TODO this doesn't seem to work properly
    converter.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata)


def convert_all_experiments_to_nwb(folder_raw: Path, experiment_type: str):
    """ function to convert all experiments within a experiment type to nwb"""
    df_sessions = ds.get_sessions_df(experiment_type)
    folder_nwb = folder_raw.parents[0] / 'nwb'
    
    #TODO Nuria - add information about microscope and light source
    microscope = Device(
        name='',
        description='',
        manufacturer='',
        model_number='',
        model_name='',
        serial_number=''
    )

    holo_light_source= LightSource(
            name='Monaco',
            description="Laser used for the holographic stim",
            manufacturer="Coherent",
            stimulation_wavelength=0., #float
            filter_description=""
        )
    
    holo_spatial_light_modulator = SpatialLightModulator(
            name="these experiments don't use a SLM",
            model='',
            resolution=0. #float
        )
    
    for index, row in df_sessions.iterrows():
        # TODO: Add behavior!

        folder_nwb_mice = folder_nwb / row.mice_name
        if not Path(folder_nwb_mice).exists():
            Path(folder_nwb_mice).mkdir(parents=True, exist_ok=True)

        # =====================================================================================================
        #                   Holographic sequential
        # =====================================================================================================
        # convert data and open file
        folder_holobmi_seq_im = Path(folder_raw) / row.session_path / 'im' / row.Holostim_seq_im
        nwbfile_holograph_seq_path = f"{folder_nwb_mice / row.mice_name}_{row.session_date}_holostim_seq.nwb"
        convert_bruker_images_to_nwb(folder_holobmi_seq_im, nwbfile_holograph_seq_path)
        io_holographic_seq = NWBHDF5IO(nwbfile_holograph_seq_path, mode="a")
        nwbfile_holographic_seq = io_holographic_seq.read()
        frame_rate = nwbfile_holographic_seq.acquisition['TwoPhotonSeries'].rate
        size_of_recording = nwbfile_holographic_seq.acquisition['TwoPhotonSeries'].data.shape[0]

        # add the device for holographic stim
        
        nwbfile_holographic_seq.add_device(microscope)
        nwbfile_holographic_seq.add_device(holo_light_source)
        nwbfile_holographic_seq.add_device(holo_spatial_light_modulator)

        # Save the neural data that was store in the mat file
        holostim_seq_data = loadmat(folder_raw / row.session_path / row.Holostim_seq_mat_file)['holoActivity']
        # select holodata that is not nan and transpose to have time x neurons
        holostim_seq_data = holostim_seq_data[:, ~np.isnan(np.sum(holostim_seq_data, 0))].T
        voltage_recording = folder_holobmi_seq_im / row.Holostim_seq_im_voltage_file
        _, _, peaks_I1, _, _, _, _, peaks_I6, peaks_I7, comments_holo = (
            svr.obtain_peaks_voltage(voltage_recording,frame_rate, size_of_recording))
        indices_for_6 = svr.obtain_indices_per_peaks(peaks_I1, peaks_I6)
        indices_for_7 = svr.obtain_indices_per_peaks(peaks_I1, peaks_I7)
        if indices_for_7.shape[0] < holostim_seq_data.shape[0]:
            holostim_seq_data = holostim_seq_data[:indices_for_7.shape[0], :]
            comments_holo.append('Holostim_seq data has more items than triggers were obtained from the voltage file')
            raise Warning(comments_holo)
        elif indices_for_7.shape[0] > holostim_seq_data.shape[0]:
            indices_for_7 = indices_for_7[:holostim_seq_data.shape[0]]
            comments_holo.append('Holostim_seq data has less items than triggers were obtained from the voltage file')
            raise Warning(comments_holo)
        else:
            comments_holo.append('conversion worked correctly')

        online_neural_data = TimeSeries(
            name="online_neural_activity",
            description=(f'neural data obtained online from {holostim_seq_data.shape[1]} neurons while'
                         f' recording for the sequential stimulation of all initial neurons'),
            data=holostim_seq_data,
            timestamps=indices_for_7.astype('float64'),
            unit="imaging frames",
            comments="".join(comments_holo)
        )
        nwbfile_holographic_seq.add_acquisition(online_neural_data)

        # obtain the holographic metadata and store it
        tree = ET.parse(folder_raw / row.session_path / row.XML_holostim_seq)
        troot = tree.getroot()
        power = []
        point = []
        index = []
        spiral_size = []
        for elem in troot.findall('PVMarkPointElement'):
            power.append(float(elem.get('UncagingLaserPower')))
            point.append(float(elem.find('PVGalvoPointElement').get('Points')[-1]))
            index.append(float(elem.find('PVGalvoPointElement').get('Indices')))
            spiral_size.append(float(elem.find('PVGalvoPointElement').get('SpiralSize')))
        if len(point) != len(index) or len(point) != holostim_seq_data.shape[1]:
            comments_holoseries = 'The number of stims locations is not consistent with data retrieved'
            Warning(comments_holoseries)
        else:
            comments_holoseries = 'All points were stimulated sequentially'
        
        #creating and storing optogenetic stimulus site, stimulation pattern (spiral scanning, unique for every neuron), and patterned series (also, unique for every neuron)
        #TODO Nuria - fill in the empty gaps
        holo_stim_site = PatternedOptogeneticStimulusSite(
            name="Holographic sequential location",
            device=holo_light_source,
            description="Sequential stimulation of all the neurons selected as initial ROIs",
            excitation_lambda=1035.,  # nm
            location="all initial ROIs",
            effector='' #ligth effector protein, not required
        )
        nwbfile_holographic_seq.add_ogen_site(holo_stim_site)
        for pt in np.arange(len(point)):
            holo_stim_pattern = SpiralScanning(
                #spiral scanning parameters should be correct
                name=str(index[pt]),
                description='',
                duration=30.0,
                number_of_stimulus_presentation=1,
                inter_stimulus_interval=2000,
                diameter=spiral_size[pt],
                height=0.0, 
                number_of_revolutions=10
            )
            nwbfile_holographic_seq.add_lab_meta_data(holo_stim_pattern)

            holo_seq_series = PatternedOptogeneticSeries(
                name='',
                rate=0., #float
                unit='', #usually watts
                description='',
                site=holo_stim_site,
                device=microscope,
                light_source=holo_light_source,
                spatial_light_modulator=holo_spatial_light_modulator,
                stimulus_pattern=holo_stim_pattern,
                pixel_rois=np.array([]) #2D array for pixels ([x, y]) 3D for voxels ([x, y, z])
            )
            nwbfile_holographic_seq.add_acquisition(holo_seq_series)

        # write and close the nwb file
        io_holographic_seq.write(nwbfile_holographic_seq)
        io_holographic_seq.close()

        # =====================================================================================================
        #                   Baseline
        # =====================================================================================================
        # convert data and open file
        folder_baseline_im = Path(folder_raw) / row.session_path / 'im' / row.Baseline_im
        nwbfile_baseline_path = f"{folder_nwb_mice / row.mice_name}_{row.session_date}_baseline.nwb"
        convert_bruker_images_to_nwb(folder_baseline_im, nwbfile_baseline_path)
        io_baseline = NWBHDF5IO(nwbfile_baseline_path, mode="a")
        nwbfile_baseline = io_baseline.read()
        frame_rate = nwbfile_baseline.acquisition['TwoPhotonSeries'].rate
        size_of_recording = nwbfile_baseline.acquisition['TwoPhotonSeries'].data.shape[0]

        baseline_data = loadmat(folder_raw / row.session_path / row.Baseline_mat_file)['baseActivity']
        baseline_data = baseline_data[:, ~np.isnan(np.sum(baseline_data, 0))].T
        voltage_recording = folder_baseline_im / row.Baseline_im_voltage_file
        _, _, peaks_I1, _, _, peaks_I4, _, _, peaks_I7, comments_baseline = (
            svr.obtain_peaks_voltage(voltage_recording,frame_rate, size_of_recording))
        indices_for_4 = svr.obtain_indices_per_peaks(peaks_I1, peaks_I4)
        indices_for_7 = svr.obtain_indices_per_peaks(peaks_I1, peaks_I7)

        if indices_for_7.shape[0] < baseline_data.shape[0]:
            baseline_data = baseline_data[:indices_for_7.shape[0], :]
            comments_baseline.append(
                'Baseline data has more items than triggers were obtained from the voltage file')
            raise Warning(comments_baseline)
        elif indices_for_7.shape[0] > baseline_data.shape[0]:
            indices_for_7 = indices_for_7[:baseline_data.shape[0]]
            comments_baseline.append(
                'Baseline data has less items than triggers were obtained from the voltage file')
            raise Warning(comments_baseline)
        else:
            comments_baseline.append('conversion worked correctly')

        online_neural_data = TimeSeries(
            name="online_neural_activity",
            description=(f'neural data obtained online from {baseline_data.shape[1]} neurons while'
                         f' recording for the calibration of the BMI'),
            data=baseline_data,
            timestamps=indices_for_7.astype('float64'),
            unit="imaging frames",
        )
        nwbfile_baseline.add_acquisition(online_neural_data)
        io_baseline.write(nwbfile_baseline)
        io_baseline.close()

        # =====================================================================================================
        #                   Pretrain
        # =====================================================================================================
        # convert data and open file
        folder_pretrain_im = Path(folder_raw) / row.session_path / 'im' / row.Pretrain_im
        nwbfile_pretrain_path = f"{folder_nwb_mice / row.mice_name}_{row.session_date}_pretrain.nwb"
        convert_bruker_images_to_nwb(folder_pretrain_im, nwbfile_pretrain_path)
        io_pretrain = NWBHDF5IO(nwbfile_pretrain_path, mode="a")
        nwbfile_pretrain = io_pretrain.read()
        frame_rate = nwbfile_pretrain.acquisition['TwoPhotonSeries'].rate
        size_of_recording = nwbfile_pretrain.acquisition['TwoPhotonSeries'].data.shape[0]

        pretrain_data = loadmat(folder_raw / row.session_path / row.Pretrain_mat_file)['data']['bmiAct'][0][0]
        pretrain_data = pretrain_data[:, :np.where(~np.isnan(pretrain_data).all(axis=0))[0][-1]].T
        voltage_recording = folder_pretrain_im / row.pretrain_im_voltage_file
        _, _, peaks_I1, _, _, peaks_I4, peaks_I5, peaks_I6, peaks_I7, comments_pretrain = (
            svr.obtain_peaks_voltage(voltage_recording,frame_rate, size_of_recording))
        indices_for_4 = svr.obtain_indices_per_peaks(peaks_I1, peaks_I4)
        indices_for_5 = svr.obtain_indices_per_peaks(peaks_I1, peaks_I5)
        indices_for_6 = svr.obtain_indices_per_peaks(peaks_I1, peaks_I6)
        indices_for_7 = svr.obtain_indices_per_peaks(peaks_I1, peaks_I7)

        if indices_for_7.shape[0] < pretrain_data.shape[0]:
            pretrain_data = pretrain_data[:indices_for_7.shape[0], :]
            comments_pretrain.append(
                'Pretrain data has more items than triggers were obtained from the voltage file')
            raise Warning(comments_pretrain)
        elif indices_for_7.shape[0] > pretrain_data.shape[0]:
            indices_for_7 = indices_for_7[:pretrain_data.shape[0]]
            comments_pretrain.append(
                'Pretrain data has less items than triggers were obtained from the voltage file')
            raise Warning(comments_pretrain)
        else:
            comments_pretrain.append('conversion worked correctly')

        online_neural_data = TimeSeries(
            name="online_neural_activity",
            description=(f'neural data obtained online from {pretrain_data.shape[1]} neurons while'
                         f' performing the pretrain of the BMI'),
            data=pretrain_data,
            timestamps=indices_for_7.astype('float64'),
            unit="imaging frames",
        )
        nwbfile_pretrain.add_acquisition(online_neural_data)
        
        #if pretrain includes holobmi the file name starts with an h. The program includes holostim consequently
        #TODO Nuria - fill in the empty gaps
        if row.mice_name[0] == "h":
            nwbfile_pretrain.add_device(microscope)
            nwbfile_pretrain.add_device(holo_light_source)
            nwbfile_pretrain.add_device(holo_spatial_light_modulator)

            pretrain_holo_stim_site = PatternedOptogeneticStimulusSite(
            name="Holographic sequential location",
            device=holo_light_source,
            description="Sequential stimulation of all the neurons selected as initial ROIs",
            excitation_lambda=1035.,  # nm
            location="all initial ROIs",
            effector=''
            )
            nwbfile_pretrain.add_ogen_site(pretrain_holo_stim_site)
            
            #in case the spiral scanning is different for every neuron than this needs a for loop 
            pretrain_holo_stim_pattern = SpiralScanning(
                name='',
                description='',
                duration=0., #float
                number_of_stimulus_presentation=0, #int
                inter_stimulus_interval=0., #float
                diameter=0., #float
                height=0., #float
                number_of_revolutions=0 #int
            )
            nwbfile_pretrain.add_lab_meta_data(pretrain_holo_stim_pattern)

            pretrain_holo_seq_series = PatternedOptogeneticSeries(
                name="Holographic sequential",
                rate=0., #float
                unit='', #usually watts
                description='',
                site=pretrain_holo_stim_site,
                device=microscope,
                light_source=holo_light_source,
                spatial_light_modulator=holo_spatial_light_modulator,
                stimulus_pattern=pretrain_holo_stim_pattern,
                pixel_rois=np.array([]) #2D array for pixels ([x, y]) 3D for voxels ([x, y, z])
            )
            nwbfile_pretrain.add_acquisition(pretrain_holo_seq_series)

        #if the file name does not start with h then holostim is not added, BMI is added regardless
        #TODO Nuria- fill in the empty gaps     
        pretrain_calibration = Calibration_metadata(
            name='',
            description='',
            category='',
            about='',
            feedback_flag=False, #bool
            ensemble_indexes=np.array([]), #1D array, dtype int, dims number of ensemble neurons,
            decoder=np.array([]), #1D array, dtype float, dims number of ensemble neurons
            target=np.array([]), #1D array, dtype float, dims number of targets
            feedback_target=np.array([]), #1D array, dtype float, dims number of audio targets
            ensemble_mean=np.array([]), #1D array, dtype float, dims number of ensemble neurons
            ensemble_sd=np.array([]) #1D array, dtype float, dims number of ensemble neurons
        )
        nwbfile_pretrain.add_lab_meta_data(pretrain_calibration)

        pretrain_parameters = Parameters_BMI(
            name='',
            description='',
            category='',
            about='',
            back_to_baseline_frames=0, #int
            prefix_window_frames=0, #int
            dff_baseline_window_frames=0, #int
            smooth_window_frames=0, #int
            cursor_zscore_bool=False, #bool
            relaxation_window_frames=0, #int
            timelimit_frames=0, #int
            timeout_window_frames=0, #int
            back_to_baseline_threshold=np.array([]), #1D array, dtype float, dims number of targets
            conditions_target=np.array([]), #1D array, dtype float, dims number of conditions
            seconds_per_reward_range=np.array([]) #1D array, dtype int, dims lower_value/higher_value (2)
        )
        nwbfile_pretrain.add_lab_meta_data(pretrain_parameters)

        pretrain_series = CaBMISeries(
            name='',
            about='',
            self_hit_counter=0, #int 
            stim_hit_counter=0, #int
            self_reward_counter=0, #int
            stim_reward_counter=0, #int
            scheduled_stim_counter=0, #int
            scheduled_reward_counter=0, #int
            trial_counter=0, #int
            number_of_hits=0, #int
            number_of_misses=0, #int
            last_frame=0, #int
            target=np.array([]), #1D array, dtype float, dims number of targets
            cursor=np.array([]), #1D array, dtype float, dims degree_freedom/BMI_frames
            cursor_audio=np.array([]), #1D array, dtype int, dims degree_freedom/BMI_frames
            raw_activity=np.array([]), #2D array, dtype float, dims number of ensemble neurons - BMi_frames
            baseline_vector=np.array([]), #2D array, dtype float, dims number of ensemble neurons - BMi_frames
            self_hits=np.array([]), #1D array, dtype bool, dims BMI_frames
            stim_hits=np.array([]), #1D array, dtype bool, dims BMI_frames
            self_reward=np.array([]), #1D array, dtype bool, dims BMI_frames
            stim_reward=np.array([]), #1D array, dtype bool, dims BMI_frames
            stim_delivery=np.array([]), #1D array, dtype bool, dims BMI_frames
            trial_start=np.array([]), #1D array, dtype bool, dims BMI_frames
            time_vector=np.array([]), #1D array, dtype float, dims BMI_frames
            scheduled_stim=np.array([]), #1D array, dtype int, dims number_stims
            scheduled_reward=np.array([]), #1D array, dtype int, dims number_rewards
        )
        nwbfile_pretrain.add_acquisition(pretrain_series)

        pretrain_roi = ROI_metadata(
            name='',
            description='',
            category='',
            about='',
            pixel_rois=np.array([]) #2D array for pixels ([x, y]) 3D for voxels ([x, y, z])
        )
        nwbfile_pretrain.add_acquisition(pretrain_roi)


        io_pretrain.write(nwbfile_pretrain)
        io_pretrain.close()

        # =====================================================================================================
        #                   BMI
        # =====================================================================================================
        # convert data and open file
        folder_bmi_im = Path(folder_raw) / row.session_path / 'im' / row.BMI_im
        nwbfile_bmi_path = f"{folder_nwb_mice / row.mice_name}_{row.session_date}_bmi.nwb"
        convert_bruker_images_to_nwb(folder_bmi_im, nwbfile_bmi_path)
        io_bmi = NWBHDF5IO(nwbfile_bmi_path, mode="a")
        nwbfile_bmi = io_bmi.read()
        frame_rate = nwbfile_bmi.acquisition['TwoPhotonSeries'].rate
        size_of_recording = nwbfile_bmi.acquisition['TwoPhotonSeries'].data.shape[0]

        bmi_data = loadmat(folder_raw / row.session_path / row.BMI_mat_file)['data']['bmiAct'][0][0]
        bmi_data = bmi_data[:, :np.where(~np.isnan(bmi_data).all(axis=0))[0][-1]].T
        voltage_recording = folder_bmi_im / row.BMI_im_voltage_file
        _, _, peaks_I1, _, _, peaks_I4, peaks_I5, peaks_I6, peaks_I7, comments_bmi = (
            svr.obtain_peaks_voltage(voltage_recording,frame_rate, size_of_recording))
        indices_for_4 = svr.obtain_indices_per_peaks(peaks_I1, peaks_I4)
        indices_for_5 = svr.obtain_indices_per_peaks(peaks_I1, peaks_I5)
        indices_for_6 = svr.obtain_indices_per_peaks(peaks_I1, peaks_I6)
        indices_for_7 = svr.obtain_indices_per_peaks(peaks_I1, peaks_I7)

        if indices_for_7.shape[0] < bmi_data.shape[0]:
            bmi_data = bmi_data[:indices_for_7.shape[0], :]
            comments_bmi.append(
                'BMI data has more items than triggers were obtained from the voltage file')
            raise Warning(comments_bmi)
        elif indices_for_7.shape[0] > bmi_data.shape[0]:
            indices_for_7 = indices_for_7[:bmi_data.shape[0]]
            comments_bmi.append(
                'BMI data has less items than triggers were obtained from the voltage file')
            raise Warning(comments_bmi)
        else:
            comments_bmi.append('conversion worked correctly')

        online_neural_data = TimeSeries(
            name="online_neural_activity",
            description=(f'neural data obtained online from {bmi_data.shape[1]} neurons while'
                         f' performing the BMI'),
            data=bmi_data,
            timestamps=indices_for_7.astype('float64'),
            unit="imaging frames",
        )
        nwbfile_bmi.add_acquisition(online_neural_data)
        
        #creating and storing BMI related data
        #TODO Nuria - fill the empty gaps
        bmi_calibration = Calibration_metadata(
            name='',
            description='',
            category='',
            about='',
            feedback_flag=False, #bool
            ensemble_indexes=np.array([]), #1D array, dtype int, dims number of ensemble neurons,
            decoder=np.array([]), #1D array, dtype float, dims number of ensemble neurons
            target=np.array([]), #1D array, dtype float, dims number of targets
            feedback_target=np.array([]), #1D array, dtype float, dims number of audio targets
            ensemble_mean=np.array([]), #1D array, dtype float, dims number of ensemble neurons
            ensemble_sd=np.array([]) #1D array, dtype float, dims number of ensemble neurons
        )
        nwbfile_bmi.add_lab_meta_data(bmi_calibration)

        bmi_parameters = Parameters_BMI(
            name='',
            description='',
            category='',
            about='',
            back_to_baseline_frames=0, #int
            prefix_window_frames=0, #int
            dff_baseline_window_frames=0, #int
            smooth_window_frames=0, #int
            cursor_zscore_bool=False, #bool
            relaxation_window_frames=0, #int
            timelimit_frames=0, #int
            timeout_window_frames=0, #int
            back_to_baseline_threshold=np.array([]), #1D array, dtype float, dims number of targets
            conditions_target=np.array([]), #1D array, dtype float, dims number of conditions
            seconds_per_reward_range=np.array([]) #1D array, dtype int, dims lower_value/higher_value (2)
        )
        nwbfile_bmi.add_lab_meta_data(bmi_parameters)

        bmi_series = CaBMISeries(
            name='',
            about='',
            self_hit_counter=0, #int 
            stim_hit_counter=0, #int
            self_reward_counter=0, #int
            stim_reward_counter=0, #int
            scheduled_stim_counter=0, #int
            scheduled_reward_counter=0, #int
            trial_counter=0, #int
            number_of_hits=0, #int
            number_of_misses=0, #int
            last_frame=0, #int
            target=np.array([]), #1D array, dtype float, dims number of targets
            cursor=np.array([]), #1D array, dtype float, dims degree_freedom/BMI_frames
            cursor_audio=np.array([]), #1D array, dtype int, dims degree_freedom/BMI_frames
            raw_activity=np.array([]), #2D array, dtype float, dims number of ensemble neurons - BMi_frames
            baseline_vector=np.array([]), #2D array, dtype float, dims number of ensemble neurons - BMi_frames
            self_hits=np.array([]), #1D array, dtype bool, dims BMI_frames
            stim_hits=np.array([]), #1D array, dtype bool, dims BMI_frames
            self_reward=np.array([]), #1D array, dtype bool, dims BMI_frames
            stim_reward=np.array([]), #1D array, dtype bool, dims BMI_frames
            stim_delivery=np.array([]), #1D array, dtype bool, dims BMI_frames
            trial_start=np.array([]), #1D array, dtype bool, dims BMI_frames
            time_vector=np.array([]), #1D array, dtype float, dims BMI_frames
            scheduled_stim=np.array([]), #1D array, dtype int, dims number_stims
            scheduled_reward=np.array([]), #1D array, dtype int, dims number_rewards
        )
        nwbfile_bmi.add_acquisition(bmi_series)

        bmi_roi = ROI_metadata(
            name='',
            description='',
            category='',
            about='',
            pixel_rois=np.array([]) #2D array for pixels ([x, y]) 3D for voxels ([x, y, z])
        )
        nwbfile_bmi.add_acquisition(bmi_roi)
        
        io_bmi.write(nwbfile_bmi)
        io_bmi.close()