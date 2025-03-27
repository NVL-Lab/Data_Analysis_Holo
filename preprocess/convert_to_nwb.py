__author__ = 'Nuria'

# __author__ = ("Nuria", "John Doe")

# make sure to be in environment with pynwb and neuroconv installed

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from scipy.io import loadmat
from pynwb import NWBFile, NWBHDF5IO, TimeSeries, ogen
from pathlib import Path
from zoneinfo import ZoneInfo
from neuroconv.converters import BrukerTiffSinglePlaneConverter

from preprocess import dataframe_sessions as ds
from preprocess import syncronize_voltage_rec as svr
from utils import analysis_constants as act

from pynwb.ophys import PlaneSegmentation, ImageSegmentation, OpticalChannel
from ndx_holographic_stimulation import (
    PatternedOptogeneticSeries,
    PatternedOptogeneticStimulusSite,
    OptogeneticStimulusPattern,
    SpiralScanning,
    TemporalFocusing,
    SpatialLightModulator,
    LightSource,
)



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

    for index, row in df_sessions.iterrows():
        folder_bmi_im = Path(folder_raw) / row.session_path / 'im' / row.BMI_im

        folder_nwb_mice = folder_nwb / row.mice_name
        if not Path(folder_nwb_mice).exists():
            Path(folder_nwb_mice).mkdir(parents=True, exist_ok=True)

        # =====================================================================================================
        #                   Holographic sequential
        # =====================================================================================================
        # convert data and open file
        folder_holobmi_seq_im = Path(folder_raw) / row.session_path / 'im' / row.Holostim_seq_im
        nwbfile_holograph_seq_path = f"{folder_nwb_mice / row.mice_name}_{row.session_date}_raw_holostim_seq.nwb"
        convert_bruker_images_to_nwb(folder_holobmi_seq_im, nwbfile_holograph_seq_path)
        io_holographic_seq = NWBHDF5IO(nwbfile_holograph_seq_path, mode="a")
        nwbfile_holographic_seq = io_holographic_seq.read()
        frame_rate = nwbfile_holographic_seq.acquisition['TwoPhotonSeries'].rate
        size_of_recording = nwbfile_holographic_seq.acquisition['TwoPhotonSeries'].data.shape[0]

        # create the device for holographic stim
        holographic_device = nwbfile_holographic_seq.create_device(
            name='Monaco',
            description="Laser used for the holographic stim",
            manufacturer="Coherent",
        )
        # The optical channel is defined, placeholders for data ###TO DO
        optical_channel = OpticalChannel(
            name = "name",
            description = "description",
            emission_lambda = 500.0
        )
        # The imaging plane is defined, placeholders for most data ###TO DO
        imaging_plane = nwbfile_holographic_seq.create_imaging_plane(
            name = "name",
            optical_channel = optical_channel,
            imaging_rate = frame_rate,
            description = "description",
            device = holographic_device,
            excitation_lambda = 1035.,
            indicator = "indicator",
            location = "location",
            grid_spacing = [0.01, 0.01],
            grid_spacing_unit = "unit",
            origin_coords = [1.0, 2.0, 3.0],
            origin_coords_unit = "unit",
        )
        #finally, ROIs are defined
        n_rois = 2
        plane_segmentation = PlaneSegmentation(
            name="PlaneSegmentation",
            description="output from segmenting my favorite imaging plane",
            imaging_plane=imaging_plane,
        )
        for _ in range(n_rois):
            plane_segmentation.add_roi(image_mask=np.zeros((10, 10)))

        if nwbfile_holographic_seq is not None:
            if "ophys" not in nwbfile_holographic_seq.processing:
                nwbfile_holographic_seq.create_processing_module("ophys", "ophys")
            nwbfile_holographic_seq.processing["ophys"].add(plane_segmentation)

        roi_table_region = plane_segmentation.create_roi_table_region(
            region=[0, 1], description="the first of two ROIs"
        )
        
        # Save the neural data that was store in the mat file
        holostim_seq_data = loadmat(folder_raw / row.session_path / row.holostim_seq_mat_file)['holoActivity']
        # select holodata that is not nan and transpose to have time x neurons
        holostim_seq_data = holostim_seq_data[:, ~np.isnan(np.sum(holostim_seq_data, 0))].T
        voltage_recording = folder_holobmi_seq_im / row.Holostim_seq_im_voltage_file
        peaks_I1, _, indices_for_6, indices_for_7, comments_holo = svr.obtain_peaks_voltage(voltage_recording, frame_rate,
                                                                             size_of_recording)
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
            comments=comments_holo
        )
        nwbfile_holographic_seq.add_acquisition(online_neural_data)

        # obtain the holographic metadata and store it (This is done using optogenetic module and not the
        # ndx-holographic which is not working
        tree = ET.parse(folder_raw / row.session_path / row.XML_holostim_seq)
        troot = tree.getroot()
        power = []
        point = []
        index = []
        for elem in troot.findall('PVMarkPointElement'):
            power.append(float(elem.get('UncagingLaserPower')))
            point.append(float(elem.find('PVGalvoPointElement').get('Points')[-1]))
            index.append(float(elem.find('PVGalvoPointElement').get('Indices')))
        if len(point) != len(index) or len(point) != holostim_seq_data.shape[1]:
            comments_holoseries = 'The number of stims locations is not consistent with data retrieved'
            Warning(comments_holoseries)
        else:
            comments_holoseries = 'All points were stimulated sequentially'

        holo_stim_seq_site = PatternedOptogeneticStimulusSite(
            name="Holographic sequential location",
            device=holographic_device,
            description="Sequential stimulation of all the neurons selected as initial ROIs",
            excitation_lambda=1035.,  # nm
            effector = "placeholder", #todo
            location="all initial ROIs",
        )
        nwbfile_holographic_seq.add_ogen_site(holo_stim_seq_site)

        holo_seq_series = PatternedOptogeneticSeries(
            name = "Holographic sequential",
            description = "Tuple with information about the index of stim, the neuron stimulated and the power",
            data = list(zip(index, point, power)),
            unit = "placeholder", #watts? todo
            timestamps = indices_for_6.astype('float64'),
            rois = roi_table_region,
            stimulus_pattern = "pattern", #todo, many possible patterns exist
            site = holo_stim_seq_site,
            light_source = "light_source", #todo    
            spatial_light_modulator = "modulator" #todo
        )
        nwbfile_holographic_seq.add_stimulus(holo_seq_series)

        # write and close the nwb file
        io_holographic_seq.write(nwbfile_holographic_seq)
        io_holographic_seq.close()

        # =====================================================================================================
        #                   Baseline
        # =====================================================================================================
        # convert data and open file
        folder_baseline_im = Path(folder_raw) / row.session_path / 'im' / row.Baseline_im
        nwbfile_baseline_path = f"{folder_nwb_mice / row.mice_name}_{row.session_date}_raw_baseline.nwb"
        convert_bruker_images_to_nwb(folder_baseline_im, nwbfile_baseline_path)
        io_baseline = NWBHDF5IO(nwbfile_baseline_path, mode="a")
        nwbfile_baseline = io_baseline.read()
        frame_rate = nwbfile_baseline.acquisition['TwoPhotonSeries'].rate
        size_of_recording = nwbfile_baseline.acquisition['TwoPhotonSeries'].data.shape[0]

        baseline_data = loadmat(folder_raw / row.session_path / row.baseline_mat_file)['baseActivity']
        baseline_data = baseline_data[:, ~np.isnan(np.sum(baseline_data, 0))].T
        voltage_recording = folder_baseline_im / row.baseline_im_voltage_file
        peaks_I1, _, _, indices_for_7, comments_baseline = svr.obtain_peaks_voltage(voltage_recording, frame_rate,
                                                                 size_of_recording)
        if indices_for_7.shape[0] < baseline_data.shape[0]:
            baseline_data = baseline_data[:indices_for_7.shape[0], :]
            comments_baseline.append('Holostim_seq data has more items than triggers were obtained from the voltage file')
            raise Warning(comments_holo)
        elif indices_for_7.shape[0] > baseline_data.shape[0]:
            indices_for_7 = indices_for_7[:baseline_data.shape[0]]
            comments_baseline.append('Holostim_seq data has less items than triggers were obtained from the voltage file')
            raise Warning(comments_holo)
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
        nwbfile_pretrain_path = f"{folder_nwb_mice / row.mice_name}_{row.session_date}_raw_pretrain.nwb"
        convert_bruker_images_to_nwb(folder_pretrain_im, nwbfile_pretrain_path)
        io_pretrain = NWBHDF5IO(nwbfile_pretrain_path, mode="a")
        nwbfile_pretrain = io_pretrain.read()
        frame_rate = nwbfile_pretrain.acquisition['TwoPhotonSeries'].rate
        size_of_recording = nwbfile_pretrain.acquisition['TwoPhotonSeries'].data.shape[0]

        pretrain_data = loadmat(folder_raw / row.session_path / row.pretrain_mat_file)['data']['bmiAct'][0][0]
        pretrain_data = pretrain_data[:, :np.where(~np.isnan(pretrain_data).all(axis=0))[0][-1]].T
        voltage_recording = folder_pretrain_im / row.pretrain_im_voltage_file
        peaks_I1, indices_for_5, indices_for_6, indices_for_7 = svr.obtain_peaks_voltage(voltage_recording, frame_rate,
                                                                                         size_of_recording)
