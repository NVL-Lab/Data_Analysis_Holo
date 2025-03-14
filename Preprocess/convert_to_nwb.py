__author__ = 'Nuria'
# __author__ = ("Nuria", "John Doe")

# make sure to be in environment with pynwb and neuroconv installed

import numpy as np
import xml.etree.ElementTree as ET

from scipy.io import loadmat
from pynwb import NWBHDF5IO, TimeSeries, ogen
from pathlib import Path
from zoneinfo import ZoneInfo
from neuroconv.converters import BrukerTiffSinglePlaneConverter

from Preprocess import dataframe_sessions as ds
from Preprocess import syncronize_voltage_rec as svr
from Utils import analysis_constants as ac

# you need to install neuroconv converter first. In this case we use:
# https://neuroconv.readthedocs.io/en/main/conversion_examples_gallery/imaging/brukertiff.html


def convert_bruker_images_to_nwb(folder_path: Path, nwbfile_path:str):
    """ function to convert a bruker tiff file recording to nwb
    :param folder_path: path to the folder containing the tiff files
    :param nwbfile_path: path where we want to store the nwb file"""
    # convert data to nwb
    converter = BrukerTiffSinglePlaneConverter(folder_path=folder_path)
    metadata = converter.get_metadata()
    session_start_time = metadata["NWBFile"]["session_start_time"]
    tzinfo = ZoneInfo("US/Eastern")
    metadata["NWBFile"].update(session_start_time=session_start_time.replace(tzinfo=tzinfo)) # TODO this doesn't seem to work properly
    converter.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata)

def convert_all_experiments_to_nwb(folder_raw: Path, experiment_type: str):
    """ function to convert all experiments within a experiment type to nwb"""
    df_sessions = ds.get_sessions_df(experiment_type)
    folder_nwb = folder_raw.parents[0] / 'nwb'

    for index, row in df_sessions.iterrows():
        folder_holobmi_seq_im = Path(folder_raw) / row.session_path / 'im' / row.Holostim_seq_im
        folder_baseline_im = Path(folder_raw) / row.session_path / 'im' / row.Baseline_im
        folder_pretrain_im = Path(folder_raw) / row.session_path / 'im' / row.Pretrain_im
        folder_bmi_im = Path(folder_raw) / row.session_path / 'im' / row.BMI_im
        folder_path_array = [folder_holobmi_seq_im, folder_baseline_im, folder_pretrain_im, folder_bmi_im]
        folder_nwb_mice = folder_nwb / row.mice_name
        if not Path(folder_nwb_mice).exists():
            Path(folder_nwb_mice).mkdir(parents=True, exist_ok=True)

        for folder_path in folder_path_array:
            parts = Path(folder_path).parts
            nwbfile_path = f"{folder_nwb_mice / row.mice_name}_{row.session_date}_raw_{parts[-2]}.nwb"
            convert_bruker_images_to_nwb(folder_path, nwbfile_path)
            io = NWBHDF5IO(nwbfile_path, mode="a")
            nwbfile = io.read()
            frame_rate = nwbfile.acquisition['TwoPhotonSeries'].rate
            size_of_recording = nwbfile.acquisition['TwoPhotonSeries'].data.shape[0]

            # create the device for holographic stim
            holographic_device = nwbfile.create_device(
                name='Monaco',
                description="Laser used for the holographic stim",
                manufacturer="Coherent",
            )

            if parts[-2] == 'holostim_seq':
                # Save the neural data that was store in the mat file
                holostim_seq_data = loadmat(folder_raw / row.session_path / row.holostim_seq_mat_file) ['holoActivity']
                # select holodata that is not nan and transpose to have time x neurons
                holostim_seq_data = holostim_seq_data[:, ~np.isnan(np.sum(holostim_seq_data,0))].T
                voltage_recording = folder_path / row.Holostim_seq_im_voltage_file
                _, indices_for_6, indices_for_7 = svr.obtain_peaks_voltage(voltage_recording, frame_rate, size_of_recording)
                if indices_for_7.shape[0] < holostim_seq_data.shape[0]:
                    holostim_seq_data = holostim_seq_data[:indices_for_7.shape[0], :]
                    comments_holo = 'Holostim_seq data has more items than triggers were obtained from the voltage file'
                    raise Warning(comments_holo)
                elif indices_for_7.shape[0] > holostim_seq_data.shape[0]:
                    indices_for_7 = indices_for_7[:holostim_seq_data.shape[0]]
                    comments_holo ='Holostim_seq data has less items than triggers were obtained from the voltage file'
                    raise Warning(comments_holo)
                else:
                    comments_holo = 'conversion worked correctly'
                online_neural_data = TimeSeries(
                    name="neural_activity",
                    description=(f'neural data obtained online from {holostim_seq_data.shape[1]} neurons while'
                                 f' recording for the calibration of the BMI'),
                    data=holostim_seq_data,
                    timestamps=indices_for_7.astype('float64'),
                    unit="imaging frames",
                    comments=comments_holo
                )
                nwbfile.add_acquisition(online_neural_data)

                # obtain the holographic metadata and store it (This is done using optogenetic module and not the
                # ndx-holographic which is not working
                tree = ET.parse(folder_raw / row.session_path / row.XML_holostim_seq)
                troot = tree.getroot()
                power = []
                point = []
                index = []
                for elem in troot.findall('PVMarkPointElement'):
                    power.append(elem.get('UncagingLaserPower'))
                    point.append(elem.find('PVGalvoPointElement').get('Points'))
                    index.append(elem.find('PVGalvoPointElement').get('Indices'))
                if len(point) != len(index) or len(point) != holostim_seq_data.shape[1]:
                    comments_holoseries = 'The number of stims locations is not consistent with data retrieved'
                    Warning(comments_holoseries)
                else:
                    comments_holoseries = 'All points were stimulated sequentially'

                ogen_stim_seq_site = ogen.OptogeneticStimulusSite(
                    name="Holographic sequential location",
                    device=holographic_device,
                    description="Sequential stimulation of all the neurons selected as initial ROIs",
                    excitation_lambda=1035.,  # nm
                    location="all initial ROIs",
                )
                ogen_seq_series = ogen.OptogeneticSeries(
                    name="Holographic sequential",
                    data=list(zip(index, point, power)),
                    site=ogen_stim_seq_site,
                    timestamp=indices_for_6,
                    comments=comments_holoseries
                )
                nwbfile.add_stimulus(ogen_seq_series)

                # write and close the nwb file
                io.write(nwbfile)
                io.close()

            elif parts[-2] == 'baseline':
                # Save the neural data that was store in the mat file
                baseline_data = loadmat(folder_raw / row.session_path / row.baseline_mat_file)['baseActivity']
                baseline_data = baseline_data[:, ~np.isnan(np.sum(baseline_data, 0))].T
                voltage_recording = folder_path / row.baseline_im_voltage_file
                _, _, indices_for_7 = svr.obtain_peaks_voltage(voltage_recording, frame_rate,
                                                                           size_of_recording)
                if indices_for_7.shape[0] != baseline_data.shape[0]:
                    raise ValueError(f' The number of Triggers {indices_for_7.shape[0]} '
                                     f'does not match the data {baseline_data.shape[0]}')

                online_neural_data = TimeSeries(
                    name="neural_activity",
                    description=(f'neural data obtained online from {baseline_data.shape[1]} neurons while'
                                 f' recording for the calibration of the BMI'),
                    data=baseline_data,
                    timestamps=indices_for_7.astype('float64'),
                    unit="imaging frames",
                )
                nwbfile.add_acquisition(online_neural_data)
                io.write(nwbfile)
                io.close()


