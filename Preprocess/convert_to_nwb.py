__author__ = 'Nuria'
# __author__ = ("Nuria", "John Doe")

# make sure to be in environment with pynwb and neuroconv installed

import numpy as np

from scipy.io import loadmat
from pynwb import NWBHDF5IO, TimeSeries
from pathlib import Path
from zoneinfo import ZoneInfo
from neuroconv.converters import BrukerTiffSinglePlaneConverter

from Preprocess import dataframe_sessions as ds
from Utils import analysis_constants as ac

# you need to install neuroconv converter first. In this case we use:
# https://neuroconv.readthedocs.io/en/main/conversion_examples_gallery/imaging/brukertiff.html


def convert_to_nwb_from_bruker(folder_path: Path, nwbfile_path:str):
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
    ## TODO add the voltage data recordings to all of these guys
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
            convert_to_nwb_from_bruker(folder_path, nwbfile_path)
            io = NWBHDF5IO(nwbfile_path, mode="a")
            nwbfile = io.read()
            frame_rate = nwbfile.acquisition['TwoPhotonSeries'].rate
            size_of_recording = nwbfile.acquisition['TwoPhotonSeries'].data.shape[0]


            if parts[-2] == 'holostim_seq':
                holostim_seq_data = loadmat(folder_raw / row.session_path / row.holostim_seq_mat_file) ['holoActivity']
                # select holodata that is not nan and transpose to have time x neurons
                holostim_seq_data = holostim_seq_data[:, ~np.isnan(np.sum(holostim_seq_data,0))].T
                timestamps = np.arange(holostim_seq_data.shape[0])
                holostim_seq_time_series = TimeSeries(
                    name="neural_activity",
                    description=("neural data obtained online from {} neurons while holographically "
                                "stimulating those neurons").format(holostim_seq_data.shape[1]),
                    data=holostim_seq_data,
                    timestamps=timestamps,
                    unit="n.a.",
                )
                nwbfile.add_acquisition(holostim_seq_time_series)
                io.write(nwbfile)
                io.close()

            elif parts[-2] == 'baseline':
                baseline_data = loadmat(folder_raw / row.session_path / row.baseline_mat_file)['baseActivity']
                baseline_data = baseline_data[:, ~np.isnan(np.sum(baseline_data, 0))].T
                # TODO need to select the correct time stamps from voltage rec input 1 (microscope frames)
                #  timestamps = np.arange(baseline_data.shape[0])

                baseline_time_series = TimeSeries(
                    name="neural_activity",
                    description=("neural data obtained online from {} neurons while recording for  "
                                 "the calibration of the BMI").format(baseline_data.shape[1]),
                    data=baseline_data,
                    timestamps=timestamps,
                    unit="n.a.",
                )
                nwbfile.add_acquisition(baseline_time_series)
                io.write(nwbfile)
                io.close()


