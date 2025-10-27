__author__ = ('Saul', 'Pardhu', 'Andrea')

from pathlib import Path
import pandas as pd
import numpy as np
import cv2

def convert_to_ms(df: pd.DataFrame) -> int:
    """ 
        Convert hrs, min, sec, mill columns to total milliseconds and create a new column 'time_ms' 
        :param df: corresponding dataframe
    """
    return (df['hrs']*3600000 + df['min']*60000 + df['sec']*1000 + df['msec'])

def synchronize_two_to_one(camera_df: pd.DataFrame, microscope_df: pd.DataFrame):
    """
        Synchronize 2 camera frames to 1 microscope frame.
        Camera always leads → only take camera frames before or equal to each microscope frame to reduce jetter effect
    """
    synced_data = []
    camera_used = set()
    expected_ratio = 2

    for i, micro_time in enumerate(microscope_df['microscope_time_in_ms']):
        # Filter camera frames before or equal to microscope timestamp
        valid_cams = camera_df[camera_df['camera_time_in_ms'] <= micro_time]

        if len(valid_cams) < expected_ratio:
            continue  # skip if not enough camera frames before microscope frame

        # Take the last 2 camera frames before microscope frame
        mapped_cams = valid_cams.tail(expected_ratio)['camera_time_in_ms'].tolist()

        synced_data.append({
        'microscope_frame': microscope_df.loc[i, 'frame'],
        'microscope_time_in_ms': micro_time,
        'camera_times': mapped_cams
        })

        # Mark used frames
        camera_used.update(valid_cams.tail(expected_ratio).index.tolist())

        synced_df = pd.DataFrame(synced_data)
        unused_cameras = camera_df.drop(index=list(camera_used)).reset_index(drop=True)

    return synced_df, unused_cameras

def sync_behavior(microscope_data_path: Path, camera_data_path: Path, behavior_data_path: Path, video_data_path: Path) -> pd.DataFrame:
    """
        Synchronizes camera frames and behavior with microscope frames
            :param microscope_data_path: path to the microscope data file
            :param camera_data_path: path to the camera data file
            :param behavior_data_path: path to the microscope data file
            :return: corresponding camara frames and behavior to microscope data
    """
    
    # Loading files
    microscope_data = pd.read_csv(microscope_data_path, sep=r'\s+',names=['hrs','min','sec','msec','type','frame'])
    camera_data = pd.read_csv(camera_data_path, sep=r'\s+', names=['hrs','min','sec','msec'])
    behavior_data = pd.read_csv(behavior_data_path, skiprows=10, on_bad_lines='skip', sep=r'\s+', names=['code', 'time', 'action'])
    video_cap = cv2.VideoCapture(video_data_path)

    # Extract basic info from video
    video_fps = video_cap.get(cv2.CAP_PROP_FPS)*2 # 30 fps video but capture was at 60 fps, therefore multiplied by 2
    video_cap.release()
    video_duration = len(camera_data) / video_fps # 1864900ms @ 30fps = 31 min and 4.9 sec

    # Notes
    #   behavior began at 13:29:24 -> video frame between ~749 and 808 (between ~24.97 and 26.93 sec in)
    #   total behavior: 916827 ms = 15 min and 16.827 sec, total camera: 919393 ms = 15 min 19.393 sec, total microscope: 900799 ms = 15 min 0.799 sec

    # Changing timestamps to ms and dropping unnecessary data
    microscope_data['microscope_time_in_ms'] = convert_to_ms(microscope_data)
    microscope_data.drop(columns=['hrs','min','sec','msec','type'], inplace=True)
    camera_data['camera_time_in_ms'] = convert_to_ms(camera_data)
    camera_data.drop(columns=['hrs','min','sec','msec'], inplace=True)

    # Normalizing - gives events at 60 fps (check)
    for i, beh_time in enumerate(behavior_data['time']):
        norm = int(beh_time)/ 916827 # Should automate getting the last ms timestamp (916827) from behavior (due to inconsistent columns last row gives error)
        seconds = norm*video_duration
        minutes = int(seconds/60)
        rem_seconds = seconds - minutes * 60
        print(minutes, 'minutes', rem_seconds, 'seconds', behavior_data['action'][i])
    exit()

    # Extract corresponding timestamp and behavior 
    #   match timestamp of behavior with microscope/camera timestamps (populate with actual action rather than number is preferred)
    #   add a column with the relative time (instead of time of day)
    #   frame can be extracted from the video info
    #   output a 60 fps video to make sure the correspoding frames match the behavior (perhaps extract the exact frame when an easily noticeable action happens like lick)

    # Sort by time
    microscope_data = microscope_data[['microscope_time_in_ms','frame']]
    camera_data = camera_data.sort_values('camera_time_in_ms').reset_index(drop=True)
    microscope_data = microscope_data.sort_values('microscope_time_in_ms').reset_index(drop=True)

    # Synchronize
    final_synced_df, unused_camera_frames = synchronize_two_to_one(camera_data, microscope_data)
    
    # df should have columns like microscope_frame, microscope_time_ms, camera_frames, camera_times_ms, behavior 

    return final_synced_df

# Beginning of the script
if __name__=='__main__':    
    
    # Paths to all data
    base_path = Path('/data/project/nvl_lab/HoloBMI/Behavior/190930/NVI12/base/')
    microscope_data_path = base_path / 'sync_2019-09-30T13_29_11.csv'
    camera_data_path = base_path / 'video_timestamp_2019-09-30T13_29_11.csv'
    behavior_data_path = base_path / 'NVI12-2019-09-30-132924.txt'
    video_data_path = base_path / 'video_2019-09-30T13_29_11.avi'
    
    final_synced_df = sync_behavior(microscope_data_path, camera_data_path, behavior_data_path, video_data_path)
    #final_synced_df.to_csv('final_sync.csv', index=False, header=False)

    print(final_synced_df)
