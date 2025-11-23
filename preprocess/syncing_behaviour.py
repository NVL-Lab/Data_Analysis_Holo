__author__ = ('Saul', 'Pardhu', 'Andrea')

from pathlib import Path
import pandas as pd
import numpy as np
import cv2


def convert_to_ms(df: pd.DataFrame) -> int:
    """Convert hrs, min, sec, msec columns to total milliseconds."""
    return (df['hrs']*3600000 + df['min']*60000 + df['sec']*1000 + df['msec'])

def map_camera_to_video_time(camera_df, video_duration):
    """
        Converting camera timestamps as video timestamps
    """
    camera_df['camera_video_format_ms'] = camera_df['camera_time_in_ms'] - camera_df['camera_time_in_ms'].min()
    
    return camera_df

def synchronize_two_to_one(camera_df: pd.DataFrame, microscope_df: pd.DataFrame):
    """Synchronize 2 camera frames per microscope frame (camera leads)."""
    synced_data = []
    camera_used = set()
    expected_ratio = 2

    for i, micro_time in enumerate(microscope_df['microscope_time_in_ms']):
         # Filter camera frames before or equal to microscope timestamp
        valid_cams = camera_df[camera_df['camera_time_in_ms'] <= micro_time]

        if len(valid_cams) < expected_ratio:
            continue # skip if not enough camera frames before microscope frame
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
    unused_cameras = camera_df.drop(index=list(camera_used)).reset_index(drop=True) # Traking Unused Camera frames
    return synced_df, unused_cameras







def map_behavior_to_video_time(behavior_df, video_duration):

    """
    Apply correct offset so behavior timestamps match camera timestamps.
    """

    max_beh_time = 916827  # ms (from behavior file due to uncertainty in text file )
    print(f"Max behavior time in minutes (task is over): {max_beh_time/60000}")

    TIME_OFFSET_SEC = video_duration - (max_beh_time / 1000)
    TIME_OFFSET_MS = TIME_OFFSET_SEC * 1000
    print("Behavior offset TimeoFF sET(sec):", TIME_OFFSET_SEC)
    print("Behavior offset (ms):", TIME_OFFSET_MS)

    #  APPLY OFFSET TO BEHAVIOR TIMESTAMP
    behavior_df["behavior_video_ms"] = behavior_df["time"] + TIME_OFFSET_MS #adding time offset to every action in 

    # Convert to hh:mm:ss format for readability
    behavior_df["seconds"] = behavior_df["behavior_video_ms"] / 1000
    behavior_df["minutes"] = (behavior_df["seconds"] // 60).astype(int)
    behavior_df["rem_seconds"] = behavior_df["seconds"] % 60

    # Print readable behavior timeline
    for _, row in behavior_df.iterrows():
        #printing the actions and there associated behaviour actions
        print(f"Time & Action: {row['minutes']:02d}:{int(row['rem_seconds']):02d}  {row['action']}")

    return behavior_df





def assign_behavior_to_camera(synced_df, camera_df, behavior_df):
    """Attach nearest previous behavior event to each synced camera-video timestamp (no jitter)."""
    behavior_times = behavior_df[['behavior_video_ms', 'action']]
    results = []

    for _, row in synced_df.iterrows():

        microscope_ms = row['microscope_time_in_ms']
        camera_times = row['camera_times']

        # Convert each camera_time_in_ms → camera_video_format_ms
        camera_video_times = [
            camera_df.loc[camera_df['camera_time_in_ms'] == c, 'camera_video_format_ms'].values[0]
            for c in camera_times
        ]

        # Get latest (max) camera video time
        latest_cam_time = max(camera_video_times)

        #Find the nearest previous behavior event
        relevant_beh = behavior_times[behavior_times['behavior_video_ms'] <= latest_cam_time]

        if not relevant_beh.empty:
            latest_action = relevant_beh.iloc[-1]['action']
            latest_action_time = relevant_beh.iloc[-1]['behavior_video_ms']
        else:
            latest_action = None
            latest_action_time = np.nan

        # Convert both to minutes for easy comparison
        camera_video_times_min = [t / 60000 for t in camera_video_times]
        latest_cam_time_min = latest_cam_time / 60000
        behavior_time_min = latest_action_time / 60000 if not np.isnan(latest_action_time) else np.nan
       
        results.append({
            'microscope_time_in_ms' : microscope_ms,
            'camera_times_ms': camera_times,
            'camera_video_format_min': camera_video_times_min,
            'latest_camera_time_min': latest_cam_time_min,
            'behavior_video_min': behavior_time_min,
            'behavior': latest_action
        })

    return pd.DataFrame(results)


def sync_behavior(microscope_data_path: Path, camera_data_path: Path, behavior_data_path: Path, video_data_path: Path) -> pd.DataFrame:
    """Main synchronization pipeline."""
    # Load files
    microscope_data = pd.read_csv(microscope_data_path, sep=r'\s+', names=['hrs','min','sec','msec','type','frame'])
    camera_data = pd.read_csv(camera_data_path, sep=r'\s+', names=['hrs','min','sec','msec'])
    behavior_data = pd.read_csv(behavior_data_path, skiprows=10, on_bad_lines='skip', sep=r'\s+', names=['code', 'time', 'action'])
    video_cap = cv2.VideoCapture(str(video_data_path))

    # Extract video info
    video_fps = video_cap.get(cv2.CAP_PROP_FPS) * 2
    video_cap.release()
    video_duration = len(camera_data) / video_fps   # seconds
    print(f"Behaviour Video FPS: {video_fps}")
    print(f" Video Duration: {video_duration} Secs or {video_duration/60} Minutes")

    # Convert to ms
    microscope_data['microscope_time_in_ms'] = convert_to_ms(microscope_data)
    microscope_data.drop(columns=['hrs','min','sec','msec','type'], inplace=True)
    camera_data['camera_time_in_ms'] = convert_to_ms(camera_data)
    camera_data.drop(columns=['hrs','min','sec','msec'], inplace=True)

    # Sort
    camera_data = camera_data.sort_values('camera_time_in_ms').reset_index(drop=True)
    microscope_data = microscope_data.sort_values('microscope_time_in_ms').reset_index(drop=True)

    # Step 1: Sync microscope to camera
    synced_df, _ = synchronize_two_to_one(camera_data, microscope_data)

    # Step 2: Map camera to video time (no normalization)
    camera_df_with_video = map_camera_to_video_time(camera_data, video_duration)
    print(camera_df_with_video.head(4),camera_df_with_video.tail(10))
    print(f'Camera_datafile duration :{camera_df_with_video.tail(1)['camera_video_format_ms']/60000}')

    # Step 3: Map behavior --> video time
    behavior_df_with_video = map_behavior_to_video_time(behavior_data, video_duration)

    # Step 4: Merge behavior with synced data
    final_synced_df = assign_behavior_to_camera(synced_df, camera_df_with_video, behavior_df_with_video)

    return final_synced_df


# Run script
if __name__ == '__main__':

    base_path = Path('/data/project/nvl_lab/HoloBMI/Behavior/190930/NVI12/base/')
    microscope_data_path = base_path / 'sync_2019-09-30T13_29_11.csv'
    camera_data_path = base_path / 'video_timestamp_2019-09-30T13_29_11.csv'
    behavior_data_path = base_path / 'NVI12-2019-09-30-132924.txt'
    video_data_path = base_path / 'video_2019-09-30T13_29_11.avi'
    

    final_df = sync_behavior(microscope_data_path, camera_data_path, behavior_data_path, video_data_path)
    print(final_df.tail(10))
    print(final_df.iloc[0])
    #final_df.to_csv('final_synced_after_offset.csv')
