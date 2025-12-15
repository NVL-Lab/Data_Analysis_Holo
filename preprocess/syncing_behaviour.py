__author__ = ('Saul', 'Pardhu', 'Andrea')

from pathlib import Path
import pandas as pd
import numpy as np
import cv2


# --------------------------------------------------
# Utility Functions
# --------------------------------------------------

def convert_to_ms(df: pd.DataFrame) -> pd.Series:
    """Convert hrs, min, sec, msec columns to milliseconds."""
    return (
        df['hrs'] * 3600000 +
        df['min'] * 60000 +
        df['sec'] * 1000 +
        df['msec']
    )


def map_camera_to_video_time(camera_df: pd.DataFrame) -> pd.DataFrame:
    """Convert camera timestamps to video-relative timestamps."""
    camera_df['camera_video_format_ms'] = (
        camera_df['camera_time_in_ms'] - camera_df['camera_time_in_ms'].min()
    )
    return camera_df


def synchronize_two_to_one(
    camera_df: pd.DataFrame,
    microscope_df: pd.DataFrame
):
    """Synchronize 2 camera frames per microscope frame."""
    synced_data = []
    camera_used = set()
    expected_ratio = 2

    for i, micro_time in enumerate(microscope_df['microscope_time_in_ms']):
        valid_cams = camera_df[
            camera_df['camera_time_in_ms'] <= micro_time
        ]

        if len(valid_cams) < expected_ratio:
            continue

        mapped_cams = (
            valid_cams
            .tail(expected_ratio)['camera_time_in_ms']
            .tolist()
        )

        synced_data.append({
            'microscope_frame': microscope_df.loc[i, 'frame'],
            'microscope_time_in_ms': micro_time,
            'camera_times': mapped_cams
        })

        camera_used.update(valid_cams.tail(expected_ratio).index.tolist())

    synced_df = pd.DataFrame(synced_data)
    unused_cameras = (
        camera_df
        .drop(index=list(camera_used))
        .reset_index(drop=True)
    )

    return synced_df, unused_cameras


def map_behavior_to_video_time(
    behavior_df: pd.DataFrame,
    video_duration: float
) -> pd.DataFrame:
    """Apply offset so behavior timestamps match video time."""

    max_beh_time = 916827  # ms (known max from behavior file)

    TIME_OFFSET_SEC = video_duration - (max_beh_time / 1000)
    TIME_OFFSET_MS = TIME_OFFSET_SEC * 1000

    behavior_df['behavior_video_ms'] = (
        behavior_df['time'] + TIME_OFFSET_MS
    )

    behavior_df['seconds'] = behavior_df['behavior_video_ms'] / 1000
    behavior_df['minutes'] = (behavior_df['seconds'] // 60).astype(int)
    behavior_df['rem_seconds'] = behavior_df['seconds'] % 60

    for _, row in behavior_df.iterrows():
        print(
            f"Time & Action: "
            f"{row['minutes']:02d}:"
            f"{int(row['rem_seconds']):02d}  "
            f"{row['action']}"
        )

    return behavior_df


def assign_behavior_to_camera(
    synced_df: pd.DataFrame,
    camera_df: pd.DataFrame,
    behavior_df: pd.DataFrame
) -> pd.DataFrame:
    """Assign nearest previous behavior to camera frames."""

    behavior_times = behavior_df[
        ['behavior_video_ms', 'action']
    ]

    results = []

    for _, row in synced_df.iterrows():
        camera_times = row['camera_times']

        camera_video_times = [
            camera_df.loc[
                camera_df['camera_time_in_ms'] == c,
                'camera_video_format_ms'
            ].values[0]
            for c in camera_times
        ]

        latest_cam_time = max(camera_video_times)

        relevant_beh = behavior_times[
            behavior_times['behavior_video_ms'] <= latest_cam_time
        ]

        if not relevant_beh.empty:
            latest_action = relevant_beh.iloc[-1]['action']
            latest_action_time = relevant_beh.iloc[-1]['behavior_video_ms']
        else:
            latest_action = None
            latest_action_time = np.nan

        results.append({
            'microscope_time_in_ms': row['microscope_time_in_ms'],
            'camera_times_ms': camera_times,
            'camera_video_format_min': [t / 60000 for t in camera_video_times],
            'latest_camera_time_min': latest_cam_time / 60000,
            'behavior_video_min': (
                latest_action_time / 60000
                if not np.isnan(latest_action_time)
                else np.nan
            ),
            'behavior': latest_action
        })

    return pd.DataFrame(results)


# --------------------------------------------------
# Main Pipeline
# --------------------------------------------------

def sync_behavior(
    microscope_data_path: Path,
    camera_data_path: Path,
    behavior_data_path: Path,
    video_data_path: Path
) -> pd.DataFrame:
    """Main synchronization pipeline."""

    microscope_data = pd.read_csv(
        microscope_data_path,
        sep=r'\s+',
        names=['hrs', 'min', 'sec', 'msec', 'type', 'frame']
    )

    camera_data = pd.read_csv(
        camera_data_path,
        sep=r'\s+',
        names=['hrs', 'min', 'sec', 'msec']
    )

    behavior_data = pd.read_csv(
        behavior_data_path,
        skiprows=10,
        on_bad_lines='skip',
        sep=r'\s+',
        names=['code', 'time', 'action']
    )

    video_cap = cv2.VideoCapture(str(video_data_path))
    video_fps = video_cap.get(cv2.CAP_PROP_FPS) * 2
    video_cap.release()

    video_duration = len(camera_data) / video_fps

    print(f"Behaviour Video FPS: {video_fps}")
    print(f"Video Duration: {video_duration} sec ({video_duration/60} min)")

    microscope_data['microscope_time_in_ms'] = convert_to_ms(microscope_data)
    camera_data['camera_time_in_ms'] = convert_to_ms(camera_data)

    microscope_data.drop(
        columns=['hrs', 'min', 'sec', 'msec', 'type'],
        inplace=True
    )
    camera_data.drop(
        columns=['hrs', 'min', 'sec', 'msec'],
        inplace=True
    )

    camera_data = camera_data.sort_values(
        'camera_time_in_ms'
    ).reset_index(drop=True)

    microscope_data = microscope_data.sort_values(
        'microscope_time_in_ms'
    ).reset_index(drop=True)

    synced_df, _ = synchronize_two_to_one(
        camera_data,
        microscope_data
    )

    camera_df_with_video = map_camera_to_video_time(camera_data)

    print(
        f"Camera data duration (min): "
        f"{camera_df_with_video['camera_video_format_ms'].iloc[-1] / 60000}"
    )

    behavior_df_with_video = map_behavior_to_video_time(
        behavior_data,
        video_duration
    )

    final_synced_df = assign_behavior_to_camera(
        synced_df,
        camera_df_with_video,
        behavior_df_with_video
    )

    return final_synced_df


# --------------------------------------------------
# Script Entry Point
# --------------------------------------------------

if __name__ == '__main__':

    base_path = Path(
        '/data/project/nvl_lab/HoloBMI/Behavior/190930/NVI12/base/'
    )

    microscope_data_path = base_path / 'sync_2019-09-30T13_29_11.csv'
    camera_data_path = base_path / 'video_timestamp_2019-09-30T13_29_11.csv'
    behavior_data_path = base_path / 'NVI12-2019-09-30-132924.txt'
    video_data_path = base_path / 'video_2019-09-30T13_29_11.avi'

    final_df = sync_behavior(
        microscope_data_path,
        camera_data_path,
        behavior_data_path,
        video_data_path
    )

    print(final_df.tail(10))
    print(final_df.iloc[0])
