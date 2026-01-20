__author__ = ('Saul', 'Pardhu', 'Andrea')

from pathlib import Path
import pandas as pd
import numpy as np
import cv2


# =============================
# BEHAVIOR MAP
# =============================
BEHAVIOR_MAP = {
    55: "touch_aux1",
    56: "touch_aux1_off",
    57: "touch_joystick",
    58: "touch_joystick_off",
    53: "lick",
    54: "lick_off",
     1: "None"
}


# =============================
# TIME UTILS
# =============================
def convert_to_ms(df: pd.DataFrame) -> pd.Series:
    return (
        df['hrs'] * 3600000 +
        df['min'] * 60000 +
        df['sec'] * 1000 +
        df['msec']
    )


def map_camera_to_video_time(camera_df):
    camera_df['camera_video_format_ms'] = (
        camera_df['camera_time_in_ms'] - camera_df['camera_time_in_ms'].min()
    )
    return camera_df


# =============================
# SYNC CAMERA  MICROSCOPE
# =============================
def synchronize_two_to_one(camera_df, microscope_df):
    synced = []
    camera_used = set()

    for micro_time in microscope_df['microscope_time_in_ms']:
        valid = camera_df[camera_df['camera_time_in_ms'] <= micro_time]
        if len(valid) < 2:
            continue

        cams = valid.tail(2)
        synced.append({
            'microscope_time_in_ms': micro_time,
            'camera_times_ms': cams['camera_time_in_ms'].tolist()
        })
        camera_used.update(cams.index.tolist())

    return pd.DataFrame(synced), camera_df.drop(index=list(camera_used))


# =============================
# BEHAVIOR  VIDEO TIME
# =============================
def map_behavior_to_video_time(behavior_df, video_duration_sec):
    max_beh_time = 916827  # ms (known end due to uncertenity in behaviour_data)
    offset_ms = (video_duration_sec * 1000) - max_beh_time

    behavior_df['behavior_video_ms'] = behavior_df['time'] + offset_ms
    behavior_df['behavior_video_min'] = behavior_df['behavior_video_ms'] / 60000
    behavior_df['behavior_action'] = behavior_df['action'].map(BEHAVIOR_MAP)

    return behavior_df


# =============================
# ASSIGN BEHAVIOR TO FRAMES
# =============================
def assign_behavior_to_camera(synced_df, camera_df, behavior_df):
    rows = []

    for _, row in synced_df.iterrows():
        cam_times = [
            camera_df.loc[camera_df['camera_time_in_ms'] == t,
                          'camera_video_format_ms'].values[0]
            for t in row['camera_times_ms']
        ]

        latest_cam_time = max(cam_times)
        valid_beh = behavior_df[behavior_df['behavior_video_ms'] <= latest_cam_time]

        if not valid_beh.empty:
            beh_row = valid_beh.iloc[-1]
            beh_code = int(beh_row['action'])
            beh_action = beh_row['behavior_action']
            beh_min = beh_row['behavior_video_min']
        else:
            beh_code = np.nan
            beh_action = None
            beh_min = np.nan

        rows.append({
            'latest_camera_time_ms': latest_cam_time,
            'behavior_video_min': beh_min,
            'behavior': beh_code,
            'behavior_action': beh_action
        })

    return pd.DataFrame(rows)


# =============================
# VIDEO CONVERT + OVERLAY
# =============================
def convert_video_to_60fps_with_overlay(
    input_video: Path,
    final_df: pd.DataFrame
) -> Path:

    cap = cv2.VideoCapture(str(input_video))
    in_fps = cap.get(cv2.CAP_PROP_FPS)
    out_fps = in_fps * 2

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video = Path.cwd() / f"{input_video.stem}_60fps_overlay.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        str(output_video),
        fourcc,
        out_fps,
        (width, height)
    )

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # duplicate frame  60 FPS
        for _ in range(2):
            video_time_ms = (frame_idx / out_fps) * 1000

            valid_beh = final_df[
                final_df['latest_camera_time_ms'] <= video_time_ms
            ]

            if not valid_beh.empty:
                label = valid_beh.iloc[-1]['behavior_action']
            else:
                label = "None"

            cv2.putText(
                frame,
                f"Behavior: {label}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            out.write(frame)
            frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Saved overlay video:", output_video)
    return output_video


# =============================
# MAIN
# =============================
if __name__ == "__main__":

    base_path = Path('/data/project/nvl_lab/HoloBMI/Behavior/190930/NVI12/base')

    microscope_data_path = base_path / 'sync_2019-09-30T13_29_11.csv'
    camera_data_path = base_path / 'video_timestamp_2019-09-30T13_29_11.csv'
    behavior_data_path = base_path / 'NVI12-2019-09-30-132924.txt'
    raw_video_path = base_path / 'video_2019-09-30T13_29_11.avi'

    microscope_df = pd.read_csv(
        microscope_data_path,
        sep=r'\s+',
        names=['hrs', 'min', 'sec', 'msec', 'type', 'frame']
    )
    camera_df = pd.read_csv(
        camera_data_path,
        sep=r'\s+',
        names=['hrs', 'min', 'sec', 'msec']
    )
    behavior_df = pd.read_csv(
        behavior_data_path,
        skiprows=10,
        sep=r'\s+',
        names=['event', 'time', 'action'],
        on_bad_lines='skip'
    )

    microscope_df['microscope_time_in_ms'] = convert_to_ms(microscope_df)
    camera_df['camera_time_in_ms'] = convert_to_ms(camera_df)

    microscope_df = microscope_df.sort_values('microscope_time_in_ms')
    camera_df = camera_df.sort_values('camera_time_in_ms')

    synced_df, _ = synchronize_two_to_one(camera_df, microscope_df)
    camera_df = map_camera_to_video_time(camera_df)

    cap = cv2.VideoCapture(str(raw_video_path))
    video_duration_sec = len(camera_df) / (cap.get(cv2.CAP_PROP_FPS) * 2)
    cap.release()

    behavior_df = map_behavior_to_video_time(behavior_df, video_duration_sec)
    final_df = assign_behavior_to_camera(synced_df, camera_df, behavior_df)

    convert_video_to_60fps_with_overlay(raw_video_path, final_df)
