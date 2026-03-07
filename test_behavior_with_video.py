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

def play_video():

    path = "video_2019-09-30T13_29_11_60fps.mp4"
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open {path}")

    print("Controls: n = next frame, p = previous (approx), q = quit")
    # Note: precise backward stepping is codec-dependent; we simulate using set().

    frame_idx = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Frame-by-frame", frame)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            frame_idx += 1
        elif key == ord('p'):
            frame_idx = max(0, frame_idx - 1)

    cap.release()
    cv2.destroyAllWindows()

# =============================
# TIME UTILS
# =============================
def convert_to_ms(df: pd.DataFrame) -> pd.Series:
    return (
        df['hr'] * 3600000 +
        df['min'] * 60000 +
        df['s'] * 1000 +
        df['ms']
    )


def map_camera_to_video_time(camera_df):
    camera_df['camera_video_format_ms'] = (
        camera_df['total_ms'] - camera_df['total_ms'].min()
    )
    return camera_df


# =============================
# SYNC CAMERA  MICROSCOPE
# =============================
def synchronize_two_to_one(camera_df, microscope_df):
    synced = []
    camera_used = set()
    for micro_time in microscope_df['total_ms']:
        valid = camera_df[camera_df['total_ms'] <= micro_time].tail(1)['total_ms']
        valid_up = camera_df[camera_df['total_ms'] >= micro_time].head(1)['total_ms']
        
        if valid.iloc[0] == valid_up.iloc[0]:
            synced.append({
                'microscope_total_ms': micro_time,
                'camera_total_ms': [valid.iloc[0]]
            })
        else:
            synced.append({
                'microscope_total_ms': micro_time,
                'camera_total_ms': [valid.iloc[0], valid_up.iloc[0]]
            })
            
        camera_used.update([valid.index[0], valid_up.index[0]])

    return pd.DataFrame(synced), camera_df.drop(index=list(camera_used))

# =============================
# BEHAVIOR  VIDEO TIME
# =============================
def map_behavior_to_video_time(behavior_df, video_duration_sec):
    max_beh_time = 916827  # ms (known end due to uncertenity in behaviour_data)
    offset_ms = (video_duration_sec * 1000) - max_beh_time - 3600 #observed difference of 3 seconds
    # 15:01.. seems to be the last full lick
    behavior_df['behavior_video_ms'] = behavior_df['time'] + offset_ms
    beh_sec = behavior_df['behavior_video_ms'] / 1000
    #behavior_df['behavior_video_floor_min'] = np.floor(beh_sec/60)
    beh_min = np.floor(beh_sec/60)
    behavior_df['behavior_video_min'] = behavior_df['behavior_video_ms'] / 60000.
    behavior_df['behavior_video_rem_sec'] = beh_sec - (beh_min * 60)
    behavior_df['behavior_action'] = behavior_df['action'].map(BEHAVIOR_MAP)

    return behavior_df


# =============================
# ASSIGN BEHAVIOR TO FRAMES
# =============================
def assign_behavior_to_camera(synced_df, camera_df, behavior_df):
    rows = []

    valid_time = -1
    for _, row in synced_df.iterrows():
        
        # Gets the camera timestamps that belong to one frame of the microscope
        # and gets the corresponding time the video has played up until the camera timestamp  
        cam_times = [
            camera_df.loc[camera_df['total_ms'] == t, 'camera_video_format_ms'].values[0]
            for t in row['camera_total_ms']
        ]

        #main_cam_time = max(cam_times)
        main_cam_time = round(sum(cam_times) / len(cam_times))
        valid_beh = behavior_df[behavior_df['behavior_video_ms'] <= main_cam_time]

        valid_time_temp = valid_beh['time'].iloc[-1]
        #valid_index = valid_beh['time'].index[-1]
        if valid_time == valid_time_temp:
            beh_code = np.nan
            beh_action = None
            beh_min = np.nan
            beh_sec = np.nan
        else:
            beh_row = valid_beh.iloc[-1]
            beh_code = int(beh_row['action'])
            beh_action = beh_row['behavior_action']
            beh_min = beh_row['behavior_video_min']
            beh_sec = beh_row['behavior_video_rem_sec']
            valid_time = valid_time_temp
        # make it so action only shows on one frame

        '''
        print(main_cam_time)
        if not valid_beh.empty:
            beh_row = valid_beh.iloc[-1]
            beh_code = int(beh_row['action'])
            beh_action = beh_row['behavior_action']
            beh_min = beh_row['behavior_video_min']
            beh_min = beh_row['behavior_video_min']
            print(valid_beh)
        else:
            beh_code = np.nan
            beh_action = None
            beh_min = np.nan
            exit()
        '''

        rows.append({
            'latest_camera_time_ms': main_cam_time,
            'behavior_video_min': beh_min,
            'behavior_video_rem_sec': beh_sec,
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
    out_fps = in_fps * 2  # double the framerate
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video = Path.cwd() / f"{input_video.stem}_2x_speed_{int(out_fps)}fps.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video), fourcc, out_fps, (width, height))

    in_idx = 0
    valid_len_temp = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Use INPUT timeline for alignment with your behavior table
        # Prefer CAP_PROP_POS_MSEC if available (works better for VFR sources)
        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC) # increases by 33.3 = (1000 / 30 fps)
        #if t_ms < 0:  # fallback for some backends
        #    t_ms = (in_idx / in_fps) * 1000.0

        valid_beh = final_df[final_df['latest_camera_time_ms'] <= t_ms]
        label = valid_beh.iloc[-1]['behavior_action'] if not valid_beh.empty else None
        #print(valid_beh)
        dif = len(valid_beh) - valid_len_temp
        if dif > 1:
            dif_beh = valid_beh.iloc[-dif:]['behavior_action']
            if dif_beh.notna().all():
                print(f'skipped {dif-1} frame(s)')
                label = "-".join(valid_beh.iloc[-dif:]['behavior_action'].astype(str))
        valid_len_temp = len(valid_beh)
        
        # Draw on a copy so overlays don't accumulate
        frame_to_write = frame.copy()
        if label != None:
            #print(label)
            cv2.putText(frame_to_write, f"Behavior: {label}",
            (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2, cv2.LINE_AA)

            out.write(frame_to_write)
        #out.write(frame_to_write)
        in_idx += 1

    cap.release()
    out.release()
    print(f"Wrote: {output_video}")
    return output_video

# =============================
# VIDEO CONVERT + OVERLAY
# =============================
def convert_video_to_60fps_with_overlay(
    input_video: Path,
    final_df: pd.DataFrame
) -> Path:

    cap = cv2.VideoCapture(str(input_video))
    in_fps = cap.get(cv2.CAP_PROP_FPS)
    out_fps = in_fps * 2  # double the framerate
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video = Path.cwd() / f"{input_video.stem}_2x_speed_{int(out_fps)}fps.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video), fourcc, out_fps, (width, height))

    in_idx = 0
    valid_len_temp = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Use INPUT timeline for alignment with your behavior table
        # Prefer CAP_PROP_POS_MSEC if available (works better for VFR sources)
        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC) # increases by 33.3 = (1000 / 30 fps)
        #if t_ms < 0:  # fallback for some backends
        #    t_ms = (in_idx / in_fps) * 1000.0

        valid_beh = final_df[final_df['latest_camera_time_ms'] <= t_ms]
        label = valid_beh.iloc[-1]['behavior_action'] if not valid_beh.empty else None
        #print(valid_beh)
        dif = len(valid_beh) - valid_len_temp
        if dif > 1:
            dif_beh = valid_beh.iloc[-dif:]['behavior_action']
            if dif_beh.notna().all():
                print(f'skipped {dif-1} frame(s)')
                label = "-".join(valid_beh.iloc[-dif:]['behavior_action'].astype(str))
        valid_len_temp = len(valid_beh)
        
        # Draw on a copy so overlays don't accumulate
        frame_to_write = frame.copy()
        if label != None:
            #print(label)
            cv2.putText(frame_to_write, f"Behavior: {label}",
            (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2, cv2.LINE_AA)

            out.write(frame_to_write)
        #out.write(frame_to_write)
        in_idx += 1

    cap.release()
    out.release()
    print(f"Wrote: {output_video}")
    return output_video

def sync_behavior_to_camera_sync(behavior_df, sync_df):
    sync_df['relative_camera_ms'] = sync_df['camera_sync_ms'] - sync_df['camera_sync_ms'].min()
    #cam_leq_indexes = [sync_df[sync_df['relative_camera_ms'] <= row.time].tail(1)['relative_camera_ms'].index[0] for row in behavior_df.itertuples()]
    #cam_leq_indexes = [sync_df[sync_df['camera_sync_ms'] <= row.time].tail(1)['camera_sync_ms'].index[0] for row in behavior_df.itertuples()]
    cam_leq_indexes = []
    for row in behavior_df.itertuples():
        sub = sync_df[sync_df['camera_sync_ms'] <= row.time]
            
        if sub.empty:
            continue
                    
        cam_leq_indexes.append(sub.tail(1).index[0])

    beh_times = [[] for _ in range(len(sync_df))]
    beh_actions = [[] for _ in range(len(sync_df))]
    beh_raw_times = [[] for _ in range(len(sync_df))]
    behavior_df['behavior_action'] = behavior_df['action'].map(BEHAVIOR_MAP)
    beh_temp = -1
    j = 0
    for i in cam_leq_indexes:
        beh_ms = behavior_df['time'].loc[j]
        raw_beh_ms = behavior_df['relative_time'].loc[j]
        beh_action = behavior_df['behavior_action'].loc[j]
        #lower_ms = sync_df['relative_camera_ms'].iloc[i]
        lower_ms = sync_df['camera_sync_ms'].iloc[i]
        #higher_ms = sync_df['relative_camera_ms'].iloc[i+1]
        higher_ms = sync_df['camera_sync_ms'].iloc[i+1]
        if lower_ms == beh_ms:
            if beh_temp != lower_ms:
                beh_times[i].append(beh_ms)
                beh_actions[i].append(beh_action)
                beh_raw_times[i].append(raw_beh_ms)
            else:
                beh_temp = lower_ms
        else:
            if abs(beh_ms - lower_ms) < abs(beh_ms - higher_ms):
                if beh_temp != lower_ms:
                    beh_times[i].append(beh_ms)
                    beh_actions[i].append(beh_action)
                    beh_raw_times[i].append(raw_beh_ms)
                else:
                    beh_temp = lower_ms
            else:
                if beh_temp != higher_ms:
                    beh_times[i+1].append(beh_ms)
                    beh_actions[i+1].append(beh_action)
                    beh_raw_times[i+1].append(raw_beh_ms)
                else:
                    beh_temp = higher_ms
        j += 1

    sync_df['behavior_sync_ms'] = beh_times
    sync_df['behavior_sync_raw_ms'] = beh_raw_times
    #sync_df['behavior_sync_relative_ms'] = [t - behavior_df['time'].loc[0] for t in beh_times]
    sync_df['behavior_sync_action'] = beh_actions

    sync_df.loc[sync_df['behavior_sync_ms'].str.len() == 0, 'behavior_sync_ms'] = None
    #sync_df.loc[sync_df['behavior_sync_relative_ms'].str.len() == 0, 'behavior_sync_relative_ms'] = None
    sync_df.loc[sync_df['behavior_sync_raw_ms'].str.len() == 0, 'behavior_sync_raw_ms'] = None
    sync_df.loc[sync_df['behavior_sync_action'].str.len() == 0, 'behavior_sync_action'] = None

    sync_df['behavior_sync_ms'] = sync_df['behavior_sync_ms'].apply(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)
    #sync_df['behavior_sync_relative_ms'] = sync_df['behavior_sync_relative_ms'].apply(lambda x: x[0] if isinstance(x, np.ndarray) and len(x) == 1 else x)
    sync_df['behavior_sync_raw_ms'] = sync_df['behavior_sync_raw_ms'].apply(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)
    sync_df['behavior_sync_action'] = sync_df['behavior_sync_action'].apply(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)

    return sync_df

def sync_camera_to_microscope(camera_df, microscope_df):
   
    cam_less = [camera_df[camera_df['total_ms'] <= row.total_ms].tail(1)['total_ms'].index[0] for row in microscope_df.itertuples()]
    
    cam_times = []
    j = 0
    for i in cam_less:
        micro_ms = microscope_df['total_ms'].loc[j]
        lower_ms = camera_df['total_ms'].iloc[i]
        higher_ms = camera_df['total_ms'].iloc[i+1]
        if lower_ms == micro_ms:
            cam_times.append(lower_ms)
        else:
            if abs(micro_ms - lower_ms) < abs(micro_ms - higher_ms):
                cam_times.append(lower_ms)
            else:
                cam_times.append(higher_ms)
        j += 1

    microscope_df['camera_sync_ms'] = cam_times
    return microscope_df

def convert_sync_camera_frames(input_video, final_df):
    cap = cv2.VideoCapture(str(input_video))
    in_fps = cap.get(cv2.CAP_PROP_FPS)
    out_fps = in_fps * 2  # double the framerate
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video = Path.cwd() / f"{input_video.stem}_{int(out_fps)}fps.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video), fourcc, out_fps, (width, height))

    # Does not ignore Nones
    #c1 = final_df['relative_camera_ms'].apply(lambda x: [x])
    #c2 = final_df['behavior_sync_ms'].apply(lambda x: x if isinstance(x, list) else [x])
    
    #c1 = final_df['relative_camera_ms'].apply(lambda x: [x] if x is not None else [])
    #c2 = final_df['behavior_sync_ms'].apply(lambda x: x if isinstance(x, list) else ([] if x is None else [x]))

    #final_df['times_for_video'] = c1 + c2

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

        #valid_beh = final_df[final_df['relative_camera_ms'] <= t_ms]
        valid_beh = final_df[final_df['camera_sync_ms'] <= t_ms]
        label = valid_beh['behavior_sync_action'].iloc[-1]

        if len(valid_beh) == 0:
            continue
        
        #print(valid_beh)
        
        # Draw on a copy so overlays don't accumulate
        frame_to_write = frame.copy()
        if label == None:
            i += 1
            cv2.putText(frame_to_write, f"Behavior: {label}",
            (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2, cv2.LINE_AA)
        out.write(frame_to_write)

    print(i) # Check how many frames this should have
    cap.release()
    out.release()
    print(f"Wrote: {output_video}")
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
        names=['hr', 'min', 's', 'ms', 'type', 'hardware_expected_ms']
    )
    camera_df = pd.read_csv(
        camera_data_path,
        sep=r'\s+',
        names=['hr', 'min', 's', 'ms']
    )

    behavior_df = pd.read_csv(
        behavior_data_path,
        skiprows=10,
        sep=r'\s+',
        names=['event', 'time', 'action'],
        on_bad_lines='skip'
    )
    behavior_start = pd.read_csv(
        behavior_data_path,
        nrows=4,
        header=None
    ).iloc[3,0]
    start_time = behavior_start.split()[-1]
    hr, minute, sec = map(int, start_time.split(':'))
    beh_start_time = (hr*3600 + minute*60 + sec)*1000 + 500 # Added 500 due to lack of ms precision
    behavior_df['relative_time'] = behavior_df['time'].copy()
    behavior_df['time'] = behavior_df['time'] + beh_start_time

    microscope_df['total_ms'] = convert_to_ms(microscope_df)
    camera_df['total_ms'] = convert_to_ms(camera_df)

    microscope_df = microscope_df.sort_values('total_ms')
    camera_df = camera_df.sort_values('total_ms')
    
    cm_sync_df = sync_camera_to_microscope(camera_df, microscope_df)
    print(cm_sync_df)

    cmb_sync_df = sync_behavior_to_camera_sync(behavior_df, cm_sync_df)
    print(cmb_sync_df)
    '''
    synced_df, _ = synchronize_two_to_one(camera_df, microscope_df)
    camera_df = map_camera_to_video_time(camera_df)
    print('synced')
    print(synced_df)
    print('camera')
    print(camera_df)
    '''
    cap = cv2.VideoCapture(str(raw_video_path))
    video_duration_sec = len(camera_df) / (cap.get(cv2.CAP_PROP_FPS) * 2)
    cap.release()

    #print(behavior_df)
    #behavior_df = map_behavior_to_video_time(behavior_df, video_duration_sec)
    #final_df = assign_behavior_to_camera(synced_df, camera_df, behavior_df)
    #print('behavior')
    #print(behavior_df)
    #print('final')
    #print(final_df)

    #convert_video_to_60fps_with_overlay(raw_video_path, final_df)
    #convert_sync_camera_frames(raw_video_path, cmb_sync_df)

    # SYNC BASED ON CAMERA TO SEE EVERYTHING BETTER
