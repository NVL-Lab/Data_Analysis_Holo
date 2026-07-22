import cv2
from pathlib import Path

def double_video_fps(video_path: Path) -> None:
    base_path = Path(video_path).parent
    video_name = Path(video_path).stem
    cap = cv2.VideoCapture(video_path)

    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    print(fourcc)
    codec = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4))
    print("Codec:", codec)
    cap.release()
    exit()

    # get FPS of input video
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps_output = fps * 2

    video_output = base_path / f'{video_name}_corrected.avi'

    # define VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'FFV1') # mp4v
    out = cv2.VideoWriter(video_output, fourcc, fps_output,
                          (int(cap.get(3)), int(cap.get(4))))

    # read and write frames for output video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    # release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print('Saved:', video_output)