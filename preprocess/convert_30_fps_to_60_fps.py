# import modules
import cv2


# open file
base_path = Path(
        '/data/project/nvl_lab/HoloBMI/Behavior/191003/NVI13/base'
)

video_path = base_path / 'video_2019-10-03T16_03_57.avi'
cap = cv2.VideoCapture(video_path)

# get FPS of input video
fps = cap.get(cv2.CAP_PROP_FPS)

# define output video and it's FPS
output_file = 'output.mp4'
output_fps = fps * 2

# define VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, output_fps,
                      (int(cap.get(3)), int(cap.get(4))))

# read and write frams for output video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    out.write(frame)

# release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Saved:", output_file)
