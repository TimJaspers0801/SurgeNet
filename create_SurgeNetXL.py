import os
import cv2
from tqdm import tqdm

OUTPUT_PATH = "SurgeNetXL"
DATASET_PATH = "GenSurgery-video-dataset"  # the video dataset
frames_file = "frames_list.txt"
zip_size = 10000

def normalize_video_name(fname: str) -> str:
    """
    Remove multiple extensions and return clean basename without extension(s).
    Replace every '#' with '?'.
    Example:
      'video#.mp4.mweb' -> 'video?'
      'video.mp4' -> 'video'
    """
    base = os.path.basename(fname)
    # Remove multiple extensions (keep only first part before dot)
    while True:
        new = os.path.splitext(base)[0]
        if new == base:  # if no change, stop
            break
        base = new
    # Replace '#' with '?'
    base = base.replace("#", "?")
    return base

if __name__ == "__main__":
    with open(frames_file, "r") as f:
        valid_frames = set(line.strip() for line in f if line.strip())

    procedures = os.listdir(DATASET_PATH)
    for proc in procedures:
        i = 0
        zipfoldercnt = 0
        for video in tqdm(os.listdir(os.path.join(DATASET_PATH, proc))):
            video_name = normalize_video_name(video)
            frame_nb = 0
            VIDEO_PATH = os.path.join(DATASET_PATH, proc, video)
            videocap = cv2.VideoCapture(VIDEO_PATH)
            fps = int(videocap.get(cv2.CAP_PROP_FPS))
            print("video number", video)
            CURR_OUTPUT_PATH = os.path.join(OUTPUT_PATH, proc, "Batch" + str(zipfoldercnt))
            k = 0
            while videocap.isOpened():
                if i >= zip_size * (1 + zipfoldercnt):
                    zipfoldercnt += 1
                    CURR_OUTPUT_PATH = os.path.join(OUTPUT_PATH, proc, "Batch" + str(zipfoldercnt))
                if not os.path.exists(CURR_OUTPUT_PATH):
                    os.makedirs(CURR_OUTPUT_PATH)
                ret, frame = videocap.read()
                if not ret:
                    break
                if k % fps == 0:
                    frame_filename = f'{video_name}_frame_{frame_nb}_Sample{int(i)}.jpg'
                    if frame_filename in valid_frames:
                        cv2.imwrite(os.path.join(CURR_OUTPUT_PATH, frame_filename), frame)
                    frame_nb += 1
                    i += 1
                    if i % 500 == 0:
                        print("writing frame", int(i))
                k += 1
            videocap.release()
