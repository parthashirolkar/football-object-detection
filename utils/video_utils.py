import cv2
from typing import List
from tqdm.auto import tqdm

def read_video(video_path: str) -> List:
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
    
        frames.append(frame)
    return frames


def save_video(output_video_frames: List, output_video_path: str):
    forucc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_video_path, forucc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))

    for frame in tqdm(output_video_frames, desc="Writing frames to output video.."):
        out.write(frame)
    out.release()