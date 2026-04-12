import cv2
import numpy as np

def read_video_frames(video_path):
    """
    Read all frames from a video file.
    Returns: list of numpy arrays (H, W, 3) in BGR order.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"Successfully read {len(frames)} frames, shape: {frames[0].shape}")
    return frames

def write_video_frames(frames, output_path, fps=30):
    """
    Write a list of frames to a video file.
    """
    if not frames:
        print("No frames to write.")
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'avc1', 'X264'
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for frame in frames:
        out.write(frame)
    out.release()
    print(f"Video saved: {output_path}")