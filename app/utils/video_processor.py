"""
Video processing and object detection module
"""

import cv2
import platform
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


class Detection:
    """Class to store detection results"""

    def __init__(self, frame_idx, class_id, class_name, bbox, confidence):
        self.frame_idx = frame_idx
        self.class_id = class_id
        self.class_name = class_name
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = confidence


def process_video(video_path, sample_rate=5):
    """
    Process video and detect golfer, club, and ball
    
    Args:
        video_path (str): Path to the video file
        sample_rate (int): Process every nth frame
        
    Returns:
        tuple: (frames, detections)
            - frames: List of processed frames
            - detections: List of Detection objects
    """
    model = YOLO("yolov8n.pt")
    class_names = model.names

    # On macOS ("Darwin"), the AVFoundation backend is often more reliable.
    # For other systems, FFMPEG is a good choice.
    backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_FFMPEG
    cap = cv2.VideoCapture(video_path, backend)
    
    if not cap.isOpened():
        backend_name = "AVFoundation" if platform.system() == "Darwin" else "FFMPEG"
        print(f"Warning: Could not open video with {backend_name} backend. Trying default.")
        cap = cv2.VideoCapture(video_path) # Fallback to default
        if not cap.isOpened():
            raise ValueError("Error opening video file with any available backend.")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count < 150:
        print(f"Short video detected ({frame_count} frames). Processing all frames.")
        sample_rate = 1

    frames = []
    detections = []

    for frame_idx in tqdm(range(0, frame_count, sample_rate), desc="Processing frames"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {frame_idx}")
            continue

        # Store original frame
        frames.append(frame)

        # Run YOLOv8 detection
        results = model(frame)

        # Process detection results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls.item())
                class_name = class_names[class_id]
                bbox = box.xyxy[0].tolist()
                confidence = box.conf.item()
                detections.append(Detection(frame_idx, class_id, class_name, bbox, confidence))

    cap.release()
    return frames, detections