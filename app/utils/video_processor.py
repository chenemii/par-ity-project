"""
Video processing and object detection module
"""

import cv2
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
    # Load YOLOv8 model
    model = YOLO("yolov8n.pt")

    # Custom class names for golf-specific objects
    class_names = model.names

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")

    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Auto-adjust sample rate based on video length
    # For short videos (less than 150 frames), don't skip any frames
    if frame_count < 150 and sample_rate > 1:
        print(f"Short video detected ({frame_count} frames). Processing all frames.")
        sample_rate = 1

    frames = []
    detections = []

    # Process frames
    for frame_idx in tqdm(range(0, frame_count, sample_rate),
                          desc="Processing frames"):
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        # Read frame
        ret, frame = cap.read()
        if not ret:
            break

        # Store original frame
        frames.append(frame)

        # Run YOLOv8 detection
        results = model(frame)

        # Process detection results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get detection information
                class_id = int(box.cls.item())
                class_name = class_names[class_id]

                # Filter for relevant objects (person, sports ball)
                if class_name in ["person", "sports ball"]:
                    bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                    confidence = box.conf.item()

                    # Create Detection object
                    detection = Detection(frame_idx, class_id, class_name,
                                          bbox, confidence)
                    detections.append(detection)

    # Release video capture
    cap.release()

    return frames, detections
