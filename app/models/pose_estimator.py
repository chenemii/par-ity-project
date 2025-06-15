"""
Pose estimation module for golf swing analysis
"""

import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

class PoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False,
                                      model_complexity=1,
                                      enable_segmentation=False,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        keypoints = []
        h, w, _ = frame.shape

        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                visibility = landmark.visibility
                keypoints.append([x, y, visibility])
        else:
            center_x, center_y = w // 2, h // 2
            for _ in range(33):
                keypoints.append([center_x, center_y, 0.0])

        return keypoints

    def close(self):
        self.pose.close()

def analyze_pose(frames):
    """
    Analyze pose in video frames
    
    Args:
        frames (list): List of video frames
        
    Returns:
        dict: Dictionary mapping frame indices to pose keypoints
    """
    pose_estimator = PoseEstimator()
    pose_data = {}

    for i, frame in enumerate(tqdm(frames, desc="Analyzing pose")):
        keypoints = pose_estimator.process_frame(frame)
        # Store all frames, even if no pose is detected
        pose_data[i] = keypoints if keypoints is not None else []

    pose_estimator.close()
    return pose_data

def calculate_joint_angles(keypoints):
    """
    Calculate joint angles from keypoints.
    
    Args:
        keypoints: List of [x, y, visibility] for each landmark
        
    Returns:
        Dictionary of joint angles
    """
    if not keypoints or len(keypoints) < 33:  # MediaPipe Pose has 33 landmarks
        return {}
    
    angles = {}
    
    # Right shoulder angle (landmarks 11, 13, 15)
    if all(keypoints[i][2] > 0.5 for i in [11, 13, 15]):
        shoulder = np.array(keypoints[11][:2])
        elbow = np.array(keypoints[13][:2])
        wrist = np.array(keypoints[15][:2])
        v1 = shoulder - elbow
        v2 = wrist - elbow
        angle = np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
        angles["right_shoulder"] = angle
    
    # Right elbow angle (landmarks 13, 15, 17)
    if all(keypoints[i][2] > 0.5 for i in [13, 15, 17]):
        upper_arm = np.array(keypoints[13][:2])
        elbow = np.array(keypoints[15][:2])
        wrist = np.array(keypoints[17][:2])
        v1 = upper_arm - elbow
        v2 = wrist - elbow
        angle = np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
        angles["right_elbow"] = angle
    
    # Right wrist angle (landmarks 15, 17, 19)
    if all(keypoints[i][2] > 0.5 for i in [15, 17, 19]):
        elbow = np.array(keypoints[15][:2])
        wrist = np.array(keypoints[17][:2])
        hand = np.array(keypoints[19][:2])
        v1 = elbow - wrist
        v2 = hand - wrist
        angle = np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
        angles["right_wrist"] = angle
    
    return angles