"""
Pose estimation module for golf swing analysis
"""

import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm


class PoseEstimator:
    """MediaPipe-based pose estimator for golf swing analysis"""

    def __init__(self):
        """Initialize the pose estimator"""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False,
                                      model_complexity=1,
                                      enable_segmentation=False,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)

    def process_frame(self, frame):
        """
        Process a single frame and extract pose landmarks
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            list: List of keypoints [x, y, visibility] or None if not detected
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        results = self.pose.process(frame_rgb)

        if not results.pose_landmarks:
            return None

        # Extract keypoints
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            # Convert normalized coordinates to pixel coordinates
            h, w, _ = frame.shape
            x, y = int(landmark.x * w), int(landmark.y * h)
            visibility = landmark.visibility
            keypoints.append([x, y, visibility])

        return keypoints

    def close(self):
        """Release resources"""
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
        if keypoints:
            pose_data[i] = keypoints

    pose_estimator.close()

    return pose_data


def calculate_joint_angles(keypoints):
    """
    Calculate joint angles from pose keypoints
    
    Args:
        keypoints (list): List of keypoints [x, y, visibility]
        
    Returns:
        dict: Dictionary of joint angles in degrees
    """
    # Define joint connections for angle calculation
    joint_connections = {
        "right_shoulder": [
            mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value,
            mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value,
            mp.solutions.pose.PoseLandmark.RIGHT_HIP.value
        ],
        "left_shoulder": [
            mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value,
            mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value,
            mp.solutions.pose.PoseLandmark.LEFT_HIP.value
        ],
        "right_elbow": [
            mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value,
            mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value,
            mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value
        ],
        "left_elbow": [
            mp.solutions.pose.PoseLandmark.LEFT_WRIST.value,
            mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value,
            mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value
        ],
        "right_hip": [
            mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value,
            mp.solutions.pose.PoseLandmark.RIGHT_HIP.value,
            mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value
        ],
        "left_hip": [
            mp.solutions.pose.PoseLandmark.LEFT_KNEE.value,
            mp.solutions.pose.PoseLandmark.LEFT_HIP.value,
            mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value
        ],
        "right_knee": [
            mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value,
            mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value,
            mp.solutions.pose.PoseLandmark.RIGHT_HIP.value
        ],
        "left_knee": [
            mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value,
            mp.solutions.pose.PoseLandmark.LEFT_KNEE.value,
            mp.solutions.pose.PoseLandmark.LEFT_HIP.value
        ]
    }

    angles = {}

    for joint_name, landmarks in joint_connections.items():
        # Get the three points that form the angle
        if all(landmarks[i] < len(keypoints) for i in range(3)):
            p1 = np.array(keypoints[landmarks[0]][:2])
            p2 = np.array(keypoints[landmarks[1]][:2])
            p3 = np.array(keypoints[landmarks[2]][:2])

            # Calculate vectors
            v1 = p1 - p2
            v2 = p3 - p2

            # Calculate angle
            cosine_angle = np.dot(
                v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            angle_degrees = np.degrees(angle)

            angles[joint_name] = angle_degrees

    return angles
