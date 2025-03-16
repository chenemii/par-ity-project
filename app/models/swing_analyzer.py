"""
Swing analysis module for golf swing segmentation and trajectory analysis
"""

import numpy as np
import cv2
from app.models.pose_estimator import calculate_joint_angles


def segment_swing(pose_data, detections, sample_rate=5):
    """
    Segment the golf swing into key phases
    
    Args:
        pose_data (dict): Dictionary mapping frame indices to pose keypoints
        detections (list): List of Detection objects
        sample_rate (int): The frame sampling rate used during processing
        
    Returns:
        dict: Dictionary mapping phase names to lists of frame indices
    """
    # Initialize swing phases
    swing_phases = {
        "setup": [],
        "backswing": [],
        "downswing": [],
        "impact": [],
        "follow_through": []
    }

    # Get frame indices with pose data
    frame_indices = sorted(pose_data.keys())

    if not frame_indices:
        return swing_phases

    # Calculate joint angles for each frame
    angles_by_frame = {}
    for idx in frame_indices:
        keypoints = pose_data[idx]
        angles = calculate_joint_angles(keypoints)
        angles_by_frame[idx] = angles

    # Analyze shoulder rotation to identify swing phases
    # This is a simplified approach - a more sophisticated algorithm would be needed for production

    # Find the frame with the maximum right shoulder angle (top of backswing)
    max_shoulder_angle = -1
    top_backswing_frame = frame_indices[0]

    for idx in frame_indices:
        angles = angles_by_frame[idx]
        if "right_shoulder" in angles and angles[
                "right_shoulder"] > max_shoulder_angle:
            max_shoulder_angle = angles["right_shoulder"]
            top_backswing_frame = idx

    # Find impact frame (when club meets ball)
    # In a real implementation, this would use club and ball detection
    impact_frame = None
    person_positions = {}

    # Extract person positions from detections
    for detection in detections:
        if detection.class_name == "person":
            frame_idx = detection.frame_idx // sample_rate  # Convert to processed frame index
            if frame_idx in frame_indices:
                person_positions[frame_idx] = detection.bbox

    # Find the frame with the most forward position (impact)
    if person_positions:
        min_x = float('inf')
        for idx, bbox in person_positions.items():
            if idx > top_backswing_frame and bbox[0] < min_x:
                min_x = bbox[0]
                impact_frame = idx

    # If impact frame not found, estimate it as 2/3 between top of backswing and end
    if impact_frame is None:
        impact_frame = frame_indices[0] + int(
            (frame_indices[-1] - top_backswing_frame) * 2 / 3)

    # Assign frames to phases
    for idx in frame_indices:
        if idx < frame_indices[len(frame_indices) // 5]:
            # First 20% of frames are setup
            swing_phases["setup"].append(idx)
        elif idx < top_backswing_frame:
            # Frames before top of backswing are backswing
            swing_phases["backswing"].append(idx)
        elif idx < impact_frame:
            # Frames between top of backswing and impact are downswing
            swing_phases["downswing"].append(idx)
        elif idx < impact_frame + 5:
            # Frames around impact
            swing_phases["impact"].append(idx)
        else:
            # Remaining frames are follow-through
            swing_phases["follow_through"].append(idx)

    return swing_phases


def analyze_trajectory(frames, detections, swing_phases, sample_rate=5):
    """
    Analyze club and ball trajectory and speed
    
    Args:
        frames (list): List of video frames
        detections (list): List of Detection objects
        swing_phases (dict): Dictionary mapping phase names to lists of frame indices
        sample_rate (int): The frame sampling rate used during processing
        
    Returns:
        dict: Dictionary mapping frame indices to trajectory data
    """
    trajectory_data = {}

    # Extract ball detections
    ball_detections = [d for d in detections if d.class_name == "sports ball"]

    # Get impact frame index
    impact_frames = swing_phases.get("impact", [])
    if not impact_frames:
        return trajectory_data

    impact_frame_idx = impact_frames[len(impact_frames) // 2]

    # Track ball trajectory after impact
    ball_trajectory = []
    ball_positions = {}

    for detection in ball_detections:
        frame_idx = detection.frame_idx // sample_rate  # Convert to processed frame index
        if frame_idx >= impact_frame_idx:
            # Calculate ball center
            x1, y1, x2, y2 = detection.bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            ball_positions[frame_idx] = (center_x, center_y)

    # Sort ball positions by frame index
    sorted_frames = sorted(ball_positions.keys())
    for idx in sorted_frames:
        ball_trajectory.append(ball_positions[idx])

    # Estimate club speed at impact
    # In a real implementation, this would use more sophisticated tracking
    club_speed = None
    if len(swing_phases.get("downswing", [])) >= 2:
        # Simplified club speed calculation
        # In reality, this would require tracking the club head specifically
        downswing_frames = swing_phases["downswing"]
        time_diff = (downswing_frames[-1] -
                     downswing_frames[0]) / 30  # Assuming 30 fps
        if time_diff > 0:
            # Simplified speed calculation (just an example)
            club_speed = 100 * (1 / time_diff)  # Arbitrary scaling

    # Populate trajectory data
    for idx in sorted(swing_phases.keys()):
        frames_in_phase = swing_phases[idx]
        for frame_idx in frames_in_phase:
            trajectory_data[frame_idx] = {
                "phase":
                idx,
                "club_speed":
                club_speed if idx == "impact" else None,
                "ball_trajectory":
                ball_trajectory
                if idx == "impact" or idx == "follow_through" else None
            }

    return trajectory_data
