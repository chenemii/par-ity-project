"""
Swing analysis module for golf swing segmentation and trajectory analysis
"""

import numpy as np
from app.models.pose_estimator import calculate_joint_angles

def segment_swing(pose_data, detections, sample_rate=1):
    swing_phases = {"setup": [], "backswing": [], "downswing": [], "impact": [], "follow_through": []}
    frame_indices = sorted(pose_data.keys())
    if not frame_indices:
        return swing_phases

    angles_by_frame = {}
    for idx in frame_indices:
        keypoints = pose_data[idx]
        angles = calculate_joint_angles(keypoints)
        angles_by_frame[idx] = angles

    setup_end = frame_indices[0]
    initial_angles = angles_by_frame[frame_indices[0]]
    initial_shoulder = initial_angles.get("right_shoulder")
    initial_wrist = initial_angles.get("right_elbow")

    for idx in frame_indices[1:]:
        angles = angles_by_frame[idx]
        shoulder = angles.get("right_shoulder")
        wrist = angles.get("right_elbow")
        if shoulder and initial_shoulder and abs(shoulder - initial_shoulder) > 10:
            setup_end = idx
            break
        if wrist and initial_wrist and abs(wrist - initial_wrist) > 10:
            setup_end = idx
            break

    max_shoulder_angle = -1
    top_backswing_frame = setup_end
    for idx in frame_indices:
        if idx < setup_end:
            continue
        shoulder = angles_by_frame[idx].get("right_shoulder")
        if shoulder and shoulder > max_shoulder_angle:
            max_shoulder_angle = shoulder
            top_backswing_frame = idx

    # Find impact frame by looking for the point where the club head is at its lowest point
    # during the downswing, before it starts rising in the follow-through
    impact_frame = None
    min_wrist_y = float('inf')
    prev_wrist_y = None
    wrist_velocities = []
    
    # First pass: collect wrist positions and calculate velocities
    wrist_positions = []
    for idx in frame_indices:
        if idx < top_backswing_frame:
            continue
        keypoints = pose_data[idx]
        if len(keypoints) > 16:
            wrist_y = keypoints[16][1]
            wrist_positions.append((idx, wrist_y))
    
    # Calculate velocities between consecutive frames
    for i in range(1, len(wrist_positions)):
        idx, wrist_y = wrist_positions[i]
        prev_idx, prev_y = wrist_positions[i-1]
        velocity = (wrist_y - prev_y) / (idx - prev_idx)
        wrist_velocities.append((idx, velocity))
    
    # Find impact as the point where velocity changes from negative (downward) to positive (upward)
    for i in range(1, len(wrist_velocities)):
        idx, velocity = wrist_velocities[i]
        prev_idx, prev_velocity = wrist_velocities[i-1]
        if prev_velocity < 0 and velocity > 0:  # Velocity changes from negative to positive
            impact_frame = prev_idx
            break
    
    # If no clear impact point found, use the frame with minimum wrist Y position
    if impact_frame is None:
        for idx, wrist_y in wrist_positions:
            if wrist_y < min_wrist_y:
                min_wrist_y = wrist_y
                impact_frame = idx

    if impact_frame is None:
        impact_frame = frame_indices[-1]

    for idx in frame_indices:
        if idx <= setup_end:
            swing_phases["setup"].append(idx)
        elif idx <= top_backswing_frame:
            swing_phases["backswing"].append(idx)
        elif idx < impact_frame:
            swing_phases["downswing"].append(idx)
        elif idx == impact_frame:
            swing_phases["impact"].append(idx)
        else:
            swing_phases["follow_through"].append(idx)

    return swing_phases

def analyze_trajectory(frames, detections, swing_phases, sample_rate=1):
    trajectory_data = {}
    if len(frames) < 150:
        sample_rate = 1

    ball_detections = [d for d in detections if d.class_name == "sports ball"]
    impact_frames = swing_phases.get("impact", [])
    if not impact_frames:
        return trajectory_data

    impact_frame_idx = impact_frames[len(impact_frames) // 2]
    ball_trajectory = []
    ball_positions = {}

    for detection in ball_detections:
        frame_idx = detection.frame_idx
        if frame_idx >= impact_frame_idx:
            x1, y1, x2, y2 = detection.bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            ball_positions[frame_idx] = (center_x, center_y)

    sorted_frames = sorted(ball_positions.keys())
    for idx in sorted_frames:
        ball_trajectory.append(ball_positions[idx])

    club_speed = None
    downswing_frames = swing_phases.get("downswing", [])
    if len(downswing_frames) >= 2:
        actual_frames_elapsed = (downswing_frames[-1] - downswing_frames[0])
        time_diff = actual_frames_elapsed / 30
        if time_diff > 0:
            club_speed = 100 * (1 / time_diff)

    for phase_name, frames_in_phase in swing_phases.items():
        for frame_idx in frames_in_phase:
            trajectory_data[frame_idx] = {
                "phase": phase_name,
                "club_speed": club_speed if phase_name == "impact" else None,
                "ball_trajectory": ball_trajectory if phase_name in ["impact", "follow_through"] else None
            }

    return trajectory_data