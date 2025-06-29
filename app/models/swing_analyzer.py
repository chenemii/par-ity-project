"""
Swing analysis module for golf swing segmentation and trajectory analysis
"""

import numpy as np
from app.models.pose_estimator import calculate_joint_angles


def find_top_of_backswing(pose_data):
    """Helper function to find the peak of backswing"""
    frame_indices = sorted(pose_data.keys())
    max_shoulder_angle = -1
    top_frame = frame_indices[0]
    
    for idx in frame_indices:
        keypoints = pose_data[idx]
        angles = calculate_joint_angles(keypoints)
        shoulder = angles.get("right_shoulder", 0)
        if shoulder > max_shoulder_angle:
            max_shoulder_angle = shoulder
            top_frame = idx
    
    return top_frame


def detect_impact_frame(pose_data, detections, sample_rate=1):
    """
    Simple impact detection: ball movement first, wrist speed fallback
    """
    frame_indices = sorted(pose_data.keys())
    if len(frame_indices) < 10:
        return None
    
    top_backswing = find_top_of_backswing(pose_data)
    downswing_frames = [f for f in frame_indices if f > top_backswing]
    
    if not downswing_frames:
        return None
    
    # Method 1: Ball movement (if we have ball detections)
    if detections:
        ball_detections = [d for d in detections if d.class_name == "sports ball"]
        ball_positions = {}
        
        # Create a mapping from original video frame indices to processed frame indices
        original_to_processed = {}
        for processed_idx in frame_indices:
            original_frame_idx = processed_idx * sample_rate
            original_to_processed[original_frame_idx] = processed_idx
        
        for detection in ball_detections:
            original_frame_idx = detection.frame_idx
            # Find the closest processed frame index
            processed_frame_idx = None
            if original_frame_idx in original_to_processed:
                processed_frame_idx = original_to_processed[original_frame_idx]
            else:
                # Find closest processed frame
                closest_original = min(original_to_processed.keys(), 
                                     key=lambda x: abs(x - original_frame_idx))
                if abs(closest_original - original_frame_idx) <= sample_rate:
                    processed_frame_idx = original_to_processed[closest_original]
            
            if processed_frame_idx and processed_frame_idx > top_backswing:
                x1, y1, x2, y2 = detection.bbox
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                ball_positions[processed_frame_idx] = (center_x, center_y)
        
        # Find first significant ball movement
        if len(ball_positions) >= 2:
            sorted_frames = sorted(ball_positions.keys())
            for i in range(1, len(sorted_frames)):
                curr_pos = ball_positions[sorted_frames[i]]
                prev_pos = ball_positions[sorted_frames[i-1]]
                movement = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
                
                if movement > 15:  # Significant movement threshold
                    print(f"Impact detected via ball movement at processed frame {sorted_frames[i]} (original frame {sorted_frames[i] * sample_rate})")
                    return sorted_frames[i]
    
    # Method 2: Wrist speed fallback (simple and reliable)
    max_wrist_speed = 0
    impact_frame = None
    
    for i in range(1, len(downswing_frames)):
        curr_frame = downswing_frames[i]
        prev_frame = downswing_frames[i-1]
        
        curr_angles = calculate_joint_angles(pose_data[curr_frame])
        prev_angles = calculate_joint_angles(pose_data[prev_frame])
        
        curr_wrist = curr_angles.get("right_wrist", 0)
        prev_wrist = prev_angles.get("right_wrist", 0)
        wrist_speed = abs(curr_wrist - prev_wrist)
        
        if wrist_speed > max_wrist_speed:
            max_wrist_speed = wrist_speed
            impact_frame = curr_frame
    
    print(f"Impact detected via wrist speed at processed frame {impact_frame} (original frame {impact_frame * sample_rate if impact_frame else 'N/A'})")
    return impact_frame or downswing_frames[len(downswing_frames) // 3]


def segment_swing_pose_based(pose_data, detections=None, sample_rate=1):
    """
    Simple swing segmentation with clean impact detection
    """
    swing_phases = {"setup": [], "backswing": [], "downswing": [], "impact": [], "follow_through": []}
    frame_indices = sorted(pose_data.keys())
    
    if not frame_indices:
        return swing_phases

    # 1. Find setup end (first significant movement)
    setup_end = frame_indices[0]
    initial_angles = calculate_joint_angles(pose_data[frame_indices[0]])
    initial_shoulder = initial_angles.get("right_shoulder", 0)
    
    for idx in frame_indices[1:]:
        angles = calculate_joint_angles(pose_data[idx])
        shoulder = angles.get("right_shoulder", 0)
        if abs(shoulder - initial_shoulder) > 10:
            setup_end = max(frame_indices[0], idx - 2)
            break

    # 2. Find top of backswing
    top_backswing = find_top_of_backswing(pose_data)

    # 3. Find impact frame
    impact_frame = detect_impact_frame(pose_data, detections, sample_rate)
    
    # Simple validation and fallback
    if not impact_frame or impact_frame <= top_backswing:
        downswing_frames = [f for f in frame_indices if f > top_backswing]
        impact_frame = downswing_frames[len(downswing_frames) // 3] if downswing_frames else top_backswing + 1

    print(f"Swing phases: Setup end={setup_end} (orig {setup_end * sample_rate}), Top backswing={top_backswing} (orig {top_backswing * sample_rate}), Impact={impact_frame} (orig {impact_frame * sample_rate if impact_frame else 'N/A'})")

    # 4. Assign phases
    for idx in frame_indices:
        if idx <= setup_end:
            swing_phases["setup"].append(idx)
        elif idx <= top_backswing:
            swing_phases["backswing"].append(idx)
        elif idx < impact_frame:
            swing_phases["downswing"].append(idx)
        elif idx == impact_frame:
            swing_phases["impact"].append(idx)
        else:
            swing_phases["follow_through"].append(idx)

    return swing_phases


# Wrapper function to maintain compatibility with existing Streamlit app
def segment_swing(pose_data, detections, sample_rate=1):
    """
    Main swing segmentation function (wrapper for pose-based approach)
    """
    return segment_swing_pose_based(pose_data, detections, sample_rate)


def analyze_trajectory(frames, detections, swing_phases, sample_rate=1):
    """
    Analyze ball trajectory and calculate club speed
    """
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
        frame_idx = detection.frame_idx // sample_rate
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
        actual_frames_elapsed = (downswing_frames[-1] - downswing_frames[0]) * sample_rate
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