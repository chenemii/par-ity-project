"""
Visualization module for creating annotated videos
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import logging
import mediapipe as mp

# Define body part groups and their colors
BODY_PART_COLORS = {
    "head": (255, 0, 0),  # Blue
    "torso": (0, 255, 0),  # Green
    "arms": (255, 165, 0),  # Orange
    "hands": (255, 0, 255),  # Magenta
    "legs": (0, 255, 255),  # Cyan
    "feet": (255, 255, 0)  # Yellow
}

# Define which landmarks belong to which body part groups
BODY_PARTS_MAPPING = {
    "head": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Nose, eyes, ears, mouth
    "torso": [11, 12, 23, 24],  # Shoulders and hips
    "arms": [11, 12, 13, 14],  # Shoulders and elbows
    "hands": [15, 16, 17, 18, 19, 20, 21,
              22],  # Wrists, pinkies, indices, thumbs
    "legs": [23, 24, 25, 26],  # Hips and knees
    "feet": [27, 28, 29, 30, 31, 32]  # Ankles, heels, foot indices
}


def create_annotated_video(video_path,
                           frames,
                           detections,
                           pose_data,
                           swing_phases,
                           trajectory_data,
                           output_dir="downloads",
                           sample_rate=5):
    """
    Create an annotated video with swing analysis visualizations
    
    Args:
        video_path (str): Path to the original video
        frames (list): List of video frames
        detections (list): List of Detection objects
        pose_data (dict): Pose estimation data
        swing_phases (dict): Swing phase segmentation data
        trajectory_data (dict): Trajectory and speed analysis data
        output_dir (str): Directory to save the output video
        sample_rate (int): The frame sampling rate used during processing
        
    Returns:
        str: Path to the annotated video
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if sample rate should be adjusted for short videos
        if len(frames) < 150 and sample_rate > 1:
            sample_rate = 1

        # Get original video filename without extension
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_dir, f"{video_name}_annotated.mp4")

        # Get video properties
        if not frames or len(frames) == 0:
            raise ValueError("No frames provided for annotation")

        height, width = frames[0].shape[:2]
        fps = 30  # Default fps
        
        # Check the original video orientation using OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open original video: {video_path}")
            
        # Read metadata from the original video if available
        rotation = 0
        # Try to get rotation metadata from the video
        if hasattr(cap, 'get') and callable(getattr(cap, 'get')):
            try:
                rotation_value = cap.get(cv2.CAP_PROP_ORIENTATION_META)
                if rotation_value == 0:  # No rotation
                    rotation = 0
                elif rotation_value == 90:  # 90 degrees clockwise
                    rotation = 270  # We'll rotate counterclockwise, so 270
                elif rotation_value == 180:  # 180 degrees
                    rotation = 180
                elif rotation_value == 270:  # 270 degrees clockwise
                    rotation = 90  # We'll rotate counterclockwise, so 90
            except:
                # If metadata reading fails, use the dimensions-based detection
                rotation = 0
        
        # If no rotation metadata or reading failed, use dimensions-based detection
        if rotation == 0:
            # Check if video is in portrait mode (height > width)
            if height > width * 1.2:  # If height is significantly greater than width
                rotation = 90  # Rotate 90 degrees counterclockwise
        
        # Close the video capture
        cap.release()
        
        # Determine output dimensions based on rotation
        output_width = width
        output_height = height
        if rotation == 90 or rotation == 270:
            # Swap dimensions for 90/270 degree rotations
            output_width, output_height = height, width
            
        # Create video writer with proper dimensions
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

        if not out.isOpened():
            raise IOError(
                f"Failed to create video writer for {output_path}. Check directory permissions."
            )

        # Process each frame
        for i, frame in enumerate(tqdm(frames,
                                       desc="Creating annotated video")):
            # Create a copy of the frame for annotations
            annotated_frame = frame.copy()
            
            # Apply rotation if needed
            if rotation == 90:
                print(f"Rotating frame {i} by 90 degrees counterclockwise")
                # Rotate 90 degrees counterclockwise
                annotated_frame = cv2.rotate(annotated_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                # Transform coordinates for detections and pose keypoints
                if i in pose_data:
                    print(f"Transforming pose data for frame {i}")
                    keypoints = pose_data[i]
                    # Debug: Check keypoints structure
                    print(f"Keypoints type: {type(keypoints)}, length: {len(keypoints)}")
                    if len(keypoints) > 0:
                        print(f"First keypoint type: {type(keypoints[0])}")
                        
                    for j in range(len(keypoints)):
                        if keypoints[j] is not None and len(keypoints[j]) >= 2:
                            try:
                                x, y = keypoints[j][0], keypoints[j][1]
                                # Fix coordinate transformation for 90-degree rotation
                                keypoints[j] = (y, width - x - 1)
                            except Exception as e:
                                print(f"Error transforming keypoint {j}: {str(e)}, value: {keypoints[j]}")
                                # Keep the keypoint as is if there's an error
                
                for detection in detections:
                    if detection.frame_idx == i * sample_rate:
                        try:
                            x1, y1, x2, y2 = detection.bbox
                            # Fix bbox coordinate transformation for 90-degree rotation
                            # The correct transformation for 90 degrees counterclockwise is:
                            # (y1, width - x2 - 1, y2, width - x1 - 1)
                            detection.bbox = (y1, width - x2 - 1, y2, width - x1 - 1)
                        except Exception as e:
                            print(f"Error transforming detection bbox: {str(e)}")
                            # Keep the bbox as is if there's an error
                        
            elif rotation == 180:
                # Rotate 180 degrees
                annotated_frame = cv2.rotate(annotated_frame, cv2.ROTATE_180)
                
                # Transform coordinates
                if i in pose_data:
                    keypoints = pose_data[i]
                    for j in range(len(keypoints)):
                        if keypoints[j] is not None and len(keypoints[j]) >= 2:
                            try:
                                x, y = keypoints[j][0], keypoints[j][1]
                                keypoints[j] = (width - x - 1, height - y - 1)
                            except Exception as e:
                                print(f"Error transforming keypoint {j}: {str(e)}")
                                # Keep the keypoint as is if there's an error
                
                for detection in detections:
                    if detection.frame_idx == i * sample_rate:
                        try:
                            x1, y1, x2, y2 = detection.bbox
                            detection.bbox = (width - x2 - 1, height - y2 - 1, width - x1 - 1, height - y1 - 1)
                        except Exception as e:
                            print(f"Error transforming detection bbox: {str(e)}")
                            # Keep the bbox as is if there's an error
                        
            elif rotation == 270:
                # Rotate 270 degrees counterclockwise (90 degrees clockwise)
                annotated_frame = cv2.rotate(annotated_frame, cv2.ROTATE_90_CLOCKWISE)
                
                # Transform coordinates
                if i in pose_data:
                    keypoints = pose_data[i]
                    for j in range(len(keypoints)):
                        if keypoints[j] is not None and len(keypoints[j]) >= 2:
                            try:
                                x, y = keypoints[j][0], keypoints[j][1]
                                # Fix coordinate transformation for 270-degree rotation
                                keypoints[j] = (height - y - 1, x)
                            except Exception as e:
                                print(f"Error transforming keypoint {j}: {str(e)}")
                                # Keep the keypoint as is if there's an error
                
                for detection in detections:
                    if detection.frame_idx == i * sample_rate:
                        try:
                            x1, y1, x2, y2 = detection.bbox
                            # Fix bbox coordinate transformation for 270-degree rotation
                            # The correct transformation for 270 degrees counterclockwise is:
                            # (height - y2 - 1, x1, height - y1 - 1, x2)
                            detection.bbox = (height - y2 - 1, x1, height - y1 - 1, x2)
                        except Exception as e:
                            print(f"Error transforming detection bbox: {str(e)}")
                            # Keep the bbox as is if there's an error

            # Draw detections
            frame_detections = [
                d for d in detections if d.frame_idx == i * sample_rate
            ]
            for detection in frame_detections:
                try:
                    # Check if bbox has exactly 4 values before unpacking
                    if not hasattr(detection, 'bbox') or not isinstance(detection.bbox, tuple) or len(detection.bbox) != 4:
                        print(f"Invalid bbox format: {getattr(detection, 'bbox', None)}")
                        continue
                        
                    x1, y1, x2, y2 = map(int, detection.bbox)

                    # Draw bounding box
                    color = (0, 255,
                             0) if detection.class_name == "person" else (0, 0,
                                                                          255)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                    # Draw label
                    label = f"{detection.class_name}: {detection.confidence:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                except Exception as e:
                    print(f"Error drawing detection: {str(e)}")
                    # Skip this detection if there's an error

            # Draw pose keypoints with different colors for different body parts
            if i in pose_data:
                keypoints = pose_data[i]

                # Draw each keypoint with its corresponding body part color
                for part_name, part_indices in BODY_PARTS_MAPPING.items():
                    color = BODY_PART_COLORS[part_name]
                    for idx in part_indices:
                        if idx < len(keypoints) and keypoints[idx] is not None and len(keypoints[idx]) >= 2:
                            try:
                                x, y = int(keypoints[idx][0]), int(keypoints[idx][1])
                                cv2.circle(annotated_frame, (x, y), 5, color, -1)
                            except Exception as e:
                                print(f"Error drawing keypoint {idx}: {str(e)}")
                                # Skip this keypoint if there's an error

                # Draw connections between keypoints
                mp_pose = mp.solutions.pose
                connections = mp_pose.POSE_CONNECTIONS

                for connection in connections:
                    start_idx, end_idx = connection

                    if (start_idx < len(keypoints) and end_idx < len(keypoints)
                            and keypoints[start_idx] is not None
                            and keypoints[end_idx] is not None
                            and len(keypoints[start_idx]) >= 2
                            and len(keypoints[end_idx]) >= 2):
                        try:
                            # Determine the color based on the body part of the start point
                            color = None
                            for part_name, part_indices in BODY_PARTS_MAPPING.items():
                                if start_idx in part_indices:
                                    color = BODY_PART_COLORS[part_name]
                                    break

                            # If no color found, use white
                            if color is None:
                                color = (255, 255, 255)

                            start_point = (int(keypoints[start_idx][0]),
                                           int(keypoints[start_idx][1]))
                            end_point = (int(keypoints[end_idx][0]),
                                         int(keypoints[end_idx][1]))

                            cv2.line(annotated_frame, start_point, end_point,
                                     color, 2)
                        except Exception as e:
                            print(f"Error drawing connection {start_idx}-{end_idx}: {str(e)}")
                            # Skip this connection if there's an error

            # Draw swing phase information
            phase = None
            for phase_name, phase_frames in swing_phases.items():
                if i in phase_frames:
                    phase = phase_name
                    break

            if phase:
                cv2.putText(annotated_frame, f"Phase: {phase}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Draw trajectory information if available
            if i in trajectory_data:
                traj_info = trajectory_data[i]
                if "club_speed" in traj_info and traj_info["club_speed"]:
                    cv2.putText(
                        annotated_frame,
                        f"Club Speed: {traj_info['club_speed']:.1f} mph",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0),
                        2)

                # Adjust ball trajectory points if we rotated the frame
                if "ball_trajectory" in traj_info and traj_info["ball_trajectory"]:
                    points = traj_info["ball_trajectory"]
                    adjusted_points = []
                    
                    # Adjust the trajectory points based on rotation
                    if rotation == 90:  # 90 degrees counterclockwise
                        for point in points:
                            try:
                                x, y = point[0], point[1]  # Access by index to avoid unpacking errors
                                adjusted_points.append((height - y - 1, x))
                            except Exception as e:
                                print(f"Error transforming trajectory point: {str(e)}")
                                # Skip this point if there's an error
                    elif rotation == 180:  # 180 degrees
                        for point in points:
                            try:
                                x, y = point[0], point[1]
                                adjusted_points.append((width - x - 1, height - y - 1))
                            except Exception as e:
                                print(f"Error transforming trajectory point: {str(e)}")
                                # Skip this point if there's an error
                    elif rotation == 270:  # 270 degrees counterclockwise
                        for point in points:
                            try:
                                x, y = point[0], point[1]
                                adjusted_points.append((y, width - x - 1))
                            except Exception as e:
                                print(f"Error transforming trajectory point: {str(e)}")
                                # Skip this point if there's an error
                    else:  # No rotation
                        adjusted_points = points
                        
                    # Draw the trajectory
                    for j in range(1, len(adjusted_points)):
                        try:
                            pt1 = (int(adjusted_points[j - 1][0]), int(adjusted_points[j - 1][1]))
                            pt2 = (int(adjusted_points[j][0]), int(adjusted_points[j][1]))
                            cv2.line(annotated_frame, pt1, pt2, (0, 255, 255), 2)
                        except Exception as e:
                            print(f"Error drawing trajectory line: {str(e)}")
                            # Skip this line if there's an error

            # Add legend for body part colors
            legend_y_start = 110
            legend_y_spacing = 30
            legend_x = 10
            legend_box_size = 20

            # Draw legend title
            cv2.putText(annotated_frame, "Body Parts Legend:",
                        (legend_x, legend_y_start - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Draw color boxes and labels for each body part
            for idx, (part_name, color) in enumerate(BODY_PART_COLORS.items()):
                y_pos = legend_y_start + idx * legend_y_spacing

                # Draw color box
                cv2.rectangle(annotated_frame,
                              (legend_x, y_pos - legend_box_size + 5),
                              (legend_x + legend_box_size, y_pos + 5), color,
                              -1)

                # Draw part name
                cv2.putText(annotated_frame, part_name.capitalize(),
                            (legend_x + legend_box_size + 10, y_pos + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Write the annotated frame to the output video
            out.write(annotated_frame)

        # Release video writer
        out.release()

        # Verify the file was created
        if not os.path.exists(output_path) or os.path.getsize(
                output_path) == 0:
            raise IOError(f"Failed to create video file at {output_path}")

        print(f"Annotated video saved to: {output_path}")
        return output_path

    except Exception as e:
        print(f"Error creating annotated video: {str(e)}")
        raise
