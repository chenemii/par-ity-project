"""
Comparison module for frame-by-frame analysis between user and pro swings
"""

import os
import cv2
import numpy as np
from tqdm import tqdm


def extract_frames(video_path, max_frames=100):
    """
    Extract frames from a video
    
    Args:
        video_path (str): Path to the video file
        max_frames (int): Maximum number of frames to extract
        
    Returns:
        list: List of extracted frames as numpy arrays
    """
    frames = []
    
    if not os.path.exists(video_path):
        raise ValueError(f"Video file not found: {video_path}")
        
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate step to get approximately max_frames
    step = max(1, total_frames // max_frames)
    
    current_frame = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if current_frame % step == 0:
            frames.append(frame)
            
        current_frame += 1
        
    cap.release()
    
    return frames


def normalize_frames(frames, target_height=480):
    """
    Normalize frames to a consistent size while maintaining aspect ratio
    
    Args:
        frames (list): List of frames
        target_height (int): Target height for normalized frames
        
    Returns:
        list: List of normalized frames
    """
    normalized_frames = []
    
    for frame in frames:
        # Get current dimensions
        h, w = frame.shape[:2]
        
        # Calculate new width to maintain aspect ratio
        target_width = int(w * (target_height / h))
        
        # Resize the frame
        resized = cv2.resize(frame, (target_width, target_height))
        normalized_frames.append(resized)
        
    return normalized_frames


def create_side_by_side_comparison(user_frames, pro_frames, output_path, fps=30):
    """
    Create a side-by-side comparison video
    
    Args:
        user_frames (list): List of user swing frames
        pro_frames (list): List of pro swing frames
        output_path (str): Path to save the comparison video
        fps (int): Frames per second for output video
        
    Returns:
        str: Path to the comparison video
    """
    if not user_frames or not pro_frames:
        raise ValueError("Both user and pro frames must be provided")
        
    # Normalize frames to same height
    user_normalized = normalize_frames(user_frames)
    pro_normalized = normalize_frames(pro_frames)
    
    # Ensure we have the same number of frames by duplicating the last frame if needed
    max_frames = max(len(user_normalized), len(pro_normalized))
    
    while len(user_normalized) < max_frames:
        user_normalized.append(user_normalized[-1])
        
    while len(pro_normalized) < max_frames:
        pro_normalized.append(pro_normalized[-1])
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Get dimensions for the combined frame
    user_h, user_w = user_normalized[0].shape[:2]
    pro_h, pro_w = pro_normalized[0].shape[:2]
    
    # Create a combined frame with padding
    padding = 20  # Pixels between the two videos
    combined_width = user_w + pro_w + padding
    combined_height = max(user_h, pro_h)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, combined_height))
    
    if not out.isOpened():
        raise IOError(f"Failed to create video writer for {output_path}")
    
    # Create the combined video
    for i in tqdm(range(min(len(user_normalized), len(pro_normalized))), desc="Creating comparison video"):
        # Create a blank canvas
        combined = np.ones((combined_height, combined_width, 3), dtype=np.uint8) * 255
        
        # Add title text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, "Your Swing", (user_w//2 - 60, 30), font, 1, (0, 0, 0), 2)
        cv2.putText(combined, "Pro Swing", (user_w + padding + pro_w//2 - 60, 30), font, 1, (0, 0, 0), 2)
        
        # Add frame number
        cv2.putText(combined, f"Frame: {i+1}/{min(len(user_normalized), len(pro_normalized))}", 
                   (10, combined_height - 10), font, 0.5, (0, 0, 0), 1)
        
        # Paste user frame
        combined[0:user_h, 0:user_w] = user_normalized[i]
        
        # Paste pro frame
        combined[0:pro_h, user_w+padding:user_w+padding+pro_w] = pro_normalized[i]
        
        # Draw vertical line between frames
        cv2.line(combined, (user_w + padding//2, 0), (user_w + padding//2, combined_height), (0, 0, 0), 2)
        
        # Write to video
        out.write(combined)
    
    out.release()
    
    return output_path


def align_swings(user_frames, pro_frames, method="manual"):
    """
    Align user and pro swings based on swing phases
    
    Args:
        user_frames (list): List of user swing frames
        pro_frames (list): List of pro swing frames
        method (str): Alignment method ('manual' or 'auto')
        
    Returns:
        tuple: Aligned user frames and pro frames
    """
    # For now, we'll use a simple frame stretching approach
    # In the future, this could be enhanced with ML-based swing phase detection
    
    # Get frame counts
    user_count = len(user_frames)
    pro_count = len(pro_frames)
    
    # If almost equal, return as-is
    if abs(user_count - pro_count) <= 5:
        return user_frames, pro_frames
    
    # If user has more frames, subsample
    if user_count > pro_count:
        indices = np.linspace(0, user_count - 1, pro_count, dtype=int)
        return [user_frames[i] for i in indices], pro_frames
    
    # If pro has more frames, subsample
    indices = np.linspace(0, pro_count - 1, user_count, dtype=int)
    return user_frames, [pro_frames[i] for i in indices]


def create_frame_by_frame_comparison(user_video_path, pro_video_path, output_dir="downloads"):
    """
    Create a frame-by-frame comparison between user and pro golfer swings
    
    Args:
        user_video_path (str): Path to the user's golf swing video
        pro_video_path (str): Path to the professional golfer's swing video
        output_dir (str): Directory to save the comparison video
        
    Returns:
        str: Path to the comparison video
    """
    # Extract frames
    user_frames = extract_frames(user_video_path)
    pro_frames = extract_frames(pro_video_path)
    
    # Align swings
    aligned_user, aligned_pro = align_swings(user_frames, pro_frames)
    
    # Create output path
    video_name = os.path.splitext(os.path.basename(user_video_path))[0]
    output_path = os.path.join(output_dir, f"{video_name}_comparison.mp4")
    
    # Create side-by-side comparison
    return create_side_by_side_comparison(aligned_user, aligned_pro, output_path) 