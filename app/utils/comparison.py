"""
Comparison module for frame-by-frame analysis between user and pro swings

CRITICAL NOTE: This module preserves the original sizes and orientations of both user and professional videos.
Frames are saved as separate image files at their original resolutions without any resizing, rotation, or distortion.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image


def ensure_color_frame(frame):
    """
    Ensure frame is in color format (3 channels)
    
    Args:
        frame (numpy.ndarray): Input frame
        
    Returns:
        numpy.ndarray: Color frame with 3 channels
    """
    if frame is None:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    # If frame is grayscale (2D), convert to color (3D)
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif len(frame.shape) == 3 and frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif len(frame.shape) == 3 and frame.shape[2] == 4:
        # Convert RGBA to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    
    return frame


def resize_frame_proportionally(frame, target_height):
    """
    Resize frame proportionally to target height while maintaining aspect ratio
    
    Args:
        frame (numpy.ndarray): Input frame
        target_height (int): Target height
        
    Returns:
        numpy.ndarray: Resized frame
    """
    # Ensure frame is in color format
    frame = ensure_color_frame(frame)
    
    h, w = frame.shape[:2]
    if h == 0:
        return np.zeros((target_height, target_height, 3), dtype=np.uint8)
    
    # Calculate new width to maintain aspect ratio
    target_width = int(w * (target_height / h))
    
    # Resize the frame
    return cv2.resize(frame, (target_width, target_height))


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
        
    # Use standard OpenCV VideoCapture with explicit settings to prevent any rotation
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # CRITICAL: Explicitly disable ALL automatic transformations
    # This prevents OpenCV from applying any rotation based on metadata
    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)  # Disable auto-orientation
        cap.set(cv2.CAP_PROP_ORIENTATION_META, 0)  # Ignore orientation metadata
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)       # Keep BGR format
    except:
        # If properties are not supported, continue without them
        pass
    
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
            # Store frame exactly as read from video - no transformations at all
            # Only verify it's a valid color frame before storing
            if frame is not None and len(frame.shape) == 3:
                frames.append(frame.copy())
            
        current_frame += 1
        
    cap.release()
    
    return frames


def extract_key_swing_frames(video_path, frames, swing_phases=None):
    """
    Extract 3 key frames from a list of processed frames.
    1. First setup frame
    2. Last backswing frame (top of backswing)
    3. First impact frame
    
    Args:
        video_path (str): Path to the original video file (used for rotation metadata).
        frames (list): List of processed video frames.
        swing_phases (dict): Dictionary mapping phase names to lists of frame indices
                             relative to the 'frames' list.
        
    Returns:
        dict: Dictionary mapping phase names to frames
    """
    key_frames = {'setup': None, 'backswing': None, 'impact': None}
    
    if not frames:
        print("Warning: No frames provided to extract_key_swing_frames.")
        return key_frames
        
    # Determine frame indices based on swing phases
    if swing_phases:
        # Get first setup frame
        setup_frames = swing_phases.get('setup', [])
        setup_idx = setup_frames[0] if setup_frames else 0
        
        # Get last backswing frame (top of backswing)
        backswing_frames = swing_phases.get('backswing', [])
        backswing_idx = backswing_frames[-1] if backswing_frames else len(frames) // 3
        
        # Get first impact frame
        impact_frames = swing_phases.get('impact', [])
        impact_idx = impact_frames[0] if impact_frames else len(frames) // 2
    else:
        # Fallback to default indices if no swing phases provided
        setup_idx = 0
        backswing_idx = len(frames) // 3
        impact_idx = len(frames) // 2
    
    print(f"Key frame indices (relative to processed frames) - Setup: {setup_idx}, Backswing: {backswing_idx}, Impact: {impact_idx}")
    print(f"These correspond to original video frames (approx) - Setup: ~{setup_idx * 1}, Backswing: ~{backswing_idx * 1}, Impact: ~{impact_idx * 1} (assuming sample_rate=1)")
    
    # Get rotation angle from the original video file
    rotation_angle = 0
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            try:
                orientation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
                if orientation == 90:
                    rotation_angle = 270  # Rotate counterclockwise
                elif orientation == 180:
                    rotation_angle = 180
                elif orientation == 270:
                    rotation_angle = 90   # Rotate counterclockwise
                print(f"Video orientation metadata: {orientation}, applying rotation: {rotation_angle}")
            except Exception as e:
                print(f"Could not read orientation metadata: {e}")
            finally:
                cap.release()
    else:
        print(f"Warning: Video path {video_path} not found for rotation check.")

    phase_indices = {'setup': setup_idx, 'backswing': backswing_idx, 'impact': impact_idx}

    for phase_name, frame_idx in phase_indices.items():
        if 0 <= frame_idx < len(frames):
            frame = frames[frame_idx].copy()
            if rotation_angle != 0:
                frame = _apply_rotation(frame, rotation_angle)
            key_frames[phase_name] = frame
            print(f"Successfully extracted {phase_name} frame from memory.")
        else:
            print(f"Failed to extract {phase_name} frame: index {frame_idx} is out of bounds for {len(frames)} frames.")

    return key_frames


def _apply_rotation(frame, rotation_angle):
    """
    Apply rotation to a frame based on angle
    """
    if rotation_angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation_angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation_angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    else:
        return frame


def generate_improvement_comments(phase):
    """
    Generate improvement comments for each swing phase in Professional/Comparison format
    
    Args:
        phase (str): The swing phase ('setup', 'backswing', 'impact')
        
    Returns:
        dict: Dictionary with 'pro_analysis' and 'comparison' keys
    """
    comments = {
        'setup': {
            'pro_analysis': [
                "Balanced stance with feet shoulder-width apart",
                "Even weight distribution on both feet",
                "Neutral grip with hands in proper position",
                "Athletic posture with slight forward bend",
                "Ball positioned correctly for club selection"
            ],
            'comparison': [
                "Compare your stance width to the pro's balanced setup",
                "Check if your weight is evenly distributed like the pro",
                "Ensure your grip matches the pro's neutral hand position",
                "Adjust your posture to match the pro's athletic stance",
                "Position the ball in your stance similar to the pro"
            ]
        },
        'backswing': {
            'pro_analysis': [
                "Full 90+ degree shoulder rotation",
                "Controlled hip turn with stable lower body",
                "Club on proper swing plane at top",
                "Consistent spine angle throughout",
                "Minimal weight shift to right side"
            ],
            'comparison': [
                "Increase your shoulder turn to match the pro's full rotation",
                "Control your hip movement like the pro's stable base",
                "Adjust your club position to match the pro's swing plane",
                "Maintain spine angle consistency like the professional",
                "Minimize weight shift compared to the pro's centered position"
            ]
        },
        'impact': {
            'pro_analysis': [
                "Weight shifted to front foot (70-80%)",
                "Hands ahead of ball at impact",
                "Square club face to target line",
                "Head behind ball with steady position",
                "Hips and shoulders aligned to target"
            ],
            'comparison': [
                "Shift more weight to your front foot like the pro",
                "Get your hands ahead of the ball like the professional",
                "Square your club face to match the pro's alignment",
                "Keep your head steady and behind the ball like the pro",
                "Align your body to the target like the professional"
            ]
        }
    }
    
    return comments.get(phase, {'pro_analysis': [], 'comparison': []})


def load_pro_reference_images(pro_images_dir="pro_reference"):
    """
    Load professional golfer reference images from directory
    
    Args:
        pro_images_dir (str): Directory containing professional reference images
        
    Returns:
        dict: Dictionary with phase names as keys and image arrays as values
    """
    # Get the absolute path to the pro_reference directory
    # This ensures it works regardless of the current working directory
    if not os.path.isabs(pro_images_dir):
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        pro_images_dir = os.path.join(script_dir, pro_images_dir)
    
    pro_frames = {}
    
    # Expected filenames for the 3 phases
    phase_files = {
        'setup': 'setup.jpg',
        'backswing': 'backswing.jpg', 
        'impact': 'impact.jpg'
    }
    
    for phase, filename in phase_files.items():
        image_path = os.path.join(pro_images_dir, filename)
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            if image is not None:
                # Ensure the image is in color format
                image = ensure_color_frame(image)
                pro_frames[phase] = image
            else:
                # Create a placeholder if image can't be loaded
                pro_frames[phase] = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            # Create a placeholder if file doesn't exist
            pro_frames[phase] = np.zeros((480, 640, 3), dtype=np.uint8)
    
    return pro_frames


def save_frame_with_orientation(frame, output_path):
    """
    Save a frame using PIL after converting from BGR to RGB.
    Ensures proper color handling and orientation.
    
    Args:
        frame (numpy.ndarray): Frame in BGR format (OpenCV)
        output_path (str): Path to save the image
    """
    try:
        if frame is None or frame.size == 0:
            # Save a black image if frame is invalid
            black = np.zeros((480, 640, 3), dtype=np.uint8)
            img = Image.fromarray(black)
            img.save(output_path, format="JPEG", quality=95)
            return
        
        # Verify frame is in color (3 channels)
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            raise ValueError(f"Frame is not in color format. Shape: {frame.shape}")
        
        # Convert BGR (OpenCV) to RGB (PIL)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create PIL image and save with high quality
        img = Image.fromarray(rgb_frame)
        img.save(output_path, format="JPEG", quality=95)
        
    except Exception as e:
        print(f"Warning: Error saving frame to {output_path}: {str(e)}")
        # Create a fallback black image
        try:
            black = np.zeros((480, 640, 3), dtype=np.uint8)
            img = Image.fromarray(black)
            img.save(output_path, format="JPEG", quality=95)
        except Exception as fallback_error:
            print(f"Error: Could not save fallback image: {str(fallback_error)}")
            raise


def create_key_frame_comparison(user_video_path, pro_video_path=None, user_swing_phases=None, pro_swing_phases=None, output_dir="downloads", use_pro_images=True):
    """
    Create separate images for 3 key frames from user and pro golfer swings
    
    IMPORTANT: This function preserves the original sizes of both user and professional frames.
    No resizing, rotation, or distortion is applied to either frame. Each frame is saved
    as a separate image file at its original resolution.
    
    Args:
        user_video_path (str): Path to the user's golf swing video
        pro_video_path (str): Path to the professional golfer's swing video (optional if use_pro_images=True)
        user_swing_phases (dict): Optional swing phase data for user video
        pro_swing_phases (dict): Optional swing phase data for pro video
        output_dir (str): Directory to save the separate images
        use_pro_images (bool): Whether to use provided pro reference images instead of video
        
    Returns:
        dict: Dictionary with phase names as keys and dictionaries containing 
              'user_image_path', 'pro_image_path', 'title', and 'comments' as values
    """
    # Extract key frames from user video
    user_frames = extract_key_swing_frames(user_video_path, user_frames, user_swing_phases)
    
    # Get pro frames either from provided images or video
    if use_pro_images:
        pro_frames = load_pro_reference_images()
    else:
        pro_frames = extract_key_swing_frames(pro_video_path, pro_frames, pro_swing_phases)
    
    # Create output directory with absolute path
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    comparison_data = {}
    phases = ['setup', 'backswing', 'impact']
    phase_titles = ['Starting Position', 'Top of Backswing', 'Impact with Ball']
    
    for i, phase in enumerate(phases):
        # Get frames for this phase
        user_frame = user_frames.get(phase, np.zeros((480, 640, 3), dtype=np.uint8))
        pro_frame = pro_frames.get(phase, np.zeros((480, 640, 3), dtype=np.uint8))
        
        # CRITICAL: Keep user frame EXACTLY as extracted - no processing at all
        # Only ensure pro frame is in color format since it comes from reference images
        pro_frame = ensure_color_frame(pro_frame)
        
        # Save user frame with original size using PIL to ensure correct orientation and color
        video_name = os.path.splitext(os.path.basename(user_video_path))[0]
        user_output_path = os.path.join(output_dir, f"{video_name}_{phase}_user.jpg")
        pro_output_path = os.path.join(output_dir, f"{video_name}_{phase}_pro.jpg")
        
        # Save user image using PIL (handles BGR->RGB and orientation)
        try:
            save_frame_with_orientation(user_frame, user_output_path)
            user_success = True
        except Exception as e:
            print(f"Warning: Failed to save user image to {user_output_path}: {e}")
            user_success = False
        # Save pro image using OpenCV (as before)
        pro_success = cv2.imwrite(pro_output_path, pro_frame)
        
        if user_success:
            print(f"Successfully saved user image: {user_output_path}")
        if not user_success:
            print(f"Warning: Failed to save user image to {user_output_path}")
        if pro_success:
            print(f"Successfully saved pro image: {pro_output_path}")
        if not pro_success:
            print(f"Warning: Failed to save pro image to {pro_output_path}")
        
        # Get improvement comments
        comments = generate_improvement_comments(phase)
        
        comparison_data[phase] = {
            'user_image_path': user_output_path,
            'pro_image_path': pro_output_path,
            'title': phase_titles[i],
            'comments': comments
        }
    
    return comparison_data


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
        # Use the color-safe resize function
        resized = resize_frame_proportionally(frame, target_height)
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
        
    # Ensure all frames are in color format
    user_frames = [ensure_color_frame(frame) for frame in user_frames]
    pro_frames = [ensure_color_frame(frame) for frame in pro_frames]
    
    # Get dimensions from first frames
    user_h, user_w = user_frames[0].shape[:2]
    pro_h, pro_w = pro_frames[0].shape[:2]
    
    # Choose target height (smaller of the two, capped at 720p)
    target_height = min(user_h, pro_h, 720)
    
    # Resize both user and pro frames proportionally to the same height
    user_resized = []
    for frame in user_frames:
        resized = resize_frame_proportionally(frame, target_height)
        user_resized.append(resized)
    
    pro_resized = []
    for frame in pro_frames:
        resized = resize_frame_proportionally(frame, target_height)
        pro_resized.append(resized)
    
    # Ensure we have the same number of frames by duplicating the last frame if needed
    max_frames = max(len(user_resized), len(pro_resized))
    
    user_aligned = user_resized.copy()
    while len(user_aligned) < max_frames:
        user_aligned.append(user_aligned[-1])
        
    while len(pro_resized) < max_frames:
        pro_resized.append(pro_resized[-1])
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Get dimensions for the combined frame using original user frame dimensions
    pro_h, pro_w = pro_resized[0].shape[:2]
    
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
    for i in tqdm(range(min(len(user_aligned), len(pro_resized))), desc="Creating comparison video"):
        # Create a blank canvas
        combined = np.ones((combined_height, combined_width, 3), dtype=np.uint8) * 255
        
        # Add title text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, "Your Swing", (user_w//2 - 60, 30), font, 1, (0, 0, 0), 2)
        cv2.putText(combined, "Pro Swing", (user_w + padding + pro_w//2 - 60, 30), font, 1, (0, 0, 0), 2)
        
        # Add frame number
        cv2.putText(combined, f"Frame: {i+1}/{min(len(user_aligned), len(pro_resized))}", 
                   (10, combined_height - 10), font, 0.5, (0, 0, 0), 1)
        
        # Paste user frame at original size and orientation
        y_offset_user = (combined_height - user_h) // 2
        combined[y_offset_user:y_offset_user + user_h, 0:user_w] = user_aligned[i]
        
        # Paste pro frame
        y_offset_pro = (combined_height - pro_h) // 2
        combined[y_offset_pro:y_offset_pro + pro_h, user_w + padding:user_w + padding + pro_w] = pro_resized[i]
        
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