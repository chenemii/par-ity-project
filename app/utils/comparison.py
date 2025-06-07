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


def extract_key_swing_frames(video_path, swing_phases=None):
    """
    Extract 3 key frames from a golf swing video:
    1. Starting position (setup)
    2. Top of backswing
    3. Impact with ball
    
    Args:
        video_path (str): Path to the video file
        swing_phases (dict): Optional swing phase data for precise frame selection
        
    Returns:
        dict: Dictionary with keys 'setup', 'backswing', 'impact' 
              and frame images as values
    """
    if not os.path.exists(video_path):
        raise ValueError(f"Video file not found: {video_path}")
        
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    key_frames = {}
    
    if swing_phases:
        # Use provided swing phase data for precise frame selection
        frame_indices = {
            'setup': swing_phases.get('setup', [0])[0] if swing_phases.get('setup') else 0,
            'backswing': swing_phases.get('backswing', [total_frames//3])[-1] if swing_phases.get('backswing') else total_frames//3,
            'impact': swing_phases.get('impact', [total_frames//2])[len(swing_phases.get('impact', [total_frames//2]))//2] if swing_phases.get('impact') else total_frames//2
        }
    else:
        # Use estimated frame positions for 3 frames
        frame_indices = {
            'setup': 0,  # First frame
            'backswing': total_frames // 3,  # 33% through
            'impact': int(total_frames * 0.6)  # 60% through
        }
    
    # Extract the specific frames
    for phase, frame_idx in frame_indices.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_idx, total_frames - 1))
        ret, frame = cap.read()
        if ret:
            # Keep original orientation - no rotation
            key_frames[phase] = frame
        else:
            # If frame extraction fails, use a black frame
            key_frames[phase] = np.zeros((480, 640, 3), dtype=np.uint8)
    
    cap.release()
    
    return key_frames


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
                pro_frames[phase] = image
            else:
                # Create a placeholder if image can't be loaded
                pro_frames[phase] = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            # Create a placeholder if file doesn't exist
            pro_frames[phase] = np.zeros((480, 640, 3), dtype=np.uint8)
    
    return pro_frames


def create_key_frame_comparison(user_video_path, pro_video_path=None, user_swing_phases=None, pro_swing_phases=None, output_dir="downloads", use_pro_images=True):
    """
    Create a comparison of 3 key frames between user and pro golfer swings
    
    Args:
        user_video_path (str): Path to the user's golf swing video
        pro_video_path (str): Path to the professional golfer's swing video (optional if use_pro_images=True)
        user_swing_phases (dict): Optional swing phase data for user video
        pro_swing_phases (dict): Optional swing phase data for pro video
        output_dir (str): Directory to save the comparison images
        use_pro_images (bool): Whether to use provided pro reference images instead of video
        
    Returns:
        dict: Dictionary with phase names as keys and image paths as values
    """
    # Extract key frames from user video
    user_frames = extract_key_swing_frames(user_video_path, user_swing_phases)
    
    # Get pro frames either from provided images or video
    if use_pro_images:
        pro_frames = load_pro_reference_images()
    else:
        pro_frames = extract_key_swing_frames(pro_video_path, pro_swing_phases)
    
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
        
        # Resize frames to consistent size while maintaining portrait orientation
        target_height = 400
        user_frame = resize_frame_maintain_aspect(user_frame, target_height)
        pro_frame = resize_frame_maintain_aspect(pro_frame, target_height)
        
        # Create side-by-side comparison
        comparison_image = create_side_by_side_image(user_frame, pro_frame, phase_titles[i])
        
        # Save the comparison image with absolute path
        video_name = os.path.splitext(os.path.basename(user_video_path))[0]
        output_path = os.path.join(output_dir, f"{video_name}_{phase}_comparison.jpg")
        
        # Ensure the image is saved successfully
        success = cv2.imwrite(output_path, comparison_image)
        if not success:
            print(f"Warning: Failed to save image to {output_path}")
        else:
            print(f"Successfully saved comparison image: {output_path}")
        
        # Get improvement comments
        comments = generate_improvement_comments(phase)
        
        comparison_data[phase] = {
            'image_path': output_path,
            'title': phase_titles[i],
            'comments': comments
        }
    
    return comparison_data


def resize_frame_maintain_aspect(frame, target_height):
    """
    Resize frame to target height while maintaining aspect ratio
    
    Args:
        frame (numpy.ndarray): Input frame
        target_height (int): Target height
        
    Returns:
        numpy.ndarray: Resized frame
    """
    h, w = frame.shape[:2]
    target_width = int(w * (target_height / h))
    return cv2.resize(frame, (target_width, target_height))


def create_side_by_side_image(user_frame, pro_frame, title):
    """
    Create a side-by-side comparison image
    
    Args:
        user_frame (numpy.ndarray): User's swing frame
        pro_frame (numpy.ndarray): Pro's swing frame
        title (str): Title for the comparison
        
    Returns:
        numpy.ndarray: Combined comparison image
    """
    # Get dimensions
    user_h, user_w = user_frame.shape[:2]
    pro_h, pro_w = pro_frame.shape[:2]
    
    # Create padding and title space
    padding = 20
    title_height = 60
    max_height = max(user_h, pro_h)
    total_width = user_w + pro_w + padding
    total_height = max_height + title_height
    
    # Create blank canvas
    canvas = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255
    
    # Add title
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_size = cv2.getTextSize(title, font, 1.2, 2)[0]
    title_x = (total_width - title_size[0]) // 2
    cv2.putText(canvas, title, (title_x, 40), font, 1.2, (0, 0, 0), 2)
    
    # Add user frame
    y_offset = title_height + (max_height - user_h) // 2
    canvas[y_offset:y_offset + user_h, 0:user_w] = user_frame
    
    # Add pro frame
    y_offset = title_height + (max_height - pro_h) // 2
    canvas[y_offset:y_offset + pro_h, user_w + padding:user_w + padding + pro_w] = pro_frame
    
    # Draw vertical separator line
    line_x = user_w + padding // 2
    cv2.line(canvas, (line_x, title_height), (line_x, total_height), (200, 200, 200), 2)
    
    return canvas


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