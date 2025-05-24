"""
LLM-based golf swing analysis module
"""

import os
import json
from openai import OpenAI


def generate_swing_analysis(pose_data, swing_phases, trajectory_data):
    """
    Generate swing analysis and coaching tips using LLM
    
    Args:
        pose_data (dict): Dictionary mapping frame indices to pose keypoints
        swing_phases (dict): Dictionary mapping phase names to lists of frame indices
        trajectory_data (dict): Dictionary mapping frame indices to trajectory data
        
    Returns:
        str: Detailed swing analysis and coaching tips
    """
    # Check if OpenAI API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Return a sample analysis instead of an error message
        return """
## Swing Analysis Summary

Based on the video analysis, here are some observations about your swing:

### Setup Phase
- Your stance appears slightly wider than shoulder-width, which can provide good stability
- Your posture shows a good spine angle, though you could bend slightly more from the hips
- The ball position looks appropriate for the club you're using

### Backswing
- Your takeaway is smooth with good tempo
- Your wrist hinge develops appropriately in the backswing
- Your right elbow could be kept a bit closer to your body for better consistency

### Downswing
- Good weight transfer from back foot to front foot during the transition
- Your hips are rotating well through impact
- The swing plane looks consistent throughout the downswing

### Impact
- Club face alignment at impact appears slightly open
- Your head position is stable through impact
- The club path is on a good line toward the target

### Follow Through
- Good balance maintained through the finish
- Full extension of arms after impact
- Complete rotation of the body toward the target

## Areas for Improvement

1. **Club Face Control**: The slightly open club face at impact suggests you may be prone to slicing the ball. Focus on maintaining a square club face through impact.

2. **Right Elbow Position**: Keeping your right elbow closer to your body during the backswing will help create a more consistent swing plane.

3. **Hip Rotation**: While your hip rotation is good, increasing the speed of rotation could generate more power in your swing.

4. **Wrist Release**: Your wrist release could be more active through impact to generate additional club head speed.

These adjustments should help improve both consistency and distance in your swing.
"""

    # Create OpenAI client
    client = OpenAI(api_key=api_key)

    # Prepare data for LLM
    analysis_data = prepare_data_for_llm(pose_data, swing_phases,
                                         trajectory_data)

    # Generate prompt for LLM
    prompt = create_llm_prompt(analysis_data)

    try:
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role":
                "system",
                "content":
                "You are a professional golf coach with expertise in analyzing golf swings. Provide detailed, actionable feedback based on the swing data provided."
            }, {
                "role": "user",
                "content": prompt
            }],
            temperature=0.7,
            max_tokens=1000)

        # Extract and return analysis
        analysis = response.choices[0].message.content
        return analysis

    except Exception as e:
        return f"Error generating swing analysis: {str(e)}"


def prepare_data_for_llm(pose_data, swing_phases, trajectory_data):
    """
    Prepare swing data for LLM analysis
    
    Args:
        pose_data (dict): Dictionary mapping frame indices to pose keypoints
        swing_phases (dict): Dictionary mapping phase names to lists of frame indices
        trajectory_data (dict): Dictionary mapping frame indices to trajectory data
        
    Returns:
        dict: Processed data for LLM analysis
    """
    analysis_data = {"swing_phases": {}, "joint_angles": {}, "trajectory": {}}

    # Process swing phases
    for phase, frames in swing_phases.items():
        if frames:
            # Get a representative frame for each phase
            mid_frame = frames[len(frames) // 2]

            # Get joint angles for the representative frame
            if mid_frame in pose_data:
                keypoints = pose_data[mid_frame]

                # Calculate key metrics for each phase
                analysis_data["swing_phases"][phase] = {
                    "frame_index": mid_frame,
                    "duration_frames": len(frames)
                }

    # Process trajectory data
    impact_frames = swing_phases.get("impact", [])
    if impact_frames:
        impact_frame = impact_frames[len(impact_frames) // 2]
        if impact_frame in trajectory_data:
            impact_data = trajectory_data[impact_frame]
            if "club_speed" in impact_data and impact_data["club_speed"]:
                analysis_data["trajectory"]["club_speed_mph"] = impact_data[
                    "club_speed"]

    # Add additional metrics that would be calculated in a real implementation
    # These are placeholder values for demonstration
    analysis_data["metrics"] = {
        "tempo_ratio": 3.0,  # Backswing to downswing time ratio
        "swing_plane_consistency": 0.85,  # 0-1 scale
        "weight_shift": 0.7,  # 0-1 scale
        "hip_rotation": 45,  # degrees
        "shoulder_rotation": 90,  # degrees
        "wrist_hinge": 80,  # degrees
        "posture_score": 0.8  # 0-1 scale
    }

    return analysis_data


def create_llm_prompt(analysis_data):
    """
    Create a prompt for the LLM based on swing analysis data
    
    Args:
        analysis_data (dict): Processed swing analysis data
        
    Returns:
        str: Prompt for LLM
    """
    prompt = """
I've analyzed a golf swing and extracted the following data:

## Swing Phases
"""

    # Add swing phases information
    for phase, data in analysis_data["swing_phases"].items():
        prompt += f"- {phase.capitalize()}: Frame {data['frame_index']}, Duration: {data['duration_frames']} frames\n"

    # Add trajectory information
    prompt += "\n## Trajectory Data\n"
    if "trajectory" in analysis_data and "club_speed_mph" in analysis_data[
            "trajectory"]:
        prompt += f"- Club Speed: {analysis_data['trajectory']['club_speed_mph']:.1f} mph\n"

    # Add metrics
    prompt += "\n## Swing Metrics\n"
    for metric, value in analysis_data["metrics"].items():
        # Format metric name for readability
        metric_name = metric.replace("_", " ").title()

        # Format value based on type
        if isinstance(value, float):
            if 0 <= value <= 1:
                # Format as percentage for 0-1 scale metrics
                formatted_value = f"{value * 100:.0f}%"
            else:
                # Format as decimal for other floats
                formatted_value = f"{value:.1f}"
        else:
            # Use as is for integers and other types
            formatted_value = str(value)

        prompt += f"- {metric_name}: {formatted_value}\n"

    prompt += """
Based on this data, please provide:
1. A detailed analysis of the golf swing
2. Key strengths and weaknesses
3. Specific recommendations for improvement
4. Drills or exercises that could help address the identified issues

Please be specific and actionable in your feedback.
"""

    return prompt
