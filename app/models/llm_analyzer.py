"""
LLM-based golf swing analysis module
"""

import json
import httpx
from openai import OpenAI
import streamlit as st
import re
import numpy as np
from app.models.pose_estimator import calculate_joint_angles


def check_llm_services():
    """
    Check which LLM services are configured
    
    Returns:
        dict: Dictionary with service availability and configuration
    """
    services = {
        'ollama': {
            'available': False,
            'config': {}
        },
        'openai': {
            'available': False,
            'config': {}
        }
    }

    # Check Ollama configuration
    try:
        ollama_url = st.secrets.get("ollama", {}).get("base_url", "")
        ollama_model = st.secrets.get("ollama", {}).get("model", "")
        if ollama_url and ollama_model:
            services['ollama']['available'] = True
            services['ollama']['config'] = {
                'base_url': ollama_url,
                'model': ollama_model
            }
    except (KeyError, FileNotFoundError, AttributeError):
        pass

    # Check OpenAI configuration
    try:
        openai_key = st.secrets.get("openai", {}).get("api_key", "")
        if openai_key:
            services['openai']['available'] = True
            services['openai']['config'] = {'api_key': openai_key}
    except (KeyError, FileNotFoundError, AttributeError):
        pass

    return services


def generate_swing_analysis(pose_data, swing_phases, trajectory_data):
    """
    Generate swing analysis and coaching tips using LLM
    
    Args:
        pose_data (dict): Dictionary mapping frame indices to pose keypoints
        swing_phases (dict): Dictionary mapping phase names to lists of frame indices
        trajectory_data (dict): Dictionary mapping frame indices to trajectory data
        
    Returns:
        str: Detailed swing analysis and coaching tips, or error message
    """
    # Check available services
    services = check_llm_services()

    # If no services are available, return error message
    if not services['ollama']['available'] and not services['openai']['available']:
        return "Error: No AI services available. Please ensure either Ollama is running or OpenAI API key is configured."

    # Prepare data for LLM
    analysis_data = prepare_data_for_llm(pose_data, swing_phases,
                                         trajectory_data)
    prompt = create_llm_prompt(analysis_data)

    # Try Ollama first if available
    if services['ollama']['available']:
        try:
            analysis = call_ollama_service(prompt,
                                           services['ollama']['config'])
            if analysis:
                return analysis
        except Exception as e:
            print(f"Error with Ollama: {str(e)}")

    # Try OpenAI if available
    if services['openai']['available']:
        try:
            analysis = call_openai_service(prompt,
                                           services['openai']['config'])
            if analysis:
                return analysis
        except Exception as e:
            print(f"Error with OpenAI: {str(e)}")

    # If both services failed, return error message
    return "Error: All AI services failed. Please check your API keys and service configurations."


def call_ollama_service(prompt, config):
    """
    Call Ollama service for analysis
    
    Args:
        prompt (str): The analysis prompt
        config (dict): Ollama configuration
        
    Returns:
        str: Analysis result or None if failed
    """
    try:
        # Create a custom httpx client
        http_client = httpx.Client()

        # Initialize OpenAI client with Ollama endpoint
        client = OpenAI(
            base_url=config['base_url'],
            api_key="ollama",  # Ollama doesn't need a real API key
            http_client=http_client)

        response = client.chat.completions.create(
            model=config['model'],
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

        return response.choices[0].message.content

    except Exception as e:
        print(f"Ollama service error: {str(e)}")
        return None


def call_openai_service(prompt, config):
    """
    Call OpenAI service for analysis
    
    Args:
        prompt (str): The analysis prompt
        config (dict): OpenAI configuration
        
    Returns:
        str: Analysis result or None if failed
    """
    try:
        # Create a custom httpx client without proxies
        http_client = httpx.Client()

        # Initialize the OpenAI client
        client = OpenAI(api_key=config['api_key'], http_client=http_client)

        try:
            # Try with GPT-4 first
            response = client.chat.completions.create(
                model="gpt-4o-mini",
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

            return response.choices[0].message.content

        except Exception as gpt4_error:
            # If there's an error with GPT-4 (like quota exceeded), try GPT-3.5
            print(
                f"Error with GPT-4: {str(gpt4_error)}. Falling back to GPT-3.5-turbo..."
            )

            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
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

                return response.choices[0].message.content

            except Exception as gpt35_error:
                print(f"Error with GPT-3.5: {str(gpt35_error)}")
                return None

    except Exception as e:
        print(f"OpenAI service error: {str(e)}")
        return None


def prepare_data_for_llm(pose_data, swing_phases, trajectory_data=None):
    """
    Prepare swing data for LLM analysis
    
    Args:
        pose_data (dict): Dictionary mapping frame indices to pose keypoints
        swing_phases (dict): Dictionary mapping phase names to lists of frame indices
        trajectory_data (dict, optional): Ball trajectory data
        
    Returns:
        dict: Formatted swing data for LLM
    """
    
    # Calculate actual biomechanical metrics from pose data
    bio_metrics = calculate_biomechanical_metrics(pose_data, swing_phases)
    
    # Calculate phase durations and timing metrics
    setup_frames = swing_phases.get("setup", [])
    backswing_frames = swing_phases.get("backswing", []) 
    downswing_frames = swing_phases.get("downswing", [])
    impact_frames = swing_phases.get("impact", [])
    follow_through_frames = swing_phases.get("follow_through", [])
    
    # Calculate tempo ratio (downswing:backswing)
    backswing_duration = len(backswing_frames) if backswing_frames else 1
    downswing_duration = len(downswing_frames) if downswing_frames else 1
    tempo_ratio = downswing_duration / backswing_duration if backswing_duration > 0 else 1.0
    
    # Calculate total swing duration and club speed estimates
    total_frames = len(setup_frames) + len(backswing_frames) + len(downswing_frames) + len(impact_frames) + len(follow_through_frames)
    
    # Estimate club speed based on downswing duration (faster downswing = higher speed)
    # Professional downswings are typically 10-15 frames at 30fps
    if downswing_duration > 0:
        speed_factor = max(0.5, min(2.0, 12.0 / downswing_duration))  # Normalize around 12 frames
        estimated_club_speed = 70 + (speed_factor * 40)  # Base 70 mph, up to 110 mph
    else:
        estimated_club_speed = 85
    
    # Process joint angles if available
    joint_angles = {}
    if pose_data:
        # Get a representative frame for joint analysis
        rep_frame = None
        if impact_frames:
            rep_frame = impact_frames[0]
        elif downswing_frames:
            rep_frame = downswing_frames[len(downswing_frames) // 2]
        elif backswing_frames:
            rep_frame = backswing_frames[-1]
        
        if rep_frame and rep_frame in pose_data:
            try:
                from app.models.pose_estimator import calculate_joint_angles
                joint_angles = calculate_joint_angles(pose_data[rep_frame])
            except Exception as e:
                print(f"Error calculating joint angles: {e}")
                joint_angles = {}
    
    # Prepare the structured data
    swing_data = {
        "swing_phases": {
            "setup": {
                "frame_count": len(setup_frames),
                "duration_ms": len(setup_frames) * 33.33  # Assuming 30fps
            },
            "backswing": {
                "frame_count": len(backswing_frames),
                "duration_ms": len(backswing_frames) * 33.33
            },
            "downswing": {
                "frame_count": len(downswing_frames),
                "duration_ms": len(downswing_frames) * 33.33
            },
            "impact": {
                "frame_count": len(impact_frames),
                "duration_ms": len(impact_frames) * 33.33
            },
            "follow_through": {
                "frame_count": len(follow_through_frames),
                "duration_ms": len(follow_through_frames) * 33.33
            }
        },
        
        "timing_metrics": {
            "tempo_ratio": round(tempo_ratio, 2),
            "total_swing_frames": total_frames,
            "total_swing_time_ms": total_frames * 33.33,
            "estimated_club_speed_mph": round(estimated_club_speed, 1)
        },
        
        "biomechanical_metrics": {
            # Core rotation metrics
            "hip_rotation_degrees": round(bio_metrics.get("hip_rotation", 25), 1),
            "shoulder_rotation_degrees": round(bio_metrics.get("shoulder_rotation", 60), 1),
            "chest_rotation_efficiency_percent": round(bio_metrics.get("chest_rotation_efficiency", 0.6) * 100, 1),
            
            # Weight transfer and stability
            "weight_shift_percent": round(bio_metrics.get("weight_shift", 0.5) * 100, 1),
            "ground_force_efficiency_percent": round(bio_metrics.get("ground_force_efficiency", 0.6) * 100, 1),
            "hip_thrust_percent": round(bio_metrics.get("hip_thrust", 0.5) * 100, 1),
            
            # Arm and club mechanics
            "arm_extension_percent": round(bio_metrics.get("arm_extension", 0.6) * 100, 1),
            "wrist_hinge_degrees": round(bio_metrics.get("wrist_hinge", 60), 1),
            "swing_plane_consistency_percent": round(bio_metrics.get("swing_plane_consistency", 0.6) * 100, 1),
            
            # Posture and stability
            "posture_score_percent": round(bio_metrics.get("posture_score", 0.6) * 100, 1),
            "head_movement_lateral_inches": round(bio_metrics.get("head_movement_lateral", 3.0), 1),
            "head_movement_vertical_inches": round(bio_metrics.get("head_movement_vertical", 2.0), 1),
            
            # Leg mechanics
            "knee_flexion_address_degrees": round(bio_metrics.get("knee_flexion_address", 25), 1),
            "knee_flexion_impact_degrees": round(bio_metrics.get("knee_flexion_impact", 30), 1),
            
            # Advanced coordination metrics
            "transition_smoothness_percent": round(bio_metrics.get("transition_smoothness", 0.6) * 100, 1),
            "kinematic_sequence_percent": round(bio_metrics.get("kinematic_sequence", 0.6) * 100, 1),
            "energy_transfer_efficiency_percent": round(bio_metrics.get("energy_transfer", 0.6) * 100, 1),
            "power_accumulation_percent": round(bio_metrics.get("power_accumulation", 0.6) * 100, 1),
            
            # Performance estimates
            "potential_distance_yards": round(bio_metrics.get("potential_distance", 200), 0),
            "speed_generation_method": bio_metrics.get("speed_generation", "Mixed")
        },
        
        "joint_angles": joint_angles,
        
        "trajectory_analysis": trajectory_data if trajectory_data else {
            "estimated_carry_distance": round(bio_metrics.get("potential_distance", 200) * 0.85, 0),
            "estimated_ball_speed": round(estimated_club_speed * 1.4, 1),  # Rough conversion
            "trajectory_type": "Mid" if bio_metrics.get("arm_extension", 0.6) > 0.7 else "Low"
        }
    }
    
    return swing_data


def create_llm_prompt(analysis_data):
    """
    Create a comprehensive prompt for LLM analysis with professional benchmarks
    
    Args:
        analysis_data (dict): Processed swing analysis data with biomechanical metrics
        
    Returns:
        str: Formatted prompt for LLM analysis
    """
    
    # Extract metrics from the new data structure
    bio_metrics = analysis_data.get("biomechanical_metrics", {})
    timing_metrics = analysis_data.get("timing_metrics", {})
    swing_phases = analysis_data.get("swing_phases", {})
    
    prompt = """# Golf Swing Analysis

## PROFESSIONAL BENCHMARKS FOR CALIBRATION
Use these professional standards as your 100% reference for scoring. These represent elite-level golf swing mechanics based on actual LPGA Tour professional analysis:

### Professional Golfer Analysis Summary (100% Reference Standards):

**Atthaya Thitikul (LPGA Tour - Elite Level):**
- Hip Rotation: 63.4°, Shoulder Rotation: 120°, Posture Score: 98.2%
- Weight Shift: 88.4%, Arm Extension: 99.8%, Wrist Hinge: 120°
- Energy Transfer: 96.1%, Power Accumulation: 100%, Potential Distance: 295 yards
- Sequential Kinematic Sequence: 100%, Swing Plane Consistency: 85%

**Nelly Korda (LPGA Tour - Elite Level):**
- Hip Rotation: 90°, Shoulder Rotation: 120°, Posture Score: 97.4%
- Weight Shift: 73.5%, Arm Extension: 96.7%, Wrist Hinge: 114.8°
- Energy Transfer: 91.2%, Power Accumulation: 100%, Potential Distance: 289 yards
- Sequential Kinematic Sequence: 100%, Swing Plane Consistency: 85%

**Demi Runas (Professional Level):**
- Hip Rotation: 63.4°, Shoulder Rotation: 120°, Posture Score: 95.9%
- Weight Shift: 63.9%, Arm Extension: 96.6%, Wrist Hinge: 93.4°
- Energy Transfer: 88.0%, Power Accumulation: 100%, Potential Distance: 286 yards
- Sequential Kinematic Sequence: 100%, Swing Plane Consistency: 85%

### **PROFESSIONAL STANDARDS CALIBRATION (100% Level):**
**Core Biomechanical Metrics:**
- **Hip Rotation**: 60-90° (Exceptional body turn and flexibility)
- **Shoulder Rotation**: 120° (Full shoulder coil for maximum power)
- **Posture Score**: 95-98% (Exceptional spine angle consistency)
- **Weight Shift**: 70-88% (Excellent weight transfer to lead side)

**Upper Body Excellence:**
- **Arm Extension**: 96-100% (Near-perfect extension at impact)
- **Wrist Hinge**: 95-120° (Optimal lag and release timing)
- **Swing Plane Consistency**: 85% (Tour-level repeatability)
- **Chest Rotation Efficiency**: 100% (Perfect coordination)

**Power & Efficiency Markers:**
- **Energy Transfer Efficiency**: 88-96% (Elite power transfer)
- **Power Accumulation**: 100% (Maximum power generation)
- **Sequential Kinematic Sequence**: 100% (Perfect body sequencing)
- **Potential Distance**: 285-295 yards (Tour-level power)

**Movement Quality Standards:**
- **Head Movement**: 2-8 inches (Controlled, minimal excessive movement)
- **Ground Force Efficiency**: 70-88% (Excellent ground interaction)
- **Hip Thrust**: 40-100% (Strong lower body drive)

### **AMATEUR REFERENCE EXAMPLES FOR CALIBRATION:**

**70% Level Skilled Amateur (Female):**
- Hip Rotation: 23.0°, Shoulder Rotation: 120° (Excellent shoulder turn, limited hip mobility)
- Posture Score: 89.5%, Weight Shift: 90.0% (Strong fundamentals)
- Arm Extension: 99.8%, Wrist Hinge: 49.4° (Great extension, needs more lag)
- Energy Transfer: 94.5%, Power Accumulation: 82.1% (Very good coordination)
- Potential Distance: 273 yards, Sequential Kinematic: 93.6%
- Head Movement: 8.0in lateral, 6.0in vertical (Excessive movement)
- Speed Generation: Mixed

**50-60% Level Amateur (Male #1 - Body-Dominant):**
- Hip Rotation: 90°, Shoulder Rotation: 84.8° (Great hip turn, limited shoulder)
- Posture Score: 90.7%, Weight Shift: 90.0% (Solid fundamentals)
- Arm Extension: 100.0%, Wrist Hinge: 66.8° (Good extension and lag)
- Energy Transfer: 91.8%, Power Accumulation: 100.0% (Strong power generation)
- Potential Distance: 290 yards, Sequential Kinematic: 100.0%
- Hip Thrust: 100.0%, Ground Force: 90.0% (Excellent lower body)
- Speed Generation: Body-dominant

**50-60% Level Amateur (Male #2 - Body-Dominant):**
- Hip Rotation: 90°, Shoulder Rotation: 120° (Excellent rotation both)
- Posture Score: 89.3%, Weight Shift: 90.0% (Good fundamentals)
- Arm Extension: 99.6%, Wrist Hinge: 52.6° (Great extension, limited lag)
- Energy Transfer: 96.7%, Power Accumulation: 100.0% (Excellent coordination)
- Potential Distance: 296 yards, Sequential Kinematic: 100.0%
- Tempo Issues: Very fast downswing (2.86 ratio vs ideal ~0.3)
- Speed Generation: Body-dominant

**50-60% Level Amateur (Female - Arms-Dominant):**
- Hip Rotation: 25°, Shoulder Rotation: 60° (Limited body rotation)
- Posture Score: 80.6%, Weight Shift: 50.0% (Needs improvement)
- Arm Extension: 94.8%, Wrist Hinge: 116.6° (Good extension, excellent lag)
- Energy Transfer: 56.8%, Power Accumulation: 89.3% (Mixed efficiency)
- Potential Distance: 241 yards, Sequential Kinematic: 66.8%
- Head Movement: 3.0in lateral, 2.0in vertical (Good head control)
- Ground Force: 50.0%, Hip Thrust: 30.0% (Weak lower body)
- Speed Generation: Arms-dominant

**CRITICAL INSIGHTS FROM AMATEUR ANALYSIS:**
1. **Hip Rotation Varies Significantly**: From 23-90° in amateurs vs 60-90° in professionals
2. **Shoulder Rotation Range**: 60-120° in amateurs, professionals consistently at 120°
3. **Wrist Hinge Compensation**: Some amateurs (116.6°) exceed professional standards to compensate for limited body rotation
4. **Power Generation Methods**: Body-dominant amateurs can achieve near-professional distances despite technical limitations
5. **Head Movement Control**: Varies dramatically (3-8 inches) - major differentiator
6. **Energy Transfer Efficiency**: Ranges from 56.8-96.7% in amateurs vs 88-96% in professionals
7. **Weight Transfer Issues**: Some amateurs struggle with weight shift (50% vs professional 70-88%)

## CURRENT SWING ANALYSIS

### Swing Phase Breakdown
""".format(
        swing_phases.get("setup", {}).get("frame_count", 44),
        swing_phases.get("backswing", {}).get("frame_count", 7), 
        swing_phases.get("downswing", {}).get("frame_count", 12),
        swing_phases.get("impact", {}).get("frame_count", 1),
        swing_phases.get("follow_through", {}).get("frame_count", 37),
        timing_metrics.get("tempo_ratio", 0.6)
    )

    # Add swing phase details
    for phase_name, phase_data in swing_phases.items():
        prompt += f"- {phase_name.title()}: {phase_data.get('frame_count', 0)} frames ({phase_data.get('duration_ms', 0):.0f}ms)\n"
    
    prompt += f"- Total Swing: {timing_metrics.get('total_swing_frames', 0)} frames ({timing_metrics.get('total_swing_time_ms', 0):.0f}ms)\n"
    prompt += f"- Tempo Ratio (down:back): {timing_metrics.get('tempo_ratio', 1.0)}\n"
    prompt += f"- Estimated Club Speed: {timing_metrics.get('estimated_club_speed_mph', 85)} mph\n"

    # Core body mechanics
    prompt += "\n### Core Body Mechanics\n"
    prompt += f"- Hip Rotation: {bio_metrics.get('hip_rotation_degrees', 25)}°\n"
    prompt += f"- Shoulder Rotation: {bio_metrics.get('shoulder_rotation_degrees', 60)}°\n"
    prompt += f"- Posture Score: {bio_metrics.get('posture_score_percent', 60)}%\n"
    prompt += f"- Weight Shift (lead foot at impact): {bio_metrics.get('weight_shift_percent', 50)}%\n"

    # Upper body mechanics
    prompt += "\n### Upper Body Mechanics\n"
    prompt += f"- Arm Extension: {bio_metrics.get('arm_extension_percent', 60)}%\n"
    prompt += f"- Wrist Hinge: {bio_metrics.get('wrist_hinge_degrees', 60)}°\n"
    prompt += f"- Shoulder Plane Consistency: {bio_metrics.get('swing_plane_consistency_percent', 60)}%\n"
    prompt += f"- Chest Rotation Efficiency: {bio_metrics.get('chest_rotation_efficiency_percent', 60)}%\n"
    prompt += f"- Head Movement (lateral): {bio_metrics.get('head_movement_lateral_inches', 3.0)}in\n"
    prompt += f"- Head Movement (vertical): {bio_metrics.get('head_movement_vertical_inches', 2.0)}in\n"

    # Lower body mechanics
    prompt += "\n### Lower Body Mechanics\n"
    prompt += f"- Knee Flexion (address): {bio_metrics.get('knee_flexion_address_degrees', 25)}°\n"
    prompt += f"- Knee Flexion (impact): {bio_metrics.get('knee_flexion_impact_degrees', 30)}°\n"
    prompt += f"- Hip Thrust (impact): {bio_metrics.get('hip_thrust_percent', 50)}%\n"
    prompt += f"- Ground Force Efficiency: {bio_metrics.get('ground_force_efficiency_percent', 60)}%\n"

    # Advanced coordination metrics
    prompt += "\n### Movement Quality & Timing\n"
    prompt += f"- Transition Smoothness: {bio_metrics.get('transition_smoothness_percent', 60)}%\n"
    prompt += f"- Sequential Kinematic Sequence: {bio_metrics.get('kinematic_sequence_percent', 60)}%\n"

    # Efficiency and power metrics
    prompt += "\n### Efficiency & Power Metrics\n"
    prompt += f"- Energy Transfer Efficiency: {bio_metrics.get('energy_transfer_efficiency_percent', 60)}%\n"
    prompt += f"- Power Accumulation: {bio_metrics.get('power_accumulation_percent', 60)}%\n"
    prompt += f"- Potential Distance: {bio_metrics.get('potential_distance_yards', 200)} yards\n"
    prompt += f"- Speed Generation Method: {bio_metrics.get('speed_generation_method', 'Mixed')}\n"

    prompt += """

## ANALYSIS INSTRUCTIONS

Using the professional benchmarks above as your calibration reference, provide your analysis in the following EXACT structured format:

**PERFORMANCE_CLASSIFICATION:** [XX%] (where XX is a percentage from 10% to 100%)

**STRENGTHS:**
• [Specific strength with direct comparison to professional benchmarks - e.g. "Your shoulder rotation shows great upper body mobility during your backswing, similar to what we see in professional swings"]
• [Another strength with benchmark comparison - e.g. "Your arm extension at impact is really strong, demonstrating excellent fundamentals"]
• [Third strength with specific metric comparison to benchmarks - e.g. "Your weight transfer demonstrates solid fundamentals in shifting from back foot to front foot through impact"]

**WEAKNESSES:**
• [Specific area describing impact without numbers - e.g. "Your hip rotation is less than optimal, which may be limiting your power generation and overall swing efficiency"]
• [Another area with impact description - e.g. "Your head movement during the swing is more than ideal, which could be affecting your accuracy and consistency"]
• [Third area with impact-focused description - e.g. "Your wrist action could use some work, which may be affecting your ability to generate lag and create powerful impact"]

**PRIORITY_IMPROVEMENTS:**
1. Topic Name - Explain what to focus on and when in swing, with encouraging tone and clear benefit explanation
2. Topic Name - What to work on with supportive guidance and positive reinforcement about potential improvements  
3. Topic Name - Area to focus on with gentle direction and realistic goals

**MANDATORY REQUIREMENTS FOR EACH SECTION:**

**For STRENGTHS** - Must include:
- EXACTLY 3 bullet points - no more, no less
- Use encouraging, positive language 
- NO numbers or statistics - focus on qualitative descriptions
- Reference professional standards without mentioning specific metrics
- Recognition when mechanics are working well
- Explain timing in swing when relevant (during backswing, at impact, etc.)
- Use supportive tone appropriate for young golfers

**For WEAKNESSES** - Must include:
- EXACTLY 3 bullet points - no more, no less
- NO numbers, degrees, or specific measurements
- Focus on the IMPACT of the issue (what it's affecting) rather than the measurement
- Use phrases like "less than optimal" or "more than ideal" instead of specific amounts
- Explain HOW the weakness affects performance (power, accuracy, consistency, etc.)
- DO NOT provide improvement suggestions - save those for the Priority Improvements section
- Frame as areas that may be affecting performance rather than deficiencies

**For PRIORITY_IMPROVEMENTS** - Must include:
- EXACTLY 3 numbered items - no more, no less
- NO header formatting like [Most Critical], [Important], [Focus Area] in the descriptions
- Use encouraging language like "try increasing" or "focus on" instead of "you need to"
- When in the swing this should happen (during downswing, backswing, at impact, etc.)
- Reference professional standards gently without excessive numerical comparisons
- Clear explanation of benefits and positive outcomes
- Maintain supportive, coaching tone throughout

**EXAMPLE ANALYSIS STRUCTURE:**

**STRENGTHS:**
• Your shoulder rotation shows great upper body mobility during your backswing, matching what we see in professional swings
• Your weight transfer demonstrates excellent fundamentals in shifting from your back foot to your front foot through impact
• Your posture maintains good stability throughout most of your swing, showing solid foundational mechanics

**WEAKNESSES:**  
• Your hip rotation is less than optimal, which may be limiting your power generation and overall swing efficiency
• Your head movement during the swing is more than ideal, which could be affecting your accuracy and consistency throughout your shots
• Your wrist action could use some work, which may be affecting your ability to generate lag and create powerful impact

**PRIORITY_IMPROVEMENTS:**
1. Hip Mobility Development - Try increasing your hip rotation during the downswing. This will help you engage your lower body more effectively and unlock substantial power gains in your swing.
2. Head Stability Enhancement - Focus on keeping your head more stable throughout your swing. This improvement will help enhance your accuracy and consistency on every shot.
3. Wrist Hinge Optimization - Work on creating more wrist angle during your backswing to improve lag and power transfer. This will help add distance and control to your shots.

PERFORMANCE CLASSIFICATION SCALE:
- **90-100%**: Professional/Tour level - Consistently meets or exceeds professional benchmarks across all metrics
- **80-89%**: Advanced amateur - Meets most professional standards with minor gaps in 1-2 areas
- **70-79%**: Skilled amateur - Solid fundamentals with some gaps from professional standards
- **60-69%**: Intermediate - Good basic mechanics but several areas need improvement to reach professional level
- **50-59%**: Developing intermediate - Basic swing structure present but multiple areas below professional standards
- **40-49%**: Advanced beginner - Some fundamentals in place but significant gaps in most areas
- **30-39%**: Beginner - Basic swing motion present but major improvements needed across most metrics
- **20-29%**: Novice - Limited swing fundamentals, extensive work needed on basic mechanics
- **10-19%**: Complete beginner - Minimal swing structure, needs comprehensive fundamental development

IMPORTANT ANALYSIS PRIORITIES (Based on Real Professional Data):
1. **PRIMARY FOCUS - Critical Biomechanical Differentiators**: 
   - Hip Rotation (Professional: 60-90°, Amateur Range: 23-90°) - MOST IMPORTANT
   - Shoulder Rotation (Professional: 120°, Amateur Range: 60-120°) - MOST IMPORTANT  
   - Sequential Kinematic Sequence (Professional: 100%, Amateur Range: 66.8-100%)
   - Energy Transfer Efficiency (Professional: 88-96%, Amateur Range: 56.8-96.7%)

2. **SECONDARY FOCUS - Power Generation Mechanics**: 
   - Power Accumulation (Professional: 100%, Amateur Range: 82.1-100%)
   - Chest Rotation Efficiency (Professional: 100%, Amateur Range: 53.7-100%)
   - Wrist Hinge (Professional: 95-120°, Amateur Range: 49.4-116.6°)
   - Swing Plane Consistency (Professional: 85%, Amateur: 70-85%)

3. **TERTIARY FOCUS - Refinement Metrics**: 
   - Posture Score (Professional: 95-98%, Amateur Range: 80.6-90.7%)
   - Arm Extension (Professional: 96-100%, Amateur Range: 94.8-100%)
   - Weight Shift (Professional: 70-88%, Amateur Range: 50-90%)
   - Ground Force Efficiency (Professional: 70-88%, Amateur Range: 50-90%)

4. **DE-EMPHASIZE - Timing Variables**: Frame counts, tempo ratios, and duration metrics vary significantly based on video capture rates and personal style preferences

**SCORING CALIBRATION GUIDELINES:**
- **Hip/Shoulder Rotation Analysis**: Compare to professional minimums (60° hip, 120° shoulder)
- **Energy Transfer <70%**: Score below 60%, reference professional range (88-96%)
- **Sequential Kinematic <80%**: Score below 70%, reference professional standards (100%)
- **Power Accumulation <90%**: Score below 80%, compare to professional benchmarks (100%)
- **Head Movement >10 inches**: Major limitation, compare to professional standards (2-8in)
- **Weight Shift <60%**: Significant weakness, reference professional range (70-88%)

IMPORTANT FORMATTING RULES:
- Use the exact headers shown above (PERFORMANCE_CLASSIFICATION, STRENGTHS, WEAKNESSES, PRIORITY_IMPROVEMENTS)
- For performance classification, use format: [XX%] where XX is the percentage (10-100)
- For strengths and weaknesses, use bullet points (•) 
- For priority improvements, use numbered format (1., 2., 3.) with priority level in brackets
- Each priority improvement must have: [Priority Level] Topic Name - Full description with professional benchmark comparisons
- **MANDATORY**: Include specific metric values and professional benchmark comparisons in every strength, weakness, and improvement
- **MANDATORY**: Reference professional standards in analysis content
- Provide clear directional guidance (more/less rotation, when in swing) rather than overly technical numerical comparisons
- Focus analysis on biomechanical consistency rather than timing variations
- **CRITICAL**: Every analysis point must tie back to the professional benchmarks provided
- Avoid absolute language like "perfect" or "flawless" - use terms like "very good" or "meets standards"

Remember: Use the professional benchmarks (Atthaya Thitikul: 63.4° hip, 120° shoulder, 96.1% energy transfer, etc.) as the foundation for ALL analysis content, not just the percentage classification. Every strength, weakness, and improvement recommendation must include specific comparisons to professional standards with clear, actionable guidance on what needs to improve and when in the swing.
"""

    return prompt


def parse_and_format_analysis(raw_analysis):
    """
    Parse the raw LLM analysis and format it into structured components
    
    Args:
        raw_analysis (str): Raw analysis text from LLM
        
    Returns:
        dict: Structured analysis with classification, strengths/weaknesses, and priorities
    """
    # Default structure
    formatted_analysis = {
        'classification': 50,  # Default to 50%
        'strengths': [],
        'weaknesses': [],
        'priority_improvements': []
    }
    
    # Extract percentage classification using the new structured format
    classification_match = re.search(r'\*\*PERFORMANCE_CLASSIFICATION:\*\*\s*\[?(\d+)%?\]?', raw_analysis, re.IGNORECASE)
    if classification_match:
        percentage = int(classification_match.group(1))
        # Ensure percentage is within valid range
        formatted_analysis['classification'] = max(10, min(100, percentage))
    else:
        # Fallback to look for standalone percentages
        percentage_patterns = [
            r'(?:Performance|Classification|Level|Score).*?(\d+)%',
            r'(\d+)%.*?(?:level|performance|classification)',
            r'classified.*?(\d+)%',
            r'(?:at|as)\s+(\d+)%'
        ]
        
        for pattern in percentage_patterns:
            match = re.search(pattern, raw_analysis, re.IGNORECASE)
            if match:
                percentage = int(match.group(1))
                formatted_analysis['classification'] = max(10, min(100, percentage))
                break
    
    # Extract strengths using the new structured format
    strengths_match = re.search(r'\*\*STRENGTHS:\*\*\s*(.*?)(?=\*\*WEAKNESSES:\*\*|\*\*PRIORITY_IMPROVEMENTS:\*\*|$)', raw_analysis, re.IGNORECASE | re.DOTALL)
    if strengths_match:
        strengths_text = strengths_match.group(1)
        # Extract bullet points
        strength_items = re.findall(r'•\s*([^\n•]+)', strengths_text)
        formatted_analysis['strengths'] = [item.strip() for item in strength_items if item.strip()]
    
    # Extract weaknesses using the new structured format
    weaknesses_match = re.search(r'\*\*WEAKNESSES:\*\*\s*(.*?)(?=\*\*PRIORITY_IMPROVEMENTS:\*\*|$)', raw_analysis, re.IGNORECASE | re.DOTALL)
    if weaknesses_match:
        weaknesses_text = weaknesses_match.group(1)
        # Extract bullet points
        weakness_items = re.findall(r'•\s*([^\n•]+)', weaknesses_text)
        formatted_analysis['weaknesses'] = [item.strip() for item in weakness_items if item.strip()]
    
    # Extract priority improvements using the new structured format
    priority_match = re.search(r'\*\*PRIORITY_IMPROVEMENTS:\*\*\s*(.*?)$', raw_analysis, re.IGNORECASE | re.DOTALL)
    if priority_match:
        priority_text = priority_match.group(1)
        # Extract numbered items with priority levels and descriptions
        priority_items = re.findall(r'(\d+)\.\s*\[(.*?)\]\s*(.*?)(?=\d+\.\s*\[|\Z)', priority_text, re.DOTALL)
        for num, priority_level, description in priority_items[:3]:  # Limit to 3
            # Clean up the description
            description = description.strip()
            # Remove any trailing incomplete sentences
            if description.endswith('...') or len(description.split('.')[-1].strip()) < 5:
                sentences = description.split('.')
                if len(sentences) > 1:
                    description = '.'.join(sentences[:-1]) + '.'
            
            formatted_analysis['priority_improvements'].append({
                'rank': int(num),
                'priority_level': priority_level.strip(),
                'description': f"[{priority_level.strip()}] {description}"
            })
    
    # Fallback parsing if structured format wasn't used
    if not formatted_analysis['strengths']:
        # Try original parsing methods for strengths
        strengths_patterns = [
            r'(?:Strengths|Strong Points|Positives|Meets.*Standards)[\s\S]*?(?=(?:Weak|Priority|Improvement|Areas|$))',
            r'(?:Professional Level|Exceeds.*Standards)[\s\S]*?(?=(?:Below|Weak|Priority|$))'
        ]
        
        for pattern in strengths_patterns:
            match = re.search(pattern, raw_analysis, re.IGNORECASE)
            if match:
                strengths_section = match.group(0)
                strength_items = re.findall(r'[-•]\s*([^-•\n]+)', strengths_section)
                formatted_analysis['strengths'] = [item.strip() for item in strength_items[:4]]
                break
    
    if not formatted_analysis['weaknesses']:
        # Try original parsing methods for weaknesses
        weaknesses_patterns = [
            r'(?:Weaknesses|Weak|Areas.*Improvement|Priority.*Areas|Below.*Standards)[\s\S]*?(?=(?:Recommendation|Priority|$))',
            r'(?:Critical|Important|Significant.*gaps?)[\s\S]*?(?=(?:Recommendation|$))'
        ]
        
        for pattern in weaknesses_patterns:
            match = re.search(pattern, raw_analysis, re.IGNORECASE)
            if match:
                weaknesses_section = match.group(0)
                weakness_items = re.findall(r'[-•]\s*([^-•\n]+)', weaknesses_section)
                formatted_analysis['weaknesses'] = [item.strip() for item in weakness_items[:4]]
                break
    
    if not formatted_analysis['priority_improvements']:
        # Try original parsing methods for priorities
        priority_patterns = [
            r'(?:Priority.*Improvement|Critical.*Areas?)[\s\S]*?(?=(?:Recommendation|$))',
            r'(?:1\..*?2\..*?3\.)',  # Numbered list
            r'(?:Critical|Important|Fine-tuning)[\s\S]*?(?=(?:Critical|Important|Fine-tuning|$))'
        ]
        
        for pattern in priority_patterns:
            match = re.search(pattern, raw_analysis, re.IGNORECASE | re.DOTALL)
            if match:
                priority_text = match.group(0)
                # Extract numbered items with better parsing
                numbered_items = re.findall(r'(\d+)\.\s*([^1-9]+?)(?=\d+\.|$)', priority_text, re.DOTALL)
                for num, item in numbered_items[:3]:  # Limit to 3
                    # Clean up the item text
                    item = item.strip()
                    # Remove any trailing incomplete sentences
                    sentences = item.split('.')
                    if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
                        item = '.'.join(sentences[:-1]) + '.'
                    
                    formatted_analysis['priority_improvements'].append({
                        'rank': int(num),
                        'description': item
                    })
                break
    
    # If still no content found, use defaults based on classification
    if not formatted_analysis['strengths']:
        formatted_analysis['strengths'] = ['Swing analysis completed successfully']
    
    if not formatted_analysis['weaknesses']:
        formatted_analysis['weaknesses'] = ['Areas for improvement identified']
    
    if not formatted_analysis['priority_improvements']:
        percentage = formatted_analysis['classification']
        if percentage >= 80:
            formatted_analysis['priority_improvements'] = [
                {'rank': 1, 'description': '[Most Critical] Technical Refinement - Fine-tune specific mechanics to achieve consistency at the highest level.'},
                {'rank': 2, 'description': '[Important] Performance Optimization - Focus on maximizing efficiency and power transfer.'},
                {'rank': 3, 'description': '[Focus Area] Competitive Preparation - Enhance mental game and course management skills.'}
            ]
        elif percentage >= 60:
            formatted_analysis['priority_improvements'] = [
                {'rank': 1, 'description': '[Most Critical] Kinematic Sequence Enhancement - Improve body rotation coordination to generate more power and consistency.'},
                {'rank': 2, 'description': '[Important] Clubface Control - Enhance swing path consistency for better ball striking accuracy.'},
                {'rank': 3, 'description': '[Focus Area] Energy Transfer Efficiency - Optimize power transfer throughout the swing to maximize distance.'}
            ]
        elif percentage >= 40:
            formatted_analysis['priority_improvements'] = [
                {'rank': 1, 'description': '[Most Critical] Fundamental Mechanics - Establish consistent posture, grip, and setup positions.'},
                {'rank': 2, 'description': '[Important] Body Rotation Development - Improve hip and shoulder turn coordination.'},
                {'rank': 3, 'description': '[Focus Area] Weight Transfer - Develop proper weight shift from back foot to front foot during swing.'}
            ]
        else:  # Below 40%
            formatted_analysis['priority_improvements'] = [
                {'rank': 1, 'description': '[Most Critical] Basic Setup and Posture - Focus on establishing proper spine angle and athletic stance.'},
                {'rank': 2, 'description': '[Important] Fundamental Swing Motion - Develop basic backswing and downswing mechanics.'},
                {'rank': 3, 'description': '[Focus Area] Balance and Stability - Improve overall balance throughout the swing motion.'}
            ]
    
    return formatted_analysis


def display_formatted_analysis(analysis_data):
    """
    Display the formatted analysis with performance classification, strengths/weaknesses table, and priorities
    
    Args:
        analysis_data (dict): Structured analysis data from parse_and_format_analysis
    """
    # 1. Performance Classification with percentage-based progress bar
    user_percentage = analysis_data['classification']
    
    # Display classification in black bolded header
    st.markdown(f"""
    <h2 style='color: black; font-weight: bold; text-align: center; margin-bottom: 20px;'>
        🎯 Performance Score: {user_percentage}%
    </h2>
    """, unsafe_allow_html=True)
    
    # Create a visual progress bar
    progress_color = "#ff4444"  # Red for low scores
    if user_percentage >= 80:
        progress_color = "#44aa44"  # Green for high scores
    elif user_percentage >= 60:
        progress_color = "#ffdd00"  # Yellow for good scores
    elif user_percentage >= 40:
        progress_color = "#ff8800"  # Orange for medium scores
    
    # Progress bar with percentage labels
    st.markdown(f"""
    <div style='margin: 20px 0;'>
        <div style='display: flex; justify-content: space-between; font-size: 12px; color: #666; margin-bottom: 5px;'>
            <span>10% - Complete Beginner</span>
            <span>50% - Intermediate</span>
            <span>100% - Professional</span>
        </div>
        <div style='width: 100%; background-color: #f0f0f0; border-radius: 25px; height: 30px; position: relative;'>
            <div style='width: {user_percentage}%; background-color: {progress_color}; height: 30px; border-radius: 25px; 
                        display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;'>
                {user_percentage}%
            </div>
        </div>
        <div style='display: flex; justify-content: space-between; font-size: 10px; color: #888; margin-top: 5px;'>
            <span>10%</span>
            <span>20%</span>
            <span>30%</span>
            <span>40%</span>
            <span>50%</span>
            <span>60%</span>
            <span>70%</span>
            <span>80%</span>
            <span>90%</span>
            <span>100%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Performance level description based on percentage
    if user_percentage >= 90:
        level_desc = "🏆 **Professional/Tour Level** - Consistently meets or exceeds professional benchmarks"
        level_color = "#44aa44"
    elif user_percentage >= 80:
        level_desc = "🥇 **Advanced Amateur** - Meets most professional standards with minor gaps"
        level_color = "#66bb44"
    elif user_percentage >= 70:
        level_desc = "🥈 **Skilled Amateur** - Solid fundamentals with some gaps from professional standards"
        level_color = "#88cc44"
    elif user_percentage >= 60:
        level_desc = "🥉 **Intermediate** - Good basic mechanics but several areas need improvement"
        level_color = "#ffdd00"
    elif user_percentage >= 50:
        level_desc = "📈 **Developing Intermediate** - Basic swing structure present"
        level_color = "#ffcc00"
    elif user_percentage >= 40:
        level_desc = "📚 **Advanced Beginner** - Some fundamentals in place"
        level_color = "#ff8800"
    elif user_percentage >= 30:
        level_desc = "🎯 **Beginner** - Basic swing motion present but major improvements needed"
        level_color = "#ff6600"
    elif user_percentage >= 20:
        level_desc = "🌱 **Novice** - Limited swing fundamentals, extensive work needed"
        level_color = "#ff4444"
    else:
        level_desc = "🚀 **Complete Beginner** - Minimal swing structure, needs comprehensive fundamental development"
        level_color = "#ff2222"
    
    st.markdown(f"""
    <div style='text-align: center; padding: 15px; background-color: {level_color}20; 
                border-radius: 10px; margin: 20px 0; border: 2px solid {level_color};'>
        <div style='color: {level_color}; font-size: 16px; font-weight: bold;'>{level_desc}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 2. Strengths and Weaknesses Table
    st.subheader("⚖️ Strengths & Areas for Improvement")
    
    # Create two columns for the table with a visual divider
    col_left, col_divider, col_right = st.columns([5, 1, 5])
    
    with col_left:
        st.markdown("""
        <div style='background-color: #e8f5e8; padding: 15px; border-radius: 10px; height: 100%;'>
            <h4 style='color: #2d5a2d; margin-top: 0;'>✅ Strengths</h4>
        """, unsafe_allow_html=True)
        for strength in analysis_data['strengths']:
            st.markdown(f"• {strength}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col_divider:
        st.markdown("""
        <div style='width: 2px; background-color: #ddd; height: 200px; margin: 20px auto;'></div>
        """, unsafe_allow_html=True)
    
    with col_right:
        st.markdown("""
        <div style='background-color: #fff5e6; padding: 15px; border-radius: 10px; height: 100%;'>
            <h4 style='color: #cc6600; margin-top: 0;'>⚠️ Areas for Improvement</h4>
        """, unsafe_allow_html=True)
        for weakness in analysis_data['weaknesses']:
            st.markdown(f"• {weakness}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 3. Priority Improvement Areas
    st.subheader("🎯 Priority Improvement Areas")
    
    for priority in sorted(analysis_data['priority_improvements'], key=lambda x: x['rank']):
        rank = priority['rank']
        description = priority['description']
        
        # Better extraction of improvement area and description
        area = ""
        desc = description
        
        # Try different patterns to extract the main topic
        if '[Most Critical]' in description or '[Important]' in description or '[Focus Area]' in description:
            # Pattern: [Priority Level] Topic - Description
            pattern = r'\[(.*?)\]\s*(.*?)(?:\s*-\s*(.*))?$'
            match = re.search(pattern, description)
            if match:
                priority_level = match.group(1)
                area = match.group(2).strip()
                desc = match.group(3).strip() if match.group(3) else ""
        elif ':' in description:
            # Pattern: Topic: Description
            parts = description.split(':', 1)
            area = parts[0].strip()
            desc = parts[1].strip()
        elif ' - ' in description:
            # Pattern: Topic - Description
            parts = description.split(' - ', 1)
            area = parts[0].strip()
            desc = parts[1].strip()
        else:
            # Try to extract first meaningful phrase as area
            words = description.split()
            if len(words) > 5:
                # Take first 3-5 words as the area
                area = ' '.join(words[:4])
                desc = ' '.join(words[4:])
            else:
                area = description
                desc = ""
        
        # Clean up area and description
        area = area.replace('[Most Critical]', '').replace('[Important]', '').replace('[Focus Area]', '').strip()
        
        # Ensure we have meaningful content
        if not area or len(area) < 5:
            area = f"Priority {rank} Improvement"
        
        if not desc or len(desc) < 10:
            # Provide a more complete description based on the area
            if 'posture' in area.lower():
                desc = "Try working on maintaining proper spine angle and athletic stance throughout the swing for better consistency and power transfer."
            elif 'tempo' in area.lower() or 'timing' in area.lower():
                desc = "Focus on developing a smooth, consistent rhythm that allows for proper sequencing of body movements."
            elif 'rotation' in area.lower():
                desc = "Try improving the coordination and range of motion in your body turn to generate more power and accuracy."
            elif 'weight' in area.lower() or 'shift' in area.lower():
                desc = "Practice transferring weight from back foot to front foot during the swing for better balance and power."
            elif 'knee' in area.lower():
                desc = "Work on maintaining proper knee flex and stability throughout the swing for better foundation and consistency."
            elif 'hip' in area.lower():
                desc = "Focus on improving hip mobility and rotation during the downswing to enhance power generation and sequencing."
            elif 'chest' in area.lower():
                desc = "Try improving chest rotation efficiency to better coordinate upper body movement with the swing sequence."
            else:
                desc = description  # Use the full description if we can't categorize it
        
        # Display using simple bullet points instead of colored boxes
        st.markdown(f"**{rank}. {area}:** {desc}")
        
        st.write("")  # Add spacing between items


def calculate_biomechanical_metrics(pose_data, swing_phases):
    """
    Calculate biomechanical metrics from pose keypoints data
    
    Args:
        pose_data (dict): Dictionary mapping frame indices to pose keypoints
        swing_phases (dict): Dictionary mapping phase names to lists of frame indices
        
    Returns:
        dict: Calculated biomechanical metrics
    """
    metrics = {}
    
    # Get key frames for analysis
    setup_frames = swing_phases.get("setup", [])
    backswing_frames = swing_phases.get("backswing", [])
    downswing_frames = swing_phases.get("downswing", [])
    impact_frames = swing_phases.get("impact", [])
    
    # Get representative frames
    setup_frame = setup_frames[len(setup_frames) // 2] if setup_frames else None
    top_backswing_frame = backswing_frames[-1] if backswing_frames else None
    impact_frame = impact_frames[0] if impact_frames else None
    
    # MediaPipe Pose landmark indices
    # Shoulders: left(11), right(12)
    # Hips: left(23), right(24)
    # Knees: left(25), right(26)
    # Ankles: left(27), right(28)
    # Elbows: left(13), right(14)
    # Wrists: left(15), right(16)
    
    try:
        # Calculate Hip Rotation
        if setup_frame and top_backswing_frame and setup_frame in pose_data and top_backswing_frame in pose_data:
            setup_keypoints = pose_data[setup_frame]
            backswing_keypoints = pose_data[top_backswing_frame]
            
            if len(setup_keypoints) >= 33 and len(backswing_keypoints) >= 33:
                # Hip rotation calculation using hip landmarks
                setup_left_hip = np.array(setup_keypoints[23][:2])
                setup_right_hip = np.array(setup_keypoints[24][:2])
                backswing_left_hip = np.array(backswing_keypoints[23][:2])
                backswing_right_hip = np.array(backswing_keypoints[24][:2])
                
                # Calculate hip line angles
                setup_hip_vector = setup_right_hip - setup_left_hip
                backswing_hip_vector = backswing_right_hip - backswing_left_hip
                
                setup_hip_angle = np.degrees(np.arctan2(setup_hip_vector[1], setup_hip_vector[0]))
                backswing_hip_angle = np.degrees(np.arctan2(backswing_hip_vector[1], backswing_hip_vector[0]))
                
                hip_rotation = abs(backswing_hip_angle - setup_hip_angle)
                # Normalize to reasonable range (professionals typically achieve 45+ degrees)
                metrics["hip_rotation"] = min(hip_rotation, 90)
            else:
                metrics["hip_rotation"] = 25  # Lower default for incomplete data
        else:
            metrics["hip_rotation"] = 25
            
        # Calculate Shoulder Rotation
        if setup_frame and top_backswing_frame and setup_frame in pose_data and top_backswing_frame in pose_data:
            setup_keypoints = pose_data[setup_frame]
            backswing_keypoints = pose_data[top_backswing_frame]
            
            if len(setup_keypoints) >= 33 and len(backswing_keypoints) >= 33:
                # Shoulder rotation calculation
                setup_left_shoulder = np.array(setup_keypoints[11][:2])
                setup_right_shoulder = np.array(setup_keypoints[12][:2])
                backswing_left_shoulder = np.array(backswing_keypoints[11][:2])
                backswing_right_shoulder = np.array(backswing_keypoints[12][:2])
                
                setup_shoulder_vector = setup_right_shoulder - setup_left_shoulder
                backswing_shoulder_vector = backswing_right_shoulder - backswing_left_shoulder
                
                setup_shoulder_angle = np.degrees(np.arctan2(setup_shoulder_vector[1], setup_shoulder_vector[0]))
                backswing_shoulder_angle = np.degrees(np.arctan2(backswing_shoulder_vector[1], backswing_shoulder_vector[0]))
                
                shoulder_rotation = abs(backswing_shoulder_angle - setup_shoulder_angle)
                metrics["shoulder_rotation"] = min(shoulder_rotation, 120)
            else:
                metrics["shoulder_rotation"] = 60  # Lower default
        else:
            metrics["shoulder_rotation"] = 60
            
        # Calculate Weight Shift (using hip and ankle positions)
        if setup_frame and impact_frame and setup_frame in pose_data and impact_frame in pose_data:
            setup_keypoints = pose_data[setup_frame]
            impact_keypoints = pose_data[impact_frame]
            
            if len(setup_keypoints) >= 33 and len(impact_keypoints) >= 33:
                # Use center of mass approximation
                setup_left_ankle = np.array(setup_keypoints[27][:2])
                setup_right_ankle = np.array(setup_keypoints[28][:2])
                impact_left_ankle = np.array(impact_keypoints[27][:2])
                impact_right_ankle = np.array(impact_keypoints[28][:2])
                
                # Calculate weight distribution based on foot positioning
                setup_center = (setup_left_ankle + setup_right_ankle) / 2
                impact_center = (impact_left_ankle + impact_right_ankle) / 2
                
                # Weight shift calculation (simplified)
                foot_width = np.linalg.norm(setup_right_ankle - setup_left_ankle)
                if foot_width > 0:
                    weight_shift_amount = np.linalg.norm(impact_center - setup_center) / foot_width
                    # Convert to percentage (professionals typically achieve 70%+ to front foot)
                    weight_shift = min(0.5 + weight_shift_amount * 0.5, 0.9)
                else:
                    weight_shift = 0.5
                metrics["weight_shift"] = weight_shift
            else:
                metrics["weight_shift"] = 0.5  # Neutral default
        else:
            metrics["weight_shift"] = 0.5
            
        # Calculate Posture Score (spine angle consistency)
        posture_scores = []
        for frame_list in [setup_frames, backswing_frames, impact_frames]:
            if frame_list:
                frame = frame_list[len(frame_list) // 2]
                if frame in pose_data and len(pose_data[frame]) >= 33:
                    keypoints = pose_data[frame]
                    # Use shoulder and hip landmarks to estimate spine angle
                    left_shoulder = np.array(keypoints[11][:2])
                    right_shoulder = np.array(keypoints[12][:2])
                    left_hip = np.array(keypoints[23][:2])
                    right_hip = np.array(keypoints[24][:2])
                    
                    shoulder_center = (left_shoulder + right_shoulder) / 2
                    hip_center = (left_hip + right_hip) / 2
                    
                    spine_vector = shoulder_center - hip_center
                    spine_angle = np.degrees(np.arctan2(spine_vector[1], spine_vector[0]))
                    posture_scores.append(abs(spine_angle))
        
        if posture_scores:
            # Good posture = consistent spine angle across phases
            posture_consistency = 1.0 - (np.std(posture_scores) / 90.0)  # Normalize by 90 degrees
            metrics["posture_score"] = max(0.3, min(posture_consistency, 1.0))
        else:
            metrics["posture_score"] = 0.6
            
        # Calculate Arm Extension at Impact
        if impact_frame and impact_frame in pose_data and len(pose_data[impact_frame]) >= 33:
            keypoints = pose_data[impact_frame]
            right_shoulder = np.array(keypoints[12][:2])
            right_elbow = np.array(keypoints[14][:2])
            right_wrist = np.array(keypoints[16][:2])
            
            # Calculate arm extension
            upper_arm = np.linalg.norm(right_elbow - right_shoulder)
            forearm = np.linalg.norm(right_wrist - right_elbow)
            total_arm_length = upper_arm + forearm
            
            # Calculate actual distance from shoulder to wrist
            actual_distance = np.linalg.norm(right_wrist - right_shoulder)
            
            if total_arm_length > 0:
                extension_ratio = actual_distance / total_arm_length
                metrics["arm_extension"] = min(extension_ratio, 1.0)
            else:
                metrics["arm_extension"] = 0.6
        else:
            metrics["arm_extension"] = 0.6
            
        # Calculate Wrist Hinge using joint angles
        wrist_angles = []
        for frame_list in [backswing_frames, impact_frames]:
            if frame_list:
                frame = frame_list[len(frame_list) // 2]
                if frame in pose_data:
                    angles = calculate_joint_angles(pose_data[frame])
                    if "right_wrist" in angles:
                        wrist_angles.append(angles["right_wrist"])
        
        if wrist_angles:
            avg_wrist_angle = np.mean(wrist_angles)
            # Good wrist hinge is typically 80+ degrees
            metrics["wrist_hinge"] = min(avg_wrist_angle, 120)
        else:
            metrics["wrist_hinge"] = 60
            
        # Calculate Head Movement (lateral and vertical)
        if setup_frame and impact_frame and setup_frame in pose_data and impact_frame in pose_data:
            setup_keypoints = pose_data[setup_frame]
            impact_keypoints = pose_data[impact_frame]
            
            if len(setup_keypoints) >= 33 and len(impact_keypoints) >= 33:
                # Use nose landmark (index 0) for head position
                setup_head = np.array(setup_keypoints[0][:2])
                impact_head = np.array(impact_keypoints[0][:2])
                
                head_movement = np.abs(impact_head - setup_head)
                # Convert pixel movement to approximate inches (rough estimation)
                # Assume average person's head is about 9 inches, use that as scale
                if len(setup_keypoints) > 10:  # Have enough landmarks
                    head_height_pixels = abs(setup_keypoints[0][1] - setup_keypoints[10][1])  # Nose to mouth
                    if head_height_pixels > 0:
                        pixel_to_inch = 4.0 / head_height_pixels  # Approximate nose-to-mouth is 4 inches
                        lateral_movement = head_movement[0] * pixel_to_inch
                        vertical_movement = head_movement[1] * pixel_to_inch
                    else:
                        lateral_movement = 3.0
                        vertical_movement = 2.0
                else:
                    lateral_movement = 3.0
                    vertical_movement = 2.0
                
                metrics["head_movement_lateral"] = min(lateral_movement, 8.0)
                metrics["head_movement_vertical"] = min(vertical_movement, 6.0)
            else:
                metrics["head_movement_lateral"] = 3.0
                metrics["head_movement_vertical"] = 2.0
        else:
            metrics["head_movement_lateral"] = 3.0
            metrics["head_movement_vertical"] = 2.0
            
        # Calculate Knee Flexion
        knee_flexions = {}
        for phase_name, frame_list in [("address", setup_frames), ("impact", impact_frames)]:
            if frame_list:
                frame = frame_list[len(frame_list) // 2]
                if frame in pose_data and len(pose_data[frame]) >= 33:
                    keypoints = pose_data[frame]
                    # Right knee angle using hip, knee, ankle
                    right_hip = np.array(keypoints[24][:2])
                    right_knee = np.array(keypoints[26][:2])
                    right_ankle = np.array(keypoints[28][:2])
                    
                    # Calculate knee angle
                    thigh_vector = right_hip - right_knee
                    shin_vector = right_ankle - right_knee
                    
                    if np.linalg.norm(thigh_vector) > 0 and np.linalg.norm(shin_vector) > 0:
                        cos_angle = np.dot(thigh_vector, shin_vector) / (np.linalg.norm(thigh_vector) * np.linalg.norm(shin_vector))
                        cos_angle = np.clip(cos_angle, -1, 1)
                        knee_angle = np.degrees(np.arccos(cos_angle))
                        knee_flexions[phase_name] = min(knee_angle, 60)
                    else:
                        knee_flexions[phase_name] = 25
                else:
                    knee_flexions[phase_name] = 25
        
        metrics["knee_flexion_address"] = knee_flexions.get("address", 25)
        metrics["knee_flexion_impact"] = knee_flexions.get("impact", 30)
        
        # Calculate derived metrics based on quality of basic metrics
        # These are more complex and would require additional analysis
        
        # Swing Plane Consistency (based on arm and club positions across frames)
        if metrics["shoulder_rotation"] >= 80 and metrics["arm_extension"] >= 0.75:
            metrics["swing_plane_consistency"] = 0.85
        elif metrics["shoulder_rotation"] >= 60 and metrics["arm_extension"] >= 0.6:
            metrics["swing_plane_consistency"] = 0.70
        else:
            metrics["swing_plane_consistency"] = 0.55
            
        # Chest Rotation Efficiency (derived from shoulder rotation and posture)
        chest_efficiency = (metrics["shoulder_rotation"] / 90.0) * metrics["posture_score"]
        metrics["chest_rotation_efficiency"] = min(chest_efficiency, 1.0)
        
        # Hip Thrust (derived from weight shift and hip rotation)
        hip_thrust = (metrics["weight_shift"] - 0.5) * 2 * (metrics["hip_rotation"] / 45.0)
        metrics["hip_thrust"] = max(0.3, min(hip_thrust, 1.0))
        
        # Ground Force Efficiency (derived from weight shift and knee flexion consistency)
        knee_consistency = 1.0 - abs(metrics["knee_flexion_impact"] - metrics["knee_flexion_address"]) / 30.0
        ground_force = metrics["weight_shift"] * knee_consistency
        metrics["ground_force_efficiency"] = max(0.4, min(ground_force, 1.0))
        
        # Transition Smoothness (based on posture consistency and movement quality)
        head_movement_penalty = (metrics["head_movement_lateral"] + metrics["head_movement_vertical"]) / 10.0
        transition_smoothness = metrics["posture_score"] * (1.0 - head_movement_penalty)
        metrics["transition_smoothness"] = max(0.4, min(transition_smoothness, 1.0))
        
        # Sequential Kinematic Sequence (based on overall coordination)
        coordination_score = (metrics["hip_rotation"] / 45.0 + metrics["shoulder_rotation"] / 90.0 + 
                            metrics["weight_shift"] + metrics["arm_extension"]) / 4.0
        metrics["kinematic_sequence"] = max(0.5, min(coordination_score, 1.0))
        
        # Energy Transfer Efficiency (based on multiple factors)
        energy_transfer = (metrics["kinematic_sequence"] + metrics["ground_force_efficiency"] + 
                         metrics["chest_rotation_efficiency"]) / 3.0
        metrics["energy_transfer"] = max(0.4, min(energy_transfer, 1.0))
        
        # Power Accumulation (based on body mechanics)
        power_accumulation = (metrics["hip_rotation"] / 45.0 + metrics["shoulder_rotation"] / 90.0 + 
                            metrics["wrist_hinge"] / 80.0) / 3.0
        metrics["power_accumulation"] = max(0.4, min(power_accumulation, 1.0))
        
        # Potential Distance (based on power metrics and efficiency)
        base_distance = 180  # Base amateur distance
        power_multiplier = metrics["power_accumulation"] * metrics["energy_transfer"]
        potential_distance = base_distance + (power_multiplier * 120)  # Up to 300 yards for perfect mechanics
        metrics["potential_distance"] = min(potential_distance, 320)
        
        # Speed Generation Method (based on power sources)
        if metrics["hip_rotation"] >= 40 and metrics["shoulder_rotation"] >= 80:
            metrics["speed_generation"] = "Body-dominant"
        elif metrics["arm_extension"] >= 0.8 and metrics["wrist_hinge"] >= 75:
            metrics["speed_generation"] = "Arms-dominant" 
        else:
            metrics["speed_generation"] = "Mixed"
            
    except Exception as e:
        print(f"Error calculating biomechanical metrics: {str(e)}")
        # Fail here
        return None
    
    return metrics
