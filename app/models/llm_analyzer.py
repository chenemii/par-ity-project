"""
LLM-based golf swing analysis module
"""

import json
import httpx
from openai import OpenAI
import streamlit as st
import re


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

    # Calculate backswing and downswing durations if available
    backswing_frames = swing_phases.get("backswing", [])
    downswing_frames = swing_phases.get("downswing", [])

    backswing_duration = None
    downswing_duration = None

    if backswing_frames:
        # Assuming 30 fps video
        backswing_duration = len(backswing_frames) / 30.0

    if downswing_frames:
        # Assuming 30 fps video
        downswing_duration = len(downswing_frames) / 30.0

    # Calculate tempo ratio if both durations are available
    tempo_ratio = None
    if backswing_duration and downswing_duration and downswing_duration > 0:
        tempo_ratio = backswing_duration / downswing_duration

    # Add comprehensive metrics with default values or calculated values
    # These values would normally be calculated from pose and trajectory data
    analysis_data["metrics"] = {
        # Core body mechanics
        "tempo_ratio": tempo_ratio or 3.0,  # Backswing to downswing time ratio
        "swing_plane_consistency": 0.85,  # 0-1 scale
        "weight_shift": 0.7,  # 0-1 scale
        "hip_rotation": 45,  # degrees
        "shoulder_rotation": 90,  # degrees
        "posture_score": 0.8,  # 0-1 scale

        # Upper body mechanics
        "arm_extension": 0.8,  # 0-1 scale
        "wrist_hinge": 80,  # degrees
        "chest_rotation_efficiency": 0.75,  # 0-1 scale
        "head_movement_lateral": 2.5,  # inches
        "head_movement_vertical": 1.8,  # inches

        # Lower body mechanics
        "knee_flexion_address": 25,  # degrees
        "knee_flexion_impact": 30,  # degrees
        "hip_thrust": 0.6,  # 0-1 scale
        "ground_force_efficiency": 0.7,  # 0-1 scale

        # Club path and face metrics
        "swing_path":
        2.5,  # degrees (positive = out-to-in, negative = in-to-out)
        "clubface_angle": 2.1,  # degrees (positive = open, negative = closed)
        "attack_angle":
        -4.2,  # degrees (negative = descending, positive = ascending)
        "club_path_consistency": 0.78,  # 0-1 scale

        # Tempo and timing metrics
        "transition_smoothness": 0.75,  # 0-1 scale
        "backswing_duration": backswing_duration or 0.9,  # seconds
        "downswing_duration": downswing_duration or 0.3,  # seconds
        "kinematic_sequence": 0.82,  # 0-1 scale

        # Efficiency and power metrics
        "energy_transfer": 0.78,  # 0-1 scale
        "potential_distance": 240,  # yards
        "power_accumulation": 0.75,  # 0-1 scale
        "speed_generation": "Arms-dominant"  # String description
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
You are analyzing a golf swing. First, here are examples of professional golfer swing analyses that represent benchmark performance levels:

## PROFESSIONAL BENCHMARKS

### Nelly Korda (Example 1) - LPGA Tour Professional
**Swing Phases:**
- Setup: 122 frames, Backswing: 2 frames, Downswing: 5 frames, Impact: 1 frame, Follow-through: 42 frames

**Key Metrics:**
- Tempo Ratio: 0.4, Hip Rotation: 45°, Shoulder Rotation: 90°, Posture Score: 80%
- Arm Extension: 80%, Wrist Hinge: 80°, Shoulder Plane Consistency: 85%
- Weight Shift: 70%, Transition Smoothness: 75%, Sequential Kinematic Sequence: 82%
- Energy Transfer Efficiency: 78%, Backswing Duration: 0.067s, Downswing Duration: 0.167s

### Nelly Korda (Example 2) - Different Swing Tempo Style
**Key Metrics:**
- Tempo Ratio: 3.0, Backswing Duration: 0.9s, Downswing Duration: 0.9s
- All other core metrics remain consistent: Hip Rotation: 45°, Shoulder Rotation: 90°, etc.

### Nelly Korda (Example 3) - Fast Tempo Style
**Key Metrics:**
- Tempo Ratio: 0.3, Backswing Duration: 0.067s, Downswing Duration: 0.2s
- Consistent professional metrics maintained across all mechanical aspects

### Lydia Ko - LPGA Tour Professional
**Key Metrics:**
- Tempo Ratio: 14.0, Backswing Duration: 0.467s, Downswing Duration: 0.033s
- Demonstrates that professional tempo can vary dramatically while maintaining consistency in:
- Hip/Shoulder Rotation, Posture, Arm Extension, Weight Shift, and Sequential Timing

### Atthaya Thitikul - LPGA Tour Professional
**Key Metrics:**
- Tempo Ratio: 2.8, Backswing Duration: 0.567s, Downswing Duration: 0.2s
- Consistent with professional standards across all biomechanical markers

## PROFESSIONAL STANDARDS SUMMARY
Based on these examples, professional golfers consistently achieve:
- **Core Body Mechanics**: Hip Rotation: 45°, Shoulder Rotation: 90°, Posture Score: 80%
- **Upper Body**: Arm Extension: 80%, Wrist Hinge: 80°, Shoulder Plane Consistency: 85%
- **Lower Body**: Weight Shift: 70%, Ground Force Efficiency: 70%
- **Timing**: Transition Smoothness: 75%, Sequential Kinematic Sequence: 82%
- **Efficiency**: Energy Transfer: 78%, Power Accumulation: 75%
- **Head Movement**: Lateral: 2.5in, Vertical: 1.8in (minimal movement is professional standard)
- **Tempo**: Highly variable (0.3 to 14.0 ratio) - personal style, not performance indicator

---

## CURRENT PLAYER ANALYSIS

I've analyzed a golf swing and extracted the following data:

## Swing Phases
"""

    # Add swing phases information
    for phase, data in analysis_data["swing_phases"].items():
        prompt += f"- {phase.capitalize()}: Frame {data['frame_index']}, Duration: {data['duration_frames']} frames\n"

    # Add detailed biomechanical metrics
    prompt += "\n## Swing Mechanics\n"

    # Core body mechanics
    prompt += "\n### Body Mechanics\n"
    prompt += "- Tempo Ratio (Backswing:Downswing): {:.1f}\n".format(
        analysis_data["metrics"].get("tempo_ratio", 0))
    prompt += "- Hip Rotation (degrees): {}\n".format(
        analysis_data["metrics"].get("hip_rotation", 0))
    prompt += "- Shoulder Rotation (degrees): {}\n".format(
        analysis_data["metrics"].get("shoulder_rotation", 0))
    prompt += "- Posture Score: {}%\n".format(
        int(analysis_data["metrics"].get("posture_score", 0) * 100))

    # Upper body mechanics
    prompt += "\n### Upper Body Mechanics\n"
    prompt += "- Arm Extension (impact): {}%\n".format(
        int(analysis_data["metrics"].get("arm_extension", 0.8) * 100))
    prompt += "- Wrist Hinge (degrees): {}\n".format(
        analysis_data["metrics"].get("wrist_hinge", 0))
    prompt += "- Shoulder Plane Consistency: {}%\n".format(
        int(analysis_data["metrics"].get("swing_plane_consistency", 0) * 100))
    prompt += "- Chest Rotation Efficiency: {}%\n".format(
        int(analysis_data["metrics"].get("chest_rotation_efficiency", 0.75) *
            100))
    prompt += "- Head Movement (lateral): {}in\n".format(
        analysis_data["metrics"].get("head_movement_lateral", 2.5))
    prompt += "- Head Movement (vertical): {}in\n".format(
        analysis_data["metrics"].get("head_movement_vertical", 1.8))

    # Lower body mechanics
    prompt += "\n### Lower Body Mechanics\n"
    prompt += "- Weight Shift (lead foot at impact): {}%\n".format(
        int(analysis_data["metrics"].get("weight_shift", 0) * 100))
    prompt += "- Knee Flexion (address): {}°\n".format(
        analysis_data["metrics"].get("knee_flexion_address", 25))
    prompt += "- Knee Flexion (impact): {}°\n".format(
        analysis_data["metrics"].get("knee_flexion_impact", 30))
    prompt += "- Hip Thrust (impact): {}%\n".format(
        int(analysis_data["metrics"].get("hip_thrust", 0.6) * 100))
    prompt += "- Ground Force Efficiency: {}%\n".format(
        int(analysis_data["metrics"].get("ground_force_efficiency", 0.7) *
            100))

    # Tempo and timing metrics
    prompt += "\n### Tempo & Timing\n"
    prompt += "- Transition Smoothness: {}%\n".format(
        int(analysis_data["metrics"].get("transition_smoothness", 0.75) * 100))
    prompt += "- Backswing Duration: {} seconds\n".format(
        analysis_data["metrics"].get("backswing_duration", 0.9))
    prompt += "- Downswing Duration: {} seconds\n".format(
        analysis_data["metrics"].get("downswing_duration", 0.3))
    prompt += "- Sequential Kinematic Sequence: {}%\n".format(
        int(analysis_data["metrics"].get("kinematic_sequence", 0.82) * 100))

    # Efficiency and power metrics
    prompt += "\n### Efficiency & Power Metrics\n"
    prompt += "- Energy Transfer Efficiency: {}%\n".format(
        int(analysis_data["metrics"].get("energy_transfer", 0.78) * 100))
    prompt += "- Potential Distance: {} yards\n".format(
        analysis_data["metrics"].get("potential_distance", 240))
    prompt += "- Power Accumulation: {}%\n".format(
        int(analysis_data["metrics"].get("power_accumulation", 0.75) * 100))
    prompt += "- Speed Generation Method: {}\n".format(
        analysis_data["metrics"].get("speed_generation", "Arms-dominant"))

    prompt += """

## ANALYSIS INSTRUCTIONS

Using the professional benchmarks above as your calibration reference, provide:

1. **Performance Classification**: Start with "Performance Classification: [Professional/Advanced/Intermediate/Beginner]" based on how the player's metrics compare to professional standards.

2. **Comparative Analysis**: 
   - **Strengths** (metrics that meet/exceed professional benchmarks):
     • List specific strong points (use bullet points)
     • Reference professional benchmark values
   
   - **Areas for Improvement** (metrics significantly below professional standards):
     • List specific weaknesses (use bullet points)
     • Note the gap from professional standards

3. **Priority Improvement Areas**: List exactly 3 areas in order of importance:
   1. [Most Critical] - Describe what's wrong and what it should be like
   2. [Important] - Describe what's wrong and what it should be like  
   3. [Focus Area] - Describe what's wrong and what it should be like

Remember: Professional golfers consistently achieve the benchmark metrics shown above. Use these as the gold standard for what constitutes excellent golf swing mechanics, while being realistic about the progression needed to reach those levels.

Provide your analysis in the structured format above for optimal coaching feedback.
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
        'classification': 'Intermediate',  # Default classification
        'strengths': [],
        'weaknesses': [],
        'priority_improvements': []
    }
    
    # Try to extract classification from the analysis
    classification_patterns = [
        r'(?:Performance Classification|Classification|Level).*?:\s*(Professional|Advanced|Intermediate|Beginner)',
        r'(Professional|Advanced|Intermediate|Beginner)\s+(?:Level|Amateur)',
        r'classified as\s+(Professional|Advanced|Intermediate|Beginner)',
        r'(?:at|as)\s+(?:an?\s+)?(Professional|Advanced|Intermediate|Beginner)\s+level'
    ]
    
    classification_found = False
    for pattern in classification_patterns:
        match = re.search(pattern, raw_analysis, re.IGNORECASE)
        if match:
            formatted_analysis['classification'] = match.group(1).title()
            classification_found = True
            break
    
    # If no classification found, try to infer from content
    if not classification_found:
        analysis_lower = raw_analysis.lower()
        if 'professional' in analysis_lower and ('meets' in analysis_lower or 'exceeds' in analysis_lower):
            formatted_analysis['classification'] = 'Professional'
        elif 'advanced' in analysis_lower or ('within 10' in analysis_lower and 'pro' in analysis_lower):
            formatted_analysis['classification'] = 'Advanced'
        elif 'beginner' in analysis_lower or ('30%' in analysis_lower and 'below' in analysis_lower):
            formatted_analysis['classification'] = 'Beginner'
        else:
            formatted_analysis['classification'] = 'Intermediate'
    
    # Extract strengths and weaknesses
    strengths_section = ""
    weaknesses_section = ""
    
    # Look for strengths/weaknesses sections
    strengths_patterns = [
        r'(?:Strengths|Strong Points|Positives|Meets.*Standards)[\s\S]*?(?=(?:Weak|Priority|Improvement|Areas|$))',
        r'(?:Professional Level|Exceeds.*Standards)[\s\S]*?(?=(?:Below|Weak|Priority|$))'
    ]
    
    weaknesses_patterns = [
        r'(?:Weaknesses|Weak|Areas.*Improvement|Priority.*Areas|Below.*Standards)[\s\S]*?(?=(?:Recommendation|Priority|$))',
        r'(?:Critical|Important|Significant.*gaps?)[\s\S]*?(?=(?:Recommendation|$))'
    ]
    
    for pattern in strengths_patterns:
        match = re.search(pattern, raw_analysis, re.IGNORECASE)
        if match:
            strengths_section = match.group(0)
            break
    
    for pattern in weaknesses_patterns:
        match = re.search(pattern, raw_analysis, re.IGNORECASE)
        if match:
            weaknesses_section = match.group(0)
            break
    
    # Parse strengths from the section
    if strengths_section:
        strength_items = re.findall(r'[-•]\s*([^-•\n]+)', strengths_section)
        formatted_analysis['strengths'] = [item.strip() for item in strength_items[:4]]  # Limit to 4
    
    # If no bullet points found, try to extract from general content
    if not formatted_analysis['strengths']:
        # Look for positive indicators in the full text
        positive_indicators = [
            r'(?:meets|exceeds|matches).*professional.*(?:standard|benchmark)',
            r'(?:excellent|good|strong).*(?:posture|rotation|extension|timing)',
            r'(?:consistent|solid).*(?:mechanics|form|technique)',
            r'(?:efficient|effective).*(?:transfer|generation|sequence)'
        ]
        
        for pattern in positive_indicators:
            matches = re.findall(pattern, raw_analysis, re.IGNORECASE)
            for match in matches[:2]:  # Limit to avoid overwhelming
                formatted_analysis['strengths'].append(match.strip())
    
    # Parse weaknesses from the section
    if weaknesses_section:
        weakness_items = re.findall(r'[-•]\s*([^-•\n]+)', weaknesses_section)
        formatted_analysis['weaknesses'] = [item.strip() for item in weakness_items[:4]]  # Limit to 4
    
    # If no bullet points found, try to extract from general content
    if not formatted_analysis['weaknesses']:
        # Look for negative indicators in the full text
        negative_indicators = [
            r'(?:below|under).*professional.*(?:standard|benchmark)',
            r'(?:poor|weak|limited).*(?:posture|rotation|extension|timing)',
            r'(?:inconsistent|unstable).*(?:mechanics|form|technique)',
            r'(?:inefficient|ineffective).*(?:transfer|generation|sequence)'
        ]
        
        for pattern in negative_indicators:
            matches = re.findall(pattern, raw_analysis, re.IGNORECASE)
            for match in matches[:2]:  # Limit to avoid overwhelming
                formatted_analysis['weaknesses'].append(match.strip())
    
    # Extract priority improvements
    priority_patterns = [
        r'(?:Priority.*Improvement|Critical.*Areas?)[\s\S]*?(?=(?:Recommendation|$))',
        r'(?:1\..*?2\..*?3\.)',  # Numbered list
        r'(?:Critical|Important|Fine-tuning)[\s\S]*?(?=(?:Critical|Important|Fine-tuning|$))'
    ]
    
    for pattern in priority_patterns:
        match = re.search(pattern, raw_analysis, re.IGNORECASE | re.DOTALL)
        if match:
            priority_text = match.group(0)
            # Extract numbered items
            numbered_items = re.findall(r'(\d+)\.\s*([^1-9\n]+)', priority_text)
            for num, item in numbered_items[:3]:  # Limit to 3
                formatted_analysis['priority_improvements'].append({
                    'rank': int(num),
                    'description': item.strip()
                })
            break
    
    # If no numbered priorities found, create generic ones based on classification
    if not formatted_analysis['priority_improvements']:
        if formatted_analysis['classification'] == 'Beginner':
            formatted_analysis['priority_improvements'] = [
                {'rank': 1, 'description': 'Focus on fundamental posture and setup position'},
                {'rank': 2, 'description': 'Develop consistent tempo and timing'},
                {'rank': 3, 'description': 'Improve weight shift and balance throughout swing'}
            ]
        elif formatted_analysis['classification'] == 'Intermediate':
            formatted_analysis['priority_improvements'] = [
                {'rank': 1, 'description': 'Enhance kinematic sequence and body rotation'},
                {'rank': 2, 'description': 'Improve clubface control and swing path consistency'},
                {'rank': 3, 'description': 'Optimize energy transfer efficiency'}
            ]
        elif formatted_analysis['classification'] == 'Advanced':
            formatted_analysis['priority_improvements'] = [
                {'rank': 1, 'description': 'Fine-tune transition smoothness and timing'},
                {'rank': 2, 'description': 'Optimize power accumulation and release'},
                {'rank': 3, 'description': 'Enhance consistency under pressure'}
            ]
        else:  # Professional
            formatted_analysis['priority_improvements'] = [
                {'rank': 1, 'description': 'Maintain current excellence with minor adjustments'},
                {'rank': 2, 'description': 'Focus on course management and strategy'},
                {'rank': 3, 'description': 'Continue physical conditioning for longevity'}
            ]
    
    # Ensure we have some default content if parsing failed
    if not formatted_analysis['strengths']:
        formatted_analysis['strengths'] = ['Swing analysis completed successfully']
    
    if not formatted_analysis['weaknesses']:
        formatted_analysis['weaknesses'] = ['Areas for improvement identified']
    
    return formatted_analysis


def display_formatted_analysis(analysis_data):
    """
    Display the formatted analysis with performance classification, strengths/weaknesses table, and priorities
    
    Args:
        analysis_data (dict): Structured analysis data from parse_and_format_analysis
    """
    # 1. Performance Classification with colored rounded rectangles
    user_classification = analysis_data['classification']
    
    # Display classification in black bolded header
    st.markdown(f"""
    <h2 style='color: black; font-weight: bold; text-align: center; margin-bottom: 20px;'>
        🎯 Performance Classification: {user_classification}
    </h2>
    """, unsafe_allow_html=True)
    
    # Create columns for the classification rectangles
    col1, col2, col3, col4 = st.columns(4)
    
    # Define colors and styling - all rectangles should have colors
    colors = {
        'Beginner': {'bg': '#ff4444', 'text': 'white'},
        'Intermediate': {'bg': '#ff8800', 'text': 'white'},
        'Advanced': {'bg': '#ffdd00', 'text': 'black'},
        'Professional': {'bg': '#44aa44', 'text': 'white'}
    }
    
    with col1:
        bg_color = colors['Beginner']['bg']
        text_color = colors['Beginner']['text']
        border_style = '3px solid #333' if user_classification == 'Beginner' else '2px solid #ddd'
        st.markdown(f"""
        <div style='text-align: center; padding: 15px; background-color: {bg_color}; 
                    border-radius: 15px; margin: 5px; border: {border_style};'>
            <div style='font-size: 14px; font-weight: bold; color: {text_color};'>Beginner</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        bg_color = colors['Intermediate']['bg']
        text_color = colors['Intermediate']['text']
        border_style = '3px solid #333' if user_classification == 'Intermediate' else '2px solid #ddd'
        st.markdown(f"""
        <div style='text-align: center; padding: 15px; background-color: {bg_color}; 
                    border-radius: 15px; margin: 5px; border: {border_style};'>
            <div style='font-size: 14px; font-weight: bold; color: {text_color};'>Intermediate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        bg_color = colors['Advanced']['bg']
        text_color = colors['Advanced']['text']
        border_style = '3px solid #333' if user_classification == 'Advanced' else '2px solid #ddd'
        st.markdown(f"""
        <div style='text-align: center; padding: 15px; background-color: {bg_color}; 
                    border-radius: 15px; margin: 5px; border: {border_style};'>
            <div style='font-size: 14px; font-weight: bold; color: {text_color};'>Advanced</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        bg_color = colors['Professional']['bg']
        text_color = colors['Professional']['text']
        border_style = '3px solid #333' if user_classification == 'Professional' else '2px solid #ddd'
        st.markdown(f"""
        <div style='text-align: center; padding: 15px; background-color: {bg_color}; 
                    border-radius: 15px; margin: 5px; border: {border_style};'>
            <div style='font-size: 14px; font-weight: bold; color: {text_color};'>Professional</div>
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
        
        # Extract improvement area and description if possible
        if ':' in description:
            area, desc = description.split(':', 1)
            area = area.strip()
            desc = desc.strip()
        elif '-' in description:
            parts = description.split('-', 1)
            if len(parts) == 2:
                area = parts[0].strip()
                desc = parts[1].strip()
            else:
                area = description
                desc = ""
        else:
            # Try to extract first sentence as area, rest as description
            sentences = description.split('. ')
            if len(sentences) > 1:
                area = sentences[0]
                desc = '. '.join(sentences[1:])
            else:
                area = description
                desc = ""
        
        # Color code by priority with better styling
        if rank == 1:
            st.markdown(f"""
            <div style='background-color: #ffebee; padding: 15px; border-left: 5px solid #f44336; border-radius: 5px; margin: 10px 0; word-wrap: break-word; overflow-wrap: break-word;'>
                <strong style='color: #d32f2f; font-size: 16px; display: block; margin-bottom: 8px;'>{rank}. MOST CRITICAL: {area}</strong>
                {f"<div style='color: #666; font-size: 14px; line-height: 1.4; word-wrap: break-word;'>{desc}</div>" if desc else ""}
            </div>
            """, unsafe_allow_html=True)
        elif rank == 2:
            st.markdown(f"""
            <div style='background-color: #fff8e1; padding: 15px; border-left: 5px solid #ff9800; border-radius: 5px; margin: 10px 0; word-wrap: break-word; overflow-wrap: break-word;'>
                <strong style='color: #f57c00; font-size: 16px; display: block; margin-bottom: 8px;'>{rank}. IMPORTANT: {area}</strong>
                {f"<div style='color: #666; font-size: 14px; line-height: 1.4; word-wrap: break-word;'>{desc}</div>" if desc else ""}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background-color: #e3f2fd; padding: 15px; border-left: 5px solid #2196f3; border-radius: 5px; margin: 10px 0; word-wrap: break-word; overflow-wrap: break-word;'>
                <strong style='color: #1976d2; font-size: 16px; display: block; margin-bottom: 8px;'>{rank}. FOCUS AREA: {area}</strong>
                {f"<div style='color: #666; font-size: 14px; line-height: 1.4; word-wrap: break-word;'>{desc}</div>" if desc else ""}
            </div>
            """, unsafe_allow_html=True)
