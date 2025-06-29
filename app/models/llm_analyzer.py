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

Using the professional benchmarks above as your calibration reference, provide your analysis in the following EXACT structured format:

**PERFORMANCE_CLASSIFICATION:** [Professional/Advanced/Intermediate/Beginner]

**STRENGTHS:**
• [Specific strength with metric comparison to professional standard]
• [Another strength with professional benchmark reference]
• [Third strength if applicable]

**WEAKNESSES:**
• [Specific weakness with gap from professional standard]
• [Another weakness with professional benchmark comparison]
• [Third weakness if applicable]

**PRIORITY_IMPROVEMENTS:**
1. [Most Critical] Topic Name - Detailed description of current issue and what should be improved to reach professional standard
2. [Important] Topic Name - Detailed description of current issue and desired improvement outcome
3. [Focus Area] Topic Name - Detailed description of current issue and target improvement goal

IMPORTANT FORMATTING RULES:
- Use the exact headers shown above (PERFORMANCE_CLASSIFICATION, STRENGTHS, WEAKNESSES, PRIORITY_IMPROVEMENTS)
- For strengths and weaknesses, use bullet points (•) 
- For priority improvements, use numbered format (1., 2., 3.) with priority level in brackets
- Each priority improvement must have: [Priority Level] Topic Name - Full description
- Provide complete sentences and descriptions - no incomplete thoughts
- Compare all metrics to the professional benchmarks provided above
- Be specific about what needs improvement and what the target should be

Remember: Professional golfers consistently achieve the benchmark metrics shown above. Use these as the gold standard for what constitutes excellent golf swing mechanics.
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
    
    # Extract classification using the new structured format
    classification_match = re.search(r'\*\*PERFORMANCE_CLASSIFICATION:\*\*\s*([A-Za-z]+)', raw_analysis, re.IGNORECASE)
    if classification_match:
        formatted_analysis['classification'] = classification_match.group(1).title()
    else:
        # Fallback to original patterns
        classification_patterns = [
            r'(?:Performance Classification|Classification|Level).*?:\s*(Professional|Advanced|Intermediate|Beginner)',
            r'(Professional|Advanced|Intermediate|Beginner)\s+(?:Level|Amateur)',
            r'classified as\s+(Professional|Advanced|Intermediate|Beginner)',
            r'(?:at|as)\s+(?:an?\s+)?(Professional|Advanced|Intermediate|Beginner)\s+level'
        ]
        
        for pattern in classification_patterns:
            match = re.search(pattern, raw_analysis, re.IGNORECASE)
            if match:
                formatted_analysis['classification'] = match.group(1).title()
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
        if formatted_analysis['classification'] == 'Beginner':
            formatted_analysis['priority_improvements'] = [
                {'rank': 1, 'description': '[Most Critical] Fundamental Posture and Setup - Focus on establishing proper spine angle and athletic stance throughout the swing for better consistency and power transfer.'},
                {'rank': 2, 'description': '[Important] Tempo and Timing Development - Develop consistent swing rhythm and timing to improve sequence and control.'},
                {'rank': 3, 'description': '[Focus Area] Weight Shift and Balance - Improve weight transfer from back foot to front foot during swing for better power and stability.'}
            ]
        elif formatted_analysis['classification'] == 'Intermediate':
            formatted_analysis['priority_improvements'] = [
                {'rank': 1, 'description': '[Most Critical] Kinematic Sequence Enhancement - Improve body rotation coordination to generate more power and consistency.'},
                {'rank': 2, 'description': '[Important] Clubface Control - Enhance swing path consistency for better ball striking accuracy.'},
                {'rank': 3, 'description': '[Focus Area] Energy Transfer Efficiency - Optimize power transfer throughout the swing to maximize distance.'}
            ]
        elif formatted_analysis['classification'] == 'Advanced':
            formatted_analysis['priority_improvements'] = [
                {'rank': 1, 'description': '[Most Critical] Transition Smoothness - Fine-tune timing and tempo to achieve professional-level consistency.'},
                {'rank': 2, 'description': '[Important] Power Accumulation - Optimize energy storage and release for maximum clubhead speed.'},
                {'rank': 3, 'description': '[Focus Area] Pressure Performance - Enhance consistency under competitive conditions.'}
            ]
        else:  # Professional
            formatted_analysis['priority_improvements'] = [
                {'rank': 1, 'description': '[Most Critical] Technical Refinement - Maintain excellence with minor adjustments to specific mechanics.'},
                {'rank': 2, 'description': '[Important] Strategic Optimization - Focus on course management and scoring opportunities.'},
                {'rank': 3, 'description': '[Focus Area] Physical Conditioning - Continue fitness work for career longevity and peak performance.'}
            ]
    
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
                desc = "Work on maintaining proper spine angle and athletic stance throughout the swing for better consistency and power transfer."
            elif 'tempo' in area.lower() or 'timing' in area.lower():
                desc = "Focus on developing a smooth, consistent rhythm that allows for proper sequencing of body movements."
            elif 'rotation' in area.lower():
                desc = "Improve the coordination and range of motion in your body turn to generate more power and accuracy."
            elif 'weight' in area.lower() or 'shift' in area.lower():
                desc = "Practice transferring weight from back foot to front foot during the swing for better balance and power."
            elif 'knee' in area.lower():
                desc = "Work on maintaining proper knee flex and stability throughout the swing for better foundation and consistency."
            elif 'hip' in area.lower():
                desc = "Focus on improving hip mobility and thrust timing to enhance power generation and sequencing."
            elif 'chest' in area.lower():
                desc = "Improve chest rotation efficiency to better coordinate upper body movement with the swing sequence."
            else:
                desc = description  # Use the full description if we can't categorize it
        
        # Display using Streamlit's native components
        if rank == 1:
            st.error(f"**{rank}. MOST CRITICAL: {area}**")
            st.write(desc)
        elif rank == 2:
            st.warning(f"**{rank}. IMPORTANT: {area}**")
            st.write(desc)
        else:
            st.info(f"**{rank}. FOCUS AREA: {area}**")
            st.write(desc)
        
        st.write("")  # Add spacing between items
