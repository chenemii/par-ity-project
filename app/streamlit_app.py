"""
Streamlit web UI for Golf Swing Analysis
"""

import os
import sys
import tempfile
import streamlit as st
from dotenv import load_dotenv
import base64
from pathlib import Path
import shutil
import cv2
from PIL import Image
from datetime import datetime

# Load environment variables
load_dotenv()

# Add the app directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.video_downloader import download_youtube_video, download_pro_reference, cleanup_video_file, cleanup_downloads_directory
from app.utils.video_processor import process_video
from app.models.pose_estimator import analyze_pose
from app.models.swing_analyzer import segment_swing, analyze_trajectory
from app.models.llm_analyzer import generate_swing_analysis, create_llm_prompt, prepare_data_for_llm, check_llm_services, parse_and_format_analysis, display_formatted_analysis
from app.utils.visualizer import create_annotated_video
from app.utils.comparison import create_key_frame_comparison, extract_key_swing_frames

# Import RAG functionality
try:
    from app.golf_swing_rag import GolfSwingRAG
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    st.warning("RAG functionality not available. Please ensure golf_swing_rag.py is in the app directory.")

# Set page config
st.set_page_config(page_title="Par-ity Project: Golf Swing Analysis 🏌️‍♀️",
                   page_icon="🏌️‍♀️",
                   layout="wide",
                   initial_sidebar_state="collapsed")

# Custom CSS for RAG interface
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f1f8e9;
        border-left: 4px solid #4caf50;
    }
    .rag-header {
        color: #2E8B57;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_rag_system():
    """Load and initialize the RAG system (cached for performance)"""
    if not RAG_AVAILABLE:
        return None
    try:
        with st.spinner("Loading golf swing knowledge base..."):
            rag = GolfSwingRAG()
            rag.load_and_process_data()
            rag.create_embeddings()
        return rag
    except Exception as e:
        st.error(f"Error loading RAG system: {str(e)}")
        return None

def display_rag_sources(sources):
    """Display source information in an organized way"""
    if not sources:
        return
    
    st.subheader("📚 Sources")
    for i, source in enumerate(sources[:3]):  # Show top 3 sources
        with st.expander(f"Source {i+1}: {source['metadata']['title'][:60]}..."):
            st.write(f"**Similarity Score:** {source['similarity_score']:.3f}")
            st.write(f"**Source:** {source['metadata']['source']}")
            if source['metadata']['url']:
                st.write(f"**URL:** [Link]({source['metadata']['url']})")
            st.write("**Content:**")
            st.write(source['chunk'][:500] + "..." if len(source['chunk']) > 500 else source['chunk'])

def render_rag_interface():
    """Render the RAG chatbot interface"""
    # Removed header and description
    
    # Initialize RAG system
    if 'rag_system' not in st.session_state and RAG_AVAILABLE:
        st.session_state.rag_system = load_rag_system()
    
    # Initialize chat history if not exists
    if 'rag_chat_history' not in st.session_state:
        st.session_state.rag_chat_history = []
    
    if not RAG_AVAILABLE or st.session_state.get('rag_system') is None:
        st.error("RAG system is not available. Please check the setup.")
        return
    
    # Check if we have video analysis data to enhance responses
    user_swing_context = ""
    if st.session_state.get('video_analyzed') and 'analysis_data' in st.session_state:
        stored_data = st.session_state.analysis_data
        
        # Use the structured analysis_data instead of just the prompt
        if 'analysis_data' in stored_data:
            structured_analysis = stored_data['analysis_data']
            
            # Format the structured data for better RAG context
            user_swing_context = f"""

USER'S SWING ANALYSIS:

=== SWING TIMING & PHASES ===
Swing Phases:
- Setup: {structured_analysis.get('swing_phases', {}).get('setup', {}).get('frame_count', 0)} frames
- Backswing: {structured_analysis.get('swing_phases', {}).get('backswing', {}).get('frame_count', 0)} frames  
- Downswing: {structured_analysis.get('swing_phases', {}).get('downswing', {}).get('frame_count', 0)} frames
- Impact: {structured_analysis.get('swing_phases', {}).get('impact', {}).get('frame_count', 0)} frames
- Follow-through: {structured_analysis.get('swing_phases', {}).get('follow_through', {}).get('frame_count', 0)} frames

Timing Metrics:
- Tempo Ratio (down:back): {structured_analysis.get('timing_metrics', {}).get('tempo_ratio', 'N/A')}
- Estimated Club Speed: {structured_analysis.get('timing_metrics', {}).get('estimated_club_speed_mph', 'N/A')} mph
- Total Swing Time: {structured_analysis.get('timing_metrics', {}).get('total_swing_time_ms', 'N/A')} ms

=== BIOMECHANICAL METRICS ===
Core Body Mechanics:
- Hip Rotation: {structured_analysis.get('biomechanical_metrics', {}).get('hip_rotation_degrees', 'N/A')}°
- Shoulder Rotation: {structured_analysis.get('biomechanical_metrics', {}).get('shoulder_rotation_degrees', 'N/A')}°
- Posture Score: {structured_analysis.get('biomechanical_metrics', {}).get('posture_score_percent', 'N/A')}%
- Weight Shift: {structured_analysis.get('biomechanical_metrics', {}).get('weight_shift_percent', 'N/A')}%

Upper Body Mechanics:
- Arm Extension: {structured_analysis.get('biomechanical_metrics', {}).get('arm_extension_percent', 'N/A')}%
- Wrist Hinge: {structured_analysis.get('biomechanical_metrics', {}).get('wrist_hinge_degrees', 'N/A')}°
- Swing Plane Consistency: {structured_analysis.get('biomechanical_metrics', {}).get('swing_plane_consistency_percent', 'N/A')}%
- Head Movement (lateral): {structured_analysis.get('biomechanical_metrics', {}).get('head_movement_lateral_inches', 'N/A')} in
- Head Movement (vertical): {structured_analysis.get('biomechanical_metrics', {}).get('head_movement_vertical_inches', 'N/A')} in

Lower Body Mechanics:
- Hip Thrust: {structured_analysis.get('biomechanical_metrics', {}).get('hip_thrust_percent', 'N/A')}%
- Ground Force Efficiency: {structured_analysis.get('biomechanical_metrics', {}).get('ground_force_efficiency_percent', 'N/A')}%
- Knee Flexion (address): {structured_analysis.get('biomechanical_metrics', {}).get('knee_flexion_address_degrees', 'N/A')}°
- Knee Flexion (impact): {structured_analysis.get('biomechanical_metrics', {}).get('knee_flexion_impact_degrees', 'N/A')}°

Movement Quality & Coordination:
- Sequential Kinematic Sequence: {structured_analysis.get('biomechanical_metrics', {}).get('kinematic_sequence_percent', 'N/A')}%
- Energy Transfer Efficiency: {structured_analysis.get('biomechanical_metrics', {}).get('energy_transfer_efficiency_percent', 'N/A')}%
- Power Accumulation: {structured_analysis.get('biomechanical_metrics', {}).get('power_accumulation_percent', 'N/A')}%
- Transition Smoothness: {structured_analysis.get('biomechanical_metrics', {}).get('transition_smoothness_percent', 'N/A')}%

Performance Estimates:
- Potential Distance: {structured_analysis.get('biomechanical_metrics', {}).get('potential_distance_yards', 'N/A')} yards
- Speed Generation Method: {structured_analysis.get('biomechanical_metrics', {}).get('speed_generation_method', 'N/A')}

=== TRAJECTORY ANALYSIS ===
- Estimated Carry Distance: {structured_analysis.get('trajectory_analysis', {}).get('estimated_carry_distance', 'N/A')} yards
- Estimated Ball Speed: {structured_analysis.get('trajectory_analysis', {}).get('estimated_ball_speed', 'N/A')} mph
- Trajectory Type: {structured_analysis.get('trajectory_analysis', {}).get('trajectory_type', 'N/A')}
"""
            
            # Removed success message
        elif 'prompt' in stored_data:
            # Fallback to prompt if structured data not available
            user_swing_context = f"\n\nUSER'S SWING ANALYSIS:\n{stored_data['prompt']}"
            # Removed success message

    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Removed subheader
        
        # Question input (removed label)
        question = st.text_area(
            "",  # Removed label
            height=100,
            placeholder="Ask about your golf swing technique..."
        )
        
        # Removed settings section - using smart defaults instead
        
        col_submit, col_clear = st.columns([1, 1])
        with col_submit:
            submit_button = st.button("🎯 Get Answer", type="primary", use_container_width=True)
        with col_clear:
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.rag_chat_history = []
                # Don't call st.rerun() here to avoid disappearing interface
                st.success("Chat history cleared!")
        
        # Process question
        if submit_button and question.strip():
            with st.spinner("Analyzing your question and searching the knowledge base..."):
                try:
                    # Enhanced query method that includes user's swing context
                    # Use smart default for number of sources (3-5 depending on context)
                    num_sources = 5 if user_swing_context else 3  # More sources when we have swing analysis
                    result = query_with_user_context(
                        st.session_state.rag_system, 
                        question, 
                        user_swing_context,
                        top_k=num_sources
                    )
                    
                    # Add to chat history
                    st.session_state.rag_chat_history.append({
                        'question': question,
                        'response': result['response'],
                        'sources': result['sources'],
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'used_swing_context': bool(user_swing_context)
                    })
                    
                    st.success("Answer generated successfully!")
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        
        # Display chat history (simplified)
        if st.session_state.rag_chat_history:
            for i, chat in enumerate(reversed(st.session_state.rag_chat_history)):
                # Removed question numbers, timestamps, and personalization indicators
                
                # Question
                st.markdown(f'<div class="chat-message user-message"><strong>🤔 Your Question:</strong><br>{chat["question"]}</div>', 
                           unsafe_allow_html=True)
                
                # Response
                st.markdown(f'<div class="chat-message assistant-message"><strong>⛳ Expert Answer:</strong><br>{chat["response"]}</div>', 
                           unsafe_allow_html=True)
                
                # Removed sources display
                
                st.divider()
    
    with col2:
        # Removed all the About section, Tips, Personalized Questions, and metrics
        pass

def query_with_user_context(rag_system, question, user_swing_context, top_k=5):
    """Enhanced query method that includes user's swing analysis context"""
    # Search for relevant chunks
    relevant_chunks = rag_system.search_similar_chunks(question, top_k)
    
    # Generate response with enhanced context
    response = generate_enhanced_response(rag_system, question, relevant_chunks, user_swing_context)
    print(f"Response: {response}")
    
    return {
        'response': response,
        'sources': relevant_chunks,
        'query': question,
        'timestamp': datetime.now().isoformat()
    }

def generate_enhanced_response(rag_system, query, context_chunks, user_swing_context=""):
    """Generate response using OpenAI API with user's swing analysis as the main system prompt"""
    if not rag_system.openai_client:
        print("No OpenAI client found")
        return generate_enhanced_fallback_response(query, context_chunks, user_swing_context)
    
    # Prepare context from knowledge base
    knowledge_context = "\n\n".join([f"Reference Material from '{chunk['metadata']['title']}':\n{chunk['chunk']}" 
                          for chunk in context_chunks])
    
    # Use the user's swing analysis as the primary system prompt if available
    print(f"User swing context: {user_swing_context}")
    if user_swing_context:
        # Extract the actual analysis content (remove the header)
        analysis_content = user_swing_context.replace("USER'S SWING ANALYSIS:\n", "").strip()
        
        system_prompt = f"""{analysis_content}

You are a golf swing technique expert assistant analyzing this specific player's swing. 

IMPORTANT: Only reference the player's swing analysis data above if the question is directly related to swing motion biomechanics (like hip rotation, shoulder turn, weight transfer, timing, etc.). 

Do NOT reference swing analysis for questions about:
- Grip (how to hold the club)
- Setup/stance (static positioning before the swing)
- Equipment (clubs, balls, etc.)
- Course management
- Mental game
- Basic fundamentals that aren't measured during swing motion

Follow this response structure:

1. Synthesize information from the reference materials below to answer the user's question. Keep this to 2-4 sentences maximum. Start with "Based on [source name]," and provide clear, actionable advice about the technique.

2. If the question relates to swing motion biomechanics AND you found relevant measurements in the analysis above, provide specific improvement advice comparing current state to recommendations. Otherwise, provide general advice without forcing connections to unrelated swing metrics.

Reference Materials from Golf Instruction Database:
{knowledge_context}"""

        user_prompt = f"""Based on the golf instruction reference materials provided, please answer this question about golf swing technique:

{query}

Remember to:
1. Only reference my swing analysis if the question is about swing motion biomechanics
2. Synthesize expert advice concisely (2-4 sentences max)
3. Don't force connections between unrelated topics (e.g., don't mention wrist hinge when asking about grip)"""

    else:
        # Fallback to general system prompt if no swing analysis available
        system_prompt = f"""You are a golf swing technique expert assistant. You help golfers improve their swing by providing detailed, accurate advice based on professional golf instruction content.

Instructions:
- Answer questions about golf swing technique, mechanics, common problems, and solutions
- Provide specific, actionable advice when possible
- Reference relevant technical concepts when appropriate
- Be encouraging and supportive
- Synthesize information from multiple sources rather than just quoting them
- Give clear, comprehensive explanations that golfers can understand and apply

Reference Materials from Golf Instruction Database:
{knowledge_context}"""

        user_prompt = f"""Based on the golf instruction reference materials provided, please answer this question about golf swing technique:

{query}

Please provide a helpful, detailed response that synthesizes the relevant information into clear, actionable guidance."""

    print(f"System prompt: {system_prompt}")
    print(f"User prompt: {user_prompt}")
    try:
        response = rag_system.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=800,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return generate_enhanced_fallback_response(query, context_chunks, user_swing_context)

def generate_enhanced_fallback_response(query, context_chunks, user_swing_context=""):
    """Generate an enhanced fallback response when OpenAI API is not available"""
    if not context_chunks:
        return "I couldn't find specific information about that topic in the golf swing database. Could you try rephrasing your question or being more specific?"
    
    # Extract relevant information from chunks
    best_chunk = context_chunks[0]
    chunk_content = best_chunk['chunk']
    source_title = best_chunk['metadata']['title']
    
    response_parts = []
    
    # Check if question is about swing motion biomechanics vs setup/grip/equipment
    question_lower = query.lower()
    
    # Define topics that are NOT about swing motion biomechanics
    non_biomechanics_topics = [
        'grip', 'hold', 'grip pressure', 'grip size', 'grip style',
        'setup', 'stance', 'address', 'alignment', 'posture at address',
        'equipment', 'club', 'ball', 'tee', 'glove',
        'course management', 'strategy', 'mental', 'psychology',
        'warm up', 'practice', 'routine', 'pre-shot'
    ]
    
    # Check if question is about non-biomechanics topics
    is_non_biomechanics = any(topic in question_lower for topic in non_biomechanics_topics)
    
    # Part 1: Only check for relevant measurements if question is about swing motion biomechanics
    found_relevant_measurement = False
    if user_swing_context and not is_non_biomechanics:
        analysis_content = user_swing_context.replace("USER'S SWING ANALYSIS:\n", "").strip()
        analysis_lower = analysis_content.lower()
        
        # Only do specific keyword matching for biomechanics-related questions
        if "wrist" in question_lower and "hinge" in question_lower:
            # Look for wrist hinge measurements (only if asking about wrist hinge specifically)
            lines = analysis_content.split('\n')
            for line in lines:
                if 'wrist hinge' in line.lower() and ('°' in line or '%' in line):
                    import re
                    wrist_match = re.search(r'wrist hinge[:\s]*(\d+\.?\d*°)', line.lower())
                    if wrist_match:
                        response_parts.append(f"I notice that your wrist hinge is {wrist_match.group(1)} during your swing.")
                        found_relevant_measurement = True
                        break
                        
        elif "hip" in question_lower and ("rotation" in question_lower or "turn" in question_lower):
            # Look for hip rotation measurements (only if asking about hip rotation/turn)
            lines = analysis_content.split('\n')
            for line in lines:
                if 'hip rotation' in line.lower() and '°' in line:
                    import re
                    user_hip_match = re.search(r'-\s*hip rotation[:\s]*(\d+\.?\d*°)', line.lower())
                    if user_hip_match:
                        response_parts.append(f"I notice that your hip rotation is {user_hip_match.group(1)} during your swing.")
                        found_relevant_measurement = True
                        break
                        
        elif "weight" in question_lower and ("transfer" in question_lower or "shift" in question_lower):
            # Look for weight transfer measurements (only if asking about weight transfer/shift)
            lines = analysis_content.split('\n')
            for line in lines:
                if ('weight transfer' in line.lower() or 'weight shift' in line.lower()) and '%' in line:
                    import re
                    weight_match = re.search(r'weight (?:transfer|shift)[:\s]*(\d+\.?\d*%)', line.lower())
                    if weight_match:
                        response_parts.append(f"I notice that your weight transfer is {weight_match.group(1)} during the downswing.")
                        found_relevant_measurement = True
                        break
                        
        elif "shoulder" in question_lower and ("rotation" in question_lower or "turn" in question_lower):
            # Look for shoulder measurements (only if asking about shoulder rotation/turn)
            lines = analysis_content.split('\n')
            for line in lines:
                if 'shoulder rotation' in line.lower() and '°' in line:
                    import re
                    shoulder_match = re.search(r'shoulder rotation[:\s]*(\d+\.?\d*°)', line.lower())
                    if shoulder_match:
                        response_parts.append(f"I notice that your shoulder rotation is {shoulder_match.group(1)} during your swing.")
                        found_relevant_measurement = True
                        break
    
    # Part 2: Expert recommendation (synthesized from source)
    sentences = chunk_content.split('. ')
    meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 20][:3]
    expert_advice = '. '.join(meaningful_sentences[:2]) + '.'
    
    response_parts.append(f"Based on {source_title}, {expert_advice}")
    
    # Part 3: Improvement recommendation (only connect to swing analysis if relevant)
    if user_swing_context and found_relevant_measurement and not is_non_biomechanics:
        # Only provide swing-analysis-specific advice if we found relevant measurements
        analysis_content = user_swing_context.replace("USER'S SWING ANALYSIS:\n", "").strip()
        response_parts.append("Based on your current measurements compared to professional standards, focus on implementing the expert advice above to address your specific swing characteristics.")
    else:
        # For non-biomechanics questions or when no relevant measurements found
        response_parts.append("Focus on implementing this expert advice to improve your technique.")
    
    # Combine all parts
    final_response = "\n\n".join(response_parts)
    
    # Add source reference
    final_response += f"\n\n📚 **Source**: {source_title}"
    
    return final_response

# Define functions
def validate_youtube_url(url):
    """Validate if the URL is a YouTube URL"""
    return "youtube.com" in url or "youtu.be" in url


def process_uploaded_video(uploaded_file):
    """Process an uploaded video file"""
    # Create downloads directory if it doesn't exist
    os.makedirs("downloads", exist_ok=True)

    # Save uploaded file to the downloads directory
    file_path = os.path.join("downloads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    return file_path


def display_video(video_path, width=300):
    """Display a video with download option"""
    # Read video bytes
    with open(video_path, "rb") as file:
        video_bytes = file.read()

    # Create a container with custom width
    video_container = st.container()
    # Apply CSS to control the width and ensure it's centered
    video_container.markdown(f"""
        <style>
        .element-container:has(video) {{
            max-width: {width}px;
            margin: 0 auto;
        }}
        video {{
            width: 100% !important;
            height: auto !important;
        }}
        </style>
        """,
                             unsafe_allow_html=True)

    # Display video using st.video with bytes
    with video_container:
        st.video(video_bytes)

    # Show download button
    st.download_button(label="Download Video",
                       data=video_bytes,
                       file_name=os.path.basename(video_path),
                       mime="video/mp4")


# Main app
def main():
    """Main Streamlit application"""
    st.title("Par-ity Project: Golf Swing Analysis 🏌️‍♀️")
    st.write("Founded to address the gender gap in golf participation and access to quality coaching resources, Par-ity Project is a technology-driven initiative empowering girls in golf through innovative AI based swing analysis. This technology uses computer vision and machine learning algorithms to analyze golf swings and provide personalized feedback to improve technique and performance.")
    
    # Initialize session state for storing analysis results
    if 'video_analyzed' not in st.session_state:
        st.session_state.video_analyzed = False
    if 'analysis_data' not in st.session_state:
        st.session_state.analysis_data = {
            'video_path': None,
            'frames': None,
            'detections': None,
            'pose_data': None,
            'swing_phases': None,
            'trajectory_data': None,
            'sample_rate': None
        }
    if 'show_chatbot' not in st.session_state:
        st.session_state.show_chatbot = False

    # Add session cleanup - clean up old files when starting a new session
    if 'session_initialized' not in st.session_state:
        cleanup_result = cleanup_downloads_directory(keep_annotated=True)
        if cleanup_result.get('files_removed', 0) > 0:
            st.info(f"🗑️ Cleaned up {cleanup_result['files_removed']} old files ({cleanup_result['space_freed_mb']} MB freed)")
        st.session_state.session_initialized = True

    # Automatic cleanup function
    def perform_cleanup():
        """Perform automatic cleanup of temporary files"""
        cleanup_result = cleanup_downloads_directory(keep_annotated=False)
        return cleanup_result

    # Set automatic defaults (no user configuration needed)
    # Check available LLM services and enable automatically if available
    llm_services = check_llm_services()
    any_service_available = llm_services['ollama']['available'] or llm_services['openai']['available']
    
    # Automatically enable LLM analysis if services are available
    enable_gpt = any_service_available
    
    # Set default frame processing rate (1 = all frames for best accuracy)
    sample_rate = 1
    
    # Disable pro comparison feature entirely
    enable_pro_comparison = False
    pro_url = None

    # Video input options
    st.header("Video Input")
    input_option = st.radio("Choose input method:",
                            ["YouTube URL", "Upload Video"])

    video_path = None
    analyze_clicked = False

    if input_option == "YouTube URL":
        youtube_url = st.text_input("Enter YouTube URL of golf swing:")

        analyze_clicked = st.button("Analyze Swing", key="analyze_youtube")
        if youtube_url and analyze_clicked:
            if validate_youtube_url(youtube_url):
                with st.spinner("Downloading video..."):
                    try:
                        video_path = download_youtube_video(youtube_url)
                        st.success("Video downloaded successfully!")
                        display_video(video_path, width=400)
                    except Exception as e:
                        st.error(f"Error downloading video: {str(e)}")
                        st.session_state.video_analyzed = False
                        return
            else:
                st.error("Please enter a valid YouTube URL")
                st.session_state.video_analyzed = False
                return

    else:  # Upload Video
        uploaded_file = st.file_uploader("Upload a golf swing video",
                                         type=["mp4", "mov", "avi"])

        analyze_clicked = st.button("Analyze Swing", key="analyze_upload")
        if uploaded_file and analyze_clicked:
            with st.spinner("Processing uploaded video..."):
                try:
                    video_path = process_uploaded_video(uploaded_file)
                    st.success("Video uploaded successfully!")
                    display_video(video_path, width=400)
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
                    st.session_state.video_analyzed = False
                    return
    
    # Download pro reference if enabled
    if enable_pro_comparison and (video_path or st.session_state.video_analyzed):
        if not st.session_state.pro_reference_path:
            with st.spinner("Downloading professional golfer reference..."):
                try:
                    pro_path = download_pro_reference(pro_url)
                    st.session_state.pro_reference_path = pro_path
                    st.success("Professional reference downloaded successfully!")
                except Exception as e:
                    st.error(f"Error downloading pro reference: {str(e)}")
                    st.session_state.pro_reference_path = None

    # Process video if available and analyze button was clicked
    if video_path and analyze_clicked:
        try:
            # Step 1: Process video and detect objects
            with st.spinner("Processing video and detecting objects..."):
                frames, detections = process_video(video_path,
                                                   sample_rate=sample_rate)
                st.success("Video processing complete!")

            # Step 2: Analyze pose
            with st.spinner("Analyzing golfer's pose..."):
                pose_data = analyze_pose(frames)
                st.success("Pose analysis complete")

            # Step 3: Segment swing into phases
            with st.spinner("Segmenting swing phases..."):
                swing_phases = segment_swing(pose_data,
                                             detections,
                                             sample_rate=sample_rate)

            # Step 4: Analyze trajectory and speed
            with st.spinner("Analyzing trajectory and speed..."):
                trajectory_data = analyze_trajectory(frames,
                                                     detections,
                                                     swing_phases,
                                                     sample_rate=sample_rate)

            # Prepare data for LLM regardless of whether GPT is enabled
            analysis_data = prepare_data_for_llm(pose_data, swing_phases,
                                                 trajectory_data)
            prompt = create_llm_prompt(analysis_data)

            # Store analysis data in session state
            st.session_state.video_analyzed = True
            st.session_state.analysis_data = {
                'video_path': video_path,
                'frames': frames,
                'detections': detections,
                'pose_data': pose_data,
                'swing_phases': swing_phases,
                'trajectory_data': trajectory_data,
                'sample_rate': sample_rate,
                'analysis_data': analysis_data,
                'prompt': prompt
            }

            # Keep the original video file for potential annotation
            # Video will be cleaned up when user uploads a new video or session ends

            # Present the options after analysis
            st.subheader("What would you like to do next?")
            options_col1, options_col2, options_col3, options_col4 = st.columns(4)

            with options_col1:
                st.info(
                    "**Option 1: Generate Annotated Video**\n\nCreate a video with visual feedback showing your swing phases, body positioning, and key metrics."
                )

            with options_col2:
                st.info(
                    "**Option 2: Generate Improvement Recommendations**\n\nGet AI-powered analysis of your swing with specific tips for improvement."
                )
                
            with options_col3:
                st.info(
                    "**Option 3: Key Frame Analysis**\n\nExtract and review your setup, top of backswing, and impact frames with helpful comments for each phase."
                )
            
            with options_col4:
                st.info(
                    "**Option 4: Golf Swing Chatbot**\n\nAsk specific questions about golf swing technique and get expert advice from our knowledge base."
                )

        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            st.session_state.video_analyzed = False
            # Clean up on error as well
            if video_path and os.path.exists(video_path):
                cleanup_video_file(video_path)

    # Show action buttons and their results (only if analysis is complete)
    if st.session_state.video_analyzed:
        # Display the GPT prompt in an expander
        if 'prompt' in st.session_state.analysis_data:
            with st.expander("View LLM Prompt", expanded=False):
                st.code(st.session_state.analysis_data['prompt'],
                        language="text")

        # Create columns for the action buttons
        button_col1, button_col2, button_col3, button_col4 = st.columns(4)

        with button_col1:
            annotated_video_clicked = st.button("Generate Annotated Video",
                                                key="create_annotated",
                                                use_container_width=True)

        with button_col2:
            improvements_clicked = st.button("Generate Improvements",
                                             key="gpt_recommendations",
                                             use_container_width=True)
        
        with button_col3:
            keyframe_analysis_clicked = st.button("Key Frame Analysis",
                                                 key="keyframe_analysis",
                                                 use_container_width=True)
        
        with button_col4:
            chatbot_clicked = st.button("Golf Swing Chatbot",
                                       key="rag_chatbot",
                                       use_container_width=True)

        # Handle annotated video creation
        if annotated_video_clicked:
            # Reset chatbot state when other buttons are clicked
            st.session_state.show_chatbot = False
            try:
                with st.spinner("Creating annotated video..."):
                    # Create downloads directory if it doesn't exist
                    os.makedirs("downloads", exist_ok=True)

                    # Get data from session state
                    data = st.session_state.analysis_data

                    # Create the annotated video
                    output_path = create_annotated_video(
                        data['video_path'],
                        data['frames'],
                        data['detections'],
                        data['pose_data'],
                        data['swing_phases'],
                        data['trajectory_data'],
                        sample_rate=data['sample_rate'])

                    # Verify the file exists
                    if not os.path.exists(output_path):
                        raise FileNotFoundError(
                            f"Annotated video file not found at {output_path}")

                    # Store the annotated video path in session state
                    st.session_state.annotated_video_path = output_path

                # Display success message and video after spinner completes
                st.success("Annotated video created successfully!")
                display_video(output_path, width=400)

                # Show download button
                with open(output_path, "rb") as file:
                    video_bytes = file.read()
                    st.download_button(label="Download Annotated Video",
                                       data=video_bytes,
                                       file_name=os.path.basename(output_path),
                                       mime="video/mp4")

            except Exception as e:
                st.error(f"Error creating annotated video: {str(e)}")
                st.error(
                    "Please check if the downloads directory exists and is writable"
                )

        # Handle improvement recommendations generation
        if improvements_clicked:
            # Reset chatbot state when other buttons are clicked
            st.session_state.show_chatbot = False
            with st.spinner(
                    "Analyzing your swing and generating recommendations..."):
                # Get data from session state
                data = st.session_state.analysis_data
                pose_data = data['pose_data']
                swing_phases = data['swing_phases']
                trajectory_data = data['trajectory_data']

                # Generate detailed analysis with recommendations
                analysis = generate_swing_analysis(pose_data, swing_phases,
                                                   trajectory_data)

                # Display the analysis
                st.subheader("Swing Analysis and Recommendations")

                # Check available services to show appropriate message
                llm_services = check_llm_services()
                any_service_available = llm_services['ollama'][
                    'available'] or llm_services['openai']['available']

                if not any_service_available or not enable_gpt:
                    st.info(
                        "ℹ️ **Using sample analysis mode**. The recommendations below are general examples and not personalized to your specific swing."
                    )
                else:
                    if llm_services['ollama']['available'] and llm_services[
                            'openai']['available']:
                        st.info(
                            "🔄 **Analysis generated using available LLM services** (tried Ollama first, OpenAI as fallback)"
                        )
                    elif llm_services['ollama']['available']:
                        st.info("🦙 **Analysis generated using Ollama**")
                    elif llm_services['openai']['available']:
                        st.info("🤖 **Analysis generated using OpenAI**")

                # Parse and display the formatted analysis instead of raw markdown
                if "Error:" not in analysis:
                    formatted_analysis = parse_and_format_analysis(analysis)
                    display_formatted_analysis(formatted_analysis)
                else:
                    # Show error message if analysis failed
                    st.error(analysis)
        
        # Handle key frame analysis (new tab/option)
        if keyframe_analysis_clicked:
            # Reset chatbot state when other buttons are clicked
            st.session_state.show_chatbot = False
            try:
                with st.spinner("Extracting key frames from your swing..."):
                    user_video_path = st.session_state.analysis_data['video_path']
                    user_swing_phases = st.session_state.analysis_data['swing_phases']
                    frames = st.session_state.analysis_data['frames']
                    key_frames = extract_key_swing_frames(user_video_path, frames, user_swing_phases)

                st.success("Key frame analysis complete!")
                st.subheader("Key Frame Analysis: Your Swing's Critical Positions")

                # Define helpful comments for each phase
                phase_comments = {
                    'setup': [
                        "Balanced stance with feet shoulder-width apart.",
                        "Even weight distribution on both feet.",
                        "Neutral grip with hands in proper position.",
                        "Athletic posture with slight forward bend.",
                        "Ball positioned correctly for club selection."
                    ],
                    'backswing': [
                        "Full shoulder rotation with stable lower body.",
                        "Club on proper swing plane at top.",
                        "Consistent spine angle throughout.",
                        "Minimal weight shift to right side."
                    ],
                    'impact': [
                        "Weight shifted to front foot (70-80%).",
                        "Hands ahead of ball at impact.",
                        "Square club face to target line.",
                        "Head behind ball with steady position.",
                        "Hips and shoulders aligned to target."
                    ]
                }
                phase_titles = {
                    'setup': 'Starting Position',
                    'backswing': 'Top of Backswing',
                    'impact': 'Impact with Ball'
                }
                phases = ['setup', 'backswing', 'impact']
                for phase in phases:
                    st.subheader(f"{phase_titles[phase]}")
                    img_col, comment_col = st.columns([1, 1])
                    with img_col:
                        if key_frames.get(phase) is not None:
                            frame = key_frames[phase]
                            
                            # Verify frame is in color before conversion
                            if len(frame.shape) == 3 and frame.shape[2] == 3:
                                try:
                                    # Save frame to temp file for display
                                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                                    
                                    # Convert BGR (OpenCV) to RGB (PIL) format
                                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    
                                    # Debug: Log frame dimensions after extraction and color conversion
                                    height, width = rgb_frame.shape[:2]
                                    print(f"Frame dimensions for {phase}: {width}x{height}")
                                    
                                    # Resize frame proportionally for better display
                                    # Target width of 400 pixels while maintaining aspect ratio
                                    target_width = 400
                                    aspect_ratio = height / width
                                    target_height = int(target_width * aspect_ratio)
                                    
                                    pil_img = Image.fromarray(rgb_frame)
                                    # Resize the image proportionally
                                    pil_img = pil_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                                    pil_img.save(temp_file.name, format="JPEG", quality=95)
                                    
                                    # Display the image with fixed width
                                    st.image(temp_file.name, width=target_width)
                                    
                                    # Clean up temp file
                                    try:
                                        os.unlink(temp_file.name)
                                    except:
                                        pass  # Ignore cleanup errors
                                        
                                except Exception as e:
                                    st.error(f"Error displaying {phase} frame: {str(e)}")
                                    st.warning("Frame could not be displayed properly.")
                            else:
                                st.warning(f"Frame for {phase} is not in color format. Shape: {frame.shape}")
                        else:
                            st.warning("Frame not found.")
                    with comment_col:
                        st.markdown("**Comments:**")
                        for comment in phase_comments[phase]:
                            st.markdown(f"- {comment}")
                    st.markdown("---")
            except Exception as e:
                st.error(f"Error during key frame analysis: {str(e)}")
                st.info("Please ensure your video is in a supported format and try again.")
        
        # Handle RAG chatbot
        if chatbot_clicked:
            st.session_state.show_chatbot = True
        
        # Always show chatbot interface if it's active
        if st.session_state.show_chatbot:
            # Create header with close button
            header_col1, header_col2 = st.columns([3, 1])
            with header_col1:
                st.subheader("Golf Swing Technique Chatbot")
            with header_col2:
                if st.button("✕ Close Chatbot", use_container_width=True):
                    st.session_state.show_chatbot = False
                    st.rerun()
            
            render_rag_interface()


if __name__ == "__main__":
    main()
