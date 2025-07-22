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

# Set page config
st.set_page_config(page_title="Par-ity Project: Golf Swing Analysis 🏌️‍♀️",
                   page_icon="🏌️‍♀️",
                   layout="wide",
                   initial_sidebar_state="collapsed")


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
            options_col1, options_col2, options_col3 = st.columns(3)

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
        button_col1, button_col2, button_col3 = st.columns(3)

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

        # Handle annotated video creation
        if annotated_video_clicked:
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


if __name__ == "__main__":
    main()
