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

# Load environment variables
load_dotenv()

# Add the app directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.video_downloader import download_youtube_video
from app.utils.video_processor import process_video
from app.models.pose_estimator import analyze_pose
from app.models.swing_analyzer import segment_swing, analyze_trajectory
from app.models.llm_analyzer import generate_swing_analysis, create_llm_prompt, prepare_data_for_llm
from app.utils.visualizer import create_annotated_video

# Set page config
st.set_page_config(page_title="Golf Swing Analysis",
                   page_icon="🏌️",
                   layout="wide",
                   initial_sidebar_state="expanded")


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


def display_video(video_path):
    """Display a video with download option"""
    # Read video bytes
    with open(video_path, "rb") as file:
        video_bytes = file.read()

    # Display video using st.video with bytes
    st.video(video_bytes)

    # Show download button
    st.download_button(label="Download Video",
                       data=video_bytes,
                       file_name=os.path.basename(video_path),
                       mime="video/mp4")


# Main app
def main():
    """Main Streamlit application"""
    st.title("🏌️ Golf Swing Analysis")
    st.write("Analyze your golf swing using computer vision and AI")

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

    # Sidebar for configuration
    st.sidebar.title("Configuration")

    # Option to enable/disable GPT analysis
    enable_gpt = st.sidebar.checkbox("Enable GPT Analysis", value=True)

    # Frame skip rate for YOLO
    sample_rate = st.sidebar.slider(
        "Frame Skip Rate (YOLO)",
        min_value=1,
        max_value=10,
        value=5,
        help=
        "Process every Nth frame. Higher values = faster but less accurate.")

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
                        display_video(video_path)
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
                    display_video(video_path)
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
                    st.session_state.video_analyzed = False
                    return

    # Process video if available and analyze button was clicked
    if video_path and analyze_clicked:
        try:
            # Step 1: Process video and detect objects
            with st.spinner("Processing video and detecting objects..."):
                frames, detections = process_video(video_path,
                                                   sample_rate=sample_rate)
                st.success(f"Processed {len(frames)} frames")

            # Step 2: Analyze pose
            with st.spinner("Analyzing golfer's pose..."):
                pose_data = analyze_pose(frames)
                st.success("Pose analysis complete")

            # Step 3: Segment swing into phases
            with st.spinner("Segmenting swing phases..."):
                swing_phases = segment_swing(pose_data,
                                             detections,
                                             sample_rate=sample_rate)

                # Display swing phases
                st.subheader("Swing Phases")
                phase_cols = st.columns(5)
                for i, (phase,
                        frames_in_phase) in enumerate(swing_phases.items()):
                    with phase_cols[i]:
                        st.metric(label=phase.capitalize(),
                                  value=f"{len(frames_in_phase)} frames")

            # Step 4: Analyze trajectory and speed
            with st.spinner("Analyzing trajectory and speed..."):
                trajectory_data = analyze_trajectory(frames,
                                                     detections,
                                                     swing_phases,
                                                     sample_rate=sample_rate)

                # Display club speed if available
                impact_frames = swing_phases.get("impact", [])
                if impact_frames:
                    impact_frame = impact_frames[len(impact_frames) // 2]
                    if impact_frame in trajectory_data and trajectory_data[
                            impact_frame].get("club_speed"):
                        st.subheader("Club Speed")
                        st.metric(
                            label="Estimated Club Speed",
                            value=
                            f"{trajectory_data[impact_frame]['club_speed']:.1f} mph"
                        )

            # Step 5: Generate swing analysis using LLM (if enabled)
            # Prepare data for LLM regardless of whether GPT is enabled
            analysis_data = prepare_data_for_llm(pose_data, swing_phases,
                                                 trajectory_data)
            prompt = create_llm_prompt(analysis_data)

            # Display the GPT prompt
            with st.expander("View GPT Prompt"):
                st.code(prompt, language="text")

            if enable_gpt:
                with st.spinner(
                        "Generating swing analysis and coaching tips..."):
                    analysis = generate_swing_analysis(pose_data, swing_phases,
                                                       trajectory_data)

                    # Display analysis
                    st.subheader("Swing Analysis")
                    st.write(analysis)
            else:
                st.info(
                    "GPT Analysis is disabled. Enable it in the sidebar to generate coaching tips."
                )

            # Store analysis data in session state
            st.session_state.video_analyzed = True
            st.session_state.analysis_data = {
                'video_path': video_path,
                'frames': frames,
                'detections': detections,
                'pose_data': pose_data,
                'swing_phases': swing_phases,
                'trajectory_data': trajectory_data,
                'sample_rate': sample_rate
            }

        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            st.session_state.video_analyzed = False

    # Create annotated video button (only show if analysis is complete)
    if st.session_state.video_analyzed:
        st.header("Create Annotated Video")
        st.write(
            "Create a video with annotations showing the analysis results")

        # Create a separate section for the annotated video
        if st.button("Generate Annotated Video", key="create_annotated"):
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

                    st.success("Annotated video created successfully!")

                    # Display the video with download option
                    display_video(output_path)

            except Exception as e:
                st.error(f"Error creating annotated video: {str(e)}")
                st.error(
                    "Please check if the downloads directory exists and is writable"
                )


if __name__ == "__main__":
    main()
