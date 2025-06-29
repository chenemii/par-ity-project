#!/usr/bin/env python3
"""
Golf Swing Analysis - Main Application
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the app directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.video_downloader import download_youtube_video
from app.utils.video_processor import process_video
from app.models.pose_estimator import analyze_pose
from app.models.swing_analyzer import segment_swing, analyze_trajectory
from app.models.llm_analyzer import generate_swing_analysis
from app.utils.visualizer import create_annotated_video


def main():
    """Main application function"""
    print("\n===== Golf Swing Analysis =====\n")

    # Step 1: Get YouTube URL from user
    youtube_url = input("Enter YouTube URL of golf swing: ")

    # Step 2: Configure analysis options
    enable_gpt = input(
        "\nEnable GPT analysis? (y/n, default: y): ").lower() != 'n'

    sample_rate_input = input(
        "\nFrame processing rate for YOLO (1-10, default: 1 for all frames): ")
    sample_rate = 1  # Default value - process all frames
    if sample_rate_input.isdigit():
        sample_rate = max(1, min(10, int(sample_rate_input)))

    try:
        # Step 3: Download the video
        print("\nDownloading video...")
        video_path = download_youtube_video(youtube_url)
        print(f"Video downloaded to: {video_path}")

        # Step 4: Process video and detect golfer, club, and ball
        print("\nProcessing video and detecting objects...")
        frames, detections = process_video(video_path, sample_rate=sample_rate)

        # Step 5: Analyze pose throughout the swing
        print("\nAnalyzing golfer's pose...")
        pose_data = analyze_pose(frames)

        # Step 6: Segment swing into phases
        print("\nSegmenting swing phases...")
        swing_phases = segment_swing(pose_data,
                                     detections,
                                     sample_rate=sample_rate)

        # Step 7: Analyze trajectory and speed
        print("\nAnalyzing trajectory and speed...")
        trajectory_data = analyze_trajectory(frames,
                                             detections,
                                             swing_phases,
                                             sample_rate=sample_rate)

        # Step 8: Generate swing analysis using LLM (if enabled)
        if enable_gpt:
            print("\nGenerating swing analysis and coaching tips...")
            analysis = generate_swing_analysis(pose_data, swing_phases,
                                               trajectory_data)

            # Display results
            print("\n===== Swing Analysis Results =====\n")
            print(analysis)
        else:
            print("\nGPT analysis disabled. Skipping swing evaluation.")

        # Step 9: Create annotated video (optional)
        create_video = input(
            "\nCreate annotated video? (y/n): ").lower() == 'y'
        if create_video:
            print("\nCreating annotated video...")
            output_path = create_annotated_video(video_path,
                                                 frames,
                                                 detections,
                                                 pose_data,
                                                 swing_phases,
                                                 trajectory_data,
                                                 sample_rate=sample_rate)
            print(f"Annotated video saved to: {output_path}")

        print("\nAnalysis complete!")

    except Exception as e:
        print(f"\nError: {str(e)}")
        return


if __name__ == "__main__":
    main()
