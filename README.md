# Golf Swing Analysis

A Python application that analyzes golf swings from YouTube videos using computer vision and AI.

## Features

- YouTube video retrieval and processing using yt-dlp
- Golfer, club, and ball detection using YOLOv8
- Pose estimation for swing analysis
- Swing phase segmentation (setup, backswing, downswing, impact, follow-through)
- Trajectory and speed analysis
- AI-powered swing evaluation and coaching tips
- Visual feedback with annotations
- Streamlit web interface

## Installation

1. Clone this repository
2. Run the setup script to create necessary directories:
   ```
   chmod +x setup_directories.sh
   ./setup_directories.sh
   ```
3. Create a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Edit the `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

### Command Line Interface

Run the main application:

```
python app/main.py
```

Follow the prompts to input a YouTube URL containing a golf swing recording.

### Streamlit Web Interface

Run the Streamlit web app using the provided shell script:

```
./run_streamlit.sh
```

Or manually with:

```
source .venv/bin/activate
 
```

The web interface provides:
- Options to upload a video or use a YouTube URL
- Control over frame skip rate for YOLO detection
- Toggle for enabling/disabling GPT analysis
- Interactive display of analysis results
- Option to create and view annotated videos

## File Organization

- **downloads/**: Contains both downloaded YouTube videos and annotated videos
- All videos (both original and annotated) are stored in the same directory for easy access

## Troubleshooting

If you encounter issues with the "Create Annotated Video" button:
1. Make sure you've run the setup script to create the downloads directory
2. Check that the `downloads` directory has write permissions
3. Try restarting the Streamlit app

## Requirements

- Python 3.8+
- OpenCV
- YOLOv8
- MediaPipe
- yt-dlp
- OpenAI API key
- Streamlit 