---
title: Par-ity Project
emoji: ⛳
colorFrom: blue
colorTo: green
sdk: streamlit
app_file: app/streamlit_app.py
pinned: false
---

# Golf Swing Analysis

A tool for analyzing golf swings using computer vision and AI.

## Features

- Upload or provide YouTube links to golf swing videos
- Automated swing analysis using computer vision
- Pose estimation and tracking
- Swing phase segmentation
- Club and ball trajectory analysis
- LLM-powered swing analysis and coaching tips (OpenAI GPT-4/3.5 or local Ollama models)
- Annotated video generation
- Key position comparison with professional golfer (3 critical swing positions)
- Detailed improvement recommendations with visual analysis

## Setup

1. Clone the repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Set up the necessary directories:
   ```
   ./setup_directories.sh
   ```
4. Add a reference professional golfer video:
   - Save a video of a professional golfer's swing as `pro_golfer.mp4` in the `downloads` directory
   - This will be used for the side-by-side comparison feature

5. Set up LLM services for analysis (optional):

   **Option 1: OpenAI**
   - Set your OpenAI API key in `.streamlit/secrets.toml`:
     ```toml
     [openai]
     api_key = "your-openai-api-key-here"
     ```

   **Option 2: Ollama (Local LLM)**
   - Install and run Ollama locally: https://ollama.ai/
   - Configure in `.streamlit/secrets.toml`:
     ```toml
     [ollama]
     base_url = "http://localhost:11434/v1"
     model = "llama2"  # or your preferred model
     ```

   **Option 3: Both Services**
   - Configure both in `.streamlit/secrets.toml` for automatic fallback
   - The app will try Ollama first, then OpenAI if Ollama fails

   **No Configuration**
   - The app works without any LLM configuration using sample analysis mode

   See `.streamlit/secrets.toml.example` for a complete configuration template.

## Running the Application

Run the Streamlit app:
```
./run_streamlit.sh
```

Or manually:
```
streamlit run app/streamlit_app.py
```

## Usage

1. Upload a golf swing video or provide a YouTube URL
2. Click "Analyze Swing" to process the video
3. View the swing phase breakdown and metrics
4. Generate an annotated video showing the analysis
5. Compare your swing at 3 key positions with a professional golfer:
   - Starting position (setup)
   - Top of backswing
   - Impact with ball
6. Get detailed improvement recommendations for each swing phase
7. Download comparison images and analysis results

## Technical Details

The application uses:
- YOLOv8 for object detection
- MediaPipe for pose estimation
- OpenCV for video processing
- OpenAI GPT-4/3.5 or Ollama for swing analysis
- Streamlit for the web interface

## Directory Structure

- `app/`: Main application code
  - `models/`: Analysis models
  - `utils/`: Utility functions
  - `components/`: UI components
  - `streamlit_app.py`: Main Streamlit application
- `downloads/`: Downloaded and processed videos
- `requirements.txt`: Required Python packages
- `setup_directories.sh`: Script to set up required directories
- `run_streamlit.sh`: Script to run the Streamlit app

## Notes

- For best results, use videos where the golfer is clearly visible
- Side view videos work best for analysis
- Processing time depends on video length and resolution 