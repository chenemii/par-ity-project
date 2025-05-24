# Golf Swing Analysis

A tool for analyzing golf swings using computer vision and AI.

## Features

- Upload or provide YouTube links to golf swing videos
- Automated swing analysis using computer vision
- Pose estimation and tracking
- Swing phase segmentation
- Club and ball trajectory analysis
- LLM-powered swing analysis and coaching tips
- Annotated video generation
- Side-by-side comparison with professional golfer
- Improvement recommendations from AI analysis

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

5. Set your OpenAI API key as an environment variable:
   ```
   export OPENAI_API_KEY="your-api-key"
   ```

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
5. Compare your swing side-by-side with a professional golfer
6. Get AI-powered improvement recommendations

## Technical Details

The application uses:
- YOLOv8 for object detection
- MediaPipe for pose estimation
- OpenCV for video processing
- OpenAI GPT-4 for swing analysis
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