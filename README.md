---
title: Par-ity Project
emoji: ⛳
colorFrom: blue
colorTo: green
sdk: streamlit
app_file: app/streamlit_app.py
pinned: false
---

# Par-ity Project: Golf Swing Analysis with AI Assistant ⛳🏌️‍♀️

A comprehensive golf swing analysis platform that combines computer vision-based swing analysis with an AI-powered technique assistant. This integrated system provides both automated video analysis and expert knowledge retrieval for improving your golf swing.

## Features

### 🎥 Video Analysis
- Upload or provide YouTube links to golf swing videos
- Automated swing analysis using computer vision
- Pose estimation and tracking
- Swing phase segmentation (setup, backswing, downswing, follow-through)
- Club and ball trajectory analysis
- Annotated video generation with visual feedback
- Key position comparison (setup, top of backswing, impact)
- AI-powered improvement recommendations

### 🤖 Golf Swing Technique Assistant (RAG)
- **Expert Knowledge Base**: 2,000+ professional golf instruction articles
- **Semantic Search**: Ask questions in natural language
- **Contextual Answers**: Get detailed responses with source citations
- **Interactive Chat**: Build conversations about your swing technique
- **TPI Content**: Based on Titleist Performance Institute materials

## What You Can Do

### Video Analysis Options
After uploading a video, you get 4 analysis options:

1. **Generate Annotated Video** - Visual feedback showing swing phases and metrics
2. **Generate Improvement Recommendations** - AI-powered personalized tips
3. **Key Frame Analysis** - Detailed review of critical swing positions
4. **Golf Swing Chatbot** - Ask specific technique questions

### Example Questions for the AI Assistant
- "What wrist motion happens during the downswing?"
- "I'm having trouble with my slice, can you help?"
- "What should I focus on to increase my driving distance?"
- "How do I fix my inconsistent ball striking?"
- "What physical limitations can affect my swing?"

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Directory Setup

```bash
./setup_directories.sh
```

### 3. OpenAI API Key (Optional)

For enhanced AI responses, set up an OpenAI API key:

**Option 1: Environment File**
```bash
cp .env.example .env
# Edit .env and add your API key
```

**Option 2: Streamlit Secrets**
Create `.streamlit/secrets.toml`:
```toml
[openai]
api_key = "your-openai-api-key-here"
```

**Option 3: Enter in App**
You can also enter the API key directly in the Streamlit interface.

### 4. Run the Application

```bash
cd app
streamlit run streamlit_app.py
```

Or use the convenience script:
```bash
./run_streamlit.sh
```

## How It Works

### Video Analysis Pipeline
1. **Video Processing**: Extracts frames and detects objects using YOLOv8
2. **Pose Analysis**: Uses MediaPipe for detailed body positioning
3. **Swing Segmentation**: Identifies swing phases automatically
4. **Trajectory Analysis**: Tracks club and ball movement
5. **AI Recommendations**: Generates personalized improvement tips

### RAG (Retrieval-Augmented Generation) System
1. **Knowledge Processing**: Loads and processes 2,000+ golf instruction articles
2. **Semantic Embeddings**: Creates vector representations using Sentence Transformers
3. **Smart Search**: Uses FAISS for fast similarity search
4. **Response Generation**: Combines retrieved knowledge with AI (GPT-3.5) or fallback mode

## File Structure

```
Golf_Swing_Analysis/
├── app/                                # Main application
│   ├── streamlit_app.py               # Integrated Streamlit app
│   ├── golf_swing_rag.py             # RAG system
│   ├── models/                        # Analysis models
│   ├── utils/                         # Utility functions
│   └── components/                    # UI components
├── golf_swing_articles_complete.csv   # Knowledge base (2,000+ articles)
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment variables template
├── test_rag_integration.py           # Integration test script
└── Generated files (after first run):
    ├── golf_swing_embeddings.pkl     # Cached embeddings
    ├── golf_swing_index.faiss        # Vector search index
    └── downloads/                     # Processed videos
```

## Technical Details

### Technologies Used
- **Computer Vision**: YOLOv8, MediaPipe, OpenCV
- **AI/ML**: OpenAI GPT-3.5/4, Ollama (local LLM option)
- **RAG Stack**: Sentence Transformers, FAISS, LangChain
- **Web Interface**: Streamlit
- **Data Processing**: Pandas, NumPy

### Performance Features
- **Cached Embeddings**: First-time setup creates embeddings saved for future use
- **Efficient Search**: FAISS enables fast similarity search over thousands of chunks
- **Automatic Cleanup**: Temporary files are managed automatically
- **Batch Processing**: Video frames and embeddings processed efficiently

## Usage Guide

### 1. Video Analysis Workflow
1. Choose input method (YouTube URL or file upload)
2. Click "Analyze Swing" to process the video
3. Select from 4 analysis options
4. Download results and annotated videos

### 2. AI Assistant Workflow
1. Click "Golf Swing Chatbot" after video analysis (or use standalone)
2. Ask questions about golf swing technique
3. Review detailed answers with source citations
4. Build conversations for comprehensive understanding

## Example Use Cases

### Video Analysis
- **Beginner Golfer**: Upload practice swing video → Get annotated feedback → Learn proper positions
- **Intermediate Player**: Analyze driver swing → Get AI recommendations → Focus on specific improvements
- **Coach**: Use key frame analysis → Show students critical positions → Provide visual evidence

### AI Assistant
- **Technique Questions**: "How should my weight shift during the swing?"
- **Problem Solving**: "I keep hitting fat shots with my irons, what's wrong?"
- **Learning**: "Explain the biomechanics of the golf swing"
- **Specific Issues**: "I have limited hip mobility, how does this affect my swing?"

## Troubleshooting

### First Run Setup
- Initial embedding creation takes 5-10 minutes (one-time process)
- Ensure adequate RAM (8GB+ recommended) for large knowledge base
- Video processing time depends on length and resolution

### Common Issues
- **Missing Dependencies**: Run `pip install --upgrade -r requirements.txt`
- **Import Errors**: Ensure you're running from the correct directory
- **RAG Not Available**: Check that `golf_swing_articles_complete.csv` exists
- **Video Issues**: Ensure videos are in supported formats (MP4, MOV, AVI)

### Testing Integration
Run the test script to verify everything works:
```bash
python3 test_rag_integration.py
```

## Contributing

This system is designed to be extensible:

1. **Video Analysis**: Add new computer vision models or metrics
2. **Knowledge Base**: Include additional golf instruction sources
3. **AI Models**: Experiment with different embedding models or LLMs
4. **UI/UX**: Enhance the Streamlit interface with new features

## License

This project is for educational and personal use. The golf instruction content is sourced from publicly available articles and should be attributed to original sources.

---

**Built with ❤️ to empower golfers with AI-powered analysis and expert knowledge** 