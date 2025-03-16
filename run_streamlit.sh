#!/bin/bash

# Activate the virtual environment
source .venv/bin/activate

# Run the Streamlit app
streamlit run app/streamlit_app.py

# Deactivate the virtual environment when done
deactivate 