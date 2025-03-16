#!/bin/bash

# Create downloads directory for the application
mkdir -p downloads

# Set permissions
chmod 755 downloads

echo "Directory created and permissions set:"
echo "- downloads: for storing downloaded YouTube videos and annotated videos"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file template..."
    echo "OPENAI_API_KEY=your_api_key_here" > .env
    echo ".env file created. Please edit it to add your OpenAI API key."
fi

echo "Setup complete!" 