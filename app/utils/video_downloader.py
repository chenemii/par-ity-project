"""
YouTube video downloader module using yt-dlp
"""

import os
import yt_dlp


def download_youtube_video(url, output_dir="downloads"):
    """
    Download a YouTube video from the provided URL using yt-dlp
    
    Args:
        url (str): YouTube video URL
        output_dir (str): Directory to save the downloaded video
        
    Returns:
        str: Path to the downloaded video file
        
    Raises:
        ValueError: If the URL is invalid or video is unavailable
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set output template for the downloaded file
    output_template = os.path.join(output_dir, "%(title)s.%(ext)s")

    # Configure yt-dlp options
    ydl_opts = {
        'format': 'best[ext=mp4]/best',  # Prefer mp4 format
        'outtmpl': output_template,
        'noplaylist': True,
        'quiet': False,
        'no_warnings': False,
        'ignoreerrors': False,
    }

    try:
        # Create yt-dlp object and download the video
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

            # Get the downloaded file path
            if 'entries' in info:
                # Playlist (should not happen with noplaylist=True)
                raise ValueError("Playlists are not supported")

            # Get video title and extension
            title = info.get('title', 'video')
            ext = info.get('ext', 'mp4')

            # Construct the file path
            video_path = os.path.join(output_dir, f"{title}.{ext}")

            # Check if file exists
            if not os.path.exists(video_path):
                # Try with sanitized filename
                sanitized_title = ''.join(c for c in title
                                          if c.isalnum() or c in ' ._-')
                video_path = os.path.join(output_dir,
                                          f"{sanitized_title}.{ext}")

                if not os.path.exists(video_path):
                    # If still not found, look for any mp4 file in the directory
                    mp4_files = [
                        f for f in os.listdir(output_dir) if f.endswith('.mp4')
                    ]
                    if mp4_files:
                        video_path = os.path.join(output_dir, mp4_files[0])
                    else:
                        raise ValueError("Downloaded file not found")

            return video_path

    except yt_dlp.utils.DownloadError as e:
        raise ValueError(f"Error downloading video: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error: {str(e)}")
