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


def download_pro_reference(url="https://www.youtube.com/shorts/geR666LWSHg", output_dir="downloads"):
    """
    Download a professional golfer reference video
    
    Args:
        url (str): YouTube video URL of professional golfer (default: provided reference)
        output_dir (str): Directory to save the downloaded video
        
    Returns:
        str: Path to the downloaded pro reference video file
    """
    try:
        # Create a specific filename for the pro reference
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if pro reference already exists to avoid re-downloading
        pro_file_path = os.path.join(output_dir, "pro_reference.mp4")
        if os.path.exists(pro_file_path):
            return pro_file_path
            
        # Set output template for the downloaded file with fixed name
        output_template = os.path.join(output_dir, "pro_reference.%(ext)s")
        
        # Configure yt-dlp options
        ydl_opts = {
            'format': 'best[ext=mp4]/best',  # Prefer mp4 format
            'outtmpl': output_template,
            'noplaylist': True,
            'quiet': False,
            'no_warnings': False,
            'ignoreerrors': False,
        }
        
        # Create yt-dlp object and download the video
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download=True)
            
        # Check if file exists with mp4 extension
        if os.path.exists(pro_file_path):
            return pro_file_path
        else:
            # Try other extensions
            for ext in ['webm', 'mkv']:
                alt_path = os.path.join(output_dir, f"pro_reference.{ext}")
                if os.path.exists(alt_path):
                    return alt_path
                    
            # If still not found, download as normal video and rename
            video_path = download_youtube_video(url, output_dir)
            ext = os.path.splitext(video_path)[1]
            new_path = os.path.join(output_dir, f"pro_reference{ext}")
            os.rename(video_path, new_path)
            return new_path
            
    except Exception as e:
        raise ValueError(f"Error downloading pro reference: {str(e)}")
