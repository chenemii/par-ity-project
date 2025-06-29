"""
YouTube video downloader module using yt-dlp
"""

import os
import yt_dlp


def cleanup_video_file(video_path):
    """
    Delete a specific video file after processing
    
    Args:
        video_path (str): Path to the video file to delete
        
    Returns:
        bool: True if file was deleted successfully, False otherwise
    """
    try:
        if os.path.exists(video_path):
            os.remove(video_path)
            print(f"Cleaned up video file: {video_path}")
            return True
        else:
            print(f"Video file not found for cleanup: {video_path}")
            return False
    except Exception as e:
        print(f"Error cleaning up video file {video_path}: {str(e)}")
        return False


def cleanup_downloads_directory(output_dir="downloads", keep_annotated=True):
    """
    Clean up downloaded videos from the downloads directory
    
    Args:
        output_dir (str): Directory containing downloaded videos
        keep_annotated (bool): Whether to keep annotated videos (default: True)
        
    Returns:
        dict: Cleanup results with files removed and space freed
    """
    try:
        if not os.path.exists(output_dir):
            return {"files_removed": 0, "space_freed_mb": 0}
            
        files_removed = 0
        space_freed = 0
        
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            
            # Skip if not a file
            if not os.path.isfile(file_path):
                continue
                
            # Skip annotated videos if keep_annotated is True
            if keep_annotated and "_annotated" in filename:
                continue
                
            # Skip pro reference videos (they can be reused)
            if "pro_reference" in filename:
                continue
                
            # Get file size before deletion
            try:
                file_size = os.path.getsize(file_path)
                space_freed += file_size
                
                # Remove the file
                os.remove(file_path)
                files_removed += 1
                print(f"Cleaned up: {filename}")
                
            except Exception as e:
                print(f"Error removing {filename}: {str(e)}")
                
        # Convert bytes to MB
        space_freed_mb = space_freed / (1024 * 1024)
        
        return {
            "files_removed": files_removed,
            "space_freed_mb": round(space_freed_mb, 2)
        }
        
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")
        return {"error": str(e)}


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
