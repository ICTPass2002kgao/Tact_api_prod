import os
from moviepy import VideoFileClip

def convert_video_to_audio(video_path: str, output_dir: str) -> str:
    """Converts a video file to an MP3 audio file."""
    try:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_filename = f"{base_name}.mp3"
        output_path = os.path.join(output_dir, output_filename)

        clip = VideoFileClip(video_path)
        audio_clip = clip.audio
        
        # Write the audio to an MP3 file
        audio_clip.write_audiofile(output_path)

        audio_clip.close()
        clip.close()

        return output_path

    except Exception as e:
        # Re-raise the exception to be caught in the view
        raise Exception(f"Conversion Error: {e}")