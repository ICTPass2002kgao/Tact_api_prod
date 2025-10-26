from rest_framework import serializers

class VideoUploadSerializer(serializers.Serializer):
    """
    Serializer for uploading a video file.
    The 'video_file' key must be used in the POST request.
    """
    video_file = serializers.FileField(
        help_text="The video file to convert (e.g., .mp4, .mov)."
    )