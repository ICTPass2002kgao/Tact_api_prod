import os
import tempfile
import urllib.request
import mimetypes  # ADDED: for setting Content-Type
from django.conf import settings
from django.http import FileResponse # ADDED: for streaming the file
import face_recognition
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
 
from .serializers import VideoUploadSerializer
from .utils import convert_video_to_audio

# --- Utility Function to Download File ---

def download_file_from_url(url, path):
    """Downloads a file from a public URL (Firebase, etc.) to a specified path."""
    try:
        urllib.request.urlretrieve(url, path)
        return True
    except Exception as e:
        # Log the specific download error for debugging
        print(f"Error downloading file from URL {url}: {e}")
        return False

# --- Core Face Verification Function using face_recognition (UPDATED FOR HOG) ---

def perform_face_recognition_verification(file1_path, file2_path):
    """
    Performs face verification by splitting detection and encoding, using the HOG model
    for faster face location detection, helping to prevent Railway timeouts.
    """
    
    try:
        # 1. Load the images
        img_live = face_recognition.load_image_file(file1_path)
        img_reference = face_recognition.load_image_file(file2_path)

        # 2. Face DETECTION (***CRITICAL CHANGE: SWITCHED FROM "cnn" TO "hog" FOR SPEED***)
        live_locations = face_recognition.face_locations(img_live, model="cnn")
        ref_locations = face_recognition.face_locations(img_reference, model="cnn")

        # 3. Validation: Check if a face was detected
        if not live_locations:
            error_message = "Face detection failed in the live camera image (HOG model). Ensure good lighting/pose."
            return {'matched': False, 'distance': -1, 'threshold': 0.6, 'error': error_message}
        
        if not ref_locations:
            error_message = "Face detection failed in the Firebase reference image (HOG model). Check image quality."
            return {'matched': False, 'distance': -1, 'threshold': 0.6, 'error': error_message}


        # 4. Face ENCODING (Pass the detected locations to the encoding function)
        encoding_live = face_recognition.face_encodings(img_live, known_face_locations=live_locations)
        encoding_reference = face_recognition.face_encodings(img_reference, known_face_locations=ref_locations)
        
        # Check if the encoding process failed
        if not encoding_live or not encoding_reference:
            error_message = "Face found, but feature extraction (encoding) failed unexpectedly."
            return {'matched': False, 'distance': -1, 'threshold': 0.6, 'error': error_message}
            
        # 5. Compare faces
        match_results = face_recognition.compare_faces(
            [encoding_reference[0]], # List of known faces (the reference)
            encoding_live[0]         # The unknown face (the live capture)
        )
        matched = match_results[0]

        # Calculate face distance
        face_distances = face_recognition.face_distance(
            [encoding_reference[0]], 
            encoding_live[0]
        )
        distance = face_distances[0]
        
        return {
            'matched': matched,
            'distance': float(distance),
            'threshold': 0.6, 
            'error': None
        }

    except Exception as e:
        # Catch any critical internal processing errors
        return {
            'matched': False,
            'distance': -1,
            'threshold': -1,
            'error': f'Face Recognition processing failed unexpectedly: {str(e)}'
        }


@api_view(['POST'])
def recognize_face(request):
    """
    Handles the POST request for face verification, receiving a live image file and a reference URL.
    """
    # 1. Get the files/fields from the request
    camera_uploaded_file = request.FILES.get('live_image')
    reference_image_url = request.data.get('reference_url')

    # 2. Validation: Check if both are present
    if not camera_uploaded_file:
        return Response({'error': 'Missing required file: live_image (from camera)'}, 
                        status=status.HTTP_400_BAD_REQUEST)
    if not reference_image_url:
        return Response({'error': 'Missing required field: reference_url'}, 
                        status=status.HTTP_400_BAD_REQUEST)

    # 3. Setup robust temporary file paths
    temp_dir = tempfile.gettempdir()
    unique_id = os.urandom(4).hex()
    
    camera_temp_path = os.path.join(temp_dir, f'camera_{os.getpid()}_{unique_id}_live.jpg')
    firebase_temp_path = os.path.join(temp_dir, f'firebase_{os.getpid()}_{unique_id}_ref.jpg')

    verification_result = None

    try:
        # Save uploaded live image to temp file
        with open(camera_temp_path, 'wb+') as f:
            for chunk in camera_uploaded_file.chunks():
                f.write(chunk)
        
        # Download reference image from URL to temp file
        download_success = download_file_from_url(reference_image_url, firebase_temp_path)
        if not download_success:
             return Response({'error': f'Could not download reference image from URL: {reference_image_url}'}, 
                             status=status.HTTP_400_BAD_REQUEST)

        # 4. Perform Verification
        verification_result = perform_face_recognition_verification(
            camera_temp_path, 
            firebase_temp_path
        )
        
    except Exception as e:
        # Catch file saving/IO errors
        return Response({'error': f'File or IO processing failed: {str(e)}'}, 
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    finally:
        # 5. Cleanup: Always delete the temporary files
        if os.path.exists(camera_temp_path):
            os.remove(camera_temp_path)
        if os.path.exists(firebase_temp_path):
            os.remove(firebase_temp_path)

    # 6. Return Response
    if verification_result['error']:
        # Return 200 OK with matched=False on failure, but include the error message
        return Response({
            'matched': False,
            'distance': verification_result['distance'],
            'threshold': verification_result['threshold'],
            'message': verification_result['error']
        }, status=status.HTTP_200_OK) 
    
    return Response({
        'matched': verification_result['matched'],
        'distance': verification_result['distance'],
        'threshold': verification_result['threshold'],
        'message': "Face verified successfully." if verification_result['matched'] else "Faces do not match."
    }, status=status.HTTP_200_OK)


# -----------------------------------------------------------------------------
# *** UPDATED FUNCTION TO STREAM AUDIO FILE DIRECTLY IN THE RESPONSE ***
# -----------------------------------------------------------------------------
@api_view(["POST"])
def convert_video_to_audio_api(request):
    """
    Receives a video file, converts it to audio, and streams the audio file back
    in the response body.
    """
    # 1. Validate input data using the Serializer
    serializer = VideoUploadSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    video_file = serializer.validated_data['video_file']
    
    # Define paths
    upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
    conversion_dir = os.path.join(settings.MEDIA_ROOT, 'audio_output')
    
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(conversion_dir, exist_ok=True)
    
    # Use a unique name for the temporary video file
    unique_id = os.urandom(4).hex()
    temp_video_path = os.path.join(upload_dir, f'video_{unique_id}_{video_file.name}')
    
    converted_audio_path = None

    try:
        # 2. Save the uploaded file temporarily
        with open(temp_video_path, 'wb+') as destination:
            for chunk in video_file.chunks():
                destination.write(chunk)

        # 3. Perform the conversion
        converted_audio_path = convert_video_to_audio(
            video_path=temp_video_path,
            output_dir=conversion_dir
        )
        
        # 4. Prepare the File Response
        audio_filename = os.path.basename(converted_audio_path)
        
        # Determine content type (should be 'audio/mpeg' for mp3)
        content_type, encoding = mimetypes.guess_type(converted_audio_path)
        if content_type is None:
             content_type = 'application/octet-stream'

        # Open the file in binary mode for streaming
        audio_file_handle = open(converted_audio_path, 'rb')

        # Stream the file back to the client
        response = FileResponse(
            audio_file_handle, 
            content_type=content_type,
            status=status.HTTP_201_CREATED # Use 201 to indicate creation
        )
        
        # Set headers for downloading/saving the file
        response['Content-Length'] = os.path.getsize(converted_audio_path)
        response['Content-Disposition'] = f'attachment; filename="{audio_filename}"'

        # 5. Clean up the temporary video file
        os.remove(temp_video_path)
        
        # NOTE: The converted audio file is left in MEDIA_ROOT for now.
        # In a production app, you'd use a signal or a deferred task (e.g., Celery) 
        # to delete it *after* the client has finished streaming/downloading.
        
        # 6. Return the file stream
        return response

    except Exception as e:
        # Ensure cleanup of any temp files if they exist
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if converted_audio_path and os.path.exists(converted_audio_path):
            # Optionally delete the failed conversion file
            os.remove(converted_audio_path) 
            
        return Response({
            'status': 'error',
            'message': f'Conversion failed: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)