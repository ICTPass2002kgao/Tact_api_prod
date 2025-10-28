import os
import tempfile
import urllib.request
import mimetypes 
import cv2  # NEW: For image decoding and processing
import numpy as np  # NEW: For array handling
from sklearn.metrics.pairwise import cosine_similarity  # NEW: For comparison
from insightface.app import FaceAnalysis  # NEW: The InsightFace library
from django.conf import settings
from django.http import FileResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from .serializers import VideoUploadSerializer
from .utils import convert_video_to_audio

# Global InsightFace Model Initialization
# This helps prevent re-loading the model for every single request,
# improving performance significantly.
# NOTE: In a production environment with multiple workers (e.g., Gunicorn), 
# this will be loaded once per worker process.
try:
    # Use "buffalo_l" model which is good for verification
    GLOBAL_FACE_APP = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    GLOBAL_FACE_APP.prepare(ctx_id=0)
    print("InsightFace model loaded successfully.")
except Exception as e:
    GLOBAL_FACE_APP = None
    print(f"Error loading InsightFace model: {e}")

# --- Utility Function to Download File (UNCHANGED) ---

def download_file_from_url(url, path):
    """Downloads a file from a public URL (Firebase, etc.) to a specified path."""
    try:
        urllib.request.urlretrieve(url, path)
        return True
    except Exception as e:
        # Log the specific download error for debugging
        print(f"Error downloading file from URL {url}: {e}")
        return False

# --- Core Face Verification Function using INSIGHTFACE ---

def perform_face_recognition_verification(file1_path, file2_path):
    """
    Performs face verification using InsightFace.
    """
    if GLOBAL_FACE_APP is None:
        return {
            'matched': False,
            'similarity': -1,
            'threshold': -1,
            'error': 'Face recognition service is unavailable (model failed to load).'
        }
    
    # Define a high-confidence threshold for verification
    VERIFICATION_THRESHOLD = 0.45 
    
    try:
        # --- Helper to load and process a face image ---
        def get_face_embedding(file_path, label):
            # 1. Load image using OpenCV
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError(f"Could not load image file from path: {file_path}")

            # 2. Get faces and embeddings using InsightFace
            faces = GLOBAL_FACE_APP.get(img)

            # 3. Strict Validation: Check for exactly one face
            if len(faces) == 0:
                error_message = f"Face detection failed in the {label} image. Ensure face is centered and clear."
                return None, error_message
            
            if len(faces) > 1:
                error_message = f"Multiple faces detected ({len(faces)}) in the {label} image. Only one person must be in the frame."
                return None, error_message
            
            # 4. Return the single face's embedding
            return faces[0].embedding, None

        # 1. Process Live Image
        encoding_live, error_live = get_face_embedding(file1_path, "live camera")
        if error_live:
            return {'matched': False, 'similarity': -1, 'threshold': VERIFICATION_THRESHOLD, 'error': error_live}

        # 2. Process Reference Image
        encoding_reference, error_reference = get_face_embedding(file2_path, "Firebase reference")
        if error_reference:
            return {'matched': False, 'similarity': -1, 'threshold': VERIFICATION_THRESHOLD, 'error': error_reference}
            
        # 3. Compare faces using Cosine Similarity
        # Reshape embeddings to (1, N) for numpy/sklearn comparison
        ref_vector = encoding_reference.reshape(1, -1)
        live_vector = encoding_live.reshape(1, -1)
        
        # Calculate similarity (cosine similarity ranges from -1 to 1)
        similarity = cosine_similarity(ref_vector, live_vector)[0][0]
        
        matched = similarity > VERIFICATION_THRESHOLD

        return {
            'matched': matched,
            'similarity': float(similarity),
            'threshold': VERIFICATION_THRESHOLD, 
            'error': None
        }

    except Exception as e:
        # Catch any critical internal processing errors
        return {
            'matched': False,
            'similarity': -1,
            'threshold': -1,
            'error': f'InsightFace processing failed unexpectedly: {str(e)}'
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
    
    # Use the appropriate extension (e.g., .jpg)
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
            'distance': verification_result['similarity'], # NOTE: Using similarity in distance field for Flutter app compatibility
            'threshold': verification_result['threshold'],
            'message': verification_result['error']
        }, status=status.HTTP_200_OK) 
    
    return Response({
        'matched': verification_result['matched'],
        'distance': verification_result['similarity'], # NOTE: Using similarity in distance field for Flutter app compatibility
        'threshold': verification_result['threshold'],
        'message': "Face verified successfully." if verification_result['matched'] else "Faces do not match."
    }, status=status.HTTP_200_OK)


# -----------------------------------------------------------------------------
# *** convert_video_to_audio_api REMAINS UNCHANGED ***
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
        
        # 6. Return the file stream
        return response

    except Exception as e:
        # Ensure cleanup of any temp files if they exist
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if converted_audio_path and os.path.exists(converted_audio_path):
            os.remove(converted_audio_path) 
            
        return Response({
            'status': 'error',
            'message': f'Conversion failed: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)