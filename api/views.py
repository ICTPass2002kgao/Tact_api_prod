import os
import tempfile
import urllib.request # For downloading files from the provided URL
from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from deepface import DeepFace  
from django.core.files.base import ContentFile # For handling downloaded files

# --- Utility Function to Handle DeepFace Verification ---

def perform_deepface_verification(file1_path, file2_path):
    """
    Performs face verification using DeepFace and handles common errors.
    """
    try:
        # NOTE: 'Facenet' is commonly used for verification. 
        # DeepFace will lazily download this model if it doesn't exist.
        result = DeepFace.verify(
            img1_path=file1_path,
            img2_path=file2_path,
            model_name='Facenet', # Recommended for high accuracy verification
            enforce_detection=True
        )
        return {
            'matched': result.get('verified', False),
            'distance': result.get('distance', 0.0),
            'threshold': result.get('threshold', 0.0),
            'error': None
        }
    except ValueError as e:
        # This typically catches cases where DeepFace cannot detect a face in one or both images
        error_message = str(e)
        return {
            'matched': False,
            'distance': -1,
            'threshold': -1,
            'error': f'Face detection failed: {error_message}'
        }
    except Exception as e:
        # Catch any other unexpected DeepFace errors
        return {
            'matched': False,
            'distance': -1,
            'threshold': -1,
            'error': f'DeepFace processing error: {str(e)}'
        }

def download_file_from_url(url, path):
    """Downloads a file from a URL to a specified path."""
    try:
        urllib.request.urlretrieve(url, path)
        return True
    except Exception as e:
        print(f"Error downloading file from URL {url}: {e}")
        return False


@api_view(['POST'])
def recognize_face(request):
    """
    Receives an image file (live camera image) and a reference URL 
    and uses DeepFace to verify if the faces match.
    """
    # 1. Get the files/fields from the request
    camera_uploaded_file = request.FILES.get('live_image')
    reference_image_url = request.data.get('reference_url') # The Flutter code sends the URL in 'reference_url' field

    # 2. Validation: Check if both are present
    if not camera_uploaded_file:
        return Response({'error': 'Missing required file: live_image (from camera)'}, 
                        status=status.HTTP_400_BAD_REQUEST)
    if not reference_image_url:
        return Response({'error': 'Missing required field: reference_url'}, 
                        status=status.HTTP_400_BAD_REQUEST)

    # 3. Process Files using a robust temporary path approach
    temp_dir = tempfile.gettempdir()
    
    # Define paths for the temporary files
    camera_temp_path = os.path.join(temp_dir, f'camera_{os.getpid()}_{os.urandom(4).hex()}.jpg')
    firebase_temp_path = os.path.join(temp_dir, f'firebase_{os.getpid()}_{os.urandom(4).hex()}.jpg')

    # Write the uploaded content (live image) to the temporary file
    try:
        with open(camera_temp_path, 'wb+') as f:
            for chunk in camera_uploaded_file.chunks():
                f.write(chunk)
        
        # Download the reference image from the provided URL
        download_success = download_file_from_url(reference_image_url, firebase_temp_path)
        if not download_success:
             return Response({'error': f'Could not download reference image from URL: {reference_image_url}'}, 
                             status=status.HTTP_400_BAD_REQUEST)

        # 4. Perform Verification
        verification_result = perform_deepface_verification(
            camera_temp_path, 
            firebase_temp_path
        )
        
    except Exception as e:
        # Catch file saving/disk errors
        return Response({'error': f'Processing failed: {str(e)}'}, 
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    finally:
        # 5. Cleanup: Always delete the temporary files
        if os.path.exists(camera_temp_path):
            os.remove(camera_temp_path)
        if os.path.exists(firebase_temp_path):
            os.remove(firebase_temp_path)

    # 6. Return Response
    if verification_result['error']:
        # Return a 200 OK with matched=False if detection failed (to be handled gracefully by client)
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

# --- NOTE FOR DJANGO USAGE ---
# You need to register this view in your Django project's urls.py:
#
# # urls.py example:
# from django.urls import path
# from . import views
# 
# urlpatterns = [
#     path('api/verify_faces/', views.recognize_face, name='recognize_face'),
# ]
#
# And ensure you have installed required packages:
# pip install djangorestframework deepface numpy opencv-python