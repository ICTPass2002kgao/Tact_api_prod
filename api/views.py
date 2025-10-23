import os
import tempfile
import urllib.request
import numpy as np # Needed for array handling with face_recognition
import face_recognition # NEW: Import the face_recognition library
from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

# NOTE: Since we are no longer using DeepFace, we remove its import.
# from deepface import DeepFace 
# from django.core.files.base import ContentFile 

# --- Utility Function to Download File ---
def download_file_from_url(url, path):
    """Downloads a file from a URL to a specified path."""
    try:
        # NOTE: If your reference URLs are Firebase or similar, ensure they are publicly accessible 
        # or that you handle authentication headers if needed. urllib.request is for public URLs.
        urllib.request.urlretrieve(url, path)
        return True
    except Exception as e:
        print(f"Error downloading file from URL {url}: {e}")
        return False

# --- Core Face Verification Function using face_recognition ---
def perform_face_recognition_verification(file1_path, file2_path):
    """
    Performs face verification using the face_recognition library.
    """
    try:
        # 1. Load the images
        img_live = face_recognition.load_image_file(file1_path)
        img_reference = face_recognition.load_image_file(file2_path)

        # 2. Get the face encodings
        # face_encodings() returns a list of 128-dimensional encodings found in the image.
        encoding_live = face_recognition.face_encodings(img_live)
        encoding_reference = face_recognition.face_encodings(img_reference)
        
        # 3. Validation: Check if a face was detected in both
        if not encoding_live or not encoding_reference:
            error_message = "Face detection failed: One or both images do not contain a detectable face."
            return {
                'matched': False,
                'distance': -1,
                'threshold': -1,
                'error': error_message
            }

        # 4. Compare faces
        # compare_faces returns a list of booleans indicating if each unknown face matches the reference list.
        # We only have one live face, so we check the first result.
        match_results = face_recognition.compare_faces(
            [encoding_reference[0]], # List of known faces (the reference)
            encoding_live[0]         # The unknown face (the live capture)
        )
        matched = match_results[0]

        # Calculate face distance (lower is better, threshold is typically 0.6)
        # face_distance returns a list of distances. We take the first one.
        face_distances = face_recognition.face_distance(
            [encoding_reference[0]], 
            encoding_live[0]
        )
        distance = face_distances[0]
        
        # NOTE: face_recognition internally uses a fixed threshold (around 0.6).
        # We report the standard distance metric.
        
        return {
            'matched': matched,
            'distance': distance,
            'threshold': 0.6, # Fixed default threshold used by the library
            'error': None
        }

    except Exception as e:
        # Catch any file loading or processing errors
        return {
            'matched': False,
            'distance': -1,
            'threshold': -1,
            'error': f'Face Recognition processing error: {str(e)}'
        }


@api_view(['POST'])
def recognize_face(request):
    """
    Receives an image file (live camera image) and a reference URL 
    and uses face_recognition to verify if the faces match.
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

    # 3. Process Files using a robust temporary path approach
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