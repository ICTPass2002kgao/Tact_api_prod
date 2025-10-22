import os
import tempfile
from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from deepface import DeepFace  

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
            model_name='Facenet', # Or 'VGG-Face', 'ArcFace', etc.
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


@api_view(['POST'])
def recognize_face(request):
    """
    Receives two image files (live camera image and Firebase reference image)
    and uses DeepFace to verify if the faces match.
    """
    # 1. Get the files from the request
    camera_uploaded_file = request.FILES.get('camera_image')
    firebase_uploaded_file = request.FILES.get('firebase_image')

    # 2. Validation: Check if both files are present
    if not camera_uploaded_file:
        return Response({'error': 'Missing required file: camera_image'}, 
                        status=status.HTTP_400_BAD_REQUEST)
    if not firebase_uploaded_file:
        return Response({'error': 'Missing required file: firebase_image'}, 
                        status=status.HTTP_400_BAD_REQUEST)

    # 3. Process Files using a robust temporary path approach
    # We use tempfile to ensure the files are cleaned up automatically.
    
    # NOTE: DeepFace requires a file path or in-memory numpy array.
    # We'll save the uploaded file temporarily to disk.
    
    temp_dir = tempfile.gettempdir()
    
    # Define paths for the temporary files
    camera_temp_path = os.path.join(temp_dir, f'camera_{os.getpid()}.jpg')
    firebase_temp_path = os.path.join(temp_dir, f'firebase_{os.getpid()}.jpg')

    # Write the uploaded content to the temporary files
    try:
        with open(camera_temp_path, 'wb+') as f:
            for chunk in camera_uploaded_file.chunks():
                f.write(chunk)
        
        with open(firebase_temp_path, 'wb+') as f:
            for chunk in firebase_uploaded_file.chunks():
                f.write(chunk)

        # 4. Perform Verification
        verification_result = perform_deepface_verification(
            camera_temp_path, 
            firebase_temp_path
        )
        
    except Exception as e:
        # Catch file saving/disk errors
        return Response({'error': f'File saving failed: {str(e)}'}, 
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    finally:
        # 5. Cleanup: Always delete the temporary files
        if os.path.exists(camera_temp_path):
            os.remove(camera_temp_path)
        if os.path.exists(firebase_temp_path):
            os.remove(firebase_temp_path)

    # 6. Return Response
    if verification_result['error']:
        return Response({
            'matched': False,
            'error': verification_result['error']
        }, status=status.HTTP_400_BAD_REQUEST)
    
    return Response({
        'matched': verification_result['matched'],
        'distance': verification_result['distance'],
        'threshold': verification_result['threshold'],
        'message': "Face verified successfully." if verification_result['matched'] else "Faces do not match."
    }, status=status.HTTP_200_OK)