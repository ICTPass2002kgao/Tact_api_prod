import os
import tempfile
import urllib.request
import face_recognition
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

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

# --- Core Face Verification Function using face_recognition (UPDATED) ---

def perform_face_recognition_verification(file1_path, file2_path):
    """
    Performs face verification by splitting detection and encoding.
    Uses upsampling (UPSAMPLE_LEVEL=2) to improve detection of lower-quality images.
    """
    # Use the upsampling level that worked in your local test
    UPSAMPLE_LEVEL = 2 
    
    try:
        # 1. Load the images
        img_live = face_recognition.load_image_file(file1_path)
        img_reference = face_recognition.load_image_file(file2_path)

        # 2. Face DETECTION (Using upsampling to search harder)
        live_locations = face_recognition.face_locations(img_live, number_of_times_to_upsample=UPSAMPLE_LEVEL)
        ref_locations = face_recognition.face_locations(img_reference, number_of_times_to_upsample=UPSAMPLE_LEVEL)
        
        # 3. Validation: Check if a face was detected
        if not live_locations:
            error_message = "Face detection failed in the live camera image (low quality/bad lighting)."
            return {'matched': False, 'distance': -1, 'threshold': 0.6, 'error': error_message}
        
        if not ref_locations:
            error_message = "Face detection failed in the Firebase reference image (low quality/no face)."
            return {'matched': False, 'distance': -1, 'threshold': 0.6, 'error': error_message}


        # 4. Face ENCODING (Pass the detected locations to the encoding function)
        # This step does NOT use the 'number_of_times_to_upsample' argument.
        encoding_live = face_recognition.face_encodings(img_live, known_face_locations=live_locations)
        encoding_reference = face_recognition.face_encodings(img_reference, known_face_locations=ref_locations)
        
        # Check if the encoding process failed (shouldn't happen if detection succeeded)
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