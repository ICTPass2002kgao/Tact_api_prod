import os
import shutil
import tempfile
import urllib.request
import face_recognition
import numpy as np # Used by face_recognition internally

# --- Configuration ---
# REPLACE THESE URLS with the images you want to test.
# Use your Firebase URL for the reference image and a known good image for the live test.

FILE_LIVE_IMAGE = "me.png"  # <-- REPLACE with your test image URL (e.g., a perfect selfie)
URL_REFERENCE_IMAGE = "https://firebasestorage.googleapis.com/v0/b/tact-3c612.firebasestorage.app/o/Tactso%20Branches%2FVaal%20University%20of%20technology%2FMediaOfficer%2FKgaogelo%20Mthimkhulu_1761076613576?alt=media&token=d5784e98-b3fb-4128-8ea1-b2c79592293b" # <-- Use one of your actual Firebase reference URLs

# --- Utility Function to Download File ---

def download_file_from_url(url, path):
    """Downloads a file from a public URL to a specified path."""
    try:
        urllib.request.urlretrieve(url, path)
        return True
    except Exception as e:
        print(f"ERROR: Failed to download {url}. Check URL and network. Details: {e}")
        return False

# --- Core Face Verification Function ---

# --- Core Face Verification Function (CORRECTED) ---

def perform_face_recognition_verification(file1_path, file2_path):
    """
    Performs face verification by splitting detection (where upsampling is used)
    and encoding (which uses the detected locations).
    """
    # Increase upsampling to search harder for smaller/poorer quality faces.
    # We will apply this to the detection step only.
    UPSAMPLE_LEVEL = 2 
    
    try:
        # 1. Load the images
        img_live = face_recognition.load_image_file(file1_path)
        img_reference = face_recognition.load_image_file(file2_path)

        # 2. Face DETECTION (Use upsampling here)
        # Note: We must check for face locations before proceeding to encoding.
        live_locations = face_recognition.face_locations(img_live, number_of_times_to_upsample=UPSAMPLE_LEVEL)
        ref_locations = face_recognition.face_locations(img_reference, number_of_times_to_upsample=UPSAMPLE_LEVEL)
        
        # 3. Validation: Check if a face was detected
        if not live_locations:
            return {'matched': False, 'message': 'FAILURE: No face detected in the LIVE image, even after upsampling.'}
        
        if not ref_locations:
            return {'matched': False, 'message': 'FAILURE: No face detected in the REFERENCE image, even after upsampling.'}

        # 4. Face ENCODING (Pass the detected locations to the encoding function)
        encoding_live = face_recognition.face_encodings(img_live, known_face_locations=live_locations)
        encoding_reference = face_recognition.face_encodings(img_reference, known_face_locations=ref_locations)
        
        # The locations were found, but encodings list might be empty if the landmark detection fails severely.
        if not encoding_live or not encoding_reference:
            return {'matched': False, 'message': 'FAILURE: Face found, but encoding extraction failed.'}
        
        # 5. Compare faces (using the first found face)
        match_results = face_recognition.compare_faces(
            [encoding_reference[0]], 
            encoding_live[0]        
        )
        matched = match_results[0]

        face_distances = face_recognition.face_distance(
            [encoding_reference[0]], 
            encoding_live[0]
        )
        distance = face_distances[0]
        
        return {
            'matched': matched,
            'distance': float(distance),
            'message': "Match found successfully!" if matched else f"Faces do not match. Distance: {distance:.4f}"
        }

    except Exception as e:
        return {
            'matched': False,
            'message': f'CRITICAL ERROR during processing: {str(e)}'
        }
# --- Main Execution Block ---

if __name__ == "__main__":
    temp_dir = tempfile.gettempdir()
    unique_id = os.urandom(4).hex()
    
    # Define temp file paths
    # The 'live_path' will be where we copy the local file.
    live_path_temp = os.path.join(temp_dir, f'test_live_{unique_id}.jpg')
    reference_path = os.path.join(temp_dir, f'test_ref_{unique_id}.jpg')
    
    # Get the absolute path for the local file
    live_path_source = os.path.join(os.getcwd(), FILE_LIVE_IMAGE)
    
    print("--- Starting Face Verification Local Test ---")
    
    try:
        # 1. Handle Live Image (Copy Local File to Temp Directory)
        if not os.path.exists(live_path_source):
            print(f"FATAL ERROR: Local file '{FILE_LIVE_IMAGE}' not found in the current directory.")
            exit(1)
        
        print(f"Copying local live image: {FILE_LIVE_IMAGE}")
        shutil.copyfile(live_path_source, live_path_temp)


        # 2. Download Reference Image
        print(f"Downloading Reference Image from: {URL_REFERENCE_IMAGE}")
        if not download_file_from_url(URL_REFERENCE_IMAGE, reference_path):
            exit(1)

        # 3. Perform Verification
        print("\nPerforming Face Verification...")
        result = perform_face_recognition_verification(live_path_temp, reference_path)

        # 4. Display Results
        print("\n--- TEST RESULTS ---")
        print(f"✅ Match Status: {result['matched']}")
        print(f"📝 Message: {result['message']}")
        if result.get('distance') is not None and result['distance'] != -1:
            print(f"📏 Distance: {result['distance']:.4f} (Threshold is ~0.6)")
        print("--------------------")

    finally:
        # 5. Cleanup
        if os.path.exists(live_path_temp):
            os.remove(live_path_temp)
        if os.path.exists(reference_path):
            os.remove(reference_path)