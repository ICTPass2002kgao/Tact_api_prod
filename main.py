import os
import requests
import json

# --- Configuration ---
# IMPORTANT: Both files MUST exist in the same directory as this script.
FILE_LIVE_IMAGE = "me.png"
FILE_REFERENCE_IMAGE = "me1.jpg"

# The endpoint where your Django server is running locally
LOCAL_API_ENDPOINT = "http://127.0.0.1:8000/api/verify_faces/"

# --- Main Execution Block ---

if __name__ == "__main__":
    print("--- Starting Local API Test (Connecting to Django) ---")
    
    # 1. Prepare image files for the multipart request
    try:
        live_file_path = os.path.join(os.getcwd(), FILE_LIVE_IMAGE)
        ref_file_path = os.path.join(os.getcwd(), FILE_REFERENCE_IMAGE)

        if not os.path.exists(live_file_path) or not os.path.exists(ref_file_path):
            print("FATAL ERROR: Ensure 'me.png' and 'me1.jpg' are in the current directory.")
            exit(1)

        # In a real Django API call, we only send the LIVE image as a file 
        # and the REFERENCE image as a URL string. 
        # Since we are testing locally, we'll mimic the Flutter app's inputs,
        # but for Django to properly process the 'reference_url' field, 
        # we will send the *local file path* as the field's value for a diagnostic test.
        # However, since your view is designed to download the URL, 
        # let's assume you'll use a public Firebase URL for the reference image here
        # to match the final production behavior.
        
        # NOTE: Using the URL from your previous test for production fidelity
        URL_REFERENCE_IMAGE = "https://firebasestorage.googleapis.com/v0/b/tact-3c612.firebasestorage.app/o/Tactso%20Branches%2FVaal%20University%20of%20technology%2FMediaOfficer%2FKgaogelo%20Mthimkhulu_1761076613576?alt=media&token=d5784e98-b3fb-4128-8ea1-b2c79592293b"

        # 2. Setup the POST request data
        files = {
            # The Flutter app sends the camera image under the key 'live_image'
            'live_image': (FILE_LIVE_IMAGE, open(live_file_path, 'rb'), 'image/jpeg')
        }
        data = {
            # The Flutter app sends the reference URL under the key 'reference_url'
            'reference_url': URL_REFERENCE_IMAGE
        }

        print(f"Sending POST request to: {LOCAL_API_ENDPOINT}")
        
        # 3. Send Request
        response = requests.post(LOCAL_API_ENDPOINT, data=data, files=files)
        
        # 4. Process Response
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        
        try:
            result = response.json()
        except json.JSONDecodeError:
            print(f"\nCRITICAL ERROR: Failed to decode JSON response.")
            print(f"Raw Response: {response.text}")
            result = {"matched": False, "message": "Invalid JSON response from server."}

        # 5. Display Results
        print("\n--- API TEST RESULTS ---")
        print(f"HTTP Status: {response.status_code}")
        print(f"✅ Match Status: {result.get('matched')}")
        print(f"📝 Message: {result.get('message')}")
        if 'distance' in result:
            print(f"📏 Distance: {result['distance']:.4f} (Threshold is ~0.6)")
        print("--------------------")

    except requests.exceptions.ConnectionError:
        print(f"\nNETWORK ERROR: Could not connect to the server at {LOCAL_API_ENDPOINT}")
        print("ACTION: Ensure your Django server is running via 'python manage.py runserver 127.0.0.1:8000'")
    except requests.exceptions.HTTPError as e:
        print(f"\nHTTP ERROR: Received status code {e.response.status_code}")
        print(f"Server Response Body: {e.response.text}")
    except Exception as e:
        print(f"\nAN UNEXPECTED ERROR OCCURRED: {e}")
    finally:
        # Close file handlers
        if 'live_image' in locals() and locals()['live_image'][1].closed is False:
             locals()['live_image'][1].close()