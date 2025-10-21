from django.shortcuts import render
from deepface import DeepFace
from rest_framework.decorators import api_view
from rest_framework.response import Response
#ill be using this for face recognition from flutter app wher by the other face will be captured from a camera and the other on from firebase storage


@api_view(['POST'])
def recognize_face(request):
    # Get the images from the request
    camera_image = request.FILES.get('camera_image')
    firebase_image = request.FILES.get('firebase_image')

    # Use DeepFace to analyze the images
    result = DeepFace.verify(camera_image, firebase_image, model_name='Facenet', enforce_detection=True)

    # Return the result as a JSON response
    return Response({
        'matched': result['verified'],
        'distance': result['distance'],
        'threshold': result['threshold']
    }, status=200)

# Create your views here.
