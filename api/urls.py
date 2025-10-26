
from django.urls import path   
from . import views
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('api/verify_faces/', views.recognize_face, name='recognize_face'),
    path('extract-audio/', views.convert_video_to_audio_api, name='extract-audio'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)