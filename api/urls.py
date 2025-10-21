
from django.urls import path   
from . import views
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('verify_faces/', views.recognize_face, name='recognize_face'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)