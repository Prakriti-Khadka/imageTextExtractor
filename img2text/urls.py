from django.urls import path
from . import views

urlpatterns = [
    path('', views.image_upload_view, name='image_upload'),  # Main upload view
    path('result/', views.result_view, name='result'),  # Result display view
]

