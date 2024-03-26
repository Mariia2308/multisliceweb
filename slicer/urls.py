from django.urls import path
from . import views  # Импортируем представления из текущего приложения

urlpatterns = [
    path('', views.my_view, name='my_view'),  # Используем функцию my_view из views.py
    # Другие URL-адреса...
]