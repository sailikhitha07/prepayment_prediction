from django.urls import path
from . import views

urlpatterns = [
    path('', views.predictor, name='predictor'),  # Route to render the main form
    path('submit/', views.formInfo, name='formInfo'),  # Route to handle form submission
]
