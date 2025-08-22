from django.urls import path
from . import views

urlpatterns = [
    path("login/", views.login),
    path("logout/", views.logout),
    path("refresh/", views.refresh_token),
    path("signup/", views.signup),
    path("test_token/", views.test_token),
]
