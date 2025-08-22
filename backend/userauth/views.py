from rest_framework.decorators import (
    api_view,
    authentication_classes,
    permission_classes,
)
from rest_framework.authentication import SessionAuthentication
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import get_object_or_404
from django.contrib.auth import authenticate
from .models import CustomUser
from .serializers import UserSerializer
from rest_framework.request import Request
from rest_framework.response import Response as DRFResponse
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.authentication import JWTAuthentication


@api_view(["POST"])
@authentication_classes([])
@permission_classes([])
def signup(request: Request) -> DRFResponse:
    serializer = UserSerializer(data=request.data)
    if serializer.is_valid():
        user = serializer.save()
        token = Token.objects.create(user=user)

        return Response(
            {"token": token.key, "user": serializer.data},
            status=status.HTTP_201_CREATED,
        )
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(["POST"])
@authentication_classes([])
@permission_classes([])
def login(request: Request) -> DRFResponse:
    username = request.data.get("username")
    email = request.data.get("email")
    password = request.data.get("password")
    
    # Use the original working approach - get user and check password directly
    try:
        if username:
            user = CustomUser.objects.get(username=username)
        elif email:
            user = CustomUser.objects.get(email=email)
        else:
            return Response(
                {"error": "Username or email required"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Check password using the original method
        if not user.check_password(password):
            return Response(
                {"error": "Invalid credentials"}, 
                status=status.HTTP_401_UNAUTHORIZED
            )
            
    except CustomUser.DoesNotExist:
        return Response(
            {"error": "Invalid credentials"}, 
            status=status.HTTP_401_UNAUTHORIZED
        )
    
    # Generate JWT tokens
    refresh = RefreshToken.for_user(user)
    access_token = refresh.access_token
    
    serializer = UserSerializer(user)
    return Response({
        "access": str(access_token),
        "refresh": str(refresh),
        "user": serializer.data
    }, status=status.HTTP_200_OK)


@api_view(["POST"])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def logout(request: Request) -> DRFResponse:
    try:
        refresh_token = request.data.get("refresh")
        if refresh_token:
            token = RefreshToken(refresh_token)
            token.blacklist()
        return Response("Successfully logged out", status=status.HTTP_200_OK)
    except Exception as e:
        return Response(
            {"error": "Error logging out"}, 
            status=status.HTTP_400_BAD_REQUEST
        )


@api_view(["POST"])
@authentication_classes([])
@permission_classes([])
def refresh_token(request: Request) -> DRFResponse:
    try:
        refresh_token = request.data.get("refresh")
        if not refresh_token:
            return Response(
                {"error": "Refresh token required"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        token = RefreshToken(refresh_token)
        new_access_token = token.access_token
        
        return Response({
            "access": str(new_access_token),
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response(
            {"error": "Invalid refresh token"}, 
            status=status.HTTP_401_UNAUTHORIZED
        )


@api_view(["GET"])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def test_token(request: Request) -> DRFResponse:
    return Response({
        "message": "Token is valid",
        "user": request.user.username
    })
