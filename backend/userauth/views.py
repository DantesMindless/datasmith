from typing import Dict, Any
from rest_framework.decorators import api_view, permission_classes
from rest_framework.authentication import SessionAuthentication
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework import status
from rest_framework.views import APIView
from django.shortcuts import get_object_or_404
from django.contrib.auth import authenticate
from .models import CustomUser
from .serializers import UserSerializer
from rest_framework.request import Request
from rest_framework.response import Response as DRFResponse
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.authentication import JWTAuthentication


class AuthenticationService(APIView):
    """
    Centralized authentication service for user registration, login, logout, and token management.

    Provides a unified interface for all authentication operations with proper
    error handling and JWT token management.
    """
    permission_classes = [AllowAny]  # Allow anonymous access for login/signup
    
    def post(self, request: Request) -> DRFResponse:
        """
        Handle authentication requests based on the action parameter.
        
        Args:
            request: The HTTP request containing action and authentication data
            
        Returns:
            Response with authentication result or error message
        """
        action = request.data.get('action')
        
        if action == 'signup':
            return self._handle_signup(request)
        elif action == 'login':
            return self._handle_login(request)
        elif action == 'logout':
            return self._handle_logout(request)
        elif action == 'refresh':
            return self._handle_refresh_token(request)
        else:
            return Response(
                {"error": "Invalid action. Supported actions: signup, login, logout, refresh"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
    
    def _handle_signup(self, request: Request) -> DRFResponse:
        """
        Handle user registration.
        
        Args:
            request: The HTTP request containing user data
            
        Returns:
            Response with user data and tokens or validation errors
        """
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            refresh = RefreshToken.for_user(user)
            access_token = refresh.access_token

            return Response(
                {
                    "access": str(access_token),
                    "refresh": str(refresh),
                    "user": serializer.data
                },
                status=status.HTTP_201_CREATED,
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    def _handle_login(self, request: Request) -> DRFResponse:
        """
        Handle user login with username/email and password.
        
        Args:
            request: The HTTP request containing login credentials
            
        Returns:
            Response with JWT tokens and user data or error message
        """
        username = request.data.get("username")
        email = request.data.get("email")
        password = request.data.get("password")
        
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
        
        refresh = RefreshToken.for_user(user)
        access_token = refresh.access_token
        
        serializer = UserSerializer(user)
        return Response({
            "access": str(access_token),
            "refresh": str(refresh),
            "user": serializer.data
        }, status=status.HTTP_200_OK)
    
    def _handle_logout(self, request: Request) -> DRFResponse:
        """
        Handle user logout by blacklisting the refresh token.
        
        Args:
            request: The HTTP request containing refresh token
            
        Returns:
            Response indicating logout success or failure
        """
        # This endpoint requires authentication
        if not hasattr(request, 'user') or not request.user.is_authenticated:
            return Response(
                {"error": "Authentication required"}, 
                status=status.HTTP_401_UNAUTHORIZED
            )
            
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
    
    def _handle_refresh_token(self, request: Request) -> DRFResponse:
        """
        Handle token refresh using a valid refresh token.
        
        Args:
            request: The HTTP request containing refresh token
            
        Returns:
            Response with new access token or error message
        """
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


class AuthenticatedTestView(APIView):
    """
    Test view to verify JWT token authentication.
    """
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    
    def get(self, request: Request) -> DRFResponse:
        """
        Test endpoint to verify token validity.

        Args:
            request: The HTTP request with authentication headers

        Returns:
            Response with user information if token is valid
        """
        return Response({
            "message": "Token is valid",
            "user": {
                "id": request.user.id,
                "username": request.user.username,
                "email": request.user.email
            }
        })


# Legacy function-based views for backward compatibility
@api_view(["POST"])
@permission_classes([AllowAny])
def signup(request: Request) -> DRFResponse:
    """
    Legacy signup endpoint. Redirects to AuthenticationService.
    
    Args:
        request: The HTTP request containing user data
        
    Returns:
        Response from AuthenticationService signup handler
    """
    auth_service = AuthenticationService()
    request.data['action'] = 'signup'
    return auth_service.post(request)


@api_view(["POST"])
@permission_classes([AllowAny])
def login(request: Request) -> DRFResponse:
    """
    Legacy login endpoint. Redirects to AuthenticationService.
    
    Args:
        request: The HTTP request containing login credentials
        
    Returns:
        Response from AuthenticationService login handler
    """
    auth_service = AuthenticationService()
    request.data['action'] = 'login'
    return auth_service.post(request)


@api_view(["POST"])
def logout(request: Request) -> DRFResponse:
    """
    Legacy logout endpoint. Redirects to AuthenticationService.
    
    Args:
        request: The HTTP request containing refresh token
        
    Returns:
        Response from AuthenticationService logout handler
    """
    auth_service = AuthenticationService()
    auth_service.authentication_classes = [JWTAuthentication]
    auth_service.permission_classes = [IsAuthenticated]
    request.data['action'] = 'logout'
    return auth_service.post(request)


@api_view(["POST"])
@permission_classes([AllowAny])
def refresh_token(request: Request) -> DRFResponse:
    """
    Legacy refresh token endpoint. Redirects to AuthenticationService.
    
    Args:
        request: The HTTP request containing refresh token
        
    Returns:
        Response from AuthenticationService refresh handler
    """
    auth_service = AuthenticationService()
    request.data['action'] = 'refresh'
    return auth_service.post(request)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def test_token(request: Request) -> DRFResponse:
    """
    Test endpoint to verify token validity and return user data.

    Args:
        request: The HTTP request with authentication headers

    Returns:
        Response with user information if token is valid
    """
    return Response({
        "message": "Token is valid",
        "user": {
            "id": request.user.id,
            "username": request.user.username,
            "email": request.user.email
        }
    })