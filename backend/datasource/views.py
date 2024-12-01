from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from datasource.models import DataSource
from datasource.serializers import DataSourceSerializer
from django.http import HttpRequest
from django.contrib.auth import get_user_model
from uuid import uuid4 as uuid
from typing import Optional

User = get_user_model()


class DataSourceView(APIView):
    def get(self, request: HttpRequest, id: Optional[uuid] = None) -> Response:
        if id:
            if datasource := DataSource.objects.filter(id=id).first():
                serializer = DataSourceSerializer(datasource)
                return Response(serializer.data)
            return Response("Datasource not found", status=status.HTTP_404_NOT_FOUND)
        else:
            datasources = DataSource.objects.all()
            serializer = DataSourceSerializer(datasources, many=True)
            return Response(serializer.data)

    def post(self, request: HttpRequest) -> Response:
        if not (user := User.objects.first()):
            return Response("User not found", status=status.HTTP_404_NOT_FOUND)
        data = request.data
        data["user"] = user.id
        data["created_by"] = user.id
        serializer = DataSourceSerializer(data=data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class DataSourceTestConnectionView(APIView):
    def get(self, request: HttpRequest, id: uuid) -> Response:
        if datasource := DataSource.objects.filter(id=id).first():
            if datasource.test_connection():
                return Response("Connection successful")
            return Response("Connection failed", status=status.HTTP_400_BAD_REQUEST)
        return Response("Datasource not found", status=status.HTTP_404_NOT_FOUND)
