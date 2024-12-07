from typing import Optional
from uuid import UUID

from django.contrib.auth import get_user_model
from django.http import HttpRequest
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from datasource.models import DataSource
from datasource.serializers import DataSourceSerializer
from enum import Enum

User = get_user_model()


class DatasourceResponses(Enum):
    DS_NOT_FOUND = "Datasource not found"
    DS_CONNECTIONS_SUCCESS = "Connection successful"
    DS_CONNECTIONS_FAIL = "Connection Fail"


class DataSourceView(APIView):
    def get(self, request: HttpRequest, id: Optional[UUID] = None) -> Response:
        if id:
            if datasource := DataSource.objects.filter(id=id).first():
                serializer = DataSourceSerializer(datasource)
                return Response(serializer.data)
            return Response(
                DatasourceResponses.DS_NOT_FOUND.value, status=status.HTTP_404_NOT_FOUND
            )
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

    def put(self, request: HttpRequest, id: UUID) -> Response:
        data = request.data
        if not (query := data.get("query")):
            return Response("Query not provided", status=status.HTTP_400_BAD_REQUEST)
        if datasource := DataSource.objects.filter(id=id).first():
            success, response_data, message = datasource.query(query)
            if success and response_data is not None:
                return Response(response_data)
            elif success:
                return Response(message)
            return Response(message, status=status.HTTP_400_BAD_REQUEST)
        return Response(
            DatasourceResponses.DS_NOT_FOUND.value, status=status.HTTP_404_NOT_FOUND
        )


class DataSourceDetailMetadataView(APIView):
    def get(
        self, request: HttpRequest, id: UUID, table_name: Optional[str] = None
    ) -> Response:
        """
        Retrieve metadata for a single table or tables
        """
        if datasource := DataSource.objects.filter(id=id).first():
            if table_name:
                if metadata := datasource.metadata.get(table_name):
                    return Response(metadata)
                else:
                    data = datasource.update_metadata()
                    if table := data.get(table_name):
                        return Response(table)
                    return Response("Table not found", status=status.HTTP_404_NOT_FOUND)
            else:
                if datasource.metadata and (tables_list := datasource.metadata.keys()):
                    return Response(tables_list)
                else:
                    data = datasource.update_metadata()
                    return Response(data)
        return Response(
            DatasourceResponses.DS_NOT_FOUND.value, status=status.HTTP_404_NOT_FOUND
        )

    def put(self, request: HttpRequest, id: UUID) -> Response:
        if datasource := DataSource.objects.filter(id=id).first():
            data = datasource.update_metadata()
            return Response(data)
        return Response(
            DatasourceResponses.DS_NOT_FOUND.value, status=status.HTTP_404_NOT_FOUND
        )


class DataSourceTablesMetadataView(APIView):
    def get(self, request: HttpRequest, id: UUID, schema: str) -> Response:
        """
        Retrieve tables lists
        """
        if datasource := DataSource.objects.filter(id=id).first():
            success, data, message = datasource.get_tables(schema)
            if success:
                return Response(data)
            return Response(message, status=status.HTTP_404_NOT_FOUND)
        return Response(
            DatasourceResponses.DS_NOT_FOUND.value, status=status.HTTP_404_NOT_FOUND
        )


class DataSourceSchemasMetadataView(APIView):
    def get(self, request: HttpRequest, id: UUID) -> Response:
        """
        Retrieve database schemas
        """
        if datasource := DataSource.objects.filter(id=id).first():
            success, data, message = datasource.get_schemas()
            if success:
                return Response(data)
            return Response(message, status=status.HTTP_404_NOT_FOUND)
        return Response(
            DatasourceResponses.DS_NOT_FOUND.value, status=status.HTTP_404_NOT_FOUND
        )


class DataSourceTestConnectionView(APIView):
    def get(self, request: HttpRequest, id: UUID) -> Response:
        """
        Connections test for existing DS
        """
        if datasource := DataSource.objects.filter(id=id).first():
            if datasource.test_connection():
                return Response(DatasourceResponses.DS_CONNECTIONS_SUCCESS.value)
            return Response(
                DatasourceResponses.DS_CONNECTIONS_FAIL.value,
                status=status.HTTP_400_BAD_REQUEST,
            )
        return Response(
            DatasourceResponses.DS_NOT_FOUND.value, status=status.HTTP_404_NOT_FOUND
        )

    def post(self, request: HttpRequest) -> Response:
        """
        Connections test before DS creation
        """
        if not (user := User.objects.first()):
            return Response("User not found", status=status.HTTP_404_NOT_FOUND)
        data = request.data
        data["user"] = user.id
        data["created_by"] = user.id
        serializer = DataSourceSerializer(data=data)
        if serializer.is_valid():
            data_source = DataSource(
                type=serializer.data.get("type"),
                credentials=serializer.data.get("credentials"),
            )
            if data_source.test_connection():
                return Response(DatasourceResponses.DS_CONNECTIONS_SUCCESS.value)
            return Response(
                DatasourceResponses.DS_CONNECTIONS_FAIL.value,
                status=status.HTTP_400_BAD_REQUEST,
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
