from typing import Optional
from uuid import UUID

from django.contrib.auth import get_user_model
from django.http import HttpRequest
from rest_framework import status
from rest_framework.response import Response

from datasource.models import DataSource
from datasource.serializers import DataSourceSerializer, DatasourceViewSerializer
from enum import StrEnum
from core.views import BaseAuthApiView
from django.db.models import Q
from .constants.choices import DatasourceTypeChoices

User = get_user_model()


class DatasourceResponses(StrEnum):
    DS_NOT_FOUND = "Datasource not found"
    DS_CONNECTIONS_SUCCESS = "Connection successful"
    DS_CONNECTIONS_FAIL = "Connection Fail"


class DataSourceView(BaseAuthApiView):
    def get(self, request: HttpRequest, id: Optional[UUID] = None) -> Response:
        if id:
            if (
                datasource := DataSource.objects.filter(id=id)
                .filter((Q(user_id=request.user.id) | Q(created_by=request.user.id)))
                .first()
            ):
                serializer = DatasourceViewSerializer(datasource)
                return Response(serializer.data)
            return Response(
                DatasourceResponses.DS_NOT_FOUND, status=status.HTTP_404_NOT_FOUND
            )
        else:
            datasources = DataSource.objects.all()
            serializer = DatasourceViewSerializer(datasources, many=True)
            return Response(serializer.data)

    def post(self, request: HttpRequest) -> Response:
        if not User.objects.first():
            return Response("User not found", status=status.HTTP_404_NOT_FOUND)
        data = request.data
        data["user"] = request.user.id
        data["created_by"] = request.user.id
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
            DatasourceResponses.DS_NOT_FOUND, status=status.HTTP_404_NOT_FOUND
        )


class DataSourceDetailMetadataView(BaseAuthApiView):
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
            DatasourceResponses.DS_NOT_FOUND, status=status.HTTP_404_NOT_FOUND
        )

    def put(self, request: HttpRequest, id: UUID) -> Response:
        if datasource := DataSource.objects.filter(id=id).first():
            data = datasource.update_metadata()
            return Response(data)
        return Response(
            DatasourceResponses.DS_NOT_FOUND, status=status.HTTP_404_NOT_FOUND
        )


class DataSourceTablesMetadataView(BaseAuthApiView):
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
            DatasourceResponses.DS_NOT_FOUND, status=status.HTTP_404_NOT_FOUND
        )


class DataSourceSchemasMetadataView(BaseAuthApiView):
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
            DatasourceResponses.DS_NOT_FOUND, status=status.HTTP_404_NOT_FOUND
        )


class DataSourceTestConnectionView(BaseAuthApiView):
    def get(self, request: HttpRequest, id: UUID) -> Response:
        """
        Connections test for existing DS
        """
        if datasource := DataSource.objects.filter(id=id).first():
            if datasource.test_connection():
                return Response(DatasourceResponses.DS_CONNECTIONS_SUCCESS)
            return Response(
                DatasourceResponses.DS_CONNECTIONS_FAIL,
                status=status.HTTP_400_BAD_REQUEST,
            )
        return Response(
            DatasourceResponses.DS_NOT_FOUND, status=status.HTTP_404_NOT_FOUND
        )

    def post(self, request: HttpRequest) -> Response:
        """
        Connections test before DS creation
        """
        if not User.objects.first():
            return Response("User not found", status=status.HTTP_404_NOT_FOUND)
        data = request.data
        data["user"] = request.user.id
        data["created_by"] = request.user.id
        serializer = DataSourceSerializer(data=data)
        if serializer.is_valid():
            data_source = DataSource(
                type=serializer.data.get("type"),
                credentials=serializer.data.get("credentials"),
            )
            if data_source.test_connection():
                return Response(DatasourceResponses.DS_CONNECTIONS_SUCCESS)
            return Response(
                DatasourceResponses.DS_CONNECTIONS_FAIL,
                status=status.HTTP_400_BAD_REQUEST,
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class DataSourceConnectionTypesView(BaseAuthApiView):
    def get(self, request: HttpRequest) -> Response:
        """
        Retrieve Supported Connection Adapters
        """
        return Response(
            DatasourceTypeChoices.supported_adapers(), status=status.HTTP_200_OK
        )


class DataSourceConnectionTypesFormFieldsView(BaseAuthApiView):
    def get(self, request: HttpRequest, id: DatasourceTypeChoices) -> Response:
        """
        Retrieve Supported Connection Adapters
        """
        if datasource := DatasourceTypeChoices.get_adapter(id):
            init_params = datasource.get_initial_params()

            PORTS_BY_TYPE = {
                DatasourceTypeChoices.POSTGRES: 5432,
                DatasourceTypeChoices.MYSQL: 3306,
                DatasourceTypeChoices.MONGO: 27017,
            }
            default_port = PORTS_BY_TYPE.get(id, None)
            if default_port is not None:
                init_params["port"]["initial"] = default_port
            return Response(init_params, status=status.HTTP_200_OK)
        return Response("Datasource not found", status=status.HTTP_404_NOT_FOUND)
