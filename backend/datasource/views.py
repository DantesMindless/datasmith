from typing import Optional
from uuid import UUID
import csv
import os
from datetime import datetime

from django.contrib.auth import get_user_model
from django.http import HttpRequest
from django.conf import settings
from django.core.files.base import ContentFile
from rest_framework import status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import SessionAuthentication
from rest_framework_simplejwt.authentication import JWTAuthentication

from datasource.models import DataSource
from datasource.serializers import DataSourceSerializer, DatasourceViewSerializer
from enum import StrEnum
from core.views import BaseAuthApiView
from django.db.models import Q
from .constants.choices import DatasourceTypeChoices
from .permissions import IsOwnerOrReadOnly, CanAccessDatasource

# Import permission utilities
from userauth.permissions import (
    require_permission, require_role, PermissionManager,
    DataSourcePermissionMixin
)
from userauth.models import UserRole, AccessType

User = get_user_model()


class DatasourceResponses(StrEnum):
    DS_NOT_FOUND = "Datasource not found"
    DS_CONNECTIONS_SUCCESS = "Connection successful"
    DS_CONNECTIONS_FAIL = "Connection Fail"


class DataSourceView(BaseAuthApiView, DataSourcePermissionMixin):
    """
    ViewSet for managing data sources with proper DRF authentication and permissions.
    
    Uses custom permission classes to handle access control based on ownership
    and the existing permission system.
    """
    authentication_classes = [JWTAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated, CanAccessDatasource]
    def get(self, request: HttpRequest, id: Optional[UUID] = None) -> Response:
        if id:
            # Check if user has permission to access this datasource
            if not PermissionManager.can_user_access_resource(
                request.user, 'datasource', str(id), AccessType.READ
            ):
                return Response(
                    "Insufficient permissions", status=status.HTTP_403_FORBIDDEN
                )
            
            if (
                datasource := DataSource.objects.filter(id=id, deleted=False).first()
            ):
                serializer = DatasourceViewSerializer(datasource)
                # Filter response based on user permissions
                filtered_data = self.filter_datasource_response(
                    request.user, datasource, serializer.data
                )
                return Response(filtered_data)
            return Response(
                DatasourceResponses.DS_NOT_FOUND, status=status.HTTP_404_NOT_FOUND
            )
        else:
            # Get only datasources user can access
            datasources = self.get_accessible_datasources(request.user)
            serializer = DatasourceViewSerializer(datasources, many=True)
            
            # Filter each datasource response
            filtered_data = []
            for i, datasource in enumerate(datasources):
                filtered_item = self.filter_datasource_response(
                    request.user, datasource, serializer.data[i]
                )
                filtered_data.append(filtered_item)
            
            return Response(filtered_data)

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

    def delete(self, request: HttpRequest, id: UUID) -> Response:
        if datasource := DataSource.objects.filter(id=id).first():
            if (
                datasource.user_id == request.user.id
                or datasource.created_by == request.user.id
            ):
                datasource.delete()
                return Response(
                    f"Datasource with id {id} deleted successfully",
                    status=status.HTTP_200_OK,
                )
            return Response(
                "You do not have permission to delete this datasource",
                status=status.HTTP_403_FORBIDDEN,
            )
        return Response(
            DatasourceResponses.DS_NOT_FOUND, status=status.HTTP_404_NOT_FOUND
        )

    def put_tut(self, request: HttpRequest, id: UUID) -> Response:
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

    def put(self, request: HttpRequest, id: UUID) -> Response:
        # Check datasource access permission
        if not PermissionManager.can_user_access_resource(
            request.user, 'datasource', str(id), AccessType.READ
        ):
            return Response(
                "Insufficient permissions", status=status.HTTP_403_FORBIDDEN
            )
        
        data = request.data
        if not (query := data.get("query")):
            return Response("Query not provided", status=status.HTTP_400_BAD_REQUEST)
        
        if datasource := DataSource.objects.filter(id=id, deleted=False).first():
            # Filter query based on user's column permissions
            filtered_query = PermissionManager.filter_query_columns(
                request.user, str(id), query
            )
            
            success, response_data, message = datasource.get_table_rows(filtered_query)
            if success and response_data is not None:
                # Additional filtering of response data based on column permissions
                if not request.user.has_role(UserRole.DATABASE_ADMIN):
                    # TODO: Implement column-level filtering of response data
                    pass
                return Response(response_data)
            elif success:
                return Response(message)
            return Response(message, status=status.HTTP_400_BAD_REQUEST)
        return Response(
            DatasourceResponses.DS_NOT_FOUND, status=status.HTTP_404_NOT_FOUND
        )


class DataSourceDetailMetadataView(BaseAuthApiView):
    """
    View for retrieving datasource metadata with proper authentication.
    """
    authentication_classes = [JWTAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated, CanAccessDatasource]
    def get(
        self,
        request: HttpRequest,
        id: UUID,
        schema: Optional[str],
        table_name: Optional[str] = None,
    ) -> Response:
        """
        Retrieve metadata for a single table or tables
        """
        if datasource := DataSource.objects.filter(id=id).first():
            if schema and table_name:
                # if datasource.metadata and (table := datasource.metadata.get(schema, {}).get(table_name, None)):
                #     return Response(table)
                # else:
                data = datasource.update_metadata(schema)
                if table := data.get(schema, {}).get(table_name, None):
                    return Response(table)
                return Response("Table not found", status=status.HTTP_404_NOT_FOUND)
            else:
                if datasource.metadata and (tables_list := datasource.metadata.keys()):
                    return Response(tables_list)
                else:
                    data = datasource.update_metadata()
                    tables_list = datasource.metadata.keys()
                    return Response(tables_list)
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
    """
    View for retrieving datasource table metadata with proper authentication.
    """
    authentication_classes = [JWTAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated, CanAccessDatasource]
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
    """
    View for retrieving datasource schema metadata with proper authentication.
    """
    authentication_classes = [JWTAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated, CanAccessDatasource]
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
    """
    View for testing datasource connections with proper authentication.
    """
    authentication_classes = [JWTAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]
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
    """
    View for retrieving supported connection types with proper authentication.
    """
    authentication_classes = [JWTAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]
    def get(self, request: HttpRequest) -> Response:
        """
        Retrieve Supported Connection Adapters
        """
        return Response(
            DatasourceTypeChoices.supported_adapers(), status=status.HTTP_200_OK
        )


class DataSourceConnectionTypesFormFieldsView(BaseAuthApiView):
    """
    View for retrieving connection form fields with proper authentication.
    """
    authentication_classes = [JWTAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]
    def get(self, request: HttpRequest, id: DatasourceTypeChoices) -> Response:
        """
        Retrieve Supported Connection Adapters
        """
        if datasource := DatasourceTypeChoices.get_adapter(id):
            return Response(datasource.get_initial_params(), status=status.HTTP_200_OK)
        return Response("Datasource not found", status=status.HTTP_404_NOT_FOUND)


class DataSourceExportView(BaseAuthApiView, DataSourcePermissionMixin):
    """
    View for exporting database query results to CSV and creating ML datasets.
    """
    authentication_classes = [JWTAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated, CanAccessDatasource]

    def post(self, request: HttpRequest, id: UUID) -> Response:
        """
        Export query results to CSV and create a Dataset.

        Expected request body:
        {
            "schema": "public",
            "table": "users",
            "columns": ["id", "name", "email"],  // optional, defaults to all
            "filters": "WHERE age > 18",  // optional
            "limit": 10000,  // optional, defaults to 100000
            "dataset_name": "User Export",  // optional
            "dataset_description": "Exported user data"  // optional
        }
        """
        # Check datasource access permission
        if not PermissionManager.can_user_access_resource(
            request.user, 'datasource', str(id), AccessType.READ
        ):
            return Response(
                "Insufficient permissions to export from this datasource",
                status=status.HTTP_403_FORBIDDEN
            )

        datasource = DataSource.objects.filter(id=id, deleted=False).first()
        if not datasource:
            return Response(
                DatasourceResponses.DS_NOT_FOUND,
                status=status.HTTP_404_NOT_FOUND
            )

        # Parse request data
        data = request.data
        schema = data.get("schema")
        table = data.get("table")
        columns = data.get("columns", [])
        filters = data.get("filters", "")
        limit = min(data.get("limit", 100000), 1000000)  # Cap at 1M rows
        dataset_name = data.get("dataset_name", f"{table} Export - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        dataset_description = data.get("dataset_description", f"Exported from {datasource.name}/{schema}/{table}")

        if not schema or not table:
            return Response(
                "Schema and table are required",
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Build query
            column_list = ", ".join(f'"{col}"' for col in columns) if columns else "*"
            query = f'SELECT {column_list} FROM "{schema}"."{table}"'

            if filters:
                query += f" {filters}"

            query += f" LIMIT {limit}"

            # Execute query
            datasource.connection.connect()
            success, result, message = datasource.connection.query(query)
            datasource.connection.close()

            if not success or not result:
                return Response(
                    f"Query failed: {message}",
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Check if we got any data
            if len(result) == 0:
                return Response(
                    "Query returned no results",
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Create CSV in memory
            import io
            csv_buffer = io.StringIO()

            # Get column names from first row
            fieldnames = list(result[0].keys())
            writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)

            # Write header and rows
            writer.writeheader()
            writer.writerows(result)

            # Get CSV content
            csv_content = csv_buffer.getvalue()
            csv_buffer.close()

            # Create Dataset
            from app.models.main import Dataset
            dataset = Dataset(
                name=dataset_name,
                description=dataset_description,
                created_by=request.user,
            )

            # Save CSV file
            filename = f"{schema}_{table}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            dataset.csv_file.save(
                filename,
                ContentFile(csv_content.encode('utf-8')),
                save=False
            )

            # Set metadata
            dataset.row_count = len(result)
            dataset.column_count = len(fieldnames)
            dataset.file_size = len(csv_content.encode('utf-8'))

            # Save dataset (this will trigger analysis)
            dataset.save()

            # Return response with dataset info
            return Response({
                "success": True,
                "message": "Export completed successfully",
                "dataset": {
                    "id": str(dataset.id),
                    "name": dataset.name,
                    "rows": dataset.row_count,
                    "columns": dataset.column_count,
                    "file_size": dataset.file_size,
                },
                "rows_exported": len(result)
            }, status=status.HTTP_201_CREATED)

        except Exception as e:
            import logging
            logging.error(f"Export error: {str(e)}", exc_info=True)
            return Response(
                f"Export failed: {str(e)}",
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
