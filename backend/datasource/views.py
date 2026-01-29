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
from datasource.serializers import DataSourceSerializer, DatasourceViewSerializer, DatasourceEditSerializer
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
    """
    ViewSet for managing data sources with proper DRF authentication.
    """
    authentication_classes = [JWTAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

    def get(self, request: HttpRequest, id: Optional[UUID] = None) -> Response:
        if id:
            if (
                datasource := DataSource.objects.filter(
                    id=id,
                    deleted=False,
                    user=request.user
                ).first()
            ):
                # Use edit serializer if 'edit' query param is present
                if request.query_params.get('edit') == 'true':
                    serializer = DatasourceEditSerializer(datasource)
                else:
                    serializer = DatasourceViewSerializer(datasource)
                return Response(serializer.data)
            return Response(
                DatasourceResponses.DS_NOT_FOUND, status=status.HTTP_404_NOT_FOUND
            )
        else:
            datasources = DataSource.objects.filter(
                deleted=False,
                user=request.user
            )
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

    def patch(self, request: HttpRequest, id: UUID) -> Response:
        """Update an existing datasource connection."""
        datasource = DataSource.objects.filter(
            id=id, deleted=False, user=request.user
        ).first()

        if not datasource:
            return Response(
                DatasourceResponses.DS_NOT_FOUND, status=status.HTTP_404_NOT_FOUND
            )

        # Check permission
        if datasource.user_id != request.user.id and datasource.created_by != request.user.id:
            return Response(
                "You do not have permission to update this datasource",
                status=status.HTTP_403_FORBIDDEN,
            )

        data = request.data.copy()
        # Preserve user ownership
        data["user"] = datasource.user_id
        data["created_by"] = datasource.created_by_id

        serializer = DataSourceSerializer(datasource, data=data, partial=True)

        if serializer.is_valid():
            serializer.save()
            return Response(DatasourceViewSerializer(datasource).data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request: HttpRequest, id: UUID) -> Response:
        if datasource := DataSource.objects.filter(id=id, user=request.user).first():
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
        if datasource := DataSource.objects.filter(id=id, user=request.user).first():
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
        data = request.data
        # Extract query object from request data
        query_data = data.get("query")
        if not query_data:
            return Response("Query not provided", status=status.HTTP_400_BAD_REQUEST)

        # Validate required fields for table query
        if not query_data.get("schema") or not query_data.get("table"):
            return Response("Schema and table are required", status=status.HTTP_400_BAD_REQUEST)

        if datasource := DataSource.objects.filter(id=id, deleted=False).first():
            success, response_data, message = datasource.get_table_rows(query_data)
            if success and response_data is not None:
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
    permission_classes = [IsAuthenticated]
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
        if datasource := DataSource.objects.filter(id=id, user=request.user).first():
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

        if datasource := DataSource.objects.filter(id=id, user=request.user).first():
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
    permission_classes = [IsAuthenticated]
    def get(self, request: HttpRequest, id: UUID, schema: str) -> Response:
        if datasource := DataSource.objects.filter(id=id, user=request.user).first():
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
    permission_classes = [IsAuthenticated]
    def get(self, request: HttpRequest, id: UUID) -> Response:
        if datasource := DataSource.objects.filter(id=id, user=request.user).first():
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
        if datasource := DataSource.objects.filter(id=id, user=request.user).first():
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


class DataSourceExportView(BaseAuthApiView):
    """
    View for exporting database query results to CSV and creating ML datasets.
    Supports optional SQL JOINs with other tables.
    """
    authentication_classes = [JWTAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]

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
            "dataset_description": "Exported user data",  // optional
            "join": {  // optional - for SQL JOIN
                "table": "orders",
                "left_column": "id",
                "right_column": "user_id",
                "join_type": "inner",  // "inner", "left", "right", or "outer"
                "columns": ["order_date", "total"]  // columns to select from joined table
            }
        }
        """
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
        join_config = data.get("join")  # Optional join configuration
        dataset_name = data.get("dataset_name", f"{table} Export - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        dataset_description = data.get("dataset_description", f"Exported from {datasource.name}/{schema}/{table}")

        if not schema or not table:
            return Response(
                "Schema and table are required",
                status=status.HTTP_400_BAD_REQUEST
            )

        # Validate join config if provided
        join_metadata = {}
        if join_config:
            required_join_fields = ['table', 'left_column', 'right_column']
            missing_fields = [f for f in required_join_fields if not join_config.get(f)]
            if missing_fields:
                return Response(
                    f"Join config missing required fields: {', '.join(missing_fields)}",
                    status=status.HTTP_400_BAD_REQUEST
                )

        try:
            # Determine quote character based on datasource type
            from datasource.constants.choices import DatasourceTypeChoices
            quote_char = '`' if datasource.type == DatasourceTypeChoices.MYSQL else '"'

            # Build column list with table aliases if joining
            if join_config:
                # Main table columns with alias
                if columns:
                    main_cols = [f'a.{quote_char}{col}{quote_char}' for col in columns]
                else:
                    main_cols = ['a.*']

                # Join table columns with alias
                join_cols = join_config.get('columns', [])
                if join_cols:
                    join_col_list = [f'b.{quote_char}{col}{quote_char}' for col in join_cols]
                else:
                    join_col_list = ['b.*']

                column_list = ", ".join(main_cols + join_col_list)

                # Get join configuration
                join_table = join_config['table']
                left_col = join_config['left_column']
                right_col = join_config['right_column']
                join_type = join_config.get('join_type', 'inner').lower()

                # Map join type to SQL keyword
                join_type_map = {
                    'inner': 'INNER JOIN',
                    'left': 'LEFT OUTER JOIN',
                    'right': 'RIGHT OUTER JOIN',
                    'outer': 'FULL OUTER JOIN'
                }

                # Human-readable join type names
                join_type_display_map = {
                    'inner': 'Inner',
                    'left': 'Left Outer',
                    'right': 'Right Outer',
                    'outer': 'Full Outer'
                }

                sql_join = join_type_map.get(join_type, 'INNER JOIN')
                join_type_display = join_type_display_map.get(join_type, 'Inner')

                # Note: MySQL doesn't support FULL OUTER JOIN directly
                # For MySQL with full outer join, we'd need a UNION of LEFT and RIGHT joins
                # For now, we'll raise an error for MySQL + full outer
                from datasource.constants.choices import DatasourceTypeChoices
                if join_type == 'outer' and datasource.type == DatasourceTypeChoices.MYSQL:
                    return Response(
                        "Full outer join is not directly supported in MySQL. Use separate left and right joins instead.",
                        status=status.HTTP_400_BAD_REQUEST
                    )

                # Build query with appropriate JOIN type
                query = f'''SELECT {column_list}
                    FROM {quote_char}{schema}{quote_char}.{quote_char}{table}{quote_char} a
                    {sql_join} {quote_char}{schema}{quote_char}.{quote_char}{join_table}{quote_char} b
                    ON a.{quote_char}{left_col}{quote_char} = b.{quote_char}{right_col}{quote_char}'''

                # Store join metadata
                join_metadata = {
                    'join_type': join_type,
                    'join_type_display': join_type_display,
                    'source_type': 'sql',
                    'datasource_id': str(datasource.id),
                    'datasource_name': datasource.name,
                    'schema': schema,
                    'left_table': table,
                    'left_column': left_col,
                    'right_table': join_table,
                    'right_column': right_col,
                    'join_columns': join_cols,
                    'joined_at': datetime.now().isoformat()
                }

                # Update description if not provided
                if not data.get("dataset_description"):
                    dataset_description = f"{join_type_display} join of {table} + {join_table} from {datasource.name}/{schema}"

            else:
                # Original single-table query
                column_list = ", ".join(f'{quote_char}{col}{quote_char}' for col in columns) if columns else "*"
                query = f'SELECT {column_list} FROM {quote_char}{schema}{quote_char}.{quote_char}{table}{quote_char}'

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

            # Upload to MinIO
            from core.storage_utils import upload_to_minio
            if join_config:
                filename = f"{schema}_{table}_joined_{join_config['table']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            else:
                filename = f"{schema}_{table}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            minio_key = f"datasets/{request.user.id}/{filename}"
            csv_bytes = csv_content.encode('utf-8')

            upload_to_minio(csv_bytes, minio_key, content_type='text/csv')

            # Create Dataset
            from app.models.main import Dataset
            dataset = Dataset(
                name=dataset_name,
                description=dataset_description,
                created_by=request.user,
                minio_csv_key=minio_key,
                row_count=len(result),
                column_count=len(fieldnames),
                file_size=len(csv_bytes),
                join_metadata=join_metadata if join_metadata else {}
            )

            # Save dataset (this will trigger analysis)
            dataset.save()

            # Build response
            response_data = {
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
            }

            if join_metadata:
                response_data["join_info"] = {
                    "type": join_metadata.get('join_type', 'inner'),
                    "type_display": join_metadata.get('join_type_display', 'Inner'),
                    "left_table": table,
                    "right_table": join_config['table'],
                    "on": f"{table}.{join_config['left_column']} = {join_config['table']}.{join_config['right_column']}"
                }

            return Response(response_data, status=status.HTTP_201_CREATED)

        except Exception as e:
            import logging
            logging.error(f"Export error: {str(e)}", exc_info=True)
            return Response(
                f"Export failed: {str(e)}",
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
