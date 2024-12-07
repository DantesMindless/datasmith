from typing import Any
from django.test import TestCase
from datasource.models import DataSource
from django.contrib.auth import get_user_model

User = get_user_model()


class DataSourceModelTestCase(TestCase):
    def setUp(self) -> None:
        """
        Set up the test case by creating a superuser and a DataSource instance.
        """
        user: Any = User.objects.create_superuser(
            username="admin", email="admin@example.com", password="password"
        )
        DataSource(
            name="Postgres",
            type="POSTGRES",
            description="Test Postgres connection",
            created_by=user,
            user=user,
            credentials={
                "host": "db",
                "port": 5432,
                "database": "postgres",
                "user": "user",
                "password": "datasmith",
            },
        ).save()

    def test_model_count(self) -> None:
        """
        Test that the DataSource model count matches the expected value.
        """
        count: int = DataSource.objects.count()
        self.assertEqual(count, 1)

    def test_model_postgres_adapter(self) -> None:
        """
        Test that the DataSource adapter is not None for a valid model.
        """
        ds: DataSource = DataSource.objects.first()
        self.assertIsNotNone(ds.adapter)

    def test_model_postgres_connection_success(self) -> None:
        """
        Test that the DataSource adapter can successfully establish a connection.
        """
        ds: DataSource = DataSource.objects.first()
        self.assertTrue(ds.adapter.test_connection)
