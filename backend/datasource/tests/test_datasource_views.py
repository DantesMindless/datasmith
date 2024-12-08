from django.test import TestCase
from django.contrib.auth import get_user_model
from datasource.models import DataSource
from django.db.models import Model
import json
from rest_framework.test import APIClient

User = get_user_model()


class DataSourceViewTestCase(TestCase):
    def setUp(self) -> None:
        """
        Set up the test case by creating a superuser and a DataSource instance.
        """
        self.client = APIClient()
        self.user: Model = User.objects.create_superuser(
            username="admin", email="u@u.com", password="password"
        )
        self.user_name = "u@u.com"
        self.password = "password"
        self.client.force_authenticate(user=self.user)

    def test_get_datasource_list(self) -> None:
        for i in range(1, 5):
            DataSource(
                name=f"Postgres{i}",
                type="POSTGRES",
                description="Test Postgres connection",
                user=self.user,
                created_by=self.user,
                credentials={
                    "host": "db",
                    "port": 5432,
                    "database": "postgres",
                    "user": "user",
                    "password": "datasmith",
                },
            ).save()
        """
        Test that the API endpoint returns a list of DataSource instances.
        """
        response = self.client.get("/api/datasource/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()), 4)
        DataSource.objects.all().delete()

    def test_created_datasource(self) -> None:
        data = {
            "name": "Postgres",
            "type": "POSTGRES",
            "description": "Test postgres connection",
            "credentials": {
                "host": "postgres",
                "port": 5432,
                "database": "mydatabase",
                "user": "user",
                "password": "password",
            },
        }
        response = self.client.post(
            "/api/datasource/", data=json.dumps(data), content_type="application/json"
        )
        response_data = response.json()
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response_data["name"], data["name"])

    def test_datasource_wrong_type_create_blocked(self) -> None:
        data = {
            "name": "Postgres",
            "type": "wrong type",
            "description": "Test postgres connection",
            "credentials": {
                "host": "postgres",
                "port": 5432,
                "database": "mydatabase",
                "user": "user",
                "password": "password",
            },
        }
        response = self.client.post(
            "/api/datasource/", data=json.dumps(data), content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
