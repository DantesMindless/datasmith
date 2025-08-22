from django.core.management.base import BaseCommand
from typing import Any, Dict, List


class Command(BaseCommand):
    help = "Create a superuser for the application."

    def handle(self, *args: List[Any], **options: Dict[str, Any]) -> None:
        from django.contrib.auth import get_user_model

        User = get_user_model()
        if not User.objects.filter(username="admin").exists():
            User.objects.create_superuser(
                username="admin", email="admin@datasmith.local", password="superadmin"
            )
            self.stdout.write(self.style.SUCCESS("Superuser created successfully."))
        else:
            self.stdout.write(self.style.WARNING("Superuser already exists."))
