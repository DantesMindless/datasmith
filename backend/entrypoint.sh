#!/bin/sh
set -e

echo "Updating poetry.lock and installing dependencies..."
poetry lock
poetry install --no-root

if [ "$ENV" = "development" ]; then
    echo "Running in development mode"
    # Migrations are handled manually by the user
    poetry run python manage.py makemigrations
    poetry run python manage.py migrate
    poetry run python manage.py create_superuser
fi

# No command passed → default to Daphne ASGI server (required for WebSocket support)
if [ $# -eq 0 ]; then
    echo "Starting Daphne ASGI server (WebSocket support)"
    exec poetry run daphne -b 0.0.0.0 -p 8000 datasmith.asgi:application
else
    echo "Running custom command: $@"
    exec "$@"
fi
