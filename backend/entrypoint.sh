#!/bin/sh
set -e

echo "Updating poetry.lock and installing dependencies..."
poetry lock
poetry install --no-root

if [ "$ENV" = "development" ]; then
    echo "Running in development mode"
    poetry run python manage.py makemigrations
    poetry run python manage.py migrate
    poetry run python manage.py create_superuser
fi

# No command passed → default to Django runserver or Daphne (ASGI for WebSocket support)
if [ $# -eq 0 ]; then
    if [ "$ENV" = "development" ]; then
        echo "Starting Django development server with ASGI/WebSocket support"
        exec poetry run python manage.py runserver 0.0.0.0:8000
    else
        echo "Starting Daphne ASGI server (WebSocket support)"
        exec poetry run daphne -b 0.0.0.0 -p 8000 datasmith.asgi:application
    fi
else
    echo "Running custom command: $@"
    exec "$@"
fi
